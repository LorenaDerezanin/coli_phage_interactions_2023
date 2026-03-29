#!/usr/bin/env python3
"""Reusable enrichment module for annotation-interaction analysis.

Takes any (phage binary feature matrix, host binary feature matrix,
interaction matrix) triple and produces a Fisher's exact test enrichment
table with odds ratios, p-values, and Benjamini-Hochberg corrected
significance.

The enrichment question for each (phage_feature, host_feature) pair is:
  "Are interactions where the phage carries this feature AND the host
   carries this feature enriched for lysis (positive outcome) compared
   to all other interactions?"

The 2x2 contingency table for each pair:

                    host_feature=1    host_feature=0
  phage_feature=1   a (lysis+both)    b (lysis+phage_only)
  phage_feature=0   c (lysis+host_only) d (lysis+neither)

...where a,b,c,d are the number of lytic (positive) interactions in each
cell, and the total interactions in each cell form the denominator.

We use a one-sided Fisher's exact test (alternative='greater') to test
whether the odds of lysis are higher when both features are present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of a single Fisher's exact test for one feature pair."""

    phage_feature: str
    host_feature: str
    a_lysis_both: int
    b_lysis_phage_only: int
    c_lysis_host_only: int
    d_lysis_neither: int
    n_both: int
    n_phage_only: int
    n_host_only: int
    n_neither: int
    odds_ratio: float
    p_value: float
    bh_p_value: float  # Filled after BH correction


def benjamini_hochberg(p_values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Returns adjusted p-values (q-values), maintaining the same order as input.
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH adjustment: q_i = p_i * n / rank_i, then enforce monotonicity
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adjusted = sorted_p * n / ranks

    # Enforce monotonicity: walk backwards, keeping cumulative minimum
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.empty(n, dtype=np.float64)
    result[sorted_indices] = adjusted
    return result


def compute_enrichment(
    phage_matrix: NDArray[np.int8],
    host_matrix: NDArray[np.int8],
    interaction_matrix: NDArray[np.int8],
    phage_feature_names: Sequence[str],
    host_feature_names: Sequence[str],
) -> list[EnrichmentResult]:
    """Compute Fisher's exact test enrichment for all feature pairs.

    Parameters
    ----------
    phage_matrix
        Binary matrix of shape (n_phages, n_phage_features).
        phage_matrix[i, j] == 1 means phage i has feature j.
    host_matrix
        Binary matrix of shape (n_hosts, n_host_features).
        host_matrix[i, k] == 1 means host i has feature k.
    interaction_matrix
        Binary matrix of shape (n_hosts, n_phages).
        interaction_matrix[i, j] == 1 means host i + phage j = lysis.
        interaction_matrix[i, j] == 0 means no lysis.
    phage_feature_names
        Names for the phage feature columns (length n_phage_features).
    host_feature_names
        Names for the host feature columns (length n_host_features).

    Returns
    -------
    List of EnrichmentResult, one per (phage_feature, host_feature) pair,
    with BH-corrected p-values applied across all tests.
    """
    n_hosts, n_phages = interaction_matrix.shape
    n_phage_features = phage_matrix.shape[1]
    n_host_features = host_matrix.shape[1]

    if phage_matrix.shape[0] != n_phages:
        msg = f"phage_matrix rows ({phage_matrix.shape[0]}) != interaction_matrix cols ({n_phages})"
        raise ValueError(msg)
    if host_matrix.shape[0] != n_hosts:
        msg = f"host_matrix rows ({host_matrix.shape[0]}) != interaction_matrix rows ({n_hosts})"
        raise ValueError(msg)
    if len(phage_feature_names) != n_phage_features:
        msg = f"phage_feature_names length ({len(phage_feature_names)}) != phage_matrix cols ({n_phage_features})"
        raise ValueError(msg)
    if len(host_feature_names) != n_host_features:
        msg = f"host_feature_names length ({len(host_feature_names)}) != host_matrix cols ({n_host_features})"
        raise ValueError(msg)

    total_tests = n_phage_features * n_host_features
    logger.info(
        "Running enrichment: %d phage features x %d host features = %d tests",
        n_phage_features,
        n_host_features,
        total_tests,
    )

    results: list[EnrichmentResult] = []
    p_values: list[float] = []

    for pf_idx in range(n_phage_features):
        phage_has = phage_matrix[:, pf_idx]  # (n_phages,)
        phage_lacks = 1 - phage_has

        for hf_idx in range(n_host_features):
            host_has = host_matrix[:, hf_idx]  # (n_hosts,)
            host_lacks = 1 - host_has

            # Build quadrant masks over the interaction matrix
            # interaction_matrix is (n_hosts, n_phages)
            # For each quadrant, count total pairs and lytic pairs
            # Using outer products to create (n_hosts, n_phages) masks
            mask_both = np.outer(host_has, phage_has)
            mask_phage_only = np.outer(host_lacks, phage_has)
            mask_host_only = np.outer(host_has, phage_lacks)
            mask_neither = np.outer(host_lacks, phage_lacks)

            n_both = int(mask_both.sum())
            n_phage_only = int(mask_phage_only.sum())
            n_host_only = int(mask_host_only.sum())
            n_neither = int(mask_neither.sum())

            a = int((interaction_matrix * mask_both).sum())
            b = int((interaction_matrix * mask_phage_only).sum())
            c = int((interaction_matrix * mask_host_only).sum())
            d = int((interaction_matrix * mask_neither).sum())

            # 2x2 contingency table for Fisher's test, conditioned on the
            # phage having the feature (controls for phage main effect):
            #
            #              lysis         no_lysis
            # host_has:    a             n_both - a
            # host_lacks:  b             n_phage_only - b
            #
            # This asks: among interactions where the phage carries this
            # PHROG, does the host carrying this receptor increase the
            # probability of lysis? A naive "both vs everything else"
            # table would conflate the phage main effect with the
            # interaction effect.
            table = np.array(
                [
                    [a, n_both - a],
                    [b, n_phage_only - b],
                ]
            )
            odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

            results.append(
                EnrichmentResult(
                    phage_feature=phage_feature_names[pf_idx],
                    host_feature=host_feature_names[hf_idx],
                    a_lysis_both=a,
                    b_lysis_phage_only=b,
                    c_lysis_host_only=c,
                    d_lysis_neither=d,
                    n_both=n_both,
                    n_phage_only=n_phage_only,
                    n_host_only=n_host_only,
                    n_neither=n_neither,
                    odds_ratio=odds_ratio,
                    p_value=p_value,
                    bh_p_value=0.0,  # placeholder
                )
            )
            p_values.append(p_value)

    # Apply BH correction
    bh_adjusted = benjamini_hochberg(np.array(p_values, dtype=np.float64))

    corrected_results = []
    for result, bh_p in zip(results, bh_adjusted):
        corrected_results.append(
            EnrichmentResult(
                phage_feature=result.phage_feature,
                host_feature=result.host_feature,
                a_lysis_both=result.a_lysis_both,
                b_lysis_phage_only=result.b_lysis_phage_only,
                c_lysis_host_only=result.c_lysis_host_only,
                d_lysis_neither=result.d_lysis_neither,
                n_both=result.n_both,
                n_phage_only=result.n_phage_only,
                n_host_only=result.n_host_only,
                n_neither=result.n_neither,
                odds_ratio=result.odds_ratio,
                p_value=result.p_value,
                bh_p_value=float(bh_p),
            )
        )

    n_significant = sum(1 for r in corrected_results if r.bh_p_value < 0.05)
    logger.info(
        "Enrichment complete: %d tests, %d significant (BH p < 0.05)",
        total_tests,
        n_significant,
    )

    return corrected_results


def results_to_rows(results: list[EnrichmentResult]) -> list[dict[str, object]]:
    """Convert enrichment results to a list of dicts suitable for CSV output."""
    rows = []
    for r in results:
        lysis_rate_both = r.a_lysis_both / r.n_both if r.n_both > 0 else 0.0
        n_other = r.n_phage_only + r.n_host_only + r.n_neither
        lysis_other = r.b_lysis_phage_only + r.c_lysis_host_only + r.d_lysis_neither
        lysis_rate_other = lysis_other / n_other if n_other > 0 else 0.0
        rows.append(
            {
                "phage_feature": r.phage_feature,
                "host_feature": r.host_feature,
                "n_both": r.n_both,
                "lysis_both": r.a_lysis_both,
                "lysis_rate_both": round(lysis_rate_both, 4),
                "n_other": n_other,
                "lysis_other": lysis_other,
                "lysis_rate_other": round(lysis_rate_other, 4),
                "odds_ratio": round(r.odds_ratio, 4) if np.isfinite(r.odds_ratio) else "inf",
                "p_value": r.p_value,
                "bh_p_value": r.bh_p_value,
                "significant": r.bh_p_value < 0.05,
            }
        )
    return rows


ENRICHMENT_CSV_FIELDNAMES: list[str] = [
    "phage_feature",
    "host_feature",
    "n_both",
    "lysis_both",
    "lysis_rate_both",
    "n_other",
    "lysis_other",
    "lysis_rate_other",
    "odds_ratio",
    "p_value",
    "bh_p_value",
    "significant",
]
