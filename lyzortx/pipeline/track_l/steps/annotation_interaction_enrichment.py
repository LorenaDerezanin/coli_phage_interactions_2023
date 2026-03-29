#!/usr/bin/env python3
"""Reusable enrichment module for annotation-interaction analysis.

Takes any (phage binary feature matrix, host binary feature matrix,
interaction matrix) triple and produces an enrichment table with odds
ratios, permutation-based p-values, and Benjamini-Hochberg corrected
significance.

The enrichment question for each (phage_feature, host_feature) pair is:
  "Among interactions where the phage carries this PHROG, does the host
   carrying this receptor increase the probability of lysis?"

We condition on the phage feature (controlling for the phage main effect)
and use a permutation test on host labels (controlling for the correlation
structure in the interaction matrix). Fisher's exact test is anticonservative
here because interaction matrix entries are not independent — susceptible
hosts and broad-range phages create row/column correlations that inflate
the effective sample size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

N_PERMUTATIONS = 1000
RANDOM_SEED = 42


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of a single enrichment test for one feature pair."""

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


def _lysis_rate_diff(
    interaction_matrix: NDArray[np.int8],
    host_has: NDArray[np.int8],
    phage_has: NDArray[np.int8],
    resolved_mask: NDArray[np.int8] | None = None,
) -> float:
    """Compute lysis rate difference: host_has - host_lacks, conditioned on phage_has.

    If resolved_mask is provided, only count pairs where resolved_mask[i,j] == 1.
    """
    mask_both = np.outer(host_has, phage_has)
    mask_phage_only = np.outer(1 - host_has, phage_has)
    if resolved_mask is not None:
        mask_both = mask_both * resolved_mask
        mask_phage_only = mask_phage_only * resolved_mask
    n_both = mask_both.sum()
    n_phage_only = mask_phage_only.sum()
    if n_both == 0 or n_phage_only == 0:
        return 0.0
    a = (interaction_matrix * mask_both).sum()
    b = (interaction_matrix * mask_phage_only).sum()
    return float(a / n_both - b / n_phage_only)


def _odds_ratio(a: int, b: int, n_both: int, n_phage_only: int) -> float:
    """Compute odds ratio for the conditional 2x2 table."""
    # Table: [[a, n_both - a], [b, n_phage_only - b]]
    c0 = n_both - a
    d0 = n_phage_only - b
    if c0 == 0 or b == 0:
        return float("inf") if a > 0 and d0 > 0 else 1.0
    if a == 0 or d0 == 0:
        return 0.0
    return (a * d0) / (b * c0)


def compute_enrichment(
    phage_matrix: NDArray[np.int8],
    host_matrix: NDArray[np.int8],
    interaction_matrix: NDArray[np.int8],
    phage_feature_names: Sequence[str],
    host_feature_names: Sequence[str],
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
    resolved_mask: NDArray[np.int8] | None = None,
) -> list[EnrichmentResult]:
    """Compute permutation-based enrichment for all feature pairs.

    For each (phage_feature, host_feature) pair, conditions on the phage
    having the feature and tests whether the host feature increases lysis.
    P-values are computed by permuting host labels, which preserves the
    correlation structure in the interaction matrix.

    Parameters
    ----------
    phage_matrix
        Binary matrix of shape (n_phages, n_phage_features).
    host_matrix
        Binary matrix of shape (n_hosts, n_host_features).
    interaction_matrix
        Binary matrix of shape (n_hosts, n_phages). 1=lysis, 0=no lysis.
    phage_feature_names
        Names for the phage feature columns.
    host_feature_names
        Names for the host feature columns.
    n_permutations
        Number of permutations for p-value computation.
    random_seed
        Seed for reproducible permutations.
    resolved_mask
        Optional binary matrix of shape (n_hosts, n_phages). 1=resolved
        label, 0=unresolved. If provided, unresolved pairs are excluded
        from all counts and statistics.

    Returns
    -------
    List of EnrichmentResult with BH-corrected p-values.
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
        "Running enrichment: %d phage features x %d host features = %d tests, %d permutations",
        n_phage_features,
        n_host_features,
        total_tests,
        n_permutations,
    )

    rng = np.random.RandomState(random_seed)
    results: list[EnrichmentResult] = []
    p_values: list[float] = []

    for pf_idx in range(n_phage_features):
        phage_has = phage_matrix[:, pf_idx]

        for hf_idx in range(n_host_features):
            host_has = host_matrix[:, hf_idx]
            host_lacks = 1 - host_has

            # Observed statistics (applying resolved_mask if provided)
            mask_both = np.outer(host_has, phage_has)
            mask_phage_only = np.outer(host_lacks, phage_has)
            mask_host_only = np.outer(host_has, 1 - phage_has)
            mask_neither = np.outer(host_lacks, 1 - phage_has)

            if resolved_mask is not None:
                mask_both = mask_both * resolved_mask
                mask_phage_only = mask_phage_only * resolved_mask
                mask_host_only = mask_host_only * resolved_mask
                mask_neither = mask_neither * resolved_mask

            n_both = int(mask_both.sum())
            n_phage_only = int(mask_phage_only.sum())
            n_host_only = int(mask_host_only.sum())
            n_neither = int(mask_neither.sum())

            a = int((interaction_matrix * mask_both).sum())
            b = int((interaction_matrix * mask_phage_only).sum())
            c = int((interaction_matrix * mask_host_only).sum())
            d = int((interaction_matrix * mask_neither).sum())

            obs_stat = _lysis_rate_diff(interaction_matrix, host_has, phage_has, resolved_mask)

            # Permutation p-value: shuffle host labels
            count_ge = 0
            for _ in range(n_permutations):
                host_perm = host_has[rng.permutation(n_hosts)]
                perm_stat = _lysis_rate_diff(interaction_matrix, host_perm, phage_has, resolved_mask)
                if perm_stat >= obs_stat:
                    count_ge += 1

            p_value = (count_ge + 1) / (n_permutations + 1)
            or_val = _odds_ratio(a, b, n_both, n_phage_only)

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
                    odds_ratio=or_val,
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
    """Convert enrichment results to a list of dicts suitable for CSV output.

    lysis_rate_both and lysis_rate_phage_only are the two rates from the
    conditional test (conditioned on phage having the feature). lysis_rate_diff
    is the test statistic (host_has rate - host_lacks rate).
    """
    rows = []
    for r in results:
        lysis_rate_both = r.a_lysis_both / r.n_both if r.n_both > 0 else 0.0
        lysis_rate_phage_only = r.b_lysis_phage_only / r.n_phage_only if r.n_phage_only > 0 else 0.0
        lysis_rate_diff = lysis_rate_both - lysis_rate_phage_only
        rows.append(
            {
                "phage_feature": r.phage_feature,
                "host_feature": r.host_feature,
                "n_both": r.n_both,
                "lysis_both": r.a_lysis_both,
                "lysis_rate_both": round(lysis_rate_both, 4),
                "n_phage_only": r.n_phage_only,
                "lysis_phage_only": r.b_lysis_phage_only,
                "lysis_rate_phage_only": round(lysis_rate_phage_only, 4),
                "lysis_rate_diff": round(lysis_rate_diff, 4),
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
    "n_phage_only",
    "lysis_phage_only",
    "lysis_rate_phage_only",
    "lysis_rate_diff",
    "odds_ratio",
    "p_value",
    "bh_p_value",
    "significant",
]
