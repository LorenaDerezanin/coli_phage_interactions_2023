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

    # Pre-generate permutation indices (shared across all feature pairs)
    perm_indices = np.array([rng.permutation(n_hosts) for _ in range(n_permutations)])

    effective_mask = resolved_mask if resolved_mask is not None else np.ones_like(interaction_matrix)

    for pf_idx in range(n_phage_features):
        phage_has = phage_matrix[:, pf_idx]
        phage_lacks = 1 - phage_has

        # Precompute per-host aggregates for this phage feature (vectorized over phage columns).
        # lysis_per_host_has[i] = number of lytic resolved pairs for host i among phages with the feature
        # resolved_per_host_has[i] = number of resolved pairs for host i among phages with the feature
        lysis_per_host_has = (
            (interaction_matrix * (phage_has[np.newaxis, :] * effective_mask)).sum(axis=1).astype(np.float64)
        )
        resolved_per_host_has = (phage_has[np.newaxis, :] * effective_mask).sum(axis=1).astype(np.float64)
        lysis_per_host_lacks = (
            (interaction_matrix * (phage_lacks[np.newaxis, :] * effective_mask)).sum(axis=1).astype(np.float64)
        )
        resolved_per_host_lacks = (phage_lacks[np.newaxis, :] * effective_mask).sum(axis=1).astype(np.float64)

        for hf_idx in range(n_host_features):
            host_has_f = host_matrix[:, hf_idx].astype(np.float64)
            host_lacks_f = 1.0 - host_has_f

            # Observed contingency counts via dot products
            n_both = int(host_has_f @ resolved_per_host_has)
            n_phage_only = int(host_lacks_f @ resolved_per_host_has)
            n_host_only = int(host_has_f @ resolved_per_host_lacks)
            n_neither = int(host_lacks_f @ resolved_per_host_lacks)

            a = int(host_has_f @ lysis_per_host_has)
            b = int(host_lacks_f @ lysis_per_host_has)
            c = int(host_has_f @ lysis_per_host_lacks)
            d = int(host_lacks_f @ lysis_per_host_lacks)

            # Observed test statistic
            rate_both = a / n_both if n_both > 0 else 0.0
            rate_phage_only = b / n_phage_only if n_phage_only > 0 else 0.0
            obs_stat = rate_both - rate_phage_only

            # Vectorized permutation p-value: permute host labels in batch
            host_perms = host_has_f[perm_indices]  # (n_permutations, n_hosts)
            host_lacks_perms = 1.0 - host_perms

            perm_n_both = host_perms @ resolved_per_host_has  # (n_permutations,)
            perm_n_po = host_lacks_perms @ resolved_per_host_has
            perm_a = host_perms @ lysis_per_host_has
            perm_b = host_lacks_perms @ lysis_per_host_has

            perm_rate_both = np.where(perm_n_both > 0, perm_a / perm_n_both, 0.0)
            perm_rate_po = np.where(perm_n_po > 0, perm_b / perm_n_po, 0.0)
            perm_stats = perm_rate_both - perm_rate_po

            count_ge = int((perm_stats >= obs_stat).sum())
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
