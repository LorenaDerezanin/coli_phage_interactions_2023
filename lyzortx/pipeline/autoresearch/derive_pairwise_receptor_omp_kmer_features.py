"""Derive directed receptor × OMP cross-terms using GenoPHI k-mer predictions (GT06).

Replaces the genus-level consensus mapping (8/96 OMP phages) with per-phage
k-mer-based receptor predictions (39/96 OMP phages). Uses the same directed
cross-term pattern: predicted_receptor_is_X × host_X_score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from lyzortx.pipeline.autoresearch.predict_receptor_from_kmers import (
    ReceptorPrediction,
    predict_receptors,
)

LOGGER = logging.getLogger(__name__)

PAIRWISE_PREFIX = "pair_receptor_omp_kmer__"

# OMP receptor columns in the host_surface slot.
OMP_RECEPTOR_COLUMNS = {
    "btub": "host_surface__host_receptor_btub_score",
    "fadL": "host_surface__host_receptor_fadL_score",
    "fhua": "host_surface__host_receptor_fhua_score",
    "lamB": "host_surface__host_receptor_lamB_score",
    "lptD": "host_surface__host_receptor_lptD_score",
    "nfrA": "host_surface__host_receptor_nfrA_score",
    "ompA": "host_surface__host_receptor_ompA_score",
    "ompC": "host_surface__host_receptor_ompC_score",
    "ompF": "host_surface__host_receptor_ompF_score",
    "tolC": "host_surface__host_receptor_tolC_score",
    "tsx": "host_surface__host_receptor_tsx_score",
    "yncD": "host_surface__host_receptor_yncD_score",
}

O_ANTIGEN_SCORE_COLUMN = "host_surface__host_o_antigen_score"

DEFAULT_PROTEOME_PATH = Path(
    "lyzortx/generated_outputs/autoresearch/phage_projection_cache_build/_batched/combined_queries.faa"
)
DEFAULT_DATASET_PATH = Path(".scratch/genophi/Supplementary_Datasets_S1-S7.xlsx")


def compute_pairwise_receptor_omp_kmer_features(
    design: pd.DataFrame,
    *,
    proteome_path: Path | None = None,
    dataset_path: Path | None = None,
) -> list[str]:
    """Add k-mer-based directed receptor × OMP cross-terms to the design matrix.

    Same feature pattern as derive_pairwise_receptor_omp_features.py but using
    per-phage k-mer predictions instead of genus-level consensus.
    """
    if proteome_path is None:
        proteome_path = DEFAULT_PROTEOME_PATH
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH

    if not proteome_path.exists():
        raise FileNotFoundError(f"Phage proteome not found at {proteome_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"GenoPHI Dataset S6 not found at {dataset_path}")
    if "phage" not in design.columns:
        raise ValueError("Design matrix must have a 'phage' column.")

    predictions = predict_receptors(proteome_path, dataset_path)
    phage_pred: dict[str, ReceptorPrediction] = {p.phage: p for p in predictions}

    phage_series = design["phage"].astype(str)
    added_columns: list[str] = []

    # has_receptor_assignment (any type).
    has_col = f"{PAIRWISE_PREFIX}has_receptor_assignment"
    design[has_col] = phage_series.map(
        lambda p: 1.0 if p in phage_pred and phage_pred[p].receptor_type != "unknown" else 0.0
    )
    added_columns.append(has_col)

    # Per-OMP receptor: predicted_is_X + directed cross-term.
    for omp_short, host_col in OMP_RECEPTOR_COLUMNS.items():
        predicted_col = f"{PAIRWISE_PREFIX}predicted_is_{omp_short}"
        design[predicted_col] = phage_series.map(
            lambda p, target=omp_short: (
                1.0
                if p in phage_pred
                and phage_pred[p].receptor_type == "omp"
                and phage_pred[p].predicted_receptor == target
                else 0.0
            )
        )
        added_columns.append(predicted_col)

        if host_col in design.columns:
            cross_col = f"{PAIRWISE_PREFIX}predicted_{omp_short}_x_host_{omp_short}"
            host_score = pd.to_numeric(design[host_col], errors="coerce").fillna(0.0)
            design[cross_col] = design[predicted_col] * host_score
            added_columns.append(cross_col)

    # LPS cross-term.
    lps_col = f"{PAIRWISE_PREFIX}predicted_is_lps"
    design[lps_col] = phage_series.map(
        lambda p: 1.0 if p in phage_pred and phage_pred[p].receptor_type == "lps" else 0.0
    )
    added_columns.append(lps_col)

    if O_ANTIGEN_SCORE_COLUMN in design.columns:
        cross_lps = f"{PAIRWISE_PREFIX}predicted_lps_x_host_o_antigen"
        design[cross_lps] = design[lps_col] * pd.to_numeric(design[O_ANTIGEN_SCORE_COLUMN], errors="coerce").fillna(0.0)
        added_columns.append(cross_lps)

    # Confidence as a continuous feature.
    conf_col = f"{PAIRWISE_PREFIX}prediction_confidence"
    design[conf_col] = phage_series.map(lambda p: phage_pred[p].confidence if p in phage_pred else 0.0)
    added_columns.append(conf_col)

    omp_cols = [f"{PAIRWISE_PREFIX}predicted_is_{r}" for r in OMP_RECEPTOR_COLUMNS]
    omp_count = design.loc[design[omp_cols].max(axis=1) == 1.0, "phage"].nunique()
    lps_count = design.loc[design[lps_col] == 1.0, "phage"].nunique()
    LOGGER.info(
        "Added %d k-mer-based receptor × OMP features (%d OMP phages, %d LPS phages)",
        len(added_columns),
        omp_count,
        lps_count,
    )
    return added_columns
