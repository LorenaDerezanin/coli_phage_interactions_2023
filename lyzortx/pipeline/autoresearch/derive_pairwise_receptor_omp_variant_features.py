"""Derive directed receptor × OMP allele-variant cross-terms (GT07).

Replaces the near-constant whole-gene HMM scores (CV 0.01-0.17) with binary
OMP allele cluster features from Track C (99% identity BLAST clustering).
These variant features have variance 0.08-0.25 and measurable discriminative
power (e.g., BtuB cluster 99_15: Cohen's d=0.455 for lysed vs non-lysed).

Cross-term pattern: predicted_receptor_is_X × host_X_variant_cluster_Y.
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

PAIRWISE_PREFIX = "pair_receptor_omp_variant__"

# Maps k-mer receptor short names to the prefix used in Track C variant feature columns.
RECEPTOR_TO_VARIANT_PREFIX = {
    "btub": "host_omp_receptor_btub_cluster_",
    "fadL": "host_omp_receptor_fadl_cluster_",
    "fhua": "host_omp_receptor_fhua_cluster_",
    "lamB": "host_omp_receptor_lamb_cluster_",
    "lptD": "host_omp_receptor_lptd_cluster_",
    "nfrA": "host_omp_receptor_nfra_cluster_",
    "ompA": "host_omp_receptor_ompa_cluster_",
    "ompC": "host_omp_receptor_ompc_cluster_",
    "ompF": "host_omp_receptor_ompf_cluster_",
    "tolC": "host_omp_receptor_tolc_cluster_",
    "tsx": "host_omp_receptor_tsx_cluster_",
    "yncD": "host_omp_receptor_yncd_cluster_",
}

DEFAULT_VARIANT_FEATURES_PATH = Path(
    "lyzortx/generated_outputs/track_c/omp_receptor_variant_feature_block/host_omp_receptor_variant_features_v1.csv"
)
DEFAULT_PROTEOME_PATH = Path(
    "lyzortx/generated_outputs/autoresearch/phage_projection_cache_build/_batched/combined_queries.faa"
)
DEFAULT_DATASET_PATH = Path(".scratch/genophi/Supplementary_Datasets_S1-S7.xlsx")


def compute_pairwise_receptor_omp_variant_features(
    design: pd.DataFrame,
    *,
    variant_features_path: Path | None = None,
    proteome_path: Path | None = None,
    dataset_path: Path | None = None,
) -> list[str]:
    """Add receptor × OMP allele-variant directed cross-terms to the design matrix.

    For each phage with a k-mer receptor prediction targeting an OMP receptor,
    creates cross-terms: predicted_receptor_is_X × host_X_variant_cluster_Y
    for each binary variant cluster of that receptor.
    """
    if variant_features_path is None:
        variant_features_path = DEFAULT_VARIANT_FEATURES_PATH
    if proteome_path is None:
        proteome_path = DEFAULT_PROTEOME_PATH
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH

    if not variant_features_path.exists():
        raise FileNotFoundError(f"OMP variant features not found at {variant_features_path}")
    if not proteome_path.exists():
        raise FileNotFoundError(f"Phage proteome not found at {proteome_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"GenoPHI Dataset S6 not found at {dataset_path}")
    if "phage" not in design.columns:
        raise ValueError("Design matrix must have a 'phage' column.")
    if "bacteria" not in design.columns:
        raise ValueError("Design matrix must have a 'bacteria' column.")

    # Load OMP variant features (binary cluster assignments per host).
    variant_df = pd.read_csv(variant_features_path)
    variant_df = variant_df.set_index("bacteria")
    variant_cols = [c for c in variant_df.columns if c.startswith("host_omp_receptor_")]
    LOGGER.info("Loaded %d OMP variant features for %d bacteria", len(variant_cols), len(variant_df))

    # Load k-mer receptor predictions.
    predictions = predict_receptors(proteome_path, dataset_path)
    phage_pred: dict[str, ReceptorPrediction] = {p.phage: p for p in predictions}

    phage_series = design["phage"].astype(str)
    bacteria_series = design["bacteria"].astype(str)
    added_columns: list[str] = []

    # Merge variant features into design by bacteria.
    bacteria_in_variants = set(variant_df.index)
    for vcol in variant_cols:
        mapped = bacteria_series.map(lambda b, col=vcol: variant_df.loc[b, col] if b in bacteria_in_variants else 0.0)
        design[vcol] = mapped.astype(float)

    # For each OMP receptor, create directed cross-terms.
    for receptor_short, variant_prefix in RECEPTOR_TO_VARIANT_PREFIX.items():
        # Find variant columns for this receptor.
        receptor_variant_cols = [c for c in variant_cols if c.startswith(variant_prefix)]
        if not receptor_variant_cols:
            continue

        # predicted_is_X indicator.
        predicted_col = f"{PAIRWISE_PREFIX}predicted_is_{receptor_short}"
        design[predicted_col] = phage_series.map(
            lambda p, target=receptor_short: (
                1.0
                if p in phage_pred
                and phage_pred[p].receptor_type == "omp"
                and phage_pred[p].predicted_receptor == target
                else 0.0
            )
        )
        added_columns.append(predicted_col)

        # Cross-terms: predicted_is_X × host_X_variant_cluster_Y.
        for vcol in receptor_variant_cols:
            cluster_id = vcol.replace(variant_prefix, "")
            cross_col = f"{PAIRWISE_PREFIX}predicted_{receptor_short}_x_{cluster_id}"
            design[cross_col] = design[predicted_col] * design[vcol]
            added_columns.append(cross_col)

    # LPS indicator and confidence (reuse from kmer features pattern).
    lps_col = f"{PAIRWISE_PREFIX}predicted_is_lps"
    design[lps_col] = phage_series.map(
        lambda p: 1.0 if p in phage_pred and phage_pred[p].receptor_type == "lps" else 0.0
    )
    added_columns.append(lps_col)

    conf_col = f"{PAIRWISE_PREFIX}prediction_confidence"
    design[conf_col] = phage_series.map(lambda p: phage_pred[p].confidence if p in phage_pred else 0.0)
    added_columns.append(conf_col)

    # Log summary.
    omp_indicators = [c for c in added_columns if c.startswith(f"{PAIRWISE_PREFIX}predicted_is_") and c != lps_col]
    cross_terms = [c for c in added_columns if "_x_" in c]
    omp_count = sum(1 for p in phage_pred.values() if p.receptor_type == "omp")
    LOGGER.info(
        "Added %d OMP variant cross-term features (%d indicators + %d cross-terms + 2 global; %d OMP phages)",
        len(added_columns),
        len(omp_indicators),
        len(cross_terms),
        omp_count,
    )

    # Clean up: remove the raw variant columns from design (they're host-only, not pairwise).
    for vcol in variant_cols:
        if vcol in design.columns:
            design.drop(columns=[vcol], inplace=True)

    return added_columns
