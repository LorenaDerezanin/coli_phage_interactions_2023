"""Derive label-free pairwise RBP × receptor interaction features.

For each (phage, host) pair, computes cross-term features between phage RBP
presence/count and host receptor HMM scores. These capture the physical
interaction between phage adsorption machinery and host surface receptors
without using any training labels.

The 12 host receptor scores come from phmmer against known E. coli outer
membrane proteins (from the host_surface slot). The phage RBP features come
from Pharokka annotation-based detection (from the phage_rbp_struct slot).
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Host receptor score columns from the host_surface slot (12 OMP receptors).
RECEPTOR_SCORE_COLUMNS = (
    "host_surface__host_receptor_btub_score",
    "host_surface__host_receptor_fadL_score",
    "host_surface__host_receptor_fhua_score",
    "host_surface__host_receptor_lamB_score",
    "host_surface__host_receptor_lptD_score",
    "host_surface__host_receptor_nfrA_score",
    "host_surface__host_receptor_ompA_score",
    "host_surface__host_receptor_ompC_score",
    "host_surface__host_receptor_ompF_score",
    "host_surface__host_receptor_tolC_score",
    "host_surface__host_receptor_tsx_score",
    "host_surface__host_receptor_yncD_score",
)

# Phage RBP columns from the phage_rbp_struct slot.
RBP_PRESENT_COLUMN = "phage_rbp_struct__has_annotated_rbp"
RBP_COUNT_COLUMN = "phage_rbp_struct__rbp_count"

# Prefix for pairwise features (must be added to SLOT_PREFIXES in train.py).
PAIRWISE_PREFIX = "pair_rbp_receptor__"


def _receptor_short_name(receptor_col: str) -> str:
    """Extract short receptor name from column: 'host_surface__host_receptor_btub_score' -> 'btub'."""
    return receptor_col.split("host_receptor_")[1].replace("_score", "")


def pairwise_feature_names() -> list[str]:
    """Return the ordered list of pairwise feature column names."""
    names: list[str] = []
    for receptor_col in RECEPTOR_SCORE_COLUMNS:
        short = _receptor_short_name(receptor_col)
        names.append(f"{PAIRWISE_PREFIX}has_rbp_x_{short}")
        names.append(f"{PAIRWISE_PREFIX}rbp_count_x_{short}")
    return names


def compute_pairwise_rbp_receptor_features(design: pd.DataFrame) -> list[str]:
    """Add pairwise RBP × receptor cross-term columns to the design matrix in-place.

    Requires both phage_rbp_struct and host_surface features in the design matrix.
    Returns the list of added feature column names.

    Cross-terms:
    - has_rbp × receptor_score: Does this phage have any RBP, interacted with receptor presence.
    - rbp_count × receptor_score: RBP count weighted by receptor score.
    """
    if RBP_PRESENT_COLUMN not in design.columns:
        raise ValueError(
            f"Pairwise RBP-receptor features require {RBP_PRESENT_COLUMN} (enable --include-phage-rbp-struct)."
        )

    rbp_present = pd.to_numeric(design[RBP_PRESENT_COLUMN], errors="coerce").fillna(0.0)
    rbp_count = pd.to_numeric(design[RBP_COUNT_COLUMN], errors="coerce").fillna(0.0)

    added_columns: list[str] = []
    available_receptors = [col for col in RECEPTOR_SCORE_COLUMNS if col in design.columns]
    if not available_receptors:
        raise ValueError("Pairwise RBP-receptor features require host receptor score columns in the design matrix.")

    for receptor_col in available_receptors:
        short = _receptor_short_name(receptor_col)
        receptor_score = pd.to_numeric(design[receptor_col], errors="coerce").fillna(0.0)

        has_rbp_col = f"{PAIRWISE_PREFIX}has_rbp_x_{short}"
        count_col = f"{PAIRWISE_PREFIX}rbp_count_x_{short}"

        design[has_rbp_col] = rbp_present * receptor_score
        design[count_col] = rbp_count * receptor_score

        added_columns.append(has_rbp_col)
        added_columns.append(count_col)

    LOGGER.info(
        "Added %d pairwise RBP × receptor features (%d receptors)",
        len(added_columns),
        len(available_receptors),
    )
    return added_columns
