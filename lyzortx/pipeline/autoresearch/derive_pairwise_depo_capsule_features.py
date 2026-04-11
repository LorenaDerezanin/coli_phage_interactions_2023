"""Derive pairwise depolymerase × capsule cross-term features (GT01).

For each (phage, host) pair, computes cross-terms between phage depolymerase
presence/count and host capsule HMM profile scores. Also includes per-cluster
membership features that enable LightGBM to learn cluster-specific capsule
interactions via tree splits.

Depolymerase data comes from DepoScope predictions (.scratch/deposcope/),
host capsule scores come from the host_surface slot already in the design matrix.

Feature groups:
- Phage-level: has_depo, depo_count, depo_cluster_count (3 features)
- Cluster membership: in_cluster_<N> binary indicators (41 features)
- Cross-terms: has_depo × capsule_score, depo_count × capsule_score (198 features)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

PAIRWISE_PREFIX = "pair_depo_capsule__"

# Pattern for host capsule columns in the host_surface slot.
CAPSULE_COLUMN_PREFIX = "host_surface__host_capsule_profile_"
CAPSULE_COLUMN_SUFFIX = "_score"


def _capsule_short_name(col: str) -> str:
    """Extract short capsule name: 'host_surface__host_capsule_profile_cluster_19_score' -> 'cluster_19'."""
    after_prefix = col[len(CAPSULE_COLUMN_PREFIX) :]
    return after_prefix[: -len(CAPSULE_COLUMN_SUFFIX)]


def load_deposcope_predictions(predictions_path: Path) -> dict[str, int]:
    """Load DepoScope predictions and return phage -> depolymerase count."""
    phage_depo_count: dict[str, int] = defaultdict(int)
    with open(predictions_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_depolymerase"] == "True":
                phage_depo_count[row["phage"]] += 1
    return dict(phage_depo_count)


def load_deposcope_clusters(cluster_path: Path) -> dict[str, set[str]]:
    """Load DepoScope cluster assignments.

    Returns phage -> set of cluster representative IDs.
    """
    phage_clusters: dict[str, set[str]] = defaultdict(set)
    with open(cluster_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep_id = parts[0]
            member_id = parts[1]
            # Extract phage name from compound ID: "409_P1|409_P1|prot_0043|score=1.0000"
            member_phage = member_id.split("|")[0]
            phage_clusters[member_phage].add(rep_id)
    return dict(phage_clusters)


def build_cluster_index(phage_clusters: dict[str, set[str]]) -> list[str]:
    """Build a deterministic ordered list of cluster representative IDs.

    Sorted alphabetically for reproducibility.
    """
    all_reps: set[str] = set()
    for reps in phage_clusters.values():
        all_reps.update(reps)
    return sorted(all_reps)


def build_phage_depo_lookup(
    predictions_path: Path,
    cluster_path: Path,
) -> tuple[dict[str, int], dict[str, set[str]], list[str]]:
    """Load DepoScope data and return (depo_counts, phage_clusters, cluster_index)."""
    depo_counts = load_deposcope_predictions(predictions_path)
    phage_clusters = load_deposcope_clusters(cluster_path)
    cluster_index = build_cluster_index(phage_clusters)
    LOGGER.info(
        "DepoScope: %d phages with depolymerases, %d clusters",
        len(depo_counts),
        len(cluster_index),
    )
    return depo_counts, phage_clusters, cluster_index


def compute_pairwise_depo_capsule_features(
    design: pd.DataFrame,
    *,
    deposcope_dir: Path | None = None,
) -> list[str]:
    """Add depolymerase × capsule cross-term columns to the design matrix in-place.

    Parameters
    ----------
    design : pd.DataFrame
        Pair-level design matrix with 'phage' column and host_surface capsule columns.
    deposcope_dir : Path, optional
        Directory containing DepoScope outputs (predictions.csv, depo_clusters_cluster.tsv).
        Defaults to .scratch/deposcope/ relative to the working directory.

    Returns
    -------
    list[str]
        The names of added feature columns.
    """
    if deposcope_dir is None:
        deposcope_dir = Path(".scratch/deposcope")

    predictions_path = deposcope_dir / "predictions.csv"
    cluster_path = deposcope_dir / "depo_clusters_cluster.tsv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"DepoScope predictions not found at {predictions_path}. Run DepoScope first (see GT01 task)."
        )
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"DepoScope cluster file not found at {cluster_path}. Run DepoScope clustering first (see GT01 task)."
        )

    if "phage" not in design.columns:
        raise ValueError("Design matrix must have a 'phage' column.")

    depo_counts, phage_clusters, cluster_index = build_phage_depo_lookup(predictions_path, cluster_path)

    # Find capsule columns in the design matrix.
    capsule_columns = [
        col for col in design.columns if col.startswith(CAPSULE_COLUMN_PREFIX) and col.endswith(CAPSULE_COLUMN_SUFFIX)
    ]
    if not capsule_columns:
        raise ValueError(
            "No host capsule profile columns found in design matrix. Ensure host_surface slot is included."
        )

    added_columns: list[str] = []
    phage_series = design["phage"].astype(str)

    # --- Phage-level features ---
    has_depo_col = f"{PAIRWISE_PREFIX}has_depo"
    depo_count_col = f"{PAIRWISE_PREFIX}depo_count"
    cluster_count_col = f"{PAIRWISE_PREFIX}depo_cluster_count"

    has_depo = phage_series.map(lambda p: 1.0 if p in depo_counts else 0.0)
    depo_count = phage_series.map(lambda p: float(depo_counts.get(p, 0)))
    cluster_count = phage_series.map(lambda p: float(len(phage_clusters.get(p, set()))))

    design[has_depo_col] = has_depo
    design[depo_count_col] = depo_count
    design[cluster_count_col] = cluster_count
    added_columns.extend([has_depo_col, depo_count_col, cluster_count_col])

    # --- Per-cluster membership features ---
    for i, rep_id in enumerate(cluster_index):
        col_name = f"{PAIRWISE_PREFIX}in_cluster_{i}"
        design[col_name] = phage_series.map(lambda p, r=rep_id: 1.0 if r in phage_clusters.get(p, set()) else 0.0)
        added_columns.append(col_name)

    # --- Cross-terms: has_depo × capsule_score, depo_count × capsule_score ---
    for capsule_col in capsule_columns:
        short = _capsule_short_name(capsule_col)
        capsule_score = pd.to_numeric(design[capsule_col], errors="coerce").fillna(0.0)

        has_depo_x_col = f"{PAIRWISE_PREFIX}has_depo_x_{short}"
        count_x_col = f"{PAIRWISE_PREFIX}depo_count_x_{short}"

        design[has_depo_x_col] = has_depo * capsule_score
        design[count_x_col] = depo_count * capsule_score

        added_columns.append(has_depo_x_col)
        added_columns.append(count_x_col)

    LOGGER.info(
        "Added %d pairwise depolymerase × capsule features "
        "(%d clusters, %d capsule profiles, %d/%d phages with depolymerases)",
        len(added_columns),
        len(cluster_index),
        len(capsule_columns),
        len(depo_counts),
        design["phage"].nunique(),
    )
    return added_columns
