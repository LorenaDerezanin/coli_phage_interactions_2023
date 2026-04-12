"""Derive directed receptor × OMP pairwise cross-term features (GT02).

Maps phage genera to predicted OMP receptor classes using Moriniere 2026
Table S1 genus-level consensus, then creates directed cross-terms:
predicted_receptor_is_X × host_X_score.

Unlike the undirected AX03 cross-terms (has_any_RBP × any_receptor), these
features encode "this phage is predicted to target OmpC, and this host has
OmpC score Y" — the directional signal that matters for adsorption.

Data sources:
- Moriniere 2026 Table S1 (.scratch/genophi/Table_S1_Phages.tsv)
- Guelin collection genus assignments (data/genomics/phages/guelin_collection.csv)
- Host OMP HMM scores from the host_surface slot (already in design matrix)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

PAIRWISE_PREFIX = "pair_receptor_omp__"

# OMP receptor columns in the host_surface slot and their canonical names.
# Maps receptor short name -> host_surface column name.
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

# Maps Table S1 receptor names to our OMP receptor short names.
# Only OMP protein receptors — LPS, NGR, Unknown are excluded.
S1_RECEPTOR_TO_OMP = {
    "BtuB": "btub",
    "FadL": "fadL",
    "FhuA": "fhua",
    "LamB": "lamB",
    "LptD": "lptD",
    "NfrA": "nfrA",
    "NupG": None,  # Not in our OMP panel
    "OmpA": "ompA",
    "OmpC": "ompC",
    "OmpF": "ompF",
    "OmpW": None,  # Not in our OMP panel
    "TolC": "tolC",
    "Tsx": "tsx",
    "YncD": "yncD",
}

# O-antigen score column for LPS-targeting phages.
O_ANTIGEN_SCORE_COLUMN = "host_surface__host_o_antigen_score"

# Minimum fraction of assays supporting the dominant receptor for a genus
# to be considered a clean assignment.
MIN_CONSENSUS_FRACTION = 0.6


def load_table_s1_genus_receptors(table_s1_path: Path) -> dict[str, dict[str, int]]:
    """Load Table S1 and build genus -> {receptor: count} mapping.

    Takes the primary receptor from semicolon-separated entries (e.g., "LptD;NupG" -> "LptD").
    Skips "Not assayed", "Resistant", and empty values.
    """
    genus_receptor_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(table_s1_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genus = row.get("Genus", "").strip()
            if not genus:
                continue
            for col in ("BW25113 receptor", "BL21 receptor"):
                receptor = row.get(col, "").strip()
                if receptor and receptor not in ("Not assayed", "Resistant", "Unknown"):
                    # Take primary receptor from semicolon-separated entries
                    primary = receptor.split(";")[0].strip()
                    genus_receptor_counts[genus][primary] += 1
    return {g: dict(counts) for g, counts in genus_receptor_counts.items()}


def build_genus_receptor_mapping(
    genus_receptor_counts: dict[str, dict[str, int]],
) -> dict[str, tuple[str, str, float]]:
    """Determine consensus receptor class per genus.

    Returns genus -> (receptor_name, receptor_type, consensus_fraction).
    receptor_type is "omp" for protein receptors in our panel, "lps" for LPS,
    "ngr" for unidentified protein receptors, or "unknown".
    """
    mapping: dict[str, tuple[str, str, float]] = {}
    for genus, counts in genus_receptor_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        dominant = max(counts, key=counts.get)
        fraction = counts[dominant] / total

        if fraction < MIN_CONSENSUS_FRACTION:
            continue  # Too ambiguous

        if dominant == "LPS":
            mapping[genus] = (dominant, "lps", fraction)
        elif dominant == "NGR":
            mapping[genus] = (dominant, "ngr", fraction)
        elif dominant in S1_RECEPTOR_TO_OMP:
            omp_name = S1_RECEPTOR_TO_OMP[dominant]
            if omp_name is not None:
                mapping[genus] = (dominant, "omp", fraction)
            # else: receptor not in our OMP panel (OmpW, NupG), skip
        # else: unknown receptor type, skip

    return mapping


def load_guelin_genus_mapping(guelin_path: Path) -> dict[str, str]:
    """Load phage -> genus mapping from guelin_collection.csv."""
    phage_genus: dict[str, str] = {}
    with open(guelin_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            phage = row.get("phage", "").strip()
            genus = row.get("Genus", "").strip()
            if phage and genus:
                phage_genus[phage] = genus
    return phage_genus


def build_phage_receptor_lookup(
    table_s1_path: Path,
    guelin_path: Path,
) -> tuple[dict[str, tuple[str, str, float]], dict[str, str]]:
    """Build phage -> (receptor, type, confidence) lookup.

    Returns (phage_receptor_map, coverage_log) where coverage_log maps
    each phage to a human-readable status string.
    """
    genus_receptors = load_table_s1_genus_receptors(table_s1_path)
    genus_mapping = build_genus_receptor_mapping(genus_receptors)
    phage_genus = load_guelin_genus_mapping(guelin_path)

    phage_receptor: dict[str, tuple[str, str, float]] = {}
    coverage_log: dict[str, str] = {}

    for phage, genus in phage_genus.items():
        if genus in genus_mapping:
            receptor_name, receptor_type, fraction = genus_mapping[genus]
            phage_receptor[phage] = (receptor_name, receptor_type, fraction)
            coverage_log[phage] = f"{genus} -> {receptor_name} ({receptor_type}, {fraction:.0%})"
        elif genus in genus_receptors:
            coverage_log[phage] = f"{genus} -> ambiguous (below {MIN_CONSENSUS_FRACTION:.0%} threshold)"
        else:
            coverage_log[phage] = f"{genus} -> not in Table S1"

    omp_count = sum(1 for _, t, _ in phage_receptor.values() if t == "omp")
    lps_count = sum(1 for _, t, _ in phage_receptor.values() if t == "lps")
    ngr_count = sum(1 for _, t, _ in phage_receptor.values() if t == "ngr")
    total = len(phage_genus)
    LOGGER.info(
        "Receptor mapping: %d/%d phages assigned (%d OMP, %d LPS, %d NGR), %d unknown",
        len(phage_receptor),
        total,
        omp_count,
        lps_count,
        ngr_count,
        total - len(phage_receptor),
    )
    return phage_receptor, coverage_log


def compute_pairwise_receptor_omp_features(
    design: pd.DataFrame,
    *,
    table_s1_path: Path | None = None,
    guelin_path: Path | None = None,
) -> list[str]:
    """Add directed receptor × OMP cross-term columns to the design matrix in-place.

    Parameters
    ----------
    design : pd.DataFrame
        Pair-level design matrix with 'phage' column and host_surface receptor columns.
    table_s1_path : Path, optional
        Path to Moriniere 2026 Table S1. Defaults to .scratch/genophi/Table_S1_Phages.tsv.
    guelin_path : Path, optional
        Path to guelin_collection.csv. Defaults to data/genomics/phages/guelin_collection.csv.

    Returns
    -------
    list[str]
        The names of added feature columns.
    """
    if table_s1_path is None:
        table_s1_path = Path(".scratch/genophi/Table_S1_Phages.tsv")
    if guelin_path is None:
        guelin_path = Path("data/genomics/phages/guelin_collection.csv")

    if not table_s1_path.exists():
        raise FileNotFoundError(
            f"Table S1 not found at {table_s1_path}. Download from Moriniere 2026 supplementary data."
        )
    if not guelin_path.exists():
        raise FileNotFoundError(f"Guelin collection not found at {guelin_path}.")

    if "phage" not in design.columns:
        raise ValueError("Design matrix must have a 'phage' column.")

    phage_receptor, coverage_log = build_phage_receptor_lookup(table_s1_path, guelin_path)

    phage_series = design["phage"].astype(str)
    added_columns: list[str] = []

    # --- has_receptor_assignment (any type) ---
    has_assignment_col = f"{PAIRWISE_PREFIX}has_receptor_assignment"
    design[has_assignment_col] = phage_series.map(lambda p: 1.0 if p in phage_receptor else 0.0)
    added_columns.append(has_assignment_col)

    # --- Per-OMP receptor: predicted_is_X (binary) + directed cross-term ---
    for omp_short, host_col in OMP_RECEPTOR_COLUMNS.items():
        predicted_col = f"{PAIRWISE_PREFIX}predicted_is_{omp_short}"
        design[predicted_col] = phage_series.map(
            lambda p, target=omp_short: (
                1.0
                if p in phage_receptor
                and phage_receptor[p][1] == "omp"
                and S1_RECEPTOR_TO_OMP.get(phage_receptor[p][0]) == target
                else 0.0
            )
        )
        added_columns.append(predicted_col)

        # Directed cross-term: predicted_is_X × host_X_score
        if host_col in design.columns:
            cross_col = f"{PAIRWISE_PREFIX}predicted_{omp_short}_x_host_{omp_short}"
            host_score = pd.to_numeric(design[host_col], errors="coerce").fillna(0.0)
            design[cross_col] = design[predicted_col] * host_score
            added_columns.append(cross_col)

    # --- LPS-targeting phages × O-antigen score ---
    predicted_lps_col = f"{PAIRWISE_PREFIX}predicted_is_lps"
    design[predicted_lps_col] = phage_series.map(
        lambda p: 1.0 if p in phage_receptor and phage_receptor[p][1] == "lps" else 0.0
    )
    added_columns.append(predicted_lps_col)

    if O_ANTIGEN_SCORE_COLUMN in design.columns:
        cross_lps_col = f"{PAIRWISE_PREFIX}predicted_lps_x_host_o_antigen"
        o_antigen_score = pd.to_numeric(design[O_ANTIGEN_SCORE_COLUMN], errors="coerce").fillna(0.0)
        design[cross_lps_col] = design[predicted_lps_col] * o_antigen_score
        added_columns.append(cross_lps_col)

    # Count unique phages per receptor type for logging.
    omp_phage_count = design.loc[design[has_assignment_col] == 1.0, "phage"].nunique()
    lps_phage_count = design.loc[design[predicted_lps_col] == 1.0, "phage"].nunique()
    LOGGER.info(
        "Added %d directed receptor × OMP features (%d phages with OMP assignment, %d with LPS)",
        len(added_columns),
        omp_phage_count,
        lps_phage_count,
    )
    return added_columns
