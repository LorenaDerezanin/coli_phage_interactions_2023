#!/usr/bin/env python3
"""Run PHROG x host-feature enrichment analyses for TL02.

Loads pharokka RBP and anti-defense annotations, host OMP receptor clusters,
LPS core types, and defense system subtypes, then runs three enrichment
analyses using the ST0.3 non-holdout interaction matrix:

1. RBP PHROG IDs x OMP receptor variant clusters
2. RBP PHROG IDs x LPS core type
3. Anti-defense gene PHROG IDs x defense system subtypes
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import _sha256
from lyzortx.pipeline.track_l.steps.annotation_interaction_enrichment import (
    ENRICHMENT_CSV_FIELDNAMES,
    compute_enrichment,
    results_to_rows,
)

logger = logging.getLogger(__name__)

# Data paths
CACHED_ANNOTATIONS_DIR = Path("data/annotations/pharokka")
LABEL_SET_V1_PATH = Path("lyzortx/generated_outputs/track_a/labels/label_set_v1_pairs.csv")
ST03_SPLIT_ASSIGNMENTS_PATH = Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv")
OMP_CLUSTERS_PATH = Path("data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv")
LPS_PRIMARY_PATH = Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt")
LPS_SUPPLEMENTAL_PATH = Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_host.txt")
DEFENSE_SUBTYPES_PATH = Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv")
OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/enrichment")
ST03_REQUIRED_COLUMNS: tuple[str, ...] = ("bacteria", "split_holdout")

# Receptor columns in OMP data
RECEPTOR_COLUMNS = (
    "BTUB",
    "FADL",
    "FHUA",
    "LAMB",
    "LPTD",
    "NFRA",
    "OMPA",
    "OMPC",
    "OMPF",
    "TOLC",
    "TSX",
    "YNCD",
)

# Minimum phage count for a PHROG to be included in enrichment
MIN_PHROG_PHAGE_COUNT = 2


def load_pharokka_phrog_matrices(
    cached_dir: Path,
    phages: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Load pharokka annotations and build binary PHROG matrices.

    Returns (rbp_matrix, rbp_phrog_names, anti_def_matrix, anti_def_phrog_names).
    Matrices have shape (n_phages, n_phrog_features).
    """
    from lyzortx.pipeline.track_l.steps.parse_annotations import (
        ANTI_DEFENSE_PATTERNS,
        RBP_PATTERNS,
        matches_any_pattern,
    )

    phage_set = set(phages)
    phage_to_idx = {p: i for i, p in enumerate(phages)}

    # Collect PHROG IDs per phage
    rbp_phrogs_per_phage: dict[str, set[str]] = defaultdict(set)
    anti_def_phrogs_per_phage: dict[str, set[str]] = defaultdict(set)

    tsvs = sorted(cached_dir.glob("*_cds_final_merged_output.tsv"))
    if not tsvs:
        msg = f"No cached merged TSVs found in {cached_dir}"
        raise FileNotFoundError(msg)

    for tsv in tsvs:
        phage_name = tsv.name.removesuffix("_cds_final_merged_output.tsv")
        if phage_name not in phage_set:
            continue
        with tsv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                phrog = row["phrog"]
                annot = row["annot"]
                if phrog == "No_PHROG":
                    continue
                if matches_any_pattern(annot, RBP_PATTERNS):
                    rbp_phrogs_per_phage[phage_name].add(phrog)
                if matches_any_pattern(annot, ANTI_DEFENSE_PATTERNS):
                    anti_def_phrogs_per_phage[phage_name].add(phrog)

    # Collect all PHROGs and filter by minimum phage count
    rbp_phrog_counts: dict[str, int] = defaultdict(int)
    for phrogs in rbp_phrogs_per_phage.values():
        for p in phrogs:
            rbp_phrog_counts[p] += 1
    rbp_phrog_names = sorted(p for p, c in rbp_phrog_counts.items() if c >= MIN_PHROG_PHAGE_COUNT)

    anti_def_phrog_counts: dict[str, int] = defaultdict(int)
    for phrogs in anti_def_phrogs_per_phage.values():
        for p in phrogs:
            anti_def_phrog_counts[p] += 1
    anti_def_phrog_names = sorted(p for p, c in anti_def_phrog_counts.items() if c >= MIN_PHROG_PHAGE_COUNT)

    logger.info(
        "RBP PHROGs: %d total unique, %d with >= %d phages",
        len(rbp_phrog_counts),
        len(rbp_phrog_names),
        MIN_PHROG_PHAGE_COUNT,
    )
    logger.info(
        "Anti-defense PHROGs: %d total unique, %d with >= %d phages",
        len(anti_def_phrog_counts),
        len(anti_def_phrog_names),
        MIN_PHROG_PHAGE_COUNT,
    )

    # Build binary matrices
    n_phages = len(phages)
    rbp_matrix = np.zeros((n_phages, len(rbp_phrog_names)), dtype=np.int8)
    anti_def_matrix = np.zeros((n_phages, len(anti_def_phrog_names)), dtype=np.int8)

    rbp_phrog_to_idx = {p: i for i, p in enumerate(rbp_phrog_names)}
    anti_def_phrog_to_idx = {p: i for i, p in enumerate(anti_def_phrog_names)}

    for phage_name, phrogs in rbp_phrogs_per_phage.items():
        if phage_name not in phage_to_idx:
            continue
        pidx = phage_to_idx[phage_name]
        for p in phrogs:
            if p in rbp_phrog_to_idx:
                rbp_matrix[pidx, rbp_phrog_to_idx[p]] = 1

    for phage_name, phrogs in anti_def_phrogs_per_phage.items():
        if phage_name not in phage_to_idx:
            continue
        pidx = phage_to_idx[phage_name]
        for p in phrogs:
            if p in anti_def_phrog_to_idx:
                anti_def_matrix[pidx, anti_def_phrog_to_idx[p]] = 1

    return rbp_matrix, rbp_phrog_names, anti_def_matrix, anti_def_phrog_names


def load_holdout_bacteria(st03_split_assignments_path: Path) -> set[str]:
    """Load bacteria IDs assigned to the ST0.3 holdout split."""
    rows = read_csv_rows(st03_split_assignments_path, required_columns=ST03_REQUIRED_COLUMNS)
    holdout_bacteria = {row["bacteria"] for row in rows if row["split_holdout"] == "holdout_test"}
    if not holdout_bacteria:
        raise ValueError(f"No holdout bacteria found in {st03_split_assignments_path}")
    return holdout_bacteria


def ensure_default_st03_split_path(st03_split_assignments_path: Path) -> None:
    if st03_split_assignments_path.exists():
        return
    if st03_split_assignments_path != ST03_SPLIT_ASSIGNMENTS_PATH:
        raise FileNotFoundError(f"Missing ST0.3 split assignments: {st03_split_assignments_path}")
    logger.info("ST0.3 split assignments missing at %s; rebuilding ST0.3 splits", st03_split_assignments_path)
    logger.info("Bootstrapping Steel Thread prerequisites for ST0.3")
    st01_label_policy.main([])
    st01b_confidence_tiers.main([])
    st02_build_pair_table.main([])
    st03_build_splits.main([])
    if not st03_split_assignments_path.exists():
        raise FileNotFoundError(f"ST0.3 rebuild did not produce expected split file: {st03_split_assignments_path}")


def select_non_holdout_bacteria(bacteria: Sequence[str], holdout_bacteria: set[str]) -> list[str]:
    """Return bacteria IDs that are not in the holdout split."""
    selected = [bacteria_id for bacteria_id in bacteria if bacteria_id not in holdout_bacteria]
    if not selected:
        raise ValueError("No non-holdout bacteria remain after filtering ST0.3 holdout strains.")
    return selected


def load_interaction_matrix(
    label_path: Path,
    bacteria: list[str],
    phages: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the binary interaction matrix from label set v1.

    Returns (interaction, resolved_mask) where both are shape (n_bacteria, n_phages).
    interaction[i,j] = 1 means lysis, 0 means no lysis or unresolved.
    resolved_mask[i,j] = 1 means the pair has a resolved label (0 or 1),
    0 means unresolved (empty any_lysis). Unresolved pairs should be excluded
    from enrichment statistics.
    """
    bact_to_idx = {b: i for i, b in enumerate(bacteria)}
    phage_to_idx = {p: i for i, p in enumerate(phages)}

    matrix = np.zeros((len(bacteria), len(phages)), dtype=np.int8)
    resolved = np.zeros((len(bacteria), len(phages)), dtype=np.int8)
    with label_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            b = row["bacteria"]
            p = row["phage"]
            label = row["any_lysis"].strip()
            if b not in bact_to_idx or p not in phage_to_idx:
                continue
            bi, pi = bact_to_idx[b], phage_to_idx[p]
            if label in ("1", "1.0"):
                matrix[bi, pi] = 1
                resolved[bi, pi] = 1
            elif label in ("0", "0.0"):
                resolved[bi, pi] = 1
            # else: unresolved — leave both at 0

    n_resolved = int(resolved.sum())
    total_lysis = int(matrix.sum())
    total_pairs = matrix.size
    logger.info(
        "Interaction matrix: %d bacteria x %d phages, %d resolved pairs (%d unresolved), %d lytic (%.1f%% of resolved)",
        len(bacteria),
        len(phages),
        n_resolved,
        total_pairs - n_resolved,
        total_lysis,
        100.0 * total_lysis / n_resolved if n_resolved > 0 else 0.0,
    )
    return matrix, resolved


def load_omp_receptor_host_matrix(
    omp_path: Path,
    bacteria: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build binary host matrix for OMP receptor variant clusters.

    Each column is a (receptor, cluster_id) pair. A host has 1 if it belongs
    to that cluster for that receptor, 0 otherwise.
    """
    bact_to_idx = {b: i for i, b in enumerate(bacteria)}

    # Read raw cluster assignments
    rows_by_bact: dict[str, dict[str, str]] = {}
    with omp_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            b = row["bacteria"]
            if b in bact_to_idx:
                rows_by_bact[b] = dict(row)

    # Discover (receptor, cluster) pairs with sufficient support
    cluster_counts: dict[tuple[str, str], int] = defaultdict(int)
    for b, row in rows_by_bact.items():
        for receptor in RECEPTOR_COLUMNS:
            cluster = row.get(receptor, "").strip()
            if cluster:
                cluster_counts[(receptor, cluster)] += 1

    # Keep clusters with >= 5 bacteria (same threshold as TC02)
    kept = sorted(k for k, v in cluster_counts.items() if v >= 5)
    feature_names = [f"{receptor}_{cluster}" for receptor, cluster in kept]
    feature_to_idx = {k: i for i, k in enumerate(kept)}

    logger.info("OMP receptor features: %d (receptor, cluster) pairs with >= 5 bacteria", len(kept))

    matrix = np.zeros((len(bacteria), len(kept)), dtype=np.int8)
    for b, row in rows_by_bact.items():
        bidx = bact_to_idx[b]
        for receptor in RECEPTOR_COLUMNS:
            cluster = row.get(receptor, "").strip()
            key = (receptor, cluster)
            if key in feature_to_idx:
                matrix[bidx, feature_to_idx[key]] = 1

    return matrix, feature_names


def load_lps_host_matrix(
    lps_primary_path: Path,
    lps_supplemental_path: Path,
    bacteria: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build binary host matrix for LPS core types.

    Each column is an LPS core type (R1, R2, R3, R4, K12). No_waaL is excluded
    since it represents absence of the gene, not a specific receptor target.
    """
    bact_to_idx = {b: i for i, b in enumerate(bacteria)}

    # Load both LPS files
    lps_by_bact: dict[str, str] = {}
    with lps_primary_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            b = row["bacteria"]
            if b in bact_to_idx:
                lps_by_bact[b] = row["LPS_type"]

    with lps_supplemental_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            b = row.get("Strain", row.get("bacteria", ""))
            if b in bact_to_idx and b not in lps_by_bact:
                lps_by_bact[b] = row["LPS_type"]

    # Determine LPS types (exclude No_waaL)
    lps_types = sorted(set(v for v in lps_by_bact.values() if v != "No_waaL"))
    feature_names = [f"LPS_{t}" for t in lps_types]
    type_to_idx = {t: i for i, t in enumerate(lps_types)}

    logger.info("LPS core types: %s", lps_types)

    matrix = np.zeros((len(bacteria), len(lps_types)), dtype=np.int8)
    for b, lps_type in lps_by_bact.items():
        if b in bact_to_idx and lps_type in type_to_idx:
            matrix[bact_to_idx[b], type_to_idx[lps_type]] = 1

    return matrix, feature_names


def load_defense_host_matrix(
    defense_path: Path,
    bacteria: list[str],
    min_bacteria_count: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Build binary host matrix for defense system subtypes.

    Each column is a defense system subtype. Subtypes present in fewer than
    min_bacteria_count bacteria are excluded (same variance filter as TC01).
    """
    bact_to_idx = {b: i for i, b in enumerate(bacteria)}

    with defense_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        fieldnames = reader.fieldnames
        if not fieldnames:
            msg = f"No header in {defense_path}"
            raise ValueError(msg)
        subtype_cols = [c for c in fieldnames if c != "bacteria"]

        rows_by_bact: dict[str, dict[str, str]] = {}
        for row in reader:
            b = row["bacteria"]
            if b in bact_to_idx:
                rows_by_bact[b] = dict(row)

    # Filter subtypes by minimum bacteria count
    subtype_counts: dict[str, int] = defaultdict(int)
    for row in rows_by_bact.values():
        for col in subtype_cols:
            if int(row.get(col, "0")) > 0:
                subtype_counts[col] += 1

    kept_subtypes = sorted(c for c in subtype_cols if subtype_counts.get(c, 0) >= min_bacteria_count)
    feature_names = [f"defense_{s}" for s in kept_subtypes]

    logger.info(
        "Defense subtypes: %d total, %d with >= %d bacteria",
        len(subtype_cols),
        len(kept_subtypes),
        min_bacteria_count,
    )

    matrix = np.zeros((len(bacteria), len(kept_subtypes)), dtype=np.int8)
    for b, row in rows_by_bact.items():
        if b not in bact_to_idx:
            continue
        bidx = bact_to_idx[b]
        for i, col in enumerate(kept_subtypes):
            if int(row.get(col, "0")) > 0:
                matrix[bidx, i] = 1

    return matrix, feature_names


def run_single_enrichment(
    name: str,
    phage_matrix: np.ndarray,
    host_matrix: np.ndarray,
    interaction_matrix: np.ndarray,
    phage_feature_names: list[str],
    host_feature_names: list[str],
    output_dir: Path,
    resolved_mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Run one enrichment analysis and write CSV output."""
    logger.info("Starting enrichment analysis: %s", name)
    results = compute_enrichment(
        phage_matrix=phage_matrix,
        host_matrix=host_matrix,
        interaction_matrix=interaction_matrix,
        phage_feature_names=phage_feature_names,
        host_feature_names=host_feature_names,
        resolved_mask=resolved_mask,
    )
    rows = results_to_rows(results)

    # Sort by p-value
    rows.sort(key=lambda r: (r["p_value"], r["phage_feature"], r["host_feature"]))

    out_path = output_dir / f"enrichment_{name}.csv"
    write_csv(out_path, ENRICHMENT_CSV_FIELDNAMES, rows)
    logger.info("Wrote %s: %d rows, %d significant", out_path, len(rows), sum(1 for r in rows if r["significant"]))

    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cached-annotations-dir", type=Path, default=CACHED_ANNOTATIONS_DIR)
    parser.add_argument("--label-path", type=Path, default=LABEL_SET_V1_PATH)
    parser.add_argument("--st03-split-assignments-path", type=Path, default=ST03_SPLIT_ASSIGNMENTS_PATH)
    parser.add_argument("--omp-path", type=Path, default=OMP_CLUSTERS_PATH)
    parser.add_argument("--lps-primary-path", type=Path, default=LPS_PRIMARY_PATH)
    parser.add_argument("--lps-supplemental-path", type=Path, default=LPS_SUPPLEMENTAL_PATH)
    parser.add_argument("--defense-path", type=Path, default=DEFENSE_SUBTYPES_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, holdout_bacteria: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    start_time = datetime.now(timezone.utc)
    logger.info("TL02 enrichment analysis starting at %s", start_time.isoformat(timespec="seconds"))

    # Determine shared bacteria and phage lists from the interaction matrix
    label_rows = read_csv_rows(args.label_path, required_columns=("bacteria", "phage"))
    if not label_rows:
        raise ValueError(f"No rows found in {args.label_path}")

    bacteria = sorted({row["bacteria"] for row in label_rows})
    phages = sorted({row["phage"] for row in label_rows})
    logger.info("Interaction panel before holdout exclusion: %d bacteria, %d phages", len(bacteria), len(phages))

    ensure_default_st03_split_path(args.st03_split_assignments_path)

    holdout_bacteria_set = (
        set(holdout_bacteria)
        if holdout_bacteria is not None
        else load_holdout_bacteria(args.st03_split_assignments_path)
    )
    missing_holdout_bacteria = holdout_bacteria_set - set(bacteria)
    if missing_holdout_bacteria:
        raise ValueError("Holdout bacteria missing from label table: " + ", ".join(sorted(missing_holdout_bacteria)))
    bacteria = select_non_holdout_bacteria(bacteria, holdout_bacteria_set)
    logger.info(
        "Excluded %d ST0.3 holdout bacteria; %d bacteria remain for enrichment",
        len(holdout_bacteria_set),
        len(bacteria),
    )

    # Load interaction matrix and resolved-pair mask
    interaction_matrix, resolved_mask = load_interaction_matrix(args.label_path, bacteria, phages)

    # Load phage PHROG matrices
    rbp_matrix, rbp_phrog_names, anti_def_matrix, anti_def_phrog_names = load_pharokka_phrog_matrices(
        args.cached_annotations_dir, phages
    )

    # Load host feature matrices
    omp_matrix, omp_feature_names = load_omp_receptor_host_matrix(args.omp_path, bacteria)
    lps_matrix, lps_feature_names = load_lps_host_matrix(args.lps_primary_path, args.lps_supplemental_path, bacteria)
    defense_matrix, defense_feature_names = load_defense_host_matrix(args.defense_path, bacteria)

    ensure_directory(args.output_dir)

    # Analysis 1: RBP PHROGs x OMP receptor variant clusters
    rbp_omp_rows = run_single_enrichment(
        name="rbp_phrog_x_omp_receptor",
        phage_matrix=rbp_matrix,
        host_matrix=omp_matrix,
        interaction_matrix=interaction_matrix,
        phage_feature_names=[f"RBP_PHROG_{p}" for p in rbp_phrog_names],
        host_feature_names=omp_feature_names,
        output_dir=args.output_dir,
        resolved_mask=resolved_mask,
    )

    # Analysis 2: RBP PHROGs x LPS core type
    rbp_lps_rows = run_single_enrichment(
        name="rbp_phrog_x_lps_core",
        phage_matrix=rbp_matrix,
        host_matrix=lps_matrix,
        interaction_matrix=interaction_matrix,
        phage_feature_names=[f"RBP_PHROG_{p}" for p in rbp_phrog_names],
        host_feature_names=lps_feature_names,
        output_dir=args.output_dir,
        resolved_mask=resolved_mask,
    )

    # Analysis 3: Anti-defense gene PHROGs x defense system subtypes
    antidef_defense_rows = run_single_enrichment(
        name="antidef_phrog_x_defense_subtype",
        phage_matrix=anti_def_matrix,
        host_matrix=defense_matrix,
        interaction_matrix=interaction_matrix,
        phage_feature_names=[f"ANTIDEF_PHROG_{p}" for p in anti_def_phrog_names],
        host_feature_names=defense_feature_names,
        output_dir=args.output_dir,
        resolved_mask=resolved_mask,
    )

    # Write manifest
    manifest = {
        "step": "TL02_enrichment_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": {"n_bacteria": len(bacteria), "n_phages": len(phages)},
        "holdout_exclusion": {
            "split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
            },
            "excluded_holdout_bacteria_ids": sorted(holdout_bacteria_set),
            "excluded_holdout_bacteria_count": len(holdout_bacteria_set),
        },
        "inputs": {
            "label_set_v1_pairs": {"path": str(args.label_path), "sha256": _sha256(args.label_path)},
            "cached_annotations_dir": str(args.cached_annotations_dir),
            "omp_clusters": {"path": str(args.omp_path), "sha256": _sha256(args.omp_path)},
            "lps_primary": {"path": str(args.lps_primary_path), "sha256": _sha256(args.lps_primary_path)},
            "lps_supplemental": {
                "path": str(args.lps_supplemental_path),
                "sha256": _sha256(args.lps_supplemental_path),
            },
            "defense_subtypes": {"path": str(args.defense_path), "sha256": _sha256(args.defense_path)},
        },
        "analyses": {
            "rbp_phrog_x_omp_receptor": {
                "n_phage_features": len(rbp_phrog_names),
                "n_host_features": len(omp_feature_names),
                "n_tests": len(rbp_omp_rows),
                "n_significant": sum(1 for r in rbp_omp_rows if r["significant"]),
            },
            "rbp_phrog_x_lps_core": {
                "n_phage_features": len(rbp_phrog_names),
                "n_host_features": len(lps_feature_names),
                "n_tests": len(rbp_lps_rows),
                "n_significant": sum(1 for r in rbp_lps_rows if r["significant"]),
            },
            "antidef_phrog_x_defense_subtype": {
                "n_phage_features": len(anti_def_phrog_names),
                "n_host_features": len(defense_feature_names),
                "n_tests": len(antidef_defense_rows),
                "n_significant": sum(1 for r in antidef_defense_rows if r["significant"]),
            },
        },
        "outputs": {
            "rbp_phrog_x_omp_receptor": {
                "path": str(args.output_dir / "enrichment_rbp_phrog_x_omp_receptor.csv"),
                "sha256": _sha256(args.output_dir / "enrichment_rbp_phrog_x_omp_receptor.csv"),
            },
            "rbp_phrog_x_lps_core": {
                "path": str(args.output_dir / "enrichment_rbp_phrog_x_lps_core.csv"),
                "sha256": _sha256(args.output_dir / "enrichment_rbp_phrog_x_lps_core.csv"),
            },
            "antidef_phrog_x_defense_subtype": {
                "path": str(args.output_dir / "enrichment_antidef_phrog_x_defense_subtype.csv"),
                "sha256": _sha256(args.output_dir / "enrichment_antidef_phrog_x_defense_subtype.csv"),
            },
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    end_time = datetime.now(timezone.utc)
    logger.info(
        "TL02 enrichment analysis completed at %s (elapsed: %s)",
        end_time.isoformat(timespec="seconds"),
        end_time - start_time,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
