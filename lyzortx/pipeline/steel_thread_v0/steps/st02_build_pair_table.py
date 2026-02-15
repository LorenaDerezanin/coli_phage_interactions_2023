#!/usr/bin/env python3
"""ST0.2: Build canonical pair table with labels, uncertainty, and v0 feature blocks."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

HOST_FEATURE_MAP: Sequence[Tuple[str, str]] = (
    ("Pathotype", "host_pathotype"),
    ("Clermont_Phylo", "host_clermont_phylo"),
    ("Origin", "host_origin"),
    ("LPS_type", "host_lps_type"),
    ("O-type", "host_o_type"),
    ("H-type", "host_h_type"),
    ("Collection", "host_collection"),
    ("ABC_serotype", "host_abc_serotype"),
    ("Mouse_killed_10", "host_mouse_killed_10"),
    ("Capsule_ABC", "host_capsule_abc"),
    ("Capsule_GroupIV_e", "host_capsule_groupiv_e"),
    ("Capsule_GroupIV_e_stricte", "host_capsule_groupiv_e_stricte"),
    ("Capsule_GroupIV_s", "host_capsule_groupiv_s"),
    ("Capsule_Wzy_stricte", "host_capsule_wzy_stricte"),
    ("n_defense_systems", "host_n_defense_systems"),
    ("n_infections", "host_n_infections"),
)

PHAGE_FEATURE_MAP: Sequence[Tuple[str, str]] = (
    ("Morphotype", "phage_morphotype"),
    ("Family", "phage_family"),
    ("Genus", "phage_genus"),
    ("Species", "phage_species"),
    ("Subfamily", "phage_subfamily"),
    ("Old_Family", "phage_old_family"),
    ("Old_Genus", "phage_old_genus"),
    ("Phage_host", "phage_host"),
    ("Phage_host_phylo", "phage_host_phylo"),
    ("Genome_size", "phage_genome_size"),
)

ST01B_REQUIRED_COLUMNS: Sequence[str] = (
    "bacteria",
    "phage",
    "total_obs",
    "score_1_count",
    "score_0_count",
    "score_n_count",
    "interpretable_count",
    "uninterpretable_fraction",
    "hard_label_any_lysis",
    "label_reason",
    "include_in_training",
    "uncertainty_flags",
    "uncertainty_flag_count",
    "n_distinct_replicates",
    "n_distinct_dilutions",
    "missing_obs_count",
    "positive_fraction_interpretable",
    "strict_confidence_tier",
    "strict_tier_reason",
    "strict_label",
    "strict_include_in_training",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st01b-pair-audit-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st01b_pair_confidence_audit.csv"),
        help="Input ST0.1b pair-level CSV.",
    )
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Host metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Phage metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--interaction-matrix-path",
        type=Path,
        default=Path("data/interactions/interaction_matrix.csv"),
        help="Interaction matrix CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--cv-groups-path",
        type=Path,
        default=Path("data/metadata/370+host_cross_validation_groups_1e-4.csv"),
        help="Cross-validation group mapping CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.2 artifacts.",
    )
    parser.add_argument(
        "--allow-missing-joins",
        action="store_true",
        help="Allow missing joins instead of failing (not recommended for v0).",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str = ";") -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()})
        return rows


def read_st01b_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [c for c in ST01B_REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()})
        return rows


def load_host_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_delimited_rows(path, delimiter=";")
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        bacteria = row.get("bacteria", "")
        if not bacteria:
            continue
        out[bacteria] = row
    return out


def load_phage_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_delimited_rows(path, delimiter=";")
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        phage = row.get("phage", "")
        if not phage:
            continue
        out[phage] = row
    return out


def load_cv_groups(path: Path) -> Dict[str, str]:
    rows = read_delimited_rows(path, delimiter=";")
    out: Dict[str, str] = {}
    for row in rows:
        bacteria = row.get("bacteria", "")
        group = row.get("group", "")
        if bacteria:
            out[bacteria] = group
    return out


def load_interaction_matrix(path: Path) -> Tuple[Dict[Tuple[str, str], str], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        if "bacteria" not in reader.fieldnames:
            raise ValueError(f"Missing 'bacteria' column in {path}.")
        phage_columns = [c for c in reader.fieldnames if c != "bacteria"]
        out: Dict[Tuple[str, str], str] = {}
        for row in reader:
            bacteria = (row.get("bacteria") or "").strip()
            if not bacteria:
                continue
            for phage in phage_columns:
                out[(bacteria, phage)] = (row.get(phage) or "").strip()
        return out, phage_columns


def parse_flags(raw_flags: str) -> set[str]:
    if not raw_flags:
        return set()
    return {flag for flag in raw_flags.split("|") if flag}


def to_float_or_empty(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except ValueError:
        return ""


def count_missing(values: Iterable[str]) -> int:
    return sum(1 for v in values if v == "")


def matrix_binary(score_raw: str, threshold: float) -> str:
    if score_raw == "":
        return ""
    try:
        score = float(score_raw)
    except ValueError:
        return ""
    return "1" if score >= threshold else "0"


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st01b_rows = read_st01b_rows(args.st01b_pair_audit_path)
    host_map = load_host_metadata(args.host_metadata_path)
    phage_map = load_phage_metadata(args.phage_metadata_path)
    cv_map = load_cv_groups(args.cv_groups_path)
    matrix_map, matrix_phage_columns = load_interaction_matrix(args.interaction_matrix_path)

    bacteria_values = sorted({row["bacteria"] for row in st01b_rows})
    phage_values = sorted({row["phage"] for row in st01b_rows})
    bacteria_to_idx = {bacteria: idx for idx, bacteria in enumerate(bacteria_values)}
    phage_to_idx = {phage: idx for idx, phage in enumerate(phage_values)}

    join_missing = Counter()
    hard_label_counts = Counter()
    strict_tier_counts = Counter()
    strict_label_counts = Counter()
    cv_group_counts = Counter()
    host_missing_counts = Counter()
    phage_missing_counts = Counter()

    output_rows: List[Dict[str, object]] = []
    for row in st01b_rows:
        bacteria = row["bacteria"]
        phage = row["phage"]
        host_row = host_map.get(bacteria)
        phage_row = phage_map.get(phage)
        cv_group = cv_map.get(bacteria, "")
        matrix_score_raw = matrix_map.get((bacteria, phage), "")

        if host_row is None:
            join_missing["host_metadata"] += 1
            host_row = {}
        if phage_row is None:
            join_missing["phage_metadata"] += 1
            phage_row = {}
        if cv_group == "":
            join_missing["cv_group"] += 1
        if matrix_score_raw == "":
            join_missing["interaction_matrix"] += 1

        uncertainty_flags = parse_flags(row["uncertainty_flags"])

        host_features = {
            target_col: host_row.get(source_col, "")
            for source_col, target_col in HOST_FEATURE_MAP
        }
        phage_features = {
            target_col: phage_row.get(source_col, "")
            for source_col, target_col in PHAGE_FEATURE_MAP
        }
        host_missing = count_missing(host_features.values())
        phage_missing = count_missing(phage_features.values())

        for key, val in host_features.items():
            if val == "":
                host_missing_counts[key] += 1
        for key, val in phage_features.items():
            if val == "":
                phage_missing_counts[key] += 1

        hard_label = row["hard_label_any_lysis"]
        strict_tier = row["strict_confidence_tier"]
        strict_label = row["strict_label"]

        hard_label_counts["unresolved" if hard_label == "" else hard_label] += 1
        strict_tier_counts[strict_tier] += 1
        strict_label_counts["ambiguous" if strict_label == "" else strict_label] += 1
        cv_group_counts[cv_group if cv_group != "" else "missing"] += 1

        pair_row: Dict[str, object] = {
            "pair_id": f"{bacteria}__{phage}",
            "bacteria": bacteria,
            "phage": phage,
            "bacteria_idx_v0": bacteria_to_idx[bacteria],
            "phage_idx_v0": phage_to_idx[phage],
            "cv_group": cv_group,
            "label_hard_any_lysis": hard_label,
            "label_hard_include_in_training": row["include_in_training"],
            "label_hard_reason": row["label_reason"],
            "label_strict_confidence_tier": strict_tier,
            "label_strict": strict_label,
            "label_strict_include_in_training": row["strict_include_in_training"],
            "label_strict_reason": row["strict_tier_reason"],
            "obs_total": row["total_obs"],
            "obs_score_1_count": row["score_1_count"],
            "obs_score_0_count": row["score_0_count"],
            "obs_score_n_count": row["score_n_count"],
            "obs_interpretable_count": row["interpretable_count"],
            "obs_uninterpretable_fraction": row["uninterpretable_fraction"],
            "obs_positive_fraction_interpretable": row["positive_fraction_interpretable"],
            "obs_uncertainty_flag_count": row["uncertainty_flag_count"],
            "obs_n_distinct_replicates": row["n_distinct_replicates"],
            "obs_n_distinct_dilutions": row["n_distinct_dilutions"],
            "obs_missing_obs_count": row["missing_obs_count"],
            "obs_flag_has_uninterpretable": "1" if "has_uninterpretable" in uncertainty_flags else "0",
            "obs_flag_high_uninterpretable_fraction": (
                "1" if "high_uninterpretable_fraction" in uncertainty_flags else "0"
            ),
            "obs_flag_conflicting_interpretable_observations": (
                "1" if "conflicting_interpretable_observations" in uncertainty_flags else "0"
            ),
            "obs_flag_incomplete_observation_grid": (
                "1" if "incomplete_observation_grid" in uncertainty_flags else "0"
            ),
            "obs_flag_low_interpretable_support": "1" if "low_interpretable_support" in uncertainty_flags else "0",
            "obs_flag_unresolved_label": "1" if "unresolved_label" in uncertainty_flags else "0",
            "aux_matrix_score_0_to_4": matrix_score_raw,
            "aux_matrix_nonzero": matrix_binary(matrix_score_raw, threshold=0.000001),
            "aux_matrix_ge2": matrix_binary(matrix_score_raw, threshold=2.0),
            "aux_matrix_ge3": matrix_binary(matrix_score_raw, threshold=3.0),
        }
        pair_row.update(host_features)
        pair_row["host_feature_missing_count"] = host_missing
        pair_row.update(phage_features)
        pair_row["phage_feature_missing_count"] = phage_missing

        host_phylo = host_features.get("host_clermont_phylo", "")
        phage_host_phylo = phage_features.get("phage_host_phylo", "")
        if host_phylo and phage_host_phylo:
            pair_row["pair_host_phylo_equals_phage_host_phylo"] = (
                "1" if host_phylo == phage_host_phylo else "0"
            )
        else:
            pair_row["pair_host_phylo_equals_phage_host_phylo"] = ""

        pair_row["metadata_internal_source"] = "raw_interactions_v1"
        output_rows.append(pair_row)

    critical_missing = {
        key: count
        for key, count in join_missing.items()
        if key in {"host_metadata", "phage_metadata", "cv_group"} and count > 0
    }
    if critical_missing and not args.allow_missing_joins:
        missing_summary = ", ".join(f"{k}={v}" for k, v in sorted(critical_missing.items()))
        raise ValueError(
            "Missing joins detected during ST0.2. Use --allow-missing-joins to continue. "
            f"Missing: {missing_summary}"
        )

    output_rows.sort(key=lambda r: (str(r["bacteria"]), str(r["phage"])))
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    output_csv_path = args.output_dir / "st02_pair_table.csv"
    write_csv(output_csv_path, fieldnames=fieldnames, rows=output_rows)

    total_rows = len(output_rows)
    strict_rows = strict_tier_counts["high_conf_pos"] + strict_tier_counts["high_conf_neg"]
    matrix_available = total_rows - join_missing["interaction_matrix"]

    feature_manifest = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st02_build_pair_table",
        "input_paths": {
            "st01b_pair_audit": str(args.st01b_pair_audit_path),
            "host_metadata": str(args.host_metadata_path),
            "phage_metadata": str(args.phage_metadata_path),
            "interaction_matrix": str(args.interaction_matrix_path),
            "cv_groups": str(args.cv_groups_path),
        },
        "column_groups": {
            "ids": ["pair_id", "bacteria", "phage", "bacteria_idx_v0", "phage_idx_v0", "cv_group"],
            "labels": [
                "label_hard_any_lysis",
                "label_hard_include_in_training",
                "label_hard_reason",
                "label_strict_confidence_tier",
                "label_strict",
                "label_strict_include_in_training",
                "label_strict_reason",
            ],
            "observation_summary": [
                "obs_total",
                "obs_score_1_count",
                "obs_score_0_count",
                "obs_score_n_count",
                "obs_interpretable_count",
                "obs_uninterpretable_fraction",
                "obs_positive_fraction_interpretable",
                "obs_uncertainty_flag_count",
                "obs_n_distinct_replicates",
                "obs_n_distinct_dilutions",
                "obs_missing_obs_count",
                "obs_flag_has_uninterpretable",
                "obs_flag_high_uninterpretable_fraction",
                "obs_flag_conflicting_interpretable_observations",
                "obs_flag_incomplete_observation_grid",
                "obs_flag_low_interpretable_support",
                "obs_flag_unresolved_label",
            ],
            "auxiliary_outcome_reference": [
                "aux_matrix_score_0_to_4",
                "aux_matrix_nonzero",
                "aux_matrix_ge2",
                "aux_matrix_ge3",
            ],
            "host_features": [target_col for _, target_col in HOST_FEATURE_MAP]
            + ["host_feature_missing_count"],
            "phage_features": [target_col for _, target_col in PHAGE_FEATURE_MAP]
            + ["phage_feature_missing_count"],
            "pair_features": [
                "pair_host_phylo_equals_phage_host_phylo",
            ],
            "metadata": ["metadata_internal_source"],
        },
        "notes": {
            "aux_matrix_columns": (
                "Interaction-matrix columns are auxiliary references from the same internal dataset and should not "
                "be used as model features for leakage-safe prediction."
            ),
            "ordering": "Rows are sorted by (bacteria, phage).",
        },
    }

    audit = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "row_count": total_rows,
        "distinct_bacteria_count": len(bacteria_values),
        "distinct_phage_count": len(phage_values),
        "join_missing_counts": dict(sorted(join_missing.items())),
        "hard_label_counts": {
            "positive_1": hard_label_counts["1"],
            "negative_0": hard_label_counts["0"],
            "unresolved": hard_label_counts["unresolved"],
        },
        "strict_tier_counts": dict(sorted(strict_tier_counts.items())),
        "strict_label_counts": {
            "positive_1": strict_label_counts["1"],
            "negative_0": strict_label_counts["0"],
            "ambiguous": strict_label_counts["ambiguous"],
        },
        "strict_slice_fraction": round(strict_rows / total_rows, 6) if total_rows else 0.0,
        "cv_group_unique_count": len(cv_group_counts),
        "cv_group_counts": dict(sorted(cv_group_counts.items())),
        "host_feature_missing_counts": dict(sorted(host_missing_counts.items())),
        "phage_feature_missing_counts": dict(sorted(phage_missing_counts.items())),
        "matrix_available_fraction": round(matrix_available / total_rows, 6) if total_rows else 0.0,
        "matrix_phage_column_count": len(matrix_phage_columns),
        "output_columns": fieldnames,
    }

    write_json(args.output_dir / "st02_feature_manifest.json", feature_manifest)
    write_json(args.output_dir / "st02_pair_table_audit.json", audit)

    print("ST0.2 completed.")
    print(f"- Rows: {total_rows}")
    print(f"- Distinct bacteria: {len(bacteria_values)}")
    print(f"- Distinct phage: {len(phage_values)}")
    print(f"- Strict slice rows: {strict_rows}")
    print(f"- Output table: {output_csv_path}")


if __name__ == "__main__":
    main()
