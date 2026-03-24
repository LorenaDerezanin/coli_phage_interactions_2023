#!/usr/bin/env python3
"""TE01: Build leakage-safe RBP-receptor compatibility features from a curated taxon lookup."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

OUTPUT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "lookup_available",
    "target_receptor_present",
    "protein_target_present",
    "surface_target_present",
    "receptor_cluster_matches",
)

PROTEIN_TARGET_COLUMNS: Dict[str, str] = {
    "BTUB": "BTUB",
    "FADL": "FADL",
    "FHUA": "FHUA",
    "LAMB": "LAMB",
    "LPTD": "LPTD",
    "NFRA": "NFRA",
    "OMPA": "OMPA",
    "OMPC": "OMPC",
    "OMPF": "OMPF",
    "TOLC": "TOLC",
    "TSX": "TSX",
    "YNCD": "YNCD",
}
SURFACE_TARGETS: Tuple[str, ...] = ("LPS_CORE", "O_ANTIGEN", "CAPSULE")
SUPPORTED_TARGETS: Tuple[str, ...] = (*tuple(PROTEIN_TARGET_COLUMNS.keys()), *SURFACE_TARGETS)
CAPSULE_PROXY_COLUMNS: Tuple[str, ...] = (
    "host_capsule_abc",
    "host_capsule_groupiv_e",
    "host_capsule_groupiv_e_stricte",
    "host_capsule_groupiv_s",
    "host_capsule_wzy_stricte",
)
FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "lookup_available": {
        "type": "binary",
        "transform": "1 when the phage genus or subfamily resolves to a curated lookup entry, else 0.",
    },
    "target_receptor_present": {
        "type": "binary",
        "transform": "1 when the host has at least one mapped target receptor or surface structure for the phage taxon.",
    },
    "protein_target_present": {
        "type": "binary",
        "transform": "1 when the host has a non-missing OMP/receptor cluster for at least one mapped protein target.",
    },
    "surface_target_present": {
        "type": "binary",
        "transform": "1 when the host has an annotated mapped surface-glycan target (LPS core, O-antigen, or capsule).",
    },
    "receptor_cluster_matches": {
        "type": "binary",
        "transform": (
            "1 when a mapped protein-receptor cluster for this pair was already seen among leakage-safe training "
            "positives for the same curated taxon."
        ),
    },
}


@dataclass(frozen=True)
class LookupEntry:
    match_level: str
    taxon: str
    target_receptors: Tuple[str, ...]
    target_family: str
    evidence_note: str


@dataclass(frozen=True)
class TargetState:
    present: int
    variant: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments used for leakage-safe training views.",
    )
    parser.add_argument(
        "--receptor-clusters-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"),
        help="Tab-delimited host receptor cluster assignments.",
    )
    parser.add_argument(
        "--lookup-path",
        type=Path,
        default=Path("lyzortx/pipeline/track_e/curated_inputs/genus_receptor_lookup.csv"),
        help="Curated genus/subfamily to receptor-target lookup CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block"),
        help="Directory for generated TE01 outputs.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and the manifest.",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_truthy_flag(value: str) -> bool:
    normalized = value.strip()
    if normalized in {"", "0", "0.0"}:
        return False
    try:
        return float(normalized) > 0
    except ValueError:
        return normalized.lower() not in {"false", "no", "na", "n/a"}


def _require_columns(rows: Sequence[Mapping[str, str]], path: Path, columns: Iterable[str]) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")


def resolve_training_flag_column(rows: Sequence[Mapping[str, str]], path: Path) -> str:
    candidates = ("include_in_training", "label_hard_include_in_training")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise ValueError(f"Missing training-flag column in {path}; expected one of: {', '.join(candidates)}")


def load_lookup(path: Path) -> Tuple[Dict[str, LookupEntry], Dict[str, LookupEntry]]:
    rows = read_delimited_rows(path)
    _require_columns(rows, path, ("match_level", "taxon", "target_receptors", "target_family", "evidence_note"))

    by_genus: Dict[str, LookupEntry] = {}
    by_subfamily: Dict[str, LookupEntry] = {}
    for row in rows:
        match_level = row["match_level"]
        taxon = row["taxon"]
        targets = tuple(token.strip().upper() for token in row["target_receptors"].split("|") if token.strip())
        if not taxon:
            raise ValueError(f"Empty taxon value in {path}")
        if match_level not in {"genus", "subfamily"}:
            raise ValueError(f"Unsupported match_level {match_level!r} in {path}")
        if not targets:
            raise ValueError(f"Lookup entry for {taxon!r} has no target receptors in {path}")
        unknown_targets = sorted(set(targets) - set(SUPPORTED_TARGETS))
        if unknown_targets:
            raise ValueError(f"Unsupported target receptor(s) in {path}: {', '.join(unknown_targets)}")

        entry = LookupEntry(
            match_level=match_level,
            taxon=taxon,
            target_receptors=targets,
            target_family=row["target_family"],
            evidence_note=row["evidence_note"],
        )
        target_index = by_genus if match_level == "genus" else by_subfamily
        if taxon in target_index:
            raise ValueError(f"Duplicate {match_level} lookup entry for {taxon!r} in {path}")
        target_index[taxon] = entry

    return by_genus, by_subfamily


def resolve_lookup_entry(
    *,
    phage_genus: str,
    phage_subfamily: str,
    lookup_by_genus: Mapping[str, LookupEntry],
    lookup_by_subfamily: Mapping[str, LookupEntry],
) -> Optional[LookupEntry]:
    if phage_genus and phage_genus in lookup_by_genus:
        return lookup_by_genus[phage_genus]
    if phage_subfamily and phage_subfamily in lookup_by_subfamily:
        return lookup_by_subfamily[phage_subfamily]
    return None


def build_receptor_index(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    _require_columns(rows, Path("<receptor_rows>"), ("bacteria", *PROTEIN_TARGET_COLUMNS.values()))
    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        bacteria = row.get("bacteria", "")
        if not bacteria:
            continue
        if bacteria in index:
            raise ValueError(f"Duplicate bacteria value {bacteria!r} in receptor cluster rows")
        index[bacteria] = dict(row)
    return index


def build_host_target_states(
    st02_rows: Sequence[Mapping[str, str]],
    receptor_index: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, TargetState]]:
    state_by_bacteria: Dict[str, Dict[str, TargetState]] = {}
    for row in st02_rows:
        bacteria = row.get("bacteria", "")
        if not bacteria or bacteria in state_by_bacteria:
            continue

        receptor_row = receptor_index.get(bacteria, {})
        states: Dict[str, TargetState] = {}
        for target, source_column in PROTEIN_TARGET_COLUMNS.items():
            variant = str(receptor_row.get(source_column, "")).strip()
            states[target] = TargetState(present=1 if variant else 0, variant=variant)

        lps_variant = row.get("host_lps_type", "")
        states["LPS_CORE"] = TargetState(present=1 if lps_variant else 0, variant=lps_variant)

        o_antigen_variant = row.get("host_o_type", "")
        states["O_ANTIGEN"] = TargetState(present=1 if o_antigen_variant else 0, variant=o_antigen_variant)

        capsule_variant = row.get("host_abc_serotype", "")
        capsule_proxy_present = any(_is_truthy_flag(row.get(column, "")) for column in CAPSULE_PROXY_COLUMNS)
        if capsule_variant:
            states["CAPSULE"] = TargetState(present=1, variant=capsule_variant)
        elif capsule_proxy_present:
            states["CAPSULE"] = TargetState(present=1, variant="__CAPSULE_PROXY_PRESENT__")
        else:
            states["CAPSULE"] = TargetState(present=0, variant="")

        state_by_bacteria[bacteria] = states

    return state_by_bacteria


def merge_pair_rows(
    st02_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
    host_target_states: Mapping[str, Mapping[str, TargetState]],
    lookup_by_genus: Mapping[str, LookupEntry],
    lookup_by_subfamily: Mapping[str, LookupEntry],
    training_flag_column: str,
) -> List[Dict[str, object]]:
    split_by_pair_id = {row["pair_id"]: row for row in split_rows}

    merged_rows: List[Dict[str, object]] = []
    for row in st02_rows:
        pair_id = row.get("pair_id", "")
        if not pair_id:
            raise ValueError("ST0.2 pair row is missing pair_id")
        split_row = split_by_pair_id.get(pair_id)
        if split_row is None:
            raise ValueError(f"Missing ST0.3 split assignment for pair_id {pair_id!r}")

        bacteria = row.get("bacteria", "")
        host_states = host_target_states.get(bacteria)
        if host_states is None:
            raise ValueError(f"Missing host target states for bacteria {bacteria!r}")

        lookup_entry = resolve_lookup_entry(
            phage_genus=row.get("phage_genus", ""),
            phage_subfamily=row.get("phage_subfamily", ""),
            lookup_by_genus=lookup_by_genus,
            lookup_by_subfamily=lookup_by_subfamily,
        )

        split_holdout = split_row.get("split_holdout", "")
        split_fold_raw = split_row.get("split_cv5_fold", "")
        split_fold = int(split_fold_raw) if split_fold_raw else -1

        merged_rows.append(
            {
                "pair_id": pair_id,
                "bacteria": bacteria,
                "phage": row.get("phage", ""),
                "phage_genus": row.get("phage_genus", ""),
                "phage_subfamily": row.get("phage_subfamily", ""),
                "label_hard_any_lysis": row.get("label_hard_any_lysis", ""),
                "include_in_training": row.get(training_flag_column, ""),
                "split_holdout": split_holdout,
                "split_cv5_fold": split_fold,
                "lookup_entry": lookup_entry,
                "host_target_states": host_states,
            }
        )

    merged_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    return merged_rows


def _is_training_positive(row: Mapping[str, object]) -> bool:
    return (
        str(row.get("include_in_training", "")) == "1"
        and str(row.get("label_hard_any_lysis", "")) == "1"
        and str(row.get("split_holdout", "")) == "train_non_holdout"
    )


def _scenario_key(row: Mapping[str, object]) -> str:
    if str(row.get("split_holdout", "")) == "holdout_test":
        return "holdout"
    return f"cv_fold_{int(row['split_cv5_fold'])}"


def build_training_variant_indices(
    merged_rows: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, Dict[Tuple[str, ...], Counter[str]]]]:
    fold_ids = sorted(
        {
            int(row["split_cv5_fold"])
            for row in merged_rows
            if str(row.get("split_holdout", "")) == "train_non_holdout" and int(row["split_cv5_fold"]) >= 0
        }
    )
    training_rows_by_scenario: Dict[str, List[Mapping[str, object]]] = {
        "holdout": [row for row in merged_rows if _is_training_positive(row)]
    }
    for fold_id in fold_ids:
        training_rows_by_scenario[f"cv_fold_{fold_id}"] = [
            row for row in merged_rows if _is_training_positive(row) and int(row["split_cv5_fold"]) != fold_id
        ]

    scenario_indices: Dict[str, Dict[str, Dict[Tuple[str, ...], Counter[str]]]] = {}
    for scenario, training_rows in training_rows_by_scenario.items():
        phage_variant_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        taxon_variant_counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)

        for row in training_rows:
            lookup_entry = row.get("lookup_entry")
            if not isinstance(lookup_entry, LookupEntry):
                continue
            host_states = row["host_target_states"]
            for target in lookup_entry.target_receptors:
                target_state = host_states[target]
                if not target_state.variant:
                    continue
                phage_variant_counts[(str(row["phage"]), target)][target_state.variant] += 1
                taxon_variant_counts[(lookup_entry.match_level, lookup_entry.taxon, target)][target_state.variant] += 1

        scenario_indices[scenario] = {
            "phage_variant_counts": phage_variant_counts,
            "taxon_variant_counts": taxon_variant_counts,
        }

    return scenario_indices


def build_feature_rows(
    merged_rows: Sequence[Mapping[str, object]],
    scenario_indices: Mapping[str, Mapping[str, Mapping[Tuple[str, ...], Counter[str]]]],
) -> List[Dict[str, object]]:
    feature_rows: List[Dict[str, object]] = []
    for row in merged_rows:
        lookup_entry = row.get("lookup_entry")
        output_row: Dict[str, object] = {
            "pair_id": row["pair_id"],
            "bacteria": row["bacteria"],
            "phage": row["phage"],
        }

        if not isinstance(lookup_entry, LookupEntry):
            for column in OUTPUT_FEATURE_COLUMNS:
                output_row[column] = 0
            feature_rows.append(output_row)
            continue

        host_states = row["host_target_states"]
        scenario = _scenario_key(row)
        scenario_data = scenario_indices[scenario]
        taxon_counts = scenario_data["taxon_variant_counts"]

        target_states = [host_states[target] for target in lookup_entry.target_receptors]
        protein_targets = [target for target in lookup_entry.target_receptors if target in PROTEIN_TARGET_COLUMNS]
        surface_targets = [target for target in lookup_entry.target_receptors if target in SURFACE_TARGETS]

        protein_target_present = int(any(host_states[target].present for target in protein_targets))
        surface_target_present = int(any(host_states[target].present for target in surface_targets))

        protein_cluster_match_count = 0
        for target in lookup_entry.target_receptors:
            target_state = host_states[target]
            if not target_state.variant:
                continue
            if (
                target in PROTEIN_TARGET_COLUMNS
                and target_state.variant in taxon_counts[(lookup_entry.match_level, lookup_entry.taxon, target)]
            ):
                protein_cluster_match_count += 1

        output_row.update(
            {
                "lookup_available": 1,
                "target_receptor_present": int(any(state.present for state in target_states)),
                "protein_target_present": protein_target_present,
                "surface_target_present": surface_target_present,
                "receptor_cluster_matches": int(protein_cluster_match_count > 0),
            }
        )
        feature_rows.append(output_row)

    feature_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    return feature_rows


def build_feature_metadata(lookup_path: Path) -> List[Dict[str, object]]:
    return [
        {
            "column_name": column,
            "type": FEATURE_METADATA[column]["type"],
            "source_path": str(lookup_path),
            "transform": FEATURE_METADATA[column]["transform"],
        }
        for column in OUTPUT_FEATURE_COLUMNS
    ]


def build_lookup_summary_rows(
    merged_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    phage_taxonomy: Dict[str, Dict[str, object]] = {}
    for row in merged_rows:
        phage = str(row["phage"])
        lookup_entry = row.get("lookup_entry")
        if phage in phage_taxonomy:
            continue
        if isinstance(lookup_entry, LookupEntry):
            phage_taxonomy[phage] = {
                "phage": phage,
                "match_level": lookup_entry.match_level,
                "taxon": lookup_entry.taxon,
                "target_receptors": "|".join(lookup_entry.target_receptors),
                "target_family": lookup_entry.target_family,
                "evidence_note": lookup_entry.evidence_note,
            }
        else:
            phage_taxonomy[phage] = {
                "phage": phage,
                "match_level": "",
                "taxon": "",
                "target_receptors": "",
                "target_family": "unknown",
                "evidence_note": "No curated lookup entry matched this phage genus or subfamily.",
            }
    return [phage_taxonomy[phage] for phage in sorted(phage_taxonomy)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_delimited_rows(args.st02_pair_table_path)
    split_rows = read_delimited_rows(args.st03_split_assignments_path)
    receptor_rows = read_delimited_rows(args.receptor_clusters_path, delimiter="\t")
    _require_columns(
        st02_rows,
        args.st02_pair_table_path,
        (
            "pair_id",
            "bacteria",
            "phage",
            "phage_genus",
            "phage_subfamily",
            "label_hard_any_lysis",
            "host_lps_type",
            "host_o_type",
            "host_abc_serotype",
            *CAPSULE_PROXY_COLUMNS,
        ),
    )
    _require_columns(split_rows, args.st03_split_assignments_path, ("pair_id", "split_holdout", "split_cv5_fold"))

    training_flag_column = resolve_training_flag_column(st02_rows, args.st02_pair_table_path)
    lookup_by_genus, lookup_by_subfamily = load_lookup(args.lookup_path)
    receptor_index = build_receptor_index(receptor_rows)
    host_target_states = build_host_target_states(st02_rows, receptor_index)
    merged_rows = merge_pair_rows(
        st02_rows,
        split_rows,
        host_target_states,
        lookup_by_genus,
        lookup_by_subfamily,
        training_flag_column,
    )
    scenario_indices = build_training_variant_indices(merged_rows)
    feature_rows = build_feature_rows(merged_rows, scenario_indices)
    metadata_rows = build_feature_metadata(args.lookup_path)
    lookup_summary_rows = build_lookup_summary_rows(merged_rows)

    feature_output_path = args.output_dir / f"rbp_receptor_compatibility_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"rbp_receptor_compatibility_feature_metadata_{args.version}.csv"
    lookup_summary_output_path = args.output_dir / f"rbp_receptor_lookup_summary_{args.version}.csv"
    manifest_output_path = args.output_dir / f"rbp_receptor_compatibility_manifest_{args.version}.json"

    write_csv(feature_output_path, ["pair_id", "bacteria", "phage", *OUTPUT_FEATURE_COLUMNS], feature_rows)
    write_csv(metadata_output_path, ["column_name", "type", "source_path", "transform"], metadata_rows)
    write_csv(
        lookup_summary_output_path,
        ["phage", "match_level", "taxon", "target_receptors", "target_family", "evidence_note"],
        lookup_summary_rows,
    )

    lookup_covered_phages = sum(1 for row in lookup_summary_rows if row["match_level"])
    training_positive_count = sum(1 for row in merged_rows if _is_training_positive(row))
    manifest = {
        "step_name": "build_rbp_receptor_compatibility_feature_block",
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_count": len(feature_rows),
        "distinct_bacteria_count": len({row["bacteria"] for row in feature_rows}),
        "distinct_phage_count": len({row["phage"] for row in feature_rows}),
        "feature_count": len(OUTPUT_FEATURE_COLUMNS),
        "lookup_coverage": {
            "covered_phage_count": lookup_covered_phages,
            "uncovered_phage_count": len(lookup_summary_rows) - lookup_covered_phages,
        },
        "training_positive_count": training_positive_count,
        "inputs": {
            "st02_pair_table": {"path": str(args.st02_pair_table_path), "sha256": _sha256(args.st02_pair_table_path)},
            "st03_split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
            },
            "receptor_clusters": {
                "path": str(args.receptor_clusters_path),
                "sha256": _sha256(args.receptor_clusters_path),
            },
            "lookup": {"path": str(args.lookup_path), "sha256": _sha256(args.lookup_path)},
        },
        "policies": {
            "training_flag_column": training_flag_column,
            "lookup_resolution_order": ["phage_genus", "phage_subfamily"],
        },
        "outputs": {
            "feature_csv": str(feature_output_path),
            "feature_metadata_csv": str(metadata_output_path),
            "lookup_summary_csv": str(lookup_summary_output_path),
        },
    }
    write_json(manifest_output_path, manifest)

    print(f"Wrote TE01 pairwise compatibility features to {feature_output_path}")
    print(f"- Pair rows: {len(feature_rows)}")
    print(f"- Lookup-covered phages: {lookup_covered_phages} / {len(lookup_summary_rows)}")
    print(f"- Leakage-safe training positives: {training_positive_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
