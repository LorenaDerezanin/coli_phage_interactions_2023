#!/usr/bin/env python3
"""ST0.8: Ingest ordered Tier-A external sources and summarize sequential ablation slices."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

REQUIRED_INTERNAL_COLUMNS = ("pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier")
REQUIRED_VHRDB_COLUMNS = (
    "source_native_record_id",
    "bacteria",
    "phage",
    "global_response",
    "datasource_response",
    "uncertainty",
)
REQUIRED_GENERIC_TIER_A_COLUMNS = (
    "source_native_record_id",
    "bacteria",
    "phage",
    "label_hard_any_lysis",
)
REQUIRED_SOURCE_REGISTRY_COLUMNS = ("source_id", "confidence_tier")

DATASOURCE_IDENTIFIER_COLUMNS = (
    "datasource",
    "datasource_id",
    "source_datasource_id",
    "source_id",
)
DEFAULT_VHRDB_DATASOURCE_ID = "vhrdb"
TIER_A_SOURCE_PRIORITY = ("vhrdb", "basel", "klebphacol", "gpb")


@dataclass(frozen=True)
class TierASourceSpec:
    """Configuration for one Tier A input file."""

    source_id: str
    path: Path
    required_columns: Sequence[str]
    normalizer: Callable[[Dict[str, str]], Dict[str, str]]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--source-registry-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/source_registry.csv"),
    )
    parser.add_argument(
        "--vhrdb-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/vhrdb_interactions.csv"),
    )
    parser.add_argument(
        "--basel-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/basel_interactions.csv"),
    )
    parser.add_argument(
        "--klebphacol-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/klebphacol_interactions.csv"),
    )
    parser.add_argument(
        "--gpb-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/gpb_interactions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def key_for_pair(row: Dict[str, str]) -> Tuple[str, str]:
    return (row["bacteria"], row["phage"])


def _resolve_datasource_identifier(row: Dict[str, str], fallback: str) -> str:
    return next((row.get(col, "") for col in DATASOURCE_IDENTIFIER_COLUMNS if row.get(col, "")), fallback)


def _disagreement_flag(global_response: str, datasource_response: str) -> str:
    global_response_normalized = global_response.lower()
    datasource_response_normalized = datasource_response.lower()
    return (
        "1"
        if global_response_normalized
        and datasource_response_normalized
        and global_response_normalized != datasource_response_normalized
        else "0"
    )


def _first_non_blank_value(row: Dict[str, str], columns: Sequence[str], default: str = "") -> str:
    for column in columns:
        value = row.get(column, "")
        if value:
            return value
    return default


def normalize_vhrdb_row(row: Dict[str, str]) -> Dict[str, str]:
    global_response_raw = row.get("global_response", "")
    datasource_response_raw = row.get("datasource_response", "")
    global_response_normalized = global_response_raw.lower()
    datasource_identifier = _resolve_datasource_identifier(row, DEFAULT_VHRDB_DATASOURCE_ID)
    pair_id = f"{row['bacteria']}__{row['phage']}"
    return {
        "pair_id": pair_id,
        "bacteria": row["bacteria"],
        "phage": row["phage"],
        "label_hard_any_lysis": global_response_normalized,
        "label_strict_confidence_tier": row.get("uncertainty", ""),
        "source_system": "vhrdb",
        "global_response": global_response_raw,
        "datasource_response": datasource_response_raw,
        "source_datasource_id": datasource_identifier,
        "source_native_record_id": row.get("source_native_record_id", ""),
        "source_disagreement_flag": _disagreement_flag(global_response_raw, datasource_response_raw),
        "source_uncertainty": row.get("uncertainty", ""),
        "source_strength_label": "",
    }


def normalize_generic_tier_a_row(row: Dict[str, str], source_id: str) -> Dict[str, str]:
    global_response_raw = row.get("global_response", "")
    datasource_response_raw = row.get("datasource_response", "")
    disagreement_flag = row.get("source_disagreement_flag", "") or _disagreement_flag(
        global_response_raw, datasource_response_raw
    )
    source_uncertainty = _first_non_blank_value(
        row,
        ("source_uncertainty", "label_strict_confidence_tier"),
    )
    pair_id = f"{row['bacteria']}__{row['phage']}"
    return {
        "pair_id": pair_id,
        "bacteria": row["bacteria"],
        "phage": row["phage"],
        "label_hard_any_lysis": row["label_hard_any_lysis"].lower(),
        "label_strict_confidence_tier": row.get("label_strict_confidence_tier", ""),
        "source_system": source_id,
        "global_response": global_response_raw,
        "datasource_response": datasource_response_raw,
        "source_datasource_id": _resolve_datasource_identifier(row, source_id),
        "source_native_record_id": row.get("source_native_record_id", ""),
        "source_disagreement_flag": disagreement_flag,
        "source_uncertainty": source_uncertainty,
        "source_strength_label": _first_non_blank_value(
            row,
            ("source_strength_label", "potency_label", "lysis_strength"),
        ),
    }


def build_internal_rows(internal_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "pair_id": row["pair_id"],
            "bacteria": row["bacteria"],
            "phage": row["phage"],
            "label_hard_any_lysis": row["label_hard_any_lysis"],
            "label_strict_confidence_tier": row["label_strict_confidence_tier"],
            "source_system": "internal",
            "global_response": "",
            "datasource_response": "",
            "source_datasource_id": "",
            "source_native_record_id": "",
            "source_disagreement_flag": "0",
            "source_uncertainty": "",
            "source_strength_label": "",
        }
        for row in internal_rows
    ]


def read_source_registry(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path, REQUIRED_SOURCE_REGISTRY_COLUMNS)
    return {row["source_id"]: row for row in rows}


def build_tier_a_source_specs(args: argparse.Namespace, registry_rows: Dict[str, Dict[str, str]]) -> List[TierASourceSpec]:
    path_by_source = {
        "vhrdb": args.vhrdb_path,
        "basel": args.basel_path,
        "klebphacol": args.klebphacol_path,
        "gpb": args.gpb_path,
    }
    candidate_source_ids = [
        source_id
        for source_id in TIER_A_SOURCE_PRIORITY
        if source_id in registry_rows and path_by_source[source_id].exists()
    ]

    wrong_tier = [
        source_id
        for source_id in candidate_source_ids
        if registry_rows[source_id].get("confidence_tier", "") != "A"
    ]
    if wrong_tier:
        wrong_text = ", ".join(sorted(wrong_tier))
        raise ValueError(f"Expected Tier A confidence tier for sources: {wrong_text}")

    source_specs: List[TierASourceSpec] = []
    for source_id in candidate_source_ids:
        path = path_by_source[source_id]
        if source_id == "vhrdb":
            source_specs.append(
                TierASourceSpec(
                    source_id=source_id,
                    path=path,
                    required_columns=REQUIRED_VHRDB_COLUMNS,
                    normalizer=normalize_vhrdb_row,
                )
            )
            continue
        source_specs.append(
            TierASourceSpec(
                source_id=source_id,
                path=path,
                required_columns=REQUIRED_GENERIC_TIER_A_COLUMNS,
                normalizer=lambda row, source_id=source_id: normalize_generic_tier_a_row(row, source_id),
            )
        )
    return source_specs


def load_tier_a_rows(source_specs: Sequence[TierASourceSpec]) -> List[Dict[str, str]]:
    # Duplicate (bacteria, phage) pairs across sources are kept intentionally;
    # downstream consumers may need per-source records for ablation and provenance.
    external_rows: List[Dict[str, str]] = []
    for source_spec in source_specs:
        source_rows = read_csv_rows(source_spec.path, source_spec.required_columns)
        external_rows.extend(source_spec.normalizer(row) for row in source_rows)
    return external_rows


def compute_ablation_summary(
    merged_rows: List[Dict[str, str]], tier_a_priority: Sequence[str] = TIER_A_SOURCE_PRIORITY
) -> List[Dict[str, object]]:
    internal_rows = [row for row in merged_rows if row["source_system"] == "internal"]
    internal_pairs = {key_for_pair(row) for row in internal_rows}
    cumulative_pairs = set(internal_pairs)

    output: List[Dict[str, object]] = [
        {
            "arm": "internal_only",
            "source_system": "internal",
            "pair_count": len(cumulative_pairs),
            "new_pairs_vs_internal": 0,
            "new_pairs_vs_previous_arm": 0,
        }
    ]

    for source_id in tier_a_priority:
        source_pairs = {key_for_pair(row) for row in merged_rows if row["source_system"] == source_id}
        next_pairs = cumulative_pairs | source_pairs
        output.append(
            {
                "arm": f"plus_{source_id}",
                "source_system": source_id,
                "pair_count": len(next_pairs),
                "new_pairs_vs_internal": len(next_pairs - internal_pairs),
                "new_pairs_vs_previous_arm": len(next_pairs - cumulative_pairs),
            }
        )
        cumulative_pairs = next_pairs
    return output


def compute_lift_failure_rows(merged_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    external_rows = [row for row in merged_rows if row["source_system"] != "internal"]
    by_source: Counter[str] = Counter(row["source_system"] for row in external_rows)
    by_datasource: Counter[str] = Counter(
        row.get("source_datasource_id", "") or row.get("source_system", "") or DEFAULT_VHRDB_DATASOURCE_ID
        for row in external_rows
    )
    by_tier: Counter[str] = Counter(row["source_uncertainty"] or "unknown" for row in external_rows)

    for source_id, count in sorted(by_source.items()):
        output.append({"slice_type": "source_system", "slice_value": source_id, "row_count": count})
    for datasource, count in sorted(by_datasource.items()):
        output.append({"slice_type": "datasource", "slice_value": datasource, "row_count": count})
    for tier, count in sorted(by_tier.items()):
        output.append({"slice_type": "confidence_tier", "slice_value": tier, "row_count": count})

    output.append(
        {
            "slice_type": "quality",
            "slice_value": "datasource_disagreement",
            "row_count": sum(1 for row in external_rows if row["source_disagreement_flag"] == "1"),
        }
    )
    output.append(
        {
            "slice_type": "quality",
            "slice_value": "strength_labels_present",
            "row_count": sum(1 for row in external_rows if row["source_strength_label"]),
        }
    )
    return output


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    internal_rows_raw = read_csv_rows(args.internal_pair_table_path, REQUIRED_INTERNAL_COLUMNS)
    source_registry_rows = read_source_registry(args.source_registry_path)
    source_specs = build_tier_a_source_specs(args, source_registry_rows)
    tier_a_rows = load_tier_a_rows(source_specs)

    merged_rows = [*build_internal_rows(internal_rows_raw), *tier_a_rows]
    active_source_ids = [source_spec.source_id for source_spec in source_specs]
    ablation_rows = compute_ablation_summary(merged_rows, tier_a_priority=active_source_ids)
    lift_failure_rows = compute_lift_failure_rows(merged_rows)

    merged_output_path = args.output_dir / "st08_tier_a_ingested_pairs.csv"
    ablation_output_path = args.output_dir / "st08_ablation_summary.csv"
    lift_failure_output_path = args.output_dir / "st08_lift_failure_slices.csv"
    manifest_output_path = args.output_dir / "st08_tier_a_manifest.json"

    merged_fieldnames = [
        "pair_id",
        "bacteria",
        "phage",
        "label_hard_any_lysis",
        "label_strict_confidence_tier",
        "source_system",
        "global_response",
        "datasource_response",
        "source_datasource_id",
        "source_native_record_id",
        "source_disagreement_flag",
        "source_uncertainty",
        "source_strength_label",
    ]
    write_csv(merged_output_path, fieldnames=merged_fieldnames, rows=merged_rows)
    write_csv(
        ablation_output_path,
        fieldnames=["arm", "source_system", "pair_count", "new_pairs_vs_internal", "new_pairs_vs_previous_arm"],
        rows=ablation_rows,
    )
    write_csv(
        lift_failure_output_path,
        fieldnames=["slice_type", "slice_value", "row_count"],
        rows=lift_failure_rows,
    )

    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "st08_tier_a_ingest_ablation",
            "tier_a_source_priority": list(TIER_A_SOURCE_PRIORITY),
            "active_tier_a_sources": active_source_ids,
            "input_paths": {
                "internal_pair_table": str(args.internal_pair_table_path),
                "source_registry": str(args.source_registry_path),
                **{source_spec.source_id: str(source_spec.path) for source_spec in source_specs},
            },
            "input_hashes_sha256": {
                "internal_pair_table": hashlib.sha256(args.internal_pair_table_path.read_bytes()).hexdigest(),
                "source_registry": hashlib.sha256(args.source_registry_path.read_bytes()).hexdigest(),
                **{
                    source_spec.source_id: hashlib.sha256(source_spec.path.read_bytes()).hexdigest()
                    for source_spec in source_specs
                },
            },
            "output_paths": {
                "ingested_pairs": str(merged_output_path),
                "ablation_summary": str(ablation_output_path),
                "lift_failure_slices": str(lift_failure_output_path),
            },
        },
    )


if __name__ == "__main__":
    main()
