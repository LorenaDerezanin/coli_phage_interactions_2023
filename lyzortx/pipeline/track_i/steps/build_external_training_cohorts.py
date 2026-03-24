#!/usr/bin/env python3
"""TI08: Build internal-only and external-enhanced training cohorts from TI07 rows."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round

REQUIRED_INTERNAL_COLUMNS = ("pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier")
REQUIRED_EXTERNAL_COLUMNS = (
    "pair_id",
    "bacteria",
    "phage",
    "label_hard_any_lysis",
    "label_strict_confidence_tier",
    "source_system",
    "confidence_tier",
    "training_weight",
    "external_label_confidence_tier",
    "external_label_confidence_score",
    "external_label_training_weight",
    "external_label_include_in_training",
)

TRAINING_ARM_ORDER = (
    "internal_only",
    "plus_vhrdb",
    "plus_basel",
    "plus_klebphacol",
    "plus_gpb",
    "plus_tier_b",
)
TRAINING_ARM_INDEX = {arm: idx for idx, arm in enumerate(TRAINING_ARM_ORDER)}
TIER_A_SOURCE_TO_ARM = {
    "vhrdb": "plus_vhrdb",
    "basel": "plus_basel",
    "klebphacol": "plus_klebphacol",
    "gpb": "plus_gpb",
}
TIER_B_SOURCES = {"virus_host_db", "ncbi_virus_biosample"}
OUTPUT_APPEND_COLUMNS = [
    "source_family",
    "first_training_arm",
    "first_training_arm_index",
    "confidence_tier",
    "training_weight",
    "effective_training_weight",
    "integration_status",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--external-confidence-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_i/external_label_confidence_tiers/ti07_external_label_confidence_pairs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/training_cohort_integration"),
    )
    return parser.parse_args(argv)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_training_arm_for_source(source_system: str) -> str:
    if source_system in TIER_A_SOURCE_TO_ARM:
        return TIER_A_SOURCE_TO_ARM[source_system]
    if source_system in TIER_B_SOURCES:
        return "plus_tier_b"
    raise ValueError(f"Unsupported external source_system for TI08 integration: {source_system!r}")


def source_family_for_source(source_system: str) -> str:
    if source_system in TIER_A_SOURCE_TO_ARM:
        return "tier_a"
    if source_system in TIER_B_SOURCES:
        return "tier_b"
    if source_system == "internal":
        return "internal"
    raise ValueError(f"Unsupported source_system for TI08 integration: {source_system!r}")


def load_external_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing TI07 external confidence cohort: {path}")
    return [_normalize_row(row) for row in read_csv_rows(path, REQUIRED_EXTERNAL_COLUMNS)]


def build_integrated_training_rows(
    internal_rows: Sequence[Mapping[str, str]],
    external_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, object]]:
    integrated_rows: List[Dict[str, object]] = []

    for row in internal_rows:
        normalized = _normalize_row(row)
        normalized.update(
            {
                "source_system": "internal",
                "source_family": "internal",
                "first_training_arm": "internal_only",
                "first_training_arm_index": TRAINING_ARM_INDEX["internal_only"],
                "confidence_tier": "",
                "training_weight": "",
                "effective_training_weight": 1.0,
                "integration_status": "baseline_internal",
            }
        )
        integrated_rows.append(normalized)

    for row in external_rows:
        normalized = _normalize_row(row)
        source_system = normalized["source_system"]
        include_in_training = normalized.get("external_label_include_in_training", "") == "1"
        first_arm = first_training_arm_for_source(source_system)
        normalized.update(
            {
                "source_family": source_family_for_source(source_system),
                "first_training_arm": first_arm if include_in_training else "excluded",
                "first_training_arm_index": TRAINING_ARM_INDEX[first_arm] if include_in_training else -1,
                "confidence_tier": normalized.get("confidence_tier", "")
                or normalized.get("external_label_confidence_tier", ""),
                "training_weight": normalized.get("training_weight", "")
                or normalized.get("external_label_training_weight", ""),
                "effective_training_weight": (
                    float(normalized.get("external_label_training_weight", "0") or 0.0) if include_in_training else 0.0
                ),
                "integration_status": "external_enhancer" if include_in_training else "excluded_by_confidence",
            }
        )
        integrated_rows.append(normalized)

    return sorted(
        integrated_rows,
        key=lambda row: (
            int(row["first_training_arm_index"]),
            str(row["source_system"]),
            str(row["pair_id"]),
            str(row.get("source_native_record_id", "")),
        ),
    )


def _trainable_rows_for_arm(rows: Sequence[Mapping[str, object]], arm: str) -> List[Mapping[str, object]]:
    arm_index = TRAINING_ARM_INDEX[arm]
    return [
        row
        for row in rows
        if int(row["first_training_arm_index"]) >= 0 and int(row["first_training_arm_index"]) <= arm_index
    ]


def compute_training_arm_summary(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    previous_pair_ids: set[str] = set()
    previous_row_count = 0

    for arm in TRAINING_ARM_ORDER:
        arm_rows = _trainable_rows_for_arm(rows, arm)
        pair_ids = {str(row["pair_id"]) for row in arm_rows}
        external_rows = [row for row in arm_rows if str(row["source_system"]) != "internal"]
        external_pair_ids = {str(row["pair_id"]) for row in external_rows}
        row_count = len(arm_rows)
        summary_rows.append(
            {
                "arm": arm,
                "arm_index": TRAINING_ARM_INDEX[arm],
                "source_system_added": arm.replace("plus_", "") if arm != "internal_only" else "internal",
                "cumulative_row_count": row_count,
                "cumulative_pair_count": len(pair_ids),
                "cumulative_external_row_count": len(external_rows),
                "cumulative_external_pair_count": len(external_pair_ids),
                "new_rows_vs_previous_arm": row_count - previous_row_count,
                "new_pairs_vs_previous_arm": len(pair_ids - previous_pair_ids),
                "cumulative_training_weight": safe_round(
                    sum(float(row["effective_training_weight"]) for row in arm_rows)
                ),
            }
        )
        previous_pair_ids = pair_ids
        previous_row_count = row_count

    return summary_rows


def compute_integration_status_summary(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["integration_status"]), str(row["source_system"]))
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "slice_type": "integration_status_and_source",
            "slice_value": f"{integration_status}:{source_system}",
            "row_count": count,
        }
        for (integration_status, source_system), count in sorted(counts.items())
    ]


def ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for column in OUTPUT_APPEND_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    return fieldnames


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    internal_rows = read_csv_rows(args.internal_pair_table_path, REQUIRED_INTERNAL_COLUMNS)
    external_rows = load_external_rows(args.external_confidence_path)

    integrated_rows = build_integrated_training_rows(internal_rows, external_rows)
    included_external_rows = [
        row
        for row in integrated_rows
        if str(row["source_system"]) != "internal" and int(row["first_training_arm_index"]) >= 0
    ]
    if not included_external_rows:
        raise ValueError("TI08 requires >0 external rows with external_label_include_in_training=1")
    arm_summary_rows = compute_training_arm_summary(integrated_rows)
    integration_status_rows = compute_integration_status_summary(integrated_rows)

    rows_output_path = args.output_dir / "ti08_training_cohort_rows.csv"
    arm_summary_output_path = args.output_dir / "ti08_training_arm_summary.csv"
    status_summary_output_path = args.output_dir / "ti08_integration_status_summary.csv"
    manifest_output_path = args.output_dir / "ti08_training_cohort_manifest.json"

    write_csv(rows_output_path, fieldnames=ordered_fieldnames(integrated_rows), rows=integrated_rows)
    write_csv(
        arm_summary_output_path,
        fieldnames=[
            "arm",
            "arm_index",
            "source_system_added",
            "cumulative_row_count",
            "cumulative_pair_count",
            "cumulative_external_row_count",
            "cumulative_external_pair_count",
            "new_rows_vs_previous_arm",
            "new_pairs_vs_previous_arm",
            "cumulative_training_weight",
        ],
        rows=arm_summary_rows,
    )
    write_csv(
        status_summary_output_path,
        fieldnames=["slice_type", "slice_value", "row_count"],
        rows=integration_status_rows,
    )
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_external_training_cohorts",
            "training_arm_order": list(TRAINING_ARM_ORDER),
            "external_confidence_input_present": args.external_confidence_path.exists(),
            "active_external_sources": sorted({row["source_system"] for row in external_rows}),
            "input_paths": {
                "internal_pair_table": str(args.internal_pair_table_path),
                "external_confidence_pairs": str(args.external_confidence_path),
            },
            "input_hashes_sha256": {
                "internal_pair_table": _hash_path(args.internal_pair_table_path),
                **(
                    {"external_confidence_pairs": _hash_path(args.external_confidence_path)}
                    if args.external_confidence_path.exists()
                    else {}
                ),
            },
            "output_paths": {
                "rows": str(rows_output_path),
                "arm_summary": str(arm_summary_output_path),
                "integration_status_summary": str(status_summary_output_path),
            },
        },
    )


if __name__ == "__main__":
    main()
