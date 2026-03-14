#!/usr/bin/env python3
"""ST0.8: Ingest Tier-A VHRdb rows and summarize internal-only vs +VHRdb ablation slices."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

DATASOURCE_IDENTIFIER_COLUMNS = (
    "datasource",
    "datasource_id",
    "source_datasource_id",
    "source_id",
)
DEFAULT_VHRDB_DATASOURCE_ID = "vhrdb"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--vhrdb-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/vhrdb_interactions.csv"),
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


def normalize_vhrdb_row(row: Dict[str, str]) -> Dict[str, str]:
    global_response_raw = row.get("global_response", "")
    datasource_response_raw = row.get("datasource_response", "")
    global_response_normalized = global_response_raw.lower()
    datasource_response_normalized = datasource_response_raw.lower()
    disagreement = (
        "1"
        if global_response_normalized
        and datasource_response_normalized
        and global_response_normalized != datasource_response_normalized
        else "0"
    )
    datasource_identifier = next(
        (row.get(col, "") for col in DATASOURCE_IDENTIFIER_COLUMNS if row.get(col, "")),
        DEFAULT_VHRDB_DATASOURCE_ID,
    )
    pair_id = f"{row['bacteria']}__{row['phage']}"
    return {
        "pair_id": pair_id,
        "bacteria": row["bacteria"],
        "phage": row["phage"],
        "label_hard_any_lysis": global_response_normalized,
        "label_strict_confidence_tier": row.get("uncertainty", ""),
        "source_system": "vhrdb",
        "source_datasource_id": datasource_identifier,
        "source_native_record_id": row.get("source_native_record_id", ""),
        "source_global_response": global_response_raw,
        "source_datasource_response": datasource_response_raw,
        "source_disagreement_flag": disagreement,
        "source_uncertainty": row.get("uncertainty", ""),
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
            "source_datasource_id": "",
            "source_native_record_id": "",
            "source_global_response": "",
            "source_datasource_response": "",
            "source_disagreement_flag": "0",
            "source_uncertainty": "",
        }
        for row in internal_rows
    ]


def compute_ablation_summary(merged_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    internal_rows = [row for row in merged_rows if row["source_system"] == "internal"]
    vhrdb_rows = [row for row in merged_rows if row["source_system"] == "vhrdb"]
    internal_pairs = {key_for_pair(row) for row in internal_rows}
    vhrdb_pairs = {key_for_pair(row) for row in vhrdb_rows}

    return [
        {"arm": "internal_only", "pair_count": len(internal_pairs), "new_pairs_vs_internal": 0},
        {
            "arm": "plus_vhrdb",
            "pair_count": len(internal_pairs | vhrdb_pairs),
            "new_pairs_vs_internal": len(vhrdb_pairs - internal_pairs),
        },
    ]


def compute_lift_failure_rows(merged_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    vhrdb_rows = [row for row in merged_rows if row["source_system"] == "vhrdb"]
    by_datasource: Counter[str] = Counter(
        row.get("source_datasource_id", "")
        or row.get("source_system", "")
        or DEFAULT_VHRDB_DATASOURCE_ID
        for row in vhrdb_rows
    )
    by_tier: Counter[str] = Counter(row["source_uncertainty"] or "unknown" for row in vhrdb_rows)

    for datasource, count in sorted(by_datasource.items()):
        output.append({"slice_type": "datasource", "slice_value": datasource, "row_count": count})
    for tier, count in sorted(by_tier.items()):
        output.append({"slice_type": "confidence_tier", "slice_value": tier, "row_count": count})

    output.append(
        {
            "slice_type": "quality",
            "slice_value": "datasource_disagreement",
            "row_count": sum(1 for row in vhrdb_rows if row["source_disagreement_flag"] == "1"),
        }
    )
    return output


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    internal_rows_raw = read_csv_rows(args.internal_pair_table_path, REQUIRED_INTERNAL_COLUMNS)
    vhrdb_rows_raw = read_csv_rows(args.vhrdb_path, REQUIRED_VHRDB_COLUMNS)

    merged_rows = [*build_internal_rows(internal_rows_raw), *[normalize_vhrdb_row(row) for row in vhrdb_rows_raw]]
    ablation_rows = compute_ablation_summary(merged_rows)
    lift_failure_rows = compute_lift_failure_rows(merged_rows)

    merged_output_path = args.output_dir / "st08_vhrdb_ingested_pairs.csv"
    ablation_output_path = args.output_dir / "st08_ablation_summary.csv"
    lift_failure_output_path = args.output_dir / "st08_lift_failure_slices.csv"
    manifest_output_path = args.output_dir / "st08_vhrdb_manifest.json"

    merged_fieldnames = [
        "pair_id",
        "bacteria",
        "phage",
        "label_hard_any_lysis",
        "label_strict_confidence_tier",
        "source_system",
        "source_datasource_id",
        "source_native_record_id",
        "source_global_response",
        "source_datasource_response",
        "source_disagreement_flag",
        "source_uncertainty",
    ]
    write_csv(merged_output_path, fieldnames=merged_fieldnames, rows=merged_rows)
    write_csv(ablation_output_path, fieldnames=["arm", "pair_count", "new_pairs_vs_internal"], rows=ablation_rows)
    write_csv(
        lift_failure_output_path,
        fieldnames=["slice_type", "slice_value", "row_count"],
        rows=lift_failure_rows,
    )

    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "st08_vhrdb_ingest_ablation",
            "input_paths": {
                "internal_pair_table": str(args.internal_pair_table_path),
                "vhrdb": str(args.vhrdb_path),
            },
            "input_hashes_sha256": {
                "internal_pair_table": hashlib.sha256(args.internal_pair_table_path.read_bytes()).hexdigest(),
                "vhrdb": hashlib.sha256(args.vhrdb_path.read_bytes()).hexdigest(),
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
