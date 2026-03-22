#!/usr/bin/env python3
"""TI10: Track incremental lift and failure modes by datasource and confidence tier."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round

REQUIRED_COHORT_COLUMNS = (
    "pair_id",
    "source_system",
    "first_training_arm",
    "first_training_arm_index",
    "effective_training_weight",
    "integration_status",
    "external_label_include_in_training",
    "external_label_confidence_tier",
    "source_resolution_status",
    "source_disagreement_flag",
    "source_qc_flag",
)
REQUIRED_ABLATION_COLUMNS = (
    "arm",
    "arm_index",
    "cumulative_row_count",
    "cumulative_pair_count",
    "cumulative_external_row_count",
    "cumulative_external_pair_count",
    "new_rows_vs_previous_arm",
    "new_pairs_vs_previous_arm",
    "cumulative_training_weight",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-cohort-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"),
    )
    parser.add_argument(
        "--strict-ablation-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/strict_ablation_sequence/ti09_strict_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/incremental_lift_failure_analysis"),
    )
    return parser.parse_args(argv)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def _split_pipe_values(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _source_tier_key(row: Mapping[str, str]) -> Tuple[str, str]:
    return (str(row.get("source_system", "")), str(row.get("external_label_confidence_tier", "")))


def _source_tier_label(source_system: str, confidence_tier: str) -> str:
    return f"{source_system}:{confidence_tier or 'unknown'}"


def _failure_modes_for_row(row: Mapping[str, str]) -> List[str]:
    modes: List[str] = []
    if row.get("external_label_include_in_training", "") != "1":
        modes.append("excluded_by_confidence")
    if row.get("source_disagreement_flag", "") == "1":
        modes.append("source_disagreement")
    qc_flag = row.get("source_qc_flag", "")
    if qc_flag and qc_flag != "ok":
        modes.append("non_ok_qc")
    resolution_status = set(_split_pipe_values(row.get("source_resolution_status", "")))
    if resolution_status & {"unresolved", "missing"}:
        modes.append("unresolved_entity_mapping")
    return modes or ["clean"]


def load_training_cohort_rows(path: Path) -> List[Dict[str, str]]:
    return [_normalize_row(row) for row in read_csv_rows(path, REQUIRED_COHORT_COLUMNS)]


def load_strict_ablation_rows(path: Path) -> List[Dict[str, str]]:
    return [_normalize_row(row) for row in read_csv_rows(path, REQUIRED_ABLATION_COLUMNS)]


def compute_arm_lift_rows(ablation_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    if not ablation_rows:
        raise ValueError("No strict ablation rows were supplied.")

    normalized = [_normalize_row(row) for row in ablation_rows]
    baseline_row = normalized[0]
    baseline_row_count = int(baseline_row["cumulative_row_count"])
    baseline_pair_count = int(baseline_row["cumulative_pair_count"])
    baseline_external_row_count = int(baseline_row["cumulative_external_row_count"])
    baseline_external_pair_count = int(baseline_row["cumulative_external_pair_count"])
    baseline_weight = float(baseline_row["cumulative_training_weight"] or 0.0)

    summary_rows: List[Dict[str, object]] = []
    previous_row_count = baseline_row_count
    previous_pair_count = baseline_pair_count

    for row in normalized:
        current_row_count = int(row["cumulative_row_count"])
        current_pair_count = int(row["cumulative_pair_count"])
        current_external_row_count = int(row["cumulative_external_row_count"])
        current_external_pair_count = int(row["cumulative_external_pair_count"])
        current_weight = float(row["cumulative_training_weight"] or 0.0)
        summary_rows.append(
            {
                "arm": row["arm"],
                "arm_index": int(row["arm_index"]),
                "source_system_added": row.get("source_system_added", ""),
                "cumulative_row_count": current_row_count,
                "cumulative_pair_count": current_pair_count,
                "cumulative_external_row_count": current_external_row_count,
                "cumulative_external_pair_count": current_external_pair_count,
                "new_rows_vs_previous_arm": int(row["new_rows_vs_previous_arm"]),
                "new_pairs_vs_previous_arm": int(row["new_pairs_vs_previous_arm"]),
                "cumulative_training_weight": safe_round(current_weight),
                "row_lift_vs_internal": current_row_count - baseline_row_count,
                "pair_lift_vs_internal": current_pair_count - baseline_pair_count,
                "external_row_lift_vs_internal": current_external_row_count - baseline_external_row_count,
                "external_pair_lift_vs_internal": current_external_pair_count - baseline_external_pair_count,
                "training_weight_lift_vs_internal": safe_round(current_weight - baseline_weight),
                "incremental_row_share": safe_round(
                    int(row["new_rows_vs_previous_arm"]) / previous_row_count if previous_row_count else 0.0
                ),
                "incremental_pair_share": safe_round(
                    int(row["new_pairs_vs_previous_arm"]) / previous_pair_count if previous_pair_count else 0.0
                ),
            }
        )
        previous_row_count = current_row_count
        previous_pair_count = current_pair_count
    return summary_rows


def _group_rows(rows: Sequence[Mapping[str, str]], key_fn) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(_normalize_row(row))
    return grouped


def compute_source_tier_lift_rows(
    cohort_rows: Sequence[Mapping[str, str]],
    ablation_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, object]]:
    arm_rows = {row["arm"]: _normalize_row(row) for row in ablation_rows}
    grouped_rows = _group_rows(
        [row for row in cohort_rows if row.get("source_system", "") != "internal"],
        _source_tier_key,
    )

    summary_rows: List[Dict[str, object]] = []
    for (source_system, confidence_tier), rows in sorted(grouped_rows.items()):
        included_rows = [row for row in rows if row.get("external_label_include_in_training", "") == "1"]
        first_arm = included_rows[0].get("first_training_arm", "excluded") if included_rows else "excluded"
        first_arm_row = included_rows[0] if included_rows else rows[0]
        arm_summary = arm_rows.get(first_arm)
        row_count = len(rows)
        pair_ids = {str(row["pair_id"]) for row in rows}
        training_weight = sum(float(row.get("effective_training_weight", "0") or 0.0) for row in included_rows)
        included_row_count = len(included_rows)
        excluded_row_count = row_count - included_row_count

        summary_rows.append(
            {
                "source_system": source_system,
                "confidence_tier": confidence_tier or "unknown",
                "first_training_arm": first_arm,
                "first_training_arm_index": int(first_arm_row.get("first_training_arm_index", "-1") or -1),
                "row_count": row_count,
                "pair_count": len(pair_ids),
                "training_weight": safe_round(training_weight),
                "mean_training_weight": safe_round(training_weight / included_row_count if included_row_count else 0.0),
                "included_row_count": included_row_count,
                "excluded_row_count": excluded_row_count,
                "row_exclusion_rate": safe_round(excluded_row_count / row_count if row_count else 0.0),
                "arm_cumulative_row_count": int(arm_summary["cumulative_row_count"]) if arm_summary else None,
                "arm_cumulative_pair_count": int(arm_summary["cumulative_pair_count"]) if arm_summary else None,
                "arm_new_rows_vs_previous_arm": int(arm_summary["new_rows_vs_previous_arm"]) if arm_summary else None,
                "arm_new_pairs_vs_previous_arm": int(arm_summary["new_pairs_vs_previous_arm"]) if arm_summary else None,
                "row_share_of_arm": safe_round(
                    row_count / int(arm_summary["cumulative_row_count"])
                    if arm_summary and int(arm_summary["cumulative_row_count"])
                    else 0.0
                ),
                "pair_share_of_arm": safe_round(
                    len(pair_ids) / int(arm_summary["cumulative_pair_count"])
                    if arm_summary and int(arm_summary["cumulative_pair_count"])
                    else 0.0
                ),
            }
        )
    return summary_rows


def compute_failure_mode_rows(cohort_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in cohort_rows:
        if row.get("source_system", "") == "internal":
            continue
        confidence_tier = row.get("external_label_confidence_tier", "") or "unknown"
        for mode in _failure_modes_for_row(row):
            grouped[(row.get("source_system", ""), confidence_tier, mode)].append(_normalize_row(row))

    summary_rows: List[Dict[str, object]] = []
    for (source_system, confidence_tier, mode), rows in sorted(grouped.items()):
        pair_ids = {str(row["pair_id"]) for row in rows}
        summary_rows.append(
            {
                "source_system": source_system,
                "confidence_tier": confidence_tier,
                "failure_mode": mode,
                "row_count": len(rows),
                "pair_count": len(pair_ids),
                "source_tier": _source_tier_label(source_system, confidence_tier),
            }
        )
    return summary_rows


def compute_summary_manifest(
    cohort_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, object]],
    source_tier_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    external_rows = [row for row in cohort_rows if row.get("source_system", "") != "internal"]
    failure_mode_totals = Counter()
    for row in failure_rows:
        failure_mode_totals[str(row["failure_mode"])] += int(row["row_count"])
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "build_incremental_lift_failure_analysis",
        "source_systems": sorted({row.get("source_system", "") for row in external_rows}),
        "confidence_tiers": sorted(
            {row.get("external_label_confidence_tier", "") or "unknown" for row in external_rows}
        ),
        "row_counts": {
            "internal": sum(1 for row in cohort_rows if row.get("source_system", "") == "internal"),
            "external": len(external_rows),
            "included_external": sum(
                1 for row in external_rows if row.get("external_label_include_in_training", "") == "1"
            ),
            "excluded_external": sum(
                1 for row in external_rows if row.get("external_label_include_in_training", "") != "1"
            ),
        },
        "arm_summary": {
            "arms": [row["arm"] for row in arm_rows],
            "max_row_lift_vs_internal": max((int(row["row_lift_vs_internal"]) for row in arm_rows), default=0),
            "max_pair_lift_vs_internal": max((int(row["pair_lift_vs_internal"]) for row in arm_rows), default=0),
        },
        "source_tier_summary": {
            "slice_count": len(source_tier_rows),
            "max_row_count": max((int(row["row_count"]) for row in source_tier_rows), default=0),
        },
        "failure_modes": dict(sorted(failure_mode_totals.items())),
    }


def ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    cohort_rows = load_training_cohort_rows(args.training_cohort_path)
    ablation_rows = load_strict_ablation_rows(args.strict_ablation_summary_path)

    arm_rows = compute_arm_lift_rows(ablation_rows)
    source_tier_rows = compute_source_tier_lift_rows(cohort_rows, ablation_rows)
    failure_rows = compute_failure_mode_rows(cohort_rows)
    manifest = compute_summary_manifest(cohort_rows, arm_rows, source_tier_rows, failure_rows)

    arm_output_path = args.output_dir / "ti10_incremental_lift_summary.csv"
    source_tier_output_path = args.output_dir / "ti10_source_tier_lift_summary.csv"
    failure_output_path = args.output_dir / "ti10_failure_mode_summary.csv"
    manifest_output_path = args.output_dir / "ti10_incremental_lift_manifest.json"

    write_csv(arm_output_path, fieldnames=ordered_fieldnames(arm_rows), rows=arm_rows)
    write_csv(source_tier_output_path, fieldnames=ordered_fieldnames(source_tier_rows), rows=source_tier_rows)
    write_csv(failure_output_path, fieldnames=ordered_fieldnames(failure_rows), rows=failure_rows)
    write_json(
        manifest_output_path,
        {
            **manifest,
            "input_paths": {
                "training_cohort_rows": str(args.training_cohort_path),
                "strict_ablation_summary": str(args.strict_ablation_summary_path),
            },
            "input_hashes_sha256": {
                "training_cohort_rows": _sha256(args.training_cohort_path),
                "strict_ablation_summary": _sha256(args.strict_ablation_summary_path),
            },
            "output_paths": {
                "incremental_lift_summary": str(arm_output_path),
                "source_tier_lift_summary": str(source_tier_output_path),
                "failure_mode_summary": str(failure_output_path),
            },
        },
    )


if __name__ == "__main__":
    main()
