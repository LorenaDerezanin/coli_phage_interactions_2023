#!/usr/bin/env python3
"""Validate Track A generated artifacts and integrity status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.track_a.steps import build_track_a_foundation


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a"),
        help="Track A output directory.",
    )
    parser.add_argument(
        "--run-build",
        action="store_true",
        help="Run Track A build before validating artifacts.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail if warning-level integrity checks failed.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def require_paths(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Track A artifacts: " + ", ".join(missing))


def as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for {field_name}, got {value!r}") from exc


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.run_build:
        build_track_a_foundation.main(["--output-dir", str(args.output_dir)])

    output_dir = args.output_dir
    id_map_dir = output_dir / "id_map"
    integrity_dir = output_dir / "integrity"
    cohort_dir = output_dir / "cohort"
    labels_dir = output_dir / "labels"
    qc_dir = output_dir / "qc"

    required_paths = [
        output_dir / "track_a_manifest.json",
        id_map_dir / "bacteria_id_map.csv",
        id_map_dir / "phage_id_map.csv",
        id_map_dir / "bacteria_alias_resolution.csv",
        id_map_dir / "phage_alias_resolution.csv",
        id_map_dir / "bacteria_alias_candidates.csv",
        id_map_dir / "phage_alias_candidates.csv",
        integrity_dir / "integrity_checks.csv",
        integrity_dir / "integrity_report.json",
        cohort_dir / "cohort_contracts.csv",
        cohort_dir / "cohort_contracts.json",
        labels_dir / "track_a_observations_with_ids.csv",
        labels_dir / "track_a_pair_dilution_summary.csv",
        labels_dir / "track_a_pair_observation_grid.csv",
        labels_dir / "label_set_v1_pairs.csv",
        labels_dir / "label_set_v2_pairs.csv",
        labels_dir / "label_set_v1_policy.json",
        labels_dir / "label_set_v2_policy.json",
        labels_dir / "label_set_v1_summary.json",
        labels_dir / "label_set_v2_summary.json",
        labels_dir / "label_set_v1_v2_comparison.csv",
        qc_dir / "plaque_image_qc_queue.csv",
        qc_dir / "plaque_image_qc_summary.json",
    ]
    require_paths(required_paths)

    integrity = load_json(integrity_dir / "integrity_report.json")
    error_failures = as_int(integrity.get("failed_error_check_count"), "failed_error_check_count")
    warning_failures = as_int(
        integrity.get("failed_warning_check_count"),
        "failed_warning_check_count",
    )
    if error_failures > 0:
        raise SystemExit(
            "Track A integrity check failed: "
            f"failed_error_check_count={error_failures}, failed_warning_check_count={warning_failures}"
        )
    if args.fail_on_warnings and warning_failures > 0:
        raise SystemExit(
            "Track A integrity warning gate failed: "
            f"failed_warning_check_count={warning_failures}"
        )

    cohort_rows = load_csv_rows(cohort_dir / "cohort_contracts.csv")
    expected_cohorts = {
        "raw369": 369,
        "matrix402": 402,
        "features404": 404,
    }
    actual_counts = {
        row["cohort_name"]: as_int(row["bacteria_count"], f"bacteria_count[{row['cohort_name']}]")
        for row in cohort_rows
    }
    for cohort_name, expected_count in expected_cohorts.items():
        actual = actual_counts.get(cohort_name)
        if actual is None:
            raise SystemExit(f"Track A cohort missing: {cohort_name}")
        if actual != expected_count:
            raise SystemExit(
                f"Track A cohort size mismatch for {cohort_name}: expected={expected_count}, actual={actual}"
            )

    v1_rows = load_csv_rows(labels_dir / "label_set_v1_pairs.csv")
    v2_rows = load_csv_rows(labels_dir / "label_set_v2_pairs.csv")
    comparison_rows = load_csv_rows(labels_dir / "label_set_v1_v2_comparison.csv")
    if len(v1_rows) == 0:
        raise SystemExit("Track A label_set_v1_pairs.csv is empty.")
    if len(v1_rows) != len(v2_rows):
        raise SystemExit(
            "Track A label set size mismatch: "
            f"v1={len(v1_rows)}, v2={len(v2_rows)}"
        )
    if len(v1_rows) != len(comparison_rows):
        raise SystemExit(
            "Track A comparison size mismatch: "
            f"v1={len(v1_rows)}, comparison={len(comparison_rows)}"
        )

    v1_summary = load_json(labels_dir / "label_set_v1_summary.json")
    v2_summary = load_json(labels_dir / "label_set_v2_summary.json")
    if as_int(v1_summary.get("pair_count"), "v1 pair_count") != len(v1_rows):
        raise SystemExit("Track A v1 summary pair_count does not match v1 label table.")
    if as_int(v2_summary.get("pair_count"), "v2 pair_count") != len(v2_rows):
        raise SystemExit("Track A v2 summary pair_count does not match v2 label table.")

    print("Track A integrity check passed.")
    print(f"- Output directory: {output_dir}")
    print(f"- Error failures: {error_failures}")
    print(f"- Warning failures: {warning_failures}")
    print(f"- Pair labels: {len(v1_rows)}")
    print("- Cohorts: raw369=369, matrix402=402, features404=404")


if __name__ == "__main__":
    main()
