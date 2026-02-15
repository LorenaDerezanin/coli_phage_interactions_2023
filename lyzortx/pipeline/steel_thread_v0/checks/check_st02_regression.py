#!/usr/bin/env python3
"""Regression check for ST0.2 canonical pair-table outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.2 generated artifacts.",
    )
    parser.add_argument(
        "--run-st01",
        action="store_true",
        help="Run ST0.1 before running/checking ST0.2.",
    )
    parser.add_argument(
        "--run-st01b",
        action="store_true",
        help="Run ST0.1b before running/checking ST0.2.",
    )
    parser.add_argument(
        "--run-st02",
        action="store_true",
        help="Run ST0.2 before checking regression metrics.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    audit_path = intermediate_dir / "st02_pair_table_audit.json"
    manifest_path = intermediate_dir / "st02_feature_manifest.json"
    pair_table_path = intermediate_dir / "st02_pair_table.csv"

    if not audit_path.exists() or not manifest_path.exists() or not pair_table_path.exists():
        missing = [str(p) for p in (audit_path, manifest_path, pair_table_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.2 artifacts missing. Run ST0.2 first or pass --run-st01 --run-st01b --run-st02. Missing: "
            + ", ".join(missing)
        )

    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    pair_table_sha256 = hashlib.sha256(pair_table_path.read_bytes()).hexdigest()

    return {
        "pair_table_summary": {
            "row_count": audit["row_count"],
            "distinct_bacteria_count": audit["distinct_bacteria_count"],
            "distinct_phage_count": audit["distinct_phage_count"],
            "strict_slice_fraction": audit["strict_slice_fraction"],
            "matrix_available_fraction": audit["matrix_available_fraction"],
        },
        "join_and_label_summary": {
            "join_missing_counts": audit["join_missing_counts"],
            "hard_label_counts": audit["hard_label_counts"],
            "strict_tier_counts": audit["strict_tier_counts"],
            "strict_label_counts": audit["strict_label_counts"],
            "cv_group_unique_count": audit["cv_group_unique_count"],
        },
        "schema_summary": {
            "output_column_count": len(audit["output_columns"]),
            "column_group_sizes": {
                key: len(cols) for key, cols in manifest["column_groups"].items()
            },
            "step_name": manifest["step_name"],
        },
        "artifacts": {
            "pair_table_csv_sha256": pair_table_sha256,
        },
    }


def compare_dicts(expected: Dict[str, Any], actual: Dict[str, Any], prefix: str = "") -> List[str]:
    errors: List[str] = []
    all_keys = sorted(set(expected.keys()) | set(actual.keys()))
    for key in all_keys:
        path = f"{prefix}.{key}" if prefix else key
        if key not in expected:
            errors.append(f"Unexpected key in actual: {path}")
            continue
        if key not in actual:
            errors.append(f"Missing key in actual: {path}")
            continue
        exp_val = expected[key]
        act_val = actual[key]
        if isinstance(exp_val, dict) and isinstance(act_val, dict):
            errors.extend(compare_dicts(exp_val, act_val, prefix=path))
            continue
        if exp_val != act_val:
            errors.append(f"Mismatch at {path}: expected={exp_val!r}, actual={act_val!r}")
    return errors


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.run_st01:
        st01_label_policy.main([])
    if args.run_st01b:
        st01b_confidence_tiers.main([])
    if args.run_st02:
        st02_build_pair_table.main([])

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)

    if errors:
        print("ST0.2 regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.2 regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
