#!/usr/bin/env python3
"""Regression check for ST0.3b split-suite outputs."""

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
    st03_build_splits,
    st03b_build_split_suite,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st03b_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.3b generated artifacts.",
    )
    parser.add_argument("--run-st01", action="store_true", help="Run ST0.1 before checking ST0.3b.")
    parser.add_argument("--run-st01b", action="store_true", help="Run ST0.1b before checking ST0.3b.")
    parser.add_argument("--run-st02", action="store_true", help="Run ST0.2 before checking ST0.3b.")
    parser.add_argument("--run-st03", action="store_true", help="Run ST0.3 before checking ST0.3b.")
    parser.add_argument("--run-st03b", action="store_true", help="Run ST0.3b before checking ST0.3b.")
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    audit_path = intermediate_dir / "st03b_split_suite_audit.json"
    protocol_path = intermediate_dir / "st03b_split_suite_protocol.json"
    assignments_path = intermediate_dir / "st03b_split_suite_assignments.csv"

    if not audit_path.exists() or not protocol_path.exists() or not assignments_path.exists():
        missing = [str(p) for p in (audit_path, protocol_path, assignments_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.3b artifacts missing. Run ST0.3b first or pass --run-st01 --run-st01b --run-st02 --run-st03 --run-st03b. Missing: "
            + ", ".join(missing)
        )

    audit = load_json(audit_path)
    protocol = load_json(protocol_path)
    assignment_sha256 = hashlib.sha256(assignments_path.read_bytes()).hexdigest()

    return {
        "split_suite_summary": {
            "row_count": audit["row_count"],
            "split_counts": audit["split_counts"],
            "phage_family_holdout_count": audit["phage_family_holdout_count"],
            "phage_family_total_count": audit["phage_family_total_count"],
            "phage_family_holdout_fraction_actual": audit["phage_family_holdout_fraction_actual"],
        },
        "leakage_summary": {
            "leakage_checks": audit["leakage_checks"],
            "holdout_membership": audit["holdout_membership"],
        },
        "protocol_summary": {
            "step_name": protocol["step_name"],
            "split_modes": protocol["split_modes"],
        },
        "artifacts": {
            "split_suite_assignments_csv_sha256": assignment_sha256,
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
    if args.run_st03:
        st03_build_splits.main([])
    if args.run_st03b:
        st03b_build_split_suite.main([])

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)

    if errors:
        print("ST0.3b regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.3b regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
