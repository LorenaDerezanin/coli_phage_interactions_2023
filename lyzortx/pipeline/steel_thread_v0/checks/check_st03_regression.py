#!/usr/bin/env python3
"""Regression check for ST0.3 split-assignment outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.checks._check_helpers import compare_dicts, load_json
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.3 generated artifacts.",
    )
    parser.add_argument("--run-st01", action="store_true", help="Run ST0.1 before checking ST0.3.")
    parser.add_argument("--run-st01b", action="store_true", help="Run ST0.1b before checking ST0.3.")
    parser.add_argument("--run-st02", action="store_true", help="Run ST0.2 before checking ST0.3.")
    parser.add_argument("--run-st03", action="store_true", help="Run ST0.3 before checking ST0.3.")
    return parser.parse_args(argv)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    audit_path = intermediate_dir / "st03_split_audit.json"
    protocol_path = intermediate_dir / "st03_split_protocol.json"
    assignments_path = intermediate_dir / "st03_split_assignments.csv"

    if not audit_path.exists() or not protocol_path.exists() or not assignments_path.exists():
        missing = [str(p) for p in (audit_path, protocol_path, assignments_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.3 artifacts missing. Run ST0.3 first or pass --run-st01 --run-st01b --run-st02 --run-st03. Missing: "
            + ", ".join(missing)
        )

    audit = load_json(audit_path)
    protocol = load_json(protocol_path)
    assignment_sha256 = hashlib.sha256(assignments_path.read_bytes()).hexdigest()

    return {
        "split_summary": {
            "row_count": audit["row_count"],
            "cv_group_count": audit["cv_group_count"],
            "holdout_group_count": audit["holdout_group_count"],
            "holdout_group_fraction_actual": audit["holdout_group_fraction_actual"],
            "holdout_row_counts": audit["holdout_row_counts"],
            "cv_fold_row_counts": audit["cv_fold_row_counts"],
        },
        "leakage_and_trainable_summary": {
            "leakage_checks": audit["leakage_checks"],
            "hard_trainable_holdout_counts": audit["hard_trainable_holdout_counts"],
            "strict_trainable_holdout_counts": audit["strict_trainable_holdout_counts"],
            "strict_trainable_cv_fold_counts": audit["strict_trainable_cv_fold_counts"],
        },
        "protocol_summary": {
            "step_name": protocol["step_name"],
            "split_type": protocol["split_type"],
            "n_cv_folds": protocol["split_rules"]["n_cv_folds"],
            "holdout_group_fraction": protocol["split_rules"]["holdout_group_fraction"],
            "split_salt": protocol["split_rules"]["split_salt"],
        },
        "artifacts": {
            "split_assignments_csv_sha256": assignment_sha256,
        },
    }


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

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)
    if errors:
        print("ST0.3 regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.3 regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
