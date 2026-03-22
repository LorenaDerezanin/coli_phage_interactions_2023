#!/usr/bin/env python3
"""Regression check for TF01/ST0.3c fixed split protocol outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.checks._check_helpers import compare_dicts, load_json
from lyzortx.pipeline.steel_thread_v0.steps import st03c_build_fixed_split_protocol


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st03c_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing TF01/ST0.3c generated artifacts.",
    )
    parser.add_argument("--run-st03c", action="store_true", help="Run TF01/ST0.3c before checking.")
    return parser.parse_args(argv)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    audit_path = intermediate_dir / "st03c_fixed_split_protocol_v1_audit.json"
    protocol_path = intermediate_dir / "st03c_fixed_split_protocol_v1_protocol.json"
    assignments_path = intermediate_dir / "st03c_fixed_split_protocol_v1_assignments.csv"

    if not audit_path.exists() or not protocol_path.exists() or not assignments_path.exists():
        missing = [str(p) for p in (audit_path, protocol_path, assignments_path) if not p.exists()]
        raise FileNotFoundError(
            "TF01/ST0.3c artifacts missing. Run the step first or pass --run-st03c. Missing: " + ", ".join(missing)
        )

    audit = load_json(audit_path)
    protocol = load_json(protocol_path)
    assignment_sha256 = hashlib.sha256(assignments_path.read_bytes()).hexdigest()

    return {
        "split_summary": {
            "row_count": audit["row_count"],
            "host_cluster_count": audit["host_cluster_count"],
            "host_cluster_holdout_count": audit["host_cluster_holdout_count"],
            "host_cluster_holdout_fraction_actual": audit["host_cluster_holdout_fraction_actual"],
            "phage_clade_count": audit["phage_clade_count"],
            "phage_clade_holdout_count": audit["phage_clade_holdout_count"],
            "phage_clade_holdout_fraction_actual": audit["phage_clade_holdout_fraction_actual"],
            "split_counts": audit["split_counts"],
        },
        "leakage_summary": {
            "leakage_checks": audit["leakage_checks"],
            "holdout_membership": audit["holdout_membership"],
        },
        "protocol_summary": {
            "step_name": protocol["step_name"],
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["protocol_version"],
            "split_type": protocol["split_type"],
            "host_axis": protocol["host_axis"],
            "phage_axis": protocol["phage_axis"],
            "dual_axis": protocol["dual_axis"],
        },
        "artifacts": {
            "split_assignments_csv_sha256": assignment_sha256,
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.run_st03c:
        st03c_build_fixed_split_protocol.main([])

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)

    if errors:
        print("TF01/ST0.3c regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("TF01/ST0.3c regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
