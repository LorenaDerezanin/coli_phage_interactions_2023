#!/usr/bin/env python3
"""Regression check for ST0.1b confidence-tier outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.steps import st01_label_policy, st01b_confidence_tiers


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.1b generated artifacts.",
    )
    parser.add_argument(
        "--run-st01",
        action="store_true",
        help="Run ST0.1 before running/checking ST0.1b.",
    )
    parser.add_argument(
        "--run-st01b",
        action="store_true",
        help="Run ST0.1b before checking regression metrics.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    audit_path = intermediate_dir / "st01b_confidence_audit.json"
    definition_path = intermediate_dir / "st01b_confidence_policy_definition.json"
    pair_audit_path = intermediate_dir / "st01b_pair_confidence_audit.csv"

    if not audit_path.exists() or not definition_path.exists() or not pair_audit_path.exists():
        missing = [str(p) for p in (audit_path, definition_path, pair_audit_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.1b artifacts missing. Run ST0.1b first or pass --run-st01 --run-st01b. Missing: " + ", ".join(missing)
        )

    audit = load_json(audit_path)
    definition = load_json(definition_path)
    pair_csv_sha256 = hashlib.sha256(pair_audit_path.read_bytes()).hexdigest()

    return {
        "confidence_policy": {
            "policy_name": definition["policy_name"],
            "policy_version": definition["policy_version"],
            "thresholds": audit["strict_thresholds"],
        },
        "tier_summary": {
            "total_pairs": audit["total_pairs"],
            "strict_tier_counts": audit["strict_tier_counts"],
            "strict_label_counts": audit["strict_label_counts"],
            "strict_coverage_fraction": audit["strict_coverage_fraction"],
            "strict_positive_fraction_within_strict_slice": audit["strict_positive_fraction_within_strict_slice"],
        },
        "quality_summary": {
            "strict_conflicting_counts": audit["strict_conflicting_counts"],
            "strict_has_uninterpretable_counts": audit["strict_has_uninterpretable_counts"],
        },
        "artifacts": {
            "pair_confidence_audit_csv_sha256": pair_csv_sha256,
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

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)

    if errors:
        print("ST0.1b regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.1b regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
