#!/usr/bin/env python3
"""Regression check for ST0.6 recommendation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
    st04_train_baselines,
    st05_calibrate_rank,
    st06_recommend_top3,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st06_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.6 generated artifacts.",
    )
    parser.add_argument("--run-st01", action="store_true", help="Run ST0.1 before checking ST0.6.")
    parser.add_argument("--run-st01b", action="store_true", help="Run ST0.1b before checking ST0.6.")
    parser.add_argument("--run-st02", action="store_true", help="Run ST0.2 before checking ST0.6.")
    parser.add_argument("--run-st03", action="store_true", help="Run ST0.3 before checking ST0.6.")
    parser.add_argument("--run-st04", action="store_true", help="Run ST0.4 before checking ST0.6.")
    parser.add_argument("--run-st05", action="store_true", help="Run ST0.5 before checking ST0.6.")
    parser.add_argument("--run-st06", action="store_true", help="Run ST0.6 before checking ST0.6.")
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return [
            {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            for row in reader
        ]


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    recs_path = intermediate_dir / "st06_top3_recommendations.csv"
    summary_path = intermediate_dir / "st06_recommendation_summary.json"
    if not recs_path.exists() or not summary_path.exists():
        missing = [str(p) for p in (recs_path, summary_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.6 artifacts missing. Run ST0.6 first or pass --run-st01 --run-st01b --run-st02 --run-st03 --run-st04 --run-st05 --run-st06. "
            + "Missing: "
            + ", ".join(missing)
        )

    rec_rows = read_csv_rows(recs_path)
    summary = load_json(summary_path)
    return {
        "row_counts": {
            "recommendation_rows": len(rec_rows),
            "recommended_strain_count": summary["recommendation_summary"]["recommended_strain_count"],
        },
        "summary_metrics": summary["holdout_topk_metrics"],
        "diversity_summary": {
            "diversity_relaxed_strain_count": summary["recommendation_summary"]["diversity_relaxed_strain_count"],
            "top_k": summary["parameters"]["top_k"],
            "score_column": summary["parameters"]["score_column"],
            "max_per_family": summary["parameters"]["max_per_family"],
            "diversity_mode": summary["parameters"]["diversity_mode"],
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
    if args.run_st04:
        st04_train_baselines.main([])
    if args.run_st05:
        st05_calibrate_rank.main([])
    if args.run_st06:
        st06_recommend_top3.main([])

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)
    if errors:
        print("ST0.6 regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.6 regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
