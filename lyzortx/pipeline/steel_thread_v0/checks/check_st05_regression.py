#!/usr/bin/env python3
"""Regression check for ST0.5 calibration/ranking outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.checks._check_helpers import (
    compare_dicts,
    load_json,
    read_csv_rows,
)
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
    st04_train_baselines,
    st05_calibrate_rank,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st05_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.5 generated artifacts.",
    )
    parser.add_argument("--run-st01", action="store_true", help="Run ST0.1 before checking ST0.5.")
    parser.add_argument("--run-st01b", action="store_true", help="Run ST0.1b before checking ST0.5.")
    parser.add_argument("--run-st02", action="store_true", help="Run ST0.2 before checking ST0.5.")
    parser.add_argument("--run-st03", action="store_true", help="Run ST0.3 before checking ST0.5.")
    parser.add_argument("--run-st04", action="store_true", help="Run ST0.4 before checking ST0.5.")
    parser.add_argument("--run-st05", action="store_true", help="Run ST0.5 before checking ST0.5.")
    return parser.parse_args(argv)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    summary_path = intermediate_dir / "st05_calibration_summary.csv"
    predictions_path = intermediate_dir / "st05_pair_predictions_calibrated.csv"
    ranked_path = intermediate_dir / "st05_ranked_predictions.csv"
    artifacts_path = intermediate_dir / "st05_calibration_artifacts.json"

    missing = [str(p) for p in (summary_path, predictions_path, ranked_path, artifacts_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ST0.5 artifacts missing. Run ST0.5 first or pass --run-st01 --run-st01b --run-st02 --run-st03 --run-st04 --run-st05. "
            + "Missing: "
            + ", ".join(missing)
        )

    summary_rows = read_csv_rows(summary_path)
    artifacts = load_json(artifacts_path)
    summary_by_key = {}
    for row in summary_rows:
        label_slice = row.get("label_slice", "full_label")
        key = f"{row['model']}|{row['dataset']}|{label_slice}|{row['variant']}"
        summary_by_key[key] = {
            "n": int(row["n"]),
            "positive_rate": float(row["positive_rate"]),
            "brier_score": float(row["brier_score"]),
            "log_loss": float(row["log_loss"]),
            "ece": float(row["ece"]),
        }

    return {
        "row_counts": {
            "calibration_summary_rows": len(summary_rows),
            "calibrated_predictions_rows": len(read_csv_rows(predictions_path)),
            "ranked_predictions_rows": len(read_csv_rows(ranked_path)),
        },
        "calibration_metrics": summary_by_key,
        "artifact_params": {
            "calibration_fold": artifacts["calibration_fold"],
            "models": sorted(list(artifacts["models"].keys())),
            "dummy_platt_coef": artifacts["models"]["dummy_prior"]["platt_coef"],
            "logreg_platt_coef": artifacts["models"]["logreg_host_phage"]["platt_coef"],
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
    if args.run_st04:
        st04_train_baselines.main([])
    if args.run_st05:
        st05_calibrate_rank.main([])

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual)
    if errors:
        print("ST0.5 regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.5 regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
