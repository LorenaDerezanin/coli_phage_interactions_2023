#!/usr/bin/env python3
"""Regression check for ST0.4 model-training outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.checks._check_helpers import (
    compare_dicts,
    load_json,
    read_csv_rows,
)
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import safe_round
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
    st04_train_baselines,
)

NUMERIC_TOLERANCE = 1e-5


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-baseline-path",
        type=Path,
        default=Path("lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json"),
        help="Path to expected regression baseline JSON.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Directory containing ST0.4 generated artifacts.",
    )
    parser.add_argument("--run-st01", action="store_true", help="Run ST0.1 before checking ST0.4.")
    parser.add_argument("--run-st01b", action="store_true", help="Run ST0.1b before checking ST0.4.")
    parser.add_argument("--run-st02", action="store_true", help="Run ST0.2 before checking ST0.4.")
    parser.add_argument("--run-st03", action="store_true", help="Run ST0.3 before checking ST0.4.")
    parser.add_argument("--run-st04", action="store_true", help="Run ST0.4 before checking ST0.4.")
    return parser.parse_args(argv)


def build_actual_summary(intermediate_dir: Path) -> Dict[str, Any]:
    metrics_path = intermediate_dir / "st04_model_metrics_raw.json"
    artifacts_path = intermediate_dir / "st04_model_artifacts.json"
    predictions_path = intermediate_dir / "st04_pair_predictions_raw.csv"

    if not metrics_path.exists() or not artifacts_path.exists() or not predictions_path.exists():
        missing = [str(p) for p in (metrics_path, artifacts_path, predictions_path) if not p.exists()]
        raise FileNotFoundError(
            "ST0.4 artifacts missing. Run ST0.4 first or pass --run-st01 --run-st01b --run-st02 --run-st03 --run-st04. "
            + "Missing: "
            + ", ".join(missing)
        )

    metrics = load_json(metrics_path)
    artifacts = load_json(artifacts_path)
    prediction_rows = read_csv_rows(predictions_path)
    if not prediction_rows:
        raise ValueError(f"Prediction CSV is empty: {predictions_path}")

    try:
        dummy_probs = [float(row["pred_dummy_raw"]) for row in prediction_rows]
        logreg_probs = [float(row["pred_logreg_raw"]) for row in prediction_rows]
    except (KeyError, ValueError) as exc:
        raise ValueError("Prediction CSV is missing numeric pred_dummy_raw/pred_logreg_raw columns.") from exc

    return {
        "train_summary": metrics["train_summary"],
        "feature_summary": {
            "n_vectorized_features": metrics["feature_summary"]["n_vectorized_features"],
            "categorical_feature_count": len(metrics["feature_summary"]["categorical_feature_columns"]),
            "numeric_feature_count": len(metrics["feature_summary"]["numeric_feature_columns"]),
        },
        "model_quality_summary": {
            "dummy_holdout_binary_metrics": metrics["models"]["dummy_prior"]["holdout_binary_metrics"],
            "dummy_holdout_top3_metrics": metrics["models"]["dummy_prior"]["holdout_top3_metrics"],
            "logreg_holdout_binary_metrics": metrics["models"]["logreg_host_phage"]["holdout_binary_metrics"],
            "logreg_holdout_top3_metrics": metrics["models"]["logreg_host_phage"]["holdout_top3_metrics"],
        },
        "prediction_summary": {
            "row_count": len(prediction_rows),
            "dummy_prob_mean": safe_round(sum(dummy_probs) / len(dummy_probs)),
            "dummy_prob_min": safe_round(min(dummy_probs)),
            "dummy_prob_max": safe_round(max(dummy_probs)),
            "logreg_prob_mean": safe_round(sum(logreg_probs) / len(logreg_probs)),
            "logreg_prob_min": safe_round(min(logreg_probs)),
            "logreg_prob_max": safe_round(max(logreg_probs)),
        },
        "artifact_summary": {
            "logreg_n_iter": artifacts["logreg_model"]["n_iter"],
            "logreg_intercept": artifacts["logreg_model"]["intercept"],
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

    expected = load_json(args.expected_baseline_path)
    actual = build_actual_summary(args.intermediate_dir)
    errors = compare_dicts(expected, actual, numeric_tolerance=NUMERIC_TOLERANCE)
    if errors:
        print("ST0.4 regression check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("ST0.4 regression check passed.")
    print(f"- Baseline: {args.expected_baseline_path}")
    print(f"- Intermediate: {args.intermediate_dir}")


if __name__ == "__main__":
    main()
