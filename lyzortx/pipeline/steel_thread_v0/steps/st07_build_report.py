#!/usr/bin/env python3
"""ST0.7: Build reproducible steel-thread report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st04-metrics-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st04_model_metrics_raw.json"),
        help="Input ST0.4 metrics JSON.",
    )
    parser.add_argument(
        "--st05-calibration-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st05_calibration_summary.csv"),
        help="Input ST0.5 calibration summary CSV.",
    )
    parser.add_argument(
        "--st05-ranked-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st05_ranked_predictions.csv"),
        help="Input ST0.5 ranked predictions CSV.",
    )
    parser.add_argument(
        "--st06-top3-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st06_top3_recommendations.csv"),
        help="Input ST0.6 top-3 recommendation CSV.",
    )
    parser.add_argument(
        "--st06-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st06_recommendation_summary.json"),
        help="Input ST0.6 recommendation summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0"),
        help="Output directory for final ST0.7 report artifacts.",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return [
            {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            for row in reader
        ]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_float(value: object) -> float:
    return float(value)


def build_error_analysis(
    top3_rows: List[Dict[str, str]],
    ranked_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    ranked_by_bacteria: Dict[str, List[Dict[str, str]]] = {}
    for row in ranked_rows:
        if row["split_holdout"] != "holdout_test":
            continue
        ranked_by_bacteria.setdefault(row["bacteria"], []).append(row)

    top3_by_bacteria: Dict[str, List[Dict[str, str]]] = {}
    for row in top3_rows:
        if row["split_holdout"] != "holdout_test":
            continue
        top3_by_bacteria.setdefault(row["bacteria"], []).append(row)

    error_rows: List[Dict[str, object]] = []
    for bacteria in sorted(top3_by_bacteria.keys()):
        recs = sorted(
            top3_by_bacteria[bacteria],
            key=lambda r: int(r["recommendation_rank"]),
        )
        ranked = ranked_by_bacteria.get(bacteria, [])
        if not ranked:
            continue

        rec_hit = any(r["label_hard_binary"] == "1" for r in recs)
        if rec_hit:
            continue

        positives = [r for r in ranked if r["label_hard_binary"] == "1"]
        best_true = (
            sorted(
                positives,
                key=lambda r: (-float(r["score_logreg_isotonic"]), r["phage"]),
            )[0]
            if positives
            else None
        )

        error_rows.append(
            {
                "bacteria": bacteria,
                "n_labeled_pairs_holdout": sum(1 for r in ranked if r["label_hard_binary"] != ""),
                "n_true_positive_phages_holdout": len(positives),
                "top3_recommended_phages": "|".join(r["phage"] for r in recs),
                "top3_recommended_scores": "|".join(r["score_logreg_isotonic"] for r in recs),
                "best_true_positive_phage": "" if best_true is None else best_true["phage"],
                "best_true_positive_score": "" if best_true is None else best_true["score_logreg_isotonic"],
                "best_true_positive_rank": "" if best_true is None else best_true["rank_logreg_isotonic"],
                "note": "top3_miss_on_holdout",
            }
        )

    return error_rows


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st04_metrics = load_json(args.st04_metrics_path)
    st05_summary_rows = read_csv_rows(args.st05_calibration_summary_path)
    st05_ranked_rows = read_csv_rows(args.st05_ranked_predictions_path)
    st06_top3_rows = read_csv_rows(args.st06_top3_path)
    st06_summary = load_json(args.st06_summary_path)

    metrics_rows: List[Dict[str, object]] = []
    for model_name, block in st04_metrics["models"].items():
        for metric_name, metric_value in block["holdout_binary_metrics"].items():
            metrics_rows.append(
                {
                    "metric_group": "st04_holdout_binary",
                    "metric_name": metric_name,
                    "model": model_name,
                    "split": "holdout",
                    "value": metric_value,
                }
            )
        for metric_name, metric_value in block["holdout_top3_metrics"].items():
            metrics_rows.append(
                {
                    "metric_group": "st04_holdout_top3",
                    "metric_name": metric_name,
                    "model": model_name,
                    "split": "holdout",
                    "value": metric_value,
                }
            )

    for row in st05_summary_rows:
        if row["dataset"] != "holdout":
            continue
        label_slice = row.get("label_slice", "full_label")
        for metric_name in ("brier_score", "log_loss", "ece"):
            metrics_rows.append(
                {
                    "metric_group": "st05_holdout_calibration",
                    "metric_name": f"{metric_name}__{label_slice}",
                    "model": f"{row['model']}:{row['variant']}",
                    "split": "holdout",
                    "value": row[metric_name],
                }
            )

    for label_slice, metric_block in st06_summary["holdout_topk_metrics"].items():
        for metric_name, metric_value in metric_block.items():
            metrics_rows.append(
                {
                    "metric_group": "st06_holdout_recommendation",
                    "metric_name": f"{metric_name}__{label_slice}",
                    "model": "st06_policy",
                    "split": "holdout",
                    "value": metric_value,
                }
            )

    for label_slice, ci_block in st06_summary.get("holdout_topk_bootstrap_ci", {}).items():
        for metric_name, metric_value in ci_block.items():
            metrics_rows.append(
                {
                    "metric_group": "st06_holdout_bootstrap_ci",
                    "metric_name": f"{metric_name}__{label_slice}",
                    "model": "st06_policy",
                    "split": "holdout",
                    "value": metric_value,
                }
            )

    metrics_rows.sort(key=lambda r: (str(r["metric_group"]), str(r["model"]), str(r["metric_name"])))
    top3_rows_sorted = sorted(
        st06_top3_rows,
        key=lambda r: (r["bacteria"], int(r["recommendation_rank"])),
    )
    error_rows = build_error_analysis(top3_rows_sorted, st05_ranked_rows)

    output_metrics = args.output_dir / "metrics_summary.csv"
    output_top3 = args.output_dir / "top3_recommendations.csv"
    output_calib = args.output_dir / "calibration_summary.csv"
    output_error = args.output_dir / "error_analysis.csv"
    output_manifest = args.output_dir / "run_manifest.json"

    write_csv(output_metrics, fieldnames=list(metrics_rows[0].keys()), rows=metrics_rows)
    write_csv(output_top3, fieldnames=list(top3_rows_sorted[0].keys()), rows=top3_rows_sorted)
    write_csv(output_calib, fieldnames=list(st05_summary_rows[0].keys()), rows=st05_summary_rows)

    error_fieldnames = [
        "bacteria",
        "n_labeled_pairs_holdout",
        "n_true_positive_phages_holdout",
        "top3_recommended_phages",
        "top3_recommended_scores",
        "best_true_positive_phage",
        "best_true_positive_score",
        "best_true_positive_rank",
        "note",
    ]
    write_csv(output_error, fieldnames=error_fieldnames, rows=error_rows)

    manifest = {
        "step_name": "st07_build_report",
        "inputs": {
            "st04_metrics_path": str(args.st04_metrics_path),
            "st05_calibration_summary_path": str(args.st05_calibration_summary_path),
            "st05_ranked_predictions_path": str(args.st05_ranked_predictions_path),
            "st06_top3_path": str(args.st06_top3_path),
            "st06_summary_path": str(args.st06_summary_path),
        },
        "input_hashes_sha256": {
            "st04_metrics": sha256(args.st04_metrics_path),
            "st05_calibration_summary": sha256(args.st05_calibration_summary_path),
            "st05_ranked_predictions": sha256(args.st05_ranked_predictions_path),
            "st06_top3_recommendations": sha256(args.st06_top3_path),
            "st06_summary": sha256(args.st06_summary_path),
        },
        "outputs": {
            "metrics_summary_csv": str(output_metrics),
            "top3_recommendations_csv": str(output_top3),
            "calibration_summary_csv": str(output_calib),
            "error_analysis_csv": str(output_error),
            "run_manifest_json": str(output_manifest),
        },
        "output_hashes_sha256": {
            "metrics_summary": sha256(output_metrics),
            "top3_recommendations": sha256(output_top3),
            "calibration_summary": sha256(output_calib),
            "error_analysis": sha256(output_error),
        },
        "counts": {
            "metrics_summary_rows": len(metrics_rows),
            "top3_recommendations_rows": len(top3_rows_sorted),
            "calibration_summary_rows": len(st05_summary_rows),
            "error_analysis_rows": len(error_rows),
        },
    }
    write_json(output_manifest, manifest)

    print("ST0.7 completed.")
    print(f"- Metrics summary rows: {len(metrics_rows)}")
    print(f"- Top-3 recommendation rows: {len(top3_rows_sorted)}")
    print(f"- Calibration summary rows: {len(st05_summary_rows)}")
    print(f"- Error analysis rows: {len(error_rows)}")
    print(f"- Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
