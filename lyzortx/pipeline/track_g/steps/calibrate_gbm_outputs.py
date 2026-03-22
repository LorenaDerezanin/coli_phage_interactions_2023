#!/usr/bin/env python3
"""TG02: Calibrate TG01 LightGBM probabilities with isotonic and Platt scaling."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier

SLICE_FILTERS = {
    "full_label": lambda row: row["label_hard_any_lysis"] != "",
    "strict_confidence": lambda row: row["label_hard_any_lysis"] != "" and row["is_strict_trainable"] == "1",
}

TG01_REQUIRED_COLUMNS: Sequence[str] = (
    "pair_id",
    "bacteria",
    "phage",
    "split_holdout",
    "split_cv5_fold",
    "label_hard_any_lysis",
    "prediction_context",
    "lightgbm_probability",
)

ST02_REQUIRED_COLUMNS: Sequence[str] = ("pair_id", "phage_family", "label_strict_confidence_tier")
ST03_REQUIRED_COLUMNS: Sequence[str] = ("pair_id", "is_strict_trainable")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg01-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_pair_predictions.csv"),
        help="Input TG01 pair-level prediction CSV.",
    )
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table for metadata passthrough.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments for strict-confidence flags.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg02_gbm_calibration"),
        help="Output directory for TG02 artifacts.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=0,
        help="Non-holdout CV fold used to fit the calibrators.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for ECE computation.",
    )
    parser.add_argument(
        "--platt-random-state",
        type=int,
        default=42,
        help="Random state for Platt-scaling logistic regression.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume TG01 outputs already exist instead of generating them when missing.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_prerequisite_outputs(args: argparse.Namespace) -> None:
    if args.skip_prerequisites:
        return
    if not args.tg01_predictions_path.exists():
        train_v1_binary_classifier.main([])


def ece_score(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch for ECE.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if not y_true:
        return 0.0

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (count / total) * abs(acc - conf)
    return ece


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], ece_bins: int) -> Dict[str, float]:
    return {
        "n": float(len(y_true)),
        "positive_rate": safe_round(sum(y_true) / len(y_true) if y_true else 0.0),
        "brier_score": safe_round(brier_score_loss(y_true, y_prob)),
        "log_loss": safe_round(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": safe_round(ece_score(y_true, y_prob, n_bins=ece_bins)),
    }


def rows_for_slice(rows: Sequence[Mapping[str, str]], *, slice_name: str) -> List[Dict[str, str]]:
    if slice_name not in SLICE_FILTERS:
        raise ValueError(f"Unknown slice name: {slice_name}")
    return [dict(row) for row in rows if SLICE_FILTERS[slice_name](row)]


def merge_prediction_metadata(
    tg01_rows: Sequence[Mapping[str, str]],
    st02_rows: Sequence[Mapping[str, str]],
    st03_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    st02_by_pair = {row["pair_id"]: row for row in st02_rows}
    st03_by_pair = {row["pair_id"]: row for row in st03_rows}

    merged_rows: List[Dict[str, str]] = []
    for row in tg01_rows:
        pair_id = row["pair_id"]
        st02_row = st02_by_pair.get(pair_id)
        st03_row = st03_by_pair.get(pair_id)
        if st02_row is None:
            raise KeyError(f"Missing ST0.2 metadata for pair_id {pair_id}")
        if st03_row is None:
            raise KeyError(f"Missing ST0.3 split metadata for pair_id {pair_id}")
        merged_rows.append(
            {
                **row,
                "phage_family": st02_row.get("phage_family", ""),
                "label_strict_confidence_tier": st02_row.get("label_strict_confidence_tier", ""),
                "is_strict_trainable": st03_row.get("is_strict_trainable", ""),
            }
        )
    merged_rows.sort(key=lambda row: (str(row["bacteria"]), str(row["phage"])))
    return merged_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)

    tg01_rows = read_csv_rows(args.tg01_predictions_path, required_columns=TG01_REQUIRED_COLUMNS)
    st02_rows = read_csv_rows(args.st02_pair_table_path, required_columns=ST02_REQUIRED_COLUMNS)
    st03_rows = read_csv_rows(args.st03_split_assignments_path, required_columns=ST03_REQUIRED_COLUMNS)
    if not tg01_rows:
        raise ValueError("TG01 prediction input is empty.")

    merged_rows = merge_prediction_metadata(tg01_rows, st02_rows, st03_rows)
    row_index_by_pair_id = {row["pair_id"]: idx for idx, row in enumerate(merged_rows)}
    calibration_rows = [
        row
        for row in merged_rows
        if row["prediction_context"] == "non_holdout_oof"
        and row["split_holdout"] == "train_non_holdout"
        and row["split_cv5_fold"] == str(args.calibration_fold)
        and row["label_hard_any_lysis"] != ""
    ]
    holdout_eval_rows = [
        row
        for row in merged_rows
        if row["prediction_context"] == "holdout_final"
        and row["split_holdout"] == "holdout_test"
        and row["label_hard_any_lysis"] != ""
    ]
    if not calibration_rows:
        raise ValueError("No calibration rows found for TG02.")
    if not holdout_eval_rows:
        raise ValueError("No holdout eval rows found for TG02.")

    x_calib = np.asarray([float(row["lightgbm_probability"]) for row in calibration_rows], dtype=float)
    y_calib = np.asarray([int(row["label_hard_any_lysis"]) for row in calibration_rows], dtype=int)
    if len(np.unique(y_calib)) < 2:
        raise ValueError("Calibration fold has only one class for TG02.")

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(x_calib, y_calib)
    platt = LogisticRegression(
        solver="lbfgs",
        random_state=args.platt_random_state,
        max_iter=1000,
    )
    platt.fit(x_calib.reshape(-1, 1), y_calib)

    all_raw = np.asarray([float(row["lightgbm_probability"]) for row in merged_rows], dtype=float)
    all_iso = isotonic.predict(all_raw)
    all_platt = platt.predict_proba(all_raw.reshape(-1, 1))[:, 1]

    summary_rows: List[Dict[str, Any]] = []
    for dataset_name, dataset_rows in (("calibration", calibration_rows), ("holdout", holdout_eval_rows)):
        for slice_name in ("full_label", "strict_confidence"):
            sliced_rows = rows_for_slice(dataset_rows, slice_name=slice_name)
            if not sliced_rows:
                continue
            sliced_indexes = [row_index_by_pair_id[row["pair_id"]] for row in sliced_rows]
            y_true = [int(row["label_hard_any_lysis"]) for row in sliced_rows]
            probs_raw = [float(all_raw[idx]) for idx in sliced_indexes]
            probs_iso = [float(all_iso[idx]) for idx in sliced_indexes]
            probs_platt = [float(all_platt[idx]) for idx in sliced_indexes]
            for variant_name, probs in (("raw", probs_raw), ("isotonic", probs_iso), ("platt", probs_platt)):
                metrics = binary_metrics(y_true, probs, ece_bins=args.ece_bins)
                summary_rows.append(
                    {
                        "model": "lightgbm",
                        "dataset": dataset_name,
                        "label_slice": slice_name,
                        "variant": variant_name,
                        "n": int(metrics["n"]),
                        "positive_rate": metrics["positive_rate"],
                        "brier_score": metrics["brier_score"],
                        "log_loss": metrics["log_loss"],
                        "ece": metrics["ece"],
                    }
                )

    output_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(merged_rows):
        output_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "phage_family": row["phage_family"],
                "split_holdout": row["split_holdout"],
                "split_cv5_fold": row["split_cv5_fold"],
                "prediction_context": row["prediction_context"],
                "is_strict_trainable": row["is_strict_trainable"],
                "label_hard_any_lysis": row["label_hard_any_lysis"],
                "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                "pred_lightgbm_raw": f"{all_raw[idx]:.10f}",
                "pred_lightgbm_isotonic": f"{all_iso[idx]:.10f}",
                "pred_lightgbm_platt": f"{all_platt[idx]:.10f}",
            }
        )

    ranked_rows: List[Dict[str, Any]] = []
    rows_by_bacteria: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        rows_by_bacteria[str(row["bacteria"])].append(row)
    for bacteria in sorted(rows_by_bacteria):
        ranked = sorted(
            rows_by_bacteria[bacteria],
            key=lambda row: (-float(str(row["pred_lightgbm_isotonic"])), str(row["phage"])),
        )
        for rank, row in enumerate(ranked, start=1):
            ranked_rows.append(
                {
                    "bacteria": row["bacteria"],
                    "phage": row["phage"],
                    "phage_family": row["phage_family"],
                    "rank_lightgbm_isotonic": rank,
                    "score_lightgbm_isotonic": row["pred_lightgbm_isotonic"],
                    "score_lightgbm_raw": row["pred_lightgbm_raw"],
                    "score_lightgbm_platt": row["pred_lightgbm_platt"],
                    "split_holdout": row["split_holdout"],
                    "prediction_context": row["prediction_context"],
                    "label_hard_any_lysis": row["label_hard_any_lysis"],
                    "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                }
            )

    summary_rows.sort(key=lambda row: (str(row["model"]), str(row["dataset"]), str(row["variant"])))
    output_rows.sort(key=lambda row: (str(row["bacteria"]), str(row["phage"])))
    calibration_artifacts = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "task_id": "TG02",
        "model": "lightgbm",
        "calibration_fold": args.calibration_fold,
        "raw_column": "lightgbm_probability",
        "isotonic_x_thresholds": [safe_round(float(value)) for value in isotonic.X_thresholds_.tolist()],
        "isotonic_y_thresholds": [safe_round(float(value)) for value in isotonic.y_thresholds_.tolist()],
        "platt_coef": safe_round(float(platt.coef_[0][0])),
        "platt_intercept": safe_round(float(platt.intercept_[0])),
        "inputs": {
            "tg01_predictions": {
                "path": str(args.tg01_predictions_path),
                "sha256": _sha256(args.tg01_predictions_path),
            },
            "st02_pair_table": {
                "path": str(args.st02_pair_table_path),
                "sha256": _sha256(args.st02_pair_table_path),
            },
            "st03_split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
            },
        },
    }

    write_csv(
        args.output_dir / "tg02_calibration_summary.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )
    write_csv(
        args.output_dir / "tg02_pair_predictions_calibrated.csv",
        fieldnames=list(output_rows[0].keys()),
        rows=output_rows,
    )
    write_csv(
        args.output_dir / "tg02_ranked_predictions.csv",
        fieldnames=list(ranked_rows[0].keys()),
        rows=ranked_rows,
    )
    write_json(args.output_dir / "tg02_calibration_artifacts.json", calibration_artifacts)

    print("TG02 completed.")
    print(f"- Calibration rows: {len(calibration_rows)}")
    print(f"- Holdout eval rows: {len(holdout_eval_rows)}")
    print(f"- Output calibrated predictions: {args.output_dir / 'tg02_pair_predictions_calibrated.csv'}")
    print(f"- Output ranking: {args.output_dir / 'tg02_ranked_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
