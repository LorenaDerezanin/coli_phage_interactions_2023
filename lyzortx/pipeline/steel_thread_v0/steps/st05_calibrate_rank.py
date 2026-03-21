#!/usr/bin/env python3
"""ST0.5: Calibrate probabilities and export ranked per-strain predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round

MODEL_COLUMNS = {
    "dummy_prior": "pred_dummy_raw",
    "logreg_host_phage": "pred_logreg_raw",
}

SLICE_FILTERS = {
    "full_label": lambda row: row["label_hard_binary"] != "",
    "strict_confidence": lambda row: row["label_hard_binary"] != "" and row["is_strict_trainable"] == "1",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st04-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st04_pair_predictions_raw.csv"),
        help="Input ST0.4 raw prediction CSV.",
    )
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table (for metadata passthrough in ranked output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.5 artifacts.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=0,
        help="CV fold within train_non_holdout used for calibration fit.",
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
    return parser.parse_args(argv)


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


def binary_metrics(y_true: List[int], y_prob: List[float], ece_bins: int) -> Dict[str, float]:
    return {
        "n": float(len(y_true)),
        "positive_rate": safe_round(sum(y_true) / len(y_true) if y_true else 0.0),
        "brier_score": safe_round(brier_score_loss(y_true, y_prob)),
        "log_loss": safe_round(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": safe_round(ece_score(y_true, y_prob, n_bins=ece_bins)),
    }


def rows_for_slice(rows: List[Dict[str, str]], slice_name: str) -> List[Dict[str, str]]:
    if slice_name not in SLICE_FILTERS:
        raise ValueError(f"Unknown slice name: {slice_name}")
    return [row for row in rows if SLICE_FILTERS[slice_name](row)]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st04_rows = read_csv_rows(args.st04_predictions_path)
    st02_rows = read_csv_rows(args.st02_pair_table_path)
    if not st04_rows:
        raise ValueError("ST0.4 prediction input is empty.")
    if not st02_rows:
        raise ValueError("ST0.2 input is empty.")

    st02_by_pair = {row["pair_id"]: row for row in st02_rows}
    row_index_by_pair_id = {row["pair_id"]: idx for idx, row in enumerate(st04_rows)}
    if len(st02_by_pair) != len(st04_rows):
        missing_pairs = [row["pair_id"] for row in st04_rows if row["pair_id"] not in st02_by_pair]
        if missing_pairs:
            raise ValueError(f"ST0.2 metadata missing for {len(missing_pairs)} ST0.4 rows.")

    calibration_rows = []
    holdout_eval_rows = []
    for row in st04_rows:
        label = row["label_hard_binary"]
        if label == "":
            continue
        if row["split_holdout"] == "train_non_holdout" and row["split_cv5_fold"] == str(args.calibration_fold):
            calibration_rows.append(row)
        if row["split_holdout"] == "holdout_test":
            holdout_eval_rows.append(row)

    if not calibration_rows:
        raise ValueError("No calibration rows found for ST0.5.")
    if not holdout_eval_rows:
        raise ValueError("No holdout eval rows found for ST0.5.")

    calibration_artifacts: Dict[str, object] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st05_calibrate_rank",
        "calibration_fold": args.calibration_fold,
        "models": {},
    }

    summary_rows: List[Dict[str, object]] = []
    calibrated_values_by_model: Dict[str, Dict[str, List[float]]] = {}

    for model_name, raw_col in MODEL_COLUMNS.items():
        x_calib = np.asarray([float(row[raw_col]) for row in calibration_rows], dtype=float)
        y_calib = np.asarray([int(row["label_hard_binary"]) for row in calibration_rows], dtype=int)

        if len(np.unique(y_calib)) < 2:
            raise ValueError(f"Calibration fold has only one class for model {model_name}.")

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(x_calib, y_calib)

        platt = LogisticRegression(
            solver="lbfgs",
            random_state=args.platt_random_state,
            max_iter=1000,
        )
        platt.fit(x_calib.reshape(-1, 1), y_calib)

        all_raw = np.asarray([float(row[raw_col]) for row in st04_rows], dtype=float)
        all_iso = isotonic.predict(all_raw)
        all_platt = platt.predict_proba(all_raw.reshape(-1, 1))[:, 1]
        calibrated_values_by_model[model_name] = {
            "raw": all_raw.tolist(),
            "isotonic": all_iso.tolist(),
            "platt": all_platt.tolist(),
        }

        calibration_artifacts["models"][model_name] = {
            "raw_column": raw_col,
            "isotonic_x_thresholds": [safe_round(float(v)) for v in isotonic.X_thresholds_.tolist()],
            "isotonic_y_thresholds": [safe_round(float(v)) for v in isotonic.y_thresholds_.tolist()],
            "platt_coef": safe_round(float(platt.coef_[0][0])),
            "platt_intercept": safe_round(float(platt.intercept_[0])),
        }

        for dataset_name, dataset_rows in [("calibration", calibration_rows), ("holdout", holdout_eval_rows)]:
            for slice_name in ("full_label", "strict_confidence"):
                sliced_rows = rows_for_slice(dataset_rows, slice_name=slice_name)
                if not sliced_rows:
                    continue
                sliced_idxs = [row_index_by_pair_id[row["pair_id"]] for row in sliced_rows]
                y_true = [int(row["label_hard_binary"]) for row in sliced_rows]
                probs_raw = [float(all_raw[idx]) for idx in sliced_idxs]
                probs_iso = [float(all_iso[idx]) for idx in sliced_idxs]
                probs_platt = [float(all_platt[idx]) for idx in sliced_idxs]
                for variant_name, probs in [("raw", probs_raw), ("isotonic", probs_iso), ("platt", probs_platt)]:
                    m = binary_metrics(y_true, probs, ece_bins=args.ece_bins)
                    summary_rows.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "label_slice": slice_name,
                            "variant": variant_name,
                            "n": int(m["n"]),
                            "positive_rate": m["positive_rate"],
                            "brier_score": m["brier_score"],
                            "log_loss": m["log_loss"],
                            "ece": m["ece"],
                        }
                    )

    output_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(st04_rows):
        pair_meta = st02_by_pair[row["pair_id"]]
        output_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "phage_family": pair_meta.get("phage_family", ""),
                "split_holdout": row["split_holdout"],
                "split_cv5_fold": row["split_cv5_fold"],
                "is_hard_trainable": row["is_hard_trainable"],
                "is_strict_trainable": row["is_strict_trainable"],
                "label_hard_binary": row["label_hard_binary"],
                "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                "pred_dummy_raw": f"{calibrated_values_by_model['dummy_prior']['raw'][idx]:.10f}",
                "pred_dummy_isotonic": f"{calibrated_values_by_model['dummy_prior']['isotonic'][idx]:.10f}",
                "pred_dummy_platt": f"{calibrated_values_by_model['dummy_prior']['platt'][idx]:.10f}",
                "pred_logreg_raw": f"{calibrated_values_by_model['logreg_host_phage']['raw'][idx]:.10f}",
                "pred_logreg_isotonic": f"{calibrated_values_by_model['logreg_host_phage']['isotonic'][idx]:.10f}",
                "pred_logreg_platt": f"{calibrated_values_by_model['logreg_host_phage']['platt'][idx]:.10f}",
            }
        )

    ranked_rows: List[Dict[str, object]] = []
    rows_by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        rows_by_bacteria[str(row["bacteria"])].append(row)

    for bacteria in sorted(rows_by_bacteria.keys()):
        ranked = sorted(
            rows_by_bacteria[bacteria],
            key=lambda r: (-float(str(r["pred_logreg_isotonic"])), str(r["phage"])),
        )
        for rank, row in enumerate(ranked, start=1):
            ranked_rows.append(
                {
                    "bacteria": row["bacteria"],
                    "phage": row["phage"],
                    "phage_family": row["phage_family"],
                    "rank_logreg_isotonic": rank,
                    "score_logreg_isotonic": row["pred_logreg_isotonic"],
                    "score_logreg_raw": row["pred_logreg_raw"],
                    "split_holdout": row["split_holdout"],
                    "label_hard_binary": row["label_hard_binary"],
                    "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                }
            )

    summary_rows.sort(key=lambda r: (str(r["model"]), str(r["dataset"]), str(r["variant"])))
    output_rows.sort(key=lambda r: (str(r["bacteria"]), str(r["phage"])))

    write_csv(
        args.output_dir / "st05_calibration_summary.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )
    write_csv(
        args.output_dir / "st05_pair_predictions_calibrated.csv",
        fieldnames=list(output_rows[0].keys()),
        rows=output_rows,
    )
    write_csv(
        args.output_dir / "st05_ranked_predictions.csv",
        fieldnames=list(ranked_rows[0].keys()),
        rows=ranked_rows,
    )
    write_json(args.output_dir / "st05_calibration_artifacts.json", calibration_artifacts)

    print("ST0.5 completed.")
    print(f"- Calibration rows: {len(calibration_rows)}")
    print(f"- Holdout eval rows: {len(holdout_eval_rows)}")
    print(f"- Output calibrated predictions: {args.output_dir / 'st05_pair_predictions_calibrated.csv'}")
    print(f"- Output ranking: {args.output_dir / 'st05_ranked_predictions.csv'}")


if __name__ == "__main__":
    main()
