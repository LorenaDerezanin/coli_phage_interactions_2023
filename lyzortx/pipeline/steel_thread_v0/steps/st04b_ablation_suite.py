#!/usr/bin/env python3
"""ST0.4b: Run host/phage/no-identity ablations on the locked split-suite protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

REQUIRED_ST02_COLUMNS = ("pair_id", "bacteria", "phage", "label_hard_any_lysis")
REQUIRED_ST03B_COLUMNS = ("pair_id", "split_dual_axis")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--st03b-split-suite-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03b_split_suite_assignments.csv"),
    )
    parser.add_argument(
        "--st03b-split-suite-protocol-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03b_split_suite_protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--logreg-max-iter", type=int, default=2000)
    return parser.parse_args(argv)


def read_csv_rows(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [
            {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            for row in reader
        ]


def parse_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def infer_feature_columns(columns: Iterable[str]) -> Dict[str, List[str]]:
    host_non_identity = sorted(col for col in columns if col.startswith("host_"))
    phage_non_identity = sorted(col for col in columns if col.startswith("phage_") and col != "phage")
    pair_features = sorted(col for col in columns if col.startswith("pair_"))
    return {
        "host_only": ["bacteria", *host_non_identity],
        "phage_only": ["phage", *phage_non_identity],
        "no_identity": [*host_non_identity, *phage_non_identity, *pair_features],
        "full_reference": ["bacteria", "phage", *host_non_identity, *phage_non_identity, *pair_features],
    }


def build_feature_dict(row: Dict[str, str], feature_columns: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in feature_columns:
        raw = row.get(col, "")
        if raw == "":
            continue
        parsed = parse_float(raw)
        out[col] = parsed if parsed is not None else raw
    return out


def compute_binary_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "n": float(len(y_true)),
        "positive_rate": safe_round(sum(y_true) / len(y_true)),
        "brier_score": safe_round(brier_score_loss(y_true, y_prob)),
        "log_loss": safe_round(log_loss(y_true, y_prob, labels=[0, 1])),
        "average_precision": None,
        "roc_auc": None,
    }
    if len(set(y_true)) >= 2:
        metrics["average_precision"] = safe_round(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = safe_round(roc_auc_score(y_true, y_prob))
    return metrics


def summarize_signal_sources(matrix_rows: List[Dict[str, object]]) -> List[str]:
    ap_by_model = {
        row["model"]: float(row["average_precision"])
        for row in matrix_rows
        if row["split"] == "all_eval" and row["average_precision"] not in ("", None)
    }
    sorted_models = sorted(ap_by_model.items(), key=lambda item: item[1], reverse=True)
    if not sorted_models:
        return ["No evaluable ablation metrics available."]

    best_model, best_ap = sorted_models[0]
    baseline_ap = ap_by_model.get("dummy_prior", 0.0)
    lines = [
        f"Best all-eval average precision comes from {best_model} (AP={best_ap:.3f}).",
        f"Lift versus dummy prior on all-eval is {best_ap - baseline_ap:.3f} AP.",
    ]

    no_identity_ap = ap_by_model.get("no_identity_logreg")
    host_ap = ap_by_model.get("host_only_logreg")
    phage_ap = ap_by_model.get("phage_only_logreg")
    if host_ap is not None and phage_ap is not None:
        dominant = "host identity" if host_ap >= phage_ap else "phage identity"
        lines.append(
            f"Dominant single-axis signal is {dominant} (host AP={host_ap:.3f}, phage AP={phage_ap:.3f})."
        )
    if no_identity_ap is not None:
        delta = best_ap - no_identity_ap
        lines.append(
            f"No-identity control AP={no_identity_ap:.3f}, indicating {delta:.3f} AP attributable to identity-linked features."
        )
    lines.append(
        "Failure mode check: compare host_only_holdout/phage_only_holdout/dual_holdout_test rows for the sharpest drop."
    )
    return lines


def split_membership(split_name: str) -> bool:
    return split_name in {"dual_holdout_test", "host_only_holdout", "phage_only_holdout"}


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path, REQUIRED_ST02_COLUMNS)
    st03b_rows = read_csv_rows(args.st03b_split_suite_path, REQUIRED_ST03B_COLUMNS)
    st03b_by_pair = {row["pair_id"]: row for row in st03b_rows}

    joined_rows: List[Dict[str, str]] = []
    for row in st02_rows:
        pair_id = row["pair_id"]
        if pair_id not in st03b_by_pair:
            raise ValueError(f"pair_id={pair_id} from ST0.2 missing in ST0.3b assignments")
        merged = dict(row)
        merged.update(st03b_by_pair[pair_id])
        joined_rows.append(merged)

    train_rows = [
        r
        for r in joined_rows
        if r["split_dual_axis"] == "train_non_holdout" and r["label_hard_any_lysis"] in {"0", "1"}
    ]
    eval_rows = [
        r
        for r in joined_rows
        if split_membership(r["split_dual_axis"]) and r["label_hard_any_lysis"] in {"0", "1"}
    ]
    if not train_rows or not eval_rows:
        raise ValueError("ST0.4b requires non-empty train_non_holdout and evaluation rows.")

    feature_sets = infer_feature_columns(joined_rows[0].keys())
    matrix_rows: List[Dict[str, object]] = []
    model_predictions: Dict[str, List[float]] = {}

    for name, columns in feature_sets.items():
        vectorizer = DictVectorizer(sparse=True, sort=True)
        train_dicts = [build_feature_dict(row, columns) for row in train_rows]
        y_train = [int(row["label_hard_any_lysis"]) for row in train_rows]
        X_train = vectorizer.fit_transform(train_dicts)
        X_eval = vectorizer.transform([build_feature_dict(row, columns) for row in eval_rows])
        y_eval = [int(row["label_hard_any_lysis"]) for row in eval_rows]

        model = LogisticRegression(
            C=args.logreg_c,
            max_iter=args.logreg_max_iter,
            class_weight="balanced",
            random_state=args.random_state,
            solver="liblinear",
        )
        model.fit(X_train, y_train)
        probs = [float(p) for p in model.predict_proba(X_eval)[:, 1]]
        model_key = f"{name}_logreg"
        model_predictions[model_key] = probs

        by_split_probs: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for row, prob in zip(eval_rows, probs):
            by_split_probs[row["split_dual_axis"]].append((int(row["label_hard_any_lysis"]), prob))

        all_metrics = compute_binary_metrics(y_eval, probs)
        matrix_rows.append({"model": model_key, "split": "all_eval", **all_metrics})
        for split_name in sorted(by_split_probs.keys()):
            labels = [label for label, _ in by_split_probs[split_name]]
            split_probs = [prob for _, prob in by_split_probs[split_name]]
            matrix_rows.append({"model": model_key, "split": split_name, **compute_binary_metrics(labels, split_probs)})

    dummy = DummyClassifier(strategy="prior", random_state=args.random_state)
    X_dummy = [[0] for _ in train_rows]
    y_train_dummy = [int(row["label_hard_any_lysis"]) for row in train_rows]
    dummy.fit(X_dummy, y_train_dummy)
    y_eval_dummy = [int(row["label_hard_any_lysis"]) for row in eval_rows]
    dummy_probs = [float(dummy.predict_proba([[0]])[0][1]) for _ in eval_rows]
    matrix_rows.append({"model": "dummy_prior", "split": "all_eval", **compute_binary_metrics(y_eval_dummy, dummy_probs)})

    signal_summary = summarize_signal_sources(matrix_rows)

    matrix_rows_sorted = sorted(matrix_rows, key=lambda row: (str(row["split"]), str(row["model"])))
    matrix_path = args.output_dir / "st04b_ablation_matrix.csv"
    summary_path = args.output_dir / "st04b_signal_summary.json"

    protocol_sha = hashlib.sha256(args.st03b_split_suite_protocol_path.read_bytes()).hexdigest()
    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st04b_ablation_suite",
        "input_paths": {
            "st02_pair_table": str(args.st02_pair_table_path),
            "st03b_split_suite": str(args.st03b_split_suite_path),
            "st03b_split_suite_protocol": str(args.st03b_split_suite_protocol_path),
        },
        "input_hashes_sha256": {
            "st03b_split_suite_protocol": protocol_sha,
        },
        "split_protocol_lock": {
            "split_column": "split_dual_axis",
            "train_value": "train_non_holdout",
            "evaluation_values": ["dual_holdout_test", "host_only_holdout", "phage_only_holdout"],
        },
        "feature_sets": feature_sets,
        "signal_summary_lines": signal_summary,
        "row_counts": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
    }

    write_csv(matrix_path, fieldnames=list(matrix_rows_sorted[0].keys()), rows=matrix_rows_sorted)
    write_json(summary_path, summary)

    print("ST0.4b completed.")
    print(f"- Ablation rows: {len(matrix_rows_sorted)}")
    print(f"- Matrix output: {matrix_path}")
    print(f"- Summary output: {summary_path}")


if __name__ == "__main__":
    main()
