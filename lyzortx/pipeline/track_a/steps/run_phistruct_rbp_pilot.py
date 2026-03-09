#!/usr/bin/env python3
"""Run a PHIStruct-style RBP embedding pilot on the phage-family holdout split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_a.steps.build_mechanistic_proxy_features import build_phage_proxy_rows

REQUIRED_ST02_COLUMNS: Sequence[str] = (
    "pair_id",
    "bacteria",
    "phage",
    "phage_family",
    "label_hard_any_lysis",
)
REQUIRED_ST03B_COLUMNS: Sequence[str] = ("pair_id", "split_phage_family_holdout")


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
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/phistruct_pilot"),
    )
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def read_csv_rows(path: Path, required_columns: Sequence[str], delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def parse_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def phistruct_style_embedding(phage_row: Mapping[str, str], dim: int) -> Dict[str, float]:
    fields = ["Morphotype", "Family", "Subfamily", "Genus", "Species"]
    tokens = [str(phage_row.get(field, "")).lower() for field in fields if str(phage_row.get(field, "")).strip()]
    if not tokens:
        tokens = ["missing_phage_metadata"]

    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dim):
            value = digest[idx % len(digest)]
            vector[idx] += (value / 255.0) * 2.0 - 1.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return {f"phage_struct_rbp_emb_{idx:02d}": round(val, 8) for idx, val in enumerate(vector)}


def expected_calibration_error(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not y_true:
        return 0.0

    counts = [0 for _ in range(n_bins)]
    conf_sum = [0.0 for _ in range(n_bins)]
    label_sum = [0.0 for _ in range(n_bins)]

    for label, prob in zip(y_true, y_prob):
        clipped = min(max(prob, 0.0), 1.0)
        bin_idx = min(int(clipped * n_bins), n_bins - 1)
        counts[bin_idx] += 1
        conf_sum[bin_idx] += clipped
        label_sum[bin_idx] += float(label)

    total = len(y_true)
    ece = 0.0
    for idx in range(n_bins):
        if counts[idx] == 0:
            continue
        acc = label_sum[idx] / counts[idx]
        conf = conf_sum[idx] / counts[idx]
        ece += abs(acc - conf) * (counts[idx] / total)
    return round(ece, 6)


def top_k_hit_rate(rows: Sequence[Mapping[str, object]], top_k: int) -> float:
    by_host: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_host[str(row["bacteria"])].append(row)

    hits = 0
    for host, host_rows in by_host.items():
        del host
        ranked = sorted(host_rows, key=lambda item: float(item["pred_prob"]), reverse=True)[:top_k]
        if any(int(item["label"]) == 1 for item in ranked):
            hits += 1

    return round(hits / len(by_host), 6) if by_host else 0.0


def _to_feature_dict(row: Mapping[str, str], columns: Iterable[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in columns:
        raw = row.get(col, "")
        if raw == "":
            continue
        as_float = parse_float(raw)
        out[col] = as_float if as_float is not None else raw
    return out


def evaluate_variant(
    *,
    variant_name: str,
    train_rows: Sequence[Mapping[str, str]],
    eval_rows: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
    top_k: int,
    random_state: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    vectorizer = DictVectorizer(sparse=True, sort=True)
    X_train = vectorizer.fit_transform([_to_feature_dict(row, feature_columns) for row in train_rows])
    y_train = [int(row["label_hard_any_lysis"]) for row in train_rows]

    model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear", random_state=random_state)
    model.fit(X_train, y_train)

    X_eval = vectorizer.transform([_to_feature_dict(row, feature_columns) for row in eval_rows])
    y_eval = [int(row["label_hard_any_lysis"]) for row in eval_rows]
    probs = [float(p) for p in model.predict_proba(X_eval)[:, 1]]

    prediction_rows: List[Dict[str, object]] = []
    for row, prob in zip(eval_rows, probs):
        prediction_rows.append(
            {
                "variant": variant_name,
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "phage_family": row.get("phage_family", ""),
                "label": int(row["label_hard_any_lysis"]),
                "pred_prob": round(prob, 6),
            }
        )

    metric_row: Dict[str, object] = {
        "variant": variant_name,
        "n_eval": len(y_eval),
        "positive_rate_eval": round(sum(y_eval) / len(y_eval), 6),
        "average_precision": round(float(average_precision_score(y_eval, probs)), 6),
        "roc_auc": round(float(roc_auc_score(y_eval, probs)), 6),
        "brier_score": round(float(brier_score_loss(y_eval, probs)), 6),
        "ece": expected_calibration_error(y_eval, probs, n_bins=10),
        "topk_hit_rate": top_k_hit_rate(prediction_rows, top_k=top_k),
    }
    return metric_row, prediction_rows


def build_variant_feature_columns(rows: Sequence[Mapping[str, str]], embedding_dim: int) -> Dict[str, List[str]]:
    st02_columns = set(rows[0].keys())
    host_pair_cols = sorted(col for col in st02_columns if col.startswith("host_") or col.startswith("pair_"))
    phage_non_struct_cols = [
        "phage_rbp_tail_associated_proxy",
        "phage_depolymerase_proxy",
        "phage_domain_complexity_proxy",
    ]
    structural_cols = [f"phage_struct_rbp_emb_{idx:02d}" for idx in range(embedding_dim)]
    return {
        "non_structural_rbp": [*host_pair_cols, *phage_non_struct_cols],
        "structural_rbp_embedding": [*host_pair_cols, *structural_cols],
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path, required_columns=REQUIRED_ST02_COLUMNS)
    st03b_rows = read_csv_rows(args.st03b_split_suite_path, required_columns=REQUIRED_ST03B_COLUMNS)
    phage_rows = read_csv_rows(args.phage_metadata_path, required_columns=("phage",), delimiter=";")

    st03b_by_pair = {row["pair_id"]: row for row in st03b_rows}
    phage_proxy_by_name = {row["phage"]: row for row in build_phage_proxy_rows(phage_rows)}
    phage_metadata_by_name = {row["phage"]: row for row in phage_rows}

    enriched_rows: List[Dict[str, str]] = []
    for row in st02_rows:
        if row["pair_id"] not in st03b_by_pair:
            continue
        if row["label_hard_any_lysis"] not in {"0", "1"}:
            continue

        merged = dict(row)
        merged.update(st03b_by_pair[row["pair_id"]])
        merged.update({k: str(v) for k, v in phage_proxy_by_name.get(row["phage"], {}).items()})

        emb = phistruct_style_embedding(phage_metadata_by_name.get(row["phage"], {}), dim=args.embedding_dim)
        merged.update({k: str(v) for k, v in emb.items()})
        enriched_rows.append(merged)

    train_rows = [row for row in enriched_rows if row["split_phage_family_holdout"] == "train_non_holdout"]
    eval_rows = [row for row in enriched_rows if row["split_phage_family_holdout"] == "holdout_test"]
    if not train_rows or not eval_rows:
        raise ValueError("Pilot requires non-empty train_non_holdout and holdout_test rows from split_phage_family_holdout")

    features_by_variant = build_variant_feature_columns(enriched_rows, embedding_dim=args.embedding_dim)

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for variant_name, columns in features_by_variant.items():
        metrics, preds = evaluate_variant(
            variant_name=variant_name,
            train_rows=train_rows,
            eval_rows=eval_rows,
            feature_columns=columns,
            top_k=args.top_k,
            random_state=args.random_state,
        )
        metric_rows.append(metrics)
        prediction_rows.extend(preds)

    metric_rows_sorted = sorted(metric_rows, key=lambda row: str(row["variant"]))
    prediction_rows_sorted = sorted(
        prediction_rows,
        key=lambda row: (str(row["variant"]), str(row["bacteria"]), -float(row["pred_prob"]), str(row["phage"])),
    )

    metric_by_variant = {str(row["variant"]): row for row in metric_rows_sorted}
    delta_summary = {
        "delta_structural_minus_non_structural": {
            "average_precision": round(
                float(metric_by_variant["structural_rbp_embedding"]["average_precision"])
                - float(metric_by_variant["non_structural_rbp"]["average_precision"]),
                6,
            ),
            "brier_score": round(
                float(metric_by_variant["structural_rbp_embedding"]["brier_score"])
                - float(metric_by_variant["non_structural_rbp"]["brier_score"]),
                6,
            ),
            "ece": round(
                float(metric_by_variant["structural_rbp_embedding"]["ece"])
                - float(metric_by_variant["non_structural_rbp"]["ece"]),
                6,
            ),
            "topk_hit_rate": round(
                float(metric_by_variant["structural_rbp_embedding"]["topk_hit_rate"])
                - float(metric_by_variant["non_structural_rbp"]["topk_hit_rate"]),
                6,
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step_name": "run_phistruct_rbp_pilot",
        "input_paths": {
            "st02_pair_table": str(args.st02_pair_table_path),
            "st03b_split_suite": str(args.st03b_split_suite_path),
            "phage_metadata": str(args.phage_metadata_path),
        },
        "config": {
            "embedding_dim": args.embedding_dim,
            "top_k": args.top_k,
            "random_state": args.random_state,
        },
        "row_counts": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
    }

    metrics_path = args.output_dir / "phistruct_pilot_metrics.csv"
    predictions_path = args.output_dir / "phistruct_pilot_predictions.csv"
    summary_path = args.output_dir / "phistruct_pilot_summary.json"

    write_csv(metrics_path, fieldnames=list(metric_rows_sorted[0].keys()), rows=metric_rows_sorted)
    write_csv(predictions_path, fieldnames=list(prediction_rows_sorted[0].keys()), rows=prediction_rows_sorted)
    write_json(summary_path, delta_summary)

    print(json.dumps({"metrics": str(metrics_path), "predictions": str(predictions_path), "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
