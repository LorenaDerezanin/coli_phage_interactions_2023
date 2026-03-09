#!/usr/bin/env python3
"""Run a PHIStruct-style RBP embedding pilot on the phage-family holdout split.

This pilot compares two feature variants on the same phage-family holdout protocol:
1) non_structural_rbp: mechanistic RBP/depolymerase/domain proxy features.
2) structural_rbp_embedding: sequence-derived phage embedding (k-mer + SVD) used
   as a PHIStruct-style structural proxy when explicit RBP structures are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_a.steps.build_mechanistic_proxy_features import build_phage_proxy_rows

REQUIRED_ST02_COLUMNS: Sequence[str] = ("pair_id", "bacteria", "phage", "phage_family", "label_hard_any_lysis")
REQUIRED_ST03B_COLUMNS: Sequence[str] = ("pair_id", "split_phage_family_holdout")
DNA_ALPHABET: Sequence[str] = ("A", "C", "G", "T")


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
        "--phage-fna-dir",
        type=Path,
        default=Path("data/genomics/phages/FNA"),
        help="Directory with per-phage genome FASTA files named <phage>.fna.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/phistruct_pilot"),
    )
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--kmer-size", type=int, default=4)
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


def _read_fasta_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    pieces: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            pieces.append(stripped.upper())
    return "".join(pieces)


def _all_kmers(k: int) -> List[str]:
    return ["".join(chars) for chars in product(DNA_ALPHABET, repeat=k)]


def _kmer_frequency_vector(sequence: str, kmer_list: Sequence[str], k: int) -> np.ndarray:
    valid_chars = set(DNA_ALPHABET)
    total = 0
    counts = {kmer: 0 for kmer in kmer_list}
    for idx in range(0, max(len(sequence) - k + 1, 0)):
        kmer = sequence[idx : idx + k]
        if all(ch in valid_chars for ch in kmer):
            counts[kmer] += 1
            total += 1

    if total == 0:
        return np.zeros(len(kmer_list), dtype=float)
    return np.array([counts[kmer] / total for kmer in kmer_list], dtype=float)


def build_phistruct_style_embeddings(
    *,
    phage_names: Sequence[str],
    phage_fna_dir: Path,
    embedding_dim: int,
    kmer_size: int,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    kmers = _all_kmers(kmer_size)
    vectors: List[np.ndarray] = []
    missing: List[str] = []
    for phage in phage_names:
        sequence = _read_fasta_sequence(phage_fna_dir / f"{phage}.fna")
        if not sequence:
            missing.append(phage)
            vectors.append(np.zeros(len(kmers), dtype=float))
            continue
        vectors.append(_kmer_frequency_vector(sequence, kmers, kmer_size))

    matrix = np.vstack(vectors)
    n_samples, n_features = matrix.shape
    svd_dim = max(1, min(embedding_dim, n_samples - 1, n_features))
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    reduced = svd.fit_transform(matrix)

    if svd_dim < embedding_dim:
        padded = np.zeros((n_samples, embedding_dim), dtype=float)
        padded[:, :svd_dim] = reduced
        reduced = padded

    by_phage: Dict[str, Dict[str, float]] = {}
    for idx, phage in enumerate(phage_names):
        by_phage[phage] = {f"phage_struct_rbp_emb_{col:02d}": round(float(reduced[idx, col]), 8) for col in range(embedding_dim)}
    by_phage["__metadata__"] = {
        "missing_genome_count": float(len(missing)),
        "svd_dim_effective": float(svd_dim),
    }
    return by_phage


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
    for host_rows in by_host.values():
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


def _safe_metric_binary(
    metric_name: str,
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> float:
    if metric_name == "average_precision":
        if len(set(y_true)) < 2:
            return float(sum(y_true) / len(y_true))
        return float(average_precision_score(y_true, y_prob))
    if metric_name == "roc_auc":
        if len(set(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    raise ValueError(metric_name)


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
        "average_precision": round(_safe_metric_binary("average_precision", y_eval, probs), 6),
        "roc_auc": round(_safe_metric_binary("roc_auc", y_eval, probs), 6),
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
    phage_names = sorted({row["phage"] for row in phage_rows if row.get("phage")})
    struct_embeddings = build_phistruct_style_embeddings(
        phage_names=phage_names,
        phage_fna_dir=args.phage_fna_dir,
        embedding_dim=args.embedding_dim,
        kmer_size=args.kmer_size,
        random_state=args.random_state,
    )

    enriched_rows: List[Dict[str, str]] = []
    for row in st02_rows:
        if row["pair_id"] not in st03b_by_pair or row["label_hard_any_lysis"] not in {"0", "1"}:
            continue

        merged = dict(row)
        merged.update(st03b_by_pair[row["pair_id"]])
        merged.update({k: str(v) for k, v in phage_proxy_by_name.get(row["phage"], {}).items()})
        merged.update({k: str(v) for k, v in struct_embeddings.get(row["phage"], {}).items()})
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
    metadata = struct_embeddings.get("__metadata__", {})
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
            "phage_fna_dir": str(args.phage_fna_dir),
        },
        "config": {
            "embedding_dim": args.embedding_dim,
            "kmer_size": args.kmer_size,
            "top_k": args.top_k,
            "random_state": args.random_state,
        },
        "embedding_audit": {
            "missing_genome_count": int(metadata.get("missing_genome_count", 0.0)),
            "svd_dim_effective": int(metadata.get("svd_dim_effective", 0.0)),
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
