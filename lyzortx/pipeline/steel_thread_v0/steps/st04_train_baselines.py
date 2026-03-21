#!/usr/bin/env python3
"""ST0.4: Train baseline models and emit raw pairwise probabilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import parse_float, read_csv_rows, safe_round

CATEGORICAL_FEATURE_COLUMNS = [
    "host_pathotype",
    "host_clermont_phylo",
    "host_origin",
    "host_lps_type",
    "host_o_type",
    "host_h_type",
    "host_collection",
    "host_abc_serotype",
    "phage",
    "phage_morphotype",
    "phage_family",
    "phage_genus",
    "phage_species",
    "phage_subfamily",
    "phage_old_family",
    "phage_old_genus",
    "phage_host",
    "phage_host_phylo",
    "pair_host_phylo_equals_phage_host_phylo",
]

NUMERIC_FEATURE_COLUMNS = [
    "host_mouse_killed_10",
    "host_capsule_abc",
    "host_capsule_groupiv_e",
    "host_capsule_groupiv_e_stricte",
    "host_capsule_groupiv_s",
    "host_capsule_wzy_stricte",
    "host_n_defense_systems",
    "host_n_infections",
    "phage_genome_size",
]

REQUIRED_ST02_COLUMNS = {
    "pair_id",
    "bacteria",
    "phage",
    "label_hard_any_lysis",
    "label_strict_confidence_tier",
}

REQUIRED_ST03_COLUMNS = {
    "pair_id",
    "split_holdout",
    "split_cv5_fold",
    "is_hard_trainable",
    "is_strict_trainable",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 canonical pair table.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.4 artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for baseline estimators.",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--logreg-max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for logistic regression solver.",
    )
    return parser.parse_args(argv)


def build_feature_dict(row: Dict[str, str]) -> Dict[str, object]:
    features: Dict[str, object] = {}
    for col in CATEGORICAL_FEATURE_COLUMNS:
        value = row.get(col, "")
        if value != "":
            features[col] = value
    for col in NUMERIC_FEATURE_COLUMNS:
        parsed = parse_float(row.get(col, ""))
        if parsed is not None:
            features[col] = parsed
    return features


def compute_binary_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, Optional[float]]:
    if not y_true:
        raise ValueError("No labels available for metric computation.")

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


def compute_top3_hit_rate(eval_rows: List[Dict[str, object]], prob_key: str) -> Dict[str, object]:
    by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        by_bacteria[str(row["bacteria"])].append(row)

    total = 0
    hits = 0
    susceptible_total = 0
    susceptible_hits = 0

    for bacteria, rows in by_bacteria.items():
        if not rows:
            continue
        rows_sorted = sorted(
            rows,
            key=lambda r: (-float(r[prob_key]), str(r["phage"])),
        )
        top3 = rows_sorted[:3]
        any_hit = any(int(r["label_hard"]) == 1 for r in top3)
        total += 1
        hits += 1 if any_hit else 0

        susceptible = any(int(r["label_hard"]) == 1 for r in rows_sorted)
        if susceptible:
            susceptible_total += 1
            susceptible_hits += 1 if any_hit else 0

    return {
        "strain_count": total,
        "hit_count": hits,
        "top3_hit_rate_all_strains": safe_round(hits / total if total else 0.0),
        "susceptible_strain_count": susceptible_total,
        "susceptible_hit_count": susceptible_hits,
        "top3_hit_rate_susceptible_only": safe_round(
            susceptible_hits / susceptible_total if susceptible_total else 0.0
        ),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    st03_rows = read_csv_rows(args.st03_split_assignments_path)

    if not st02_rows:
        raise ValueError("ST0.2 table is empty.")
    if not st03_rows:
        raise ValueError("ST0.3 split assignments are empty.")

    st02_columns = set(st02_rows[0].keys())
    st03_columns = set(st03_rows[0].keys())
    missing_st02 = sorted(REQUIRED_ST02_COLUMNS - st02_columns)
    missing_st03 = sorted(REQUIRED_ST03_COLUMNS - st03_columns)
    if missing_st02:
        raise ValueError(f"Missing required ST0.2 columns: {', '.join(missing_st02)}")
    if missing_st03:
        raise ValueError(f"Missing required ST0.3 columns: {', '.join(missing_st03)}")

    st02_by_pair = {row["pair_id"]: row for row in st02_rows}
    st03_by_pair = {row["pair_id"]: row for row in st03_rows}
    pair_ids = sorted(st02_by_pair.keys())
    if set(st02_by_pair.keys()) != set(st03_by_pair.keys()):
        missing_in_st03 = sorted(set(st02_by_pair.keys()) - set(st03_by_pair.keys()))
        missing_in_st02 = sorted(set(st03_by_pair.keys()) - set(st02_by_pair.keys()))
        raise ValueError(
            "Mismatch between ST0.2 and ST0.3 pair_id sets. "
            f"Missing in ST0.3: {len(missing_in_st03)}; missing in ST0.2: {len(missing_in_st02)}"
        )

    joined_rows: List[Dict[str, str]] = []
    for pair_id in pair_ids:
        merged = dict(st02_by_pair[pair_id])
        merged.update(st03_by_pair[pair_id])
        joined_rows.append(merged)

    feature_dicts = [build_feature_dict(row) for row in joined_rows]
    vectorizer = DictVectorizer(sparse=True, sort=True)

    train_indices = [
        idx
        for idx, row in enumerate(joined_rows)
        if row["split_holdout"] == "train_non_holdout" and row["is_hard_trainable"] == "1"
    ]
    holdout_eval_indices = [
        idx
        for idx, row in enumerate(joined_rows)
        if row["split_holdout"] == "holdout_test" and row["is_hard_trainable"] == "1"
    ]

    if not train_indices:
        raise ValueError("No trainable rows found for ST0.4 training.")
    if not holdout_eval_indices:
        raise ValueError("No holdout evaluation rows found for ST0.4.")

    y_train = [int(joined_rows[idx]["label_hard_any_lysis"]) for idx in train_indices]
    X_train = vectorizer.fit_transform([feature_dicts[idx] for idx in train_indices])

    dummy_model = DummyClassifier(strategy="prior", random_state=args.random_state)
    dummy_model.fit(X_train, y_train)

    logreg_model = LogisticRegression(
        C=args.logreg_c,
        max_iter=args.logreg_max_iter,
        class_weight="balanced",
        random_state=args.random_state,
        solver="liblinear",
    )
    logreg_model.fit(X_train, y_train)

    X_all = vectorizer.transform(feature_dicts)
    pred_dummy = dummy_model.predict_proba(X_all)[:, 1]
    pred_logreg = logreg_model.predict_proba(X_all)[:, 1]

    holdout_y = [int(joined_rows[idx]["label_hard_any_lysis"]) for idx in holdout_eval_indices]
    holdout_dummy = [float(pred_dummy[idx]) for idx in holdout_eval_indices]
    holdout_logreg = [float(pred_logreg[idx]) for idx in holdout_eval_indices]

    eval_rows = []
    for idx in holdout_eval_indices:
        row = joined_rows[idx]
        eval_rows.append(
            {
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "label_hard": int(row["label_hard_any_lysis"]),
                "pred_dummy_raw": float(pred_dummy[idx]),
                "pred_logreg_raw": float(pred_logreg[idx]),
            }
        )

    metrics = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_paths": {
            "st02_pair_table": str(args.st02_pair_table_path),
            "st03_split_assignments": str(args.st03_split_assignments_path),
        },
        "train_summary": {
            "train_non_holdout_hard_rows": len(train_indices),
            "holdout_hard_rows": len(holdout_eval_indices),
            "train_positive_count": int(sum(y_train)),
            "train_negative_count": int(len(y_train) - sum(y_train)),
            "holdout_positive_count": int(sum(holdout_y)),
            "holdout_negative_count": int(len(holdout_y) - sum(holdout_y)),
        },
        "feature_summary": {
            "n_vectorized_features": len(vectorizer.get_feature_names_out()),
            "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
            "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        },
        "models": {
            "dummy_prior": {
                "holdout_binary_metrics": compute_binary_metrics(holdout_y, holdout_dummy),
                "holdout_top3_metrics": compute_top3_hit_rate(eval_rows, "pred_dummy_raw"),
            },
            "logreg_host_phage": {
                "holdout_binary_metrics": compute_binary_metrics(holdout_y, holdout_logreg),
                "holdout_top3_metrics": compute_top3_hit_rate(eval_rows, "pred_logreg_raw"),
            },
        },
    }

    feature_names = list(vectorizer.get_feature_names_out())
    coefficients = list(logreg_model.coef_[0])
    ranked_coef = sorted(zip(feature_names, coefficients), key=lambda x: x[1], reverse=True)
    artifacts = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st04_train_baselines",
        "dummy_model": {
            "type": "DummyClassifier",
            "strategy": "prior",
            "class_prior": [safe_round(v) for v in dummy_model.class_prior_.tolist()],
        },
        "logreg_model": {
            "type": "LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "C": args.logreg_c,
            "max_iter": args.logreg_max_iter,
            "random_state": args.random_state,
            "n_iter": int(logreg_model.n_iter_[0]),
            "intercept": safe_round(float(logreg_model.intercept_[0])),
            "top_positive_features": [
                {"feature": feature, "coefficient": safe_round(float(coef))} for feature, coef in ranked_coef[:25]
            ],
            "top_negative_features": [
                {"feature": feature, "coefficient": safe_round(float(coef))}
                for feature, coef in sorted(ranked_coef, key=lambda x: x[1])[:25]
            ],
        },
    }

    prediction_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(joined_rows):
        label_hard = row["label_hard_any_lysis"]
        label_hard_binary = "" if label_hard == "" else int(label_hard)
        prediction_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "split_holdout": row["split_holdout"],
                "split_cv5_fold": row["split_cv5_fold"],
                "is_hard_trainable": row["is_hard_trainable"],
                "is_strict_trainable": row["is_strict_trainable"],
                "label_hard_binary": label_hard_binary,
                "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                "pred_dummy_raw": f"{float(pred_dummy[idx]):.10f}",
                "pred_logreg_raw": f"{float(pred_logreg[idx]):.10f}",
            }
        )

    output_predictions = args.output_dir / "st04_pair_predictions_raw.csv"
    write_csv(output_predictions, fieldnames=list(prediction_rows[0].keys()), rows=prediction_rows)
    write_json(args.output_dir / "st04_model_metrics_raw.json", metrics)
    write_json(args.output_dir / "st04_model_artifacts.json", artifacts)

    print("ST0.4 completed.")
    print(f"- Train rows: {len(train_indices)}")
    print(f"- Holdout eval rows: {len(holdout_eval_indices)}")
    print(f"- Vectorized feature count: {len(vectorizer.get_feature_names_out())}")
    print(f"- Output predictions: {output_predictions}")


if __name__ == "__main__":
    main()
