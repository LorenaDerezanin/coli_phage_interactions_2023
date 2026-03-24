#!/usr/bin/env python3
"""TG04: Compute SHAP explanations for Track G LightGBM recommendations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps import calibrate_gbm_outputs
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier

logger = logging.getLogger(__name__)

TG02_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "bacteria",
    "phage",
    "phage_family",
    "split_holdout",
    "prediction_context",
    "is_strict_trainable",
    "label_hard_any_lysis",
    "pred_lightgbm_raw",
    "pred_lightgbm_isotonic",
    "pred_lightgbm_platt",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg01-model-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json"),
        help="TG01 model summary JSON containing the best LightGBM hyperparameters.",
    )
    parser.add_argument(
        "--tg02-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg02_gbm_calibration/tg02_pair_predictions_calibrated.csv"),
        help="TG02 calibrated pair prediction CSV used to select recommended phages.",
    )
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table path.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments path.",
    )
    parser.add_argument(
        "--track-c-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
        help="Input Track C v1 pair table path.",
    )
    parser.add_argument(
        "--track-d-genome-kmer-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv"),
        help="Input Track D genome k-mer feature CSV.",
    )
    parser.add_argument(
        "--track-d-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_d/phage_distance_embedding/phage_distance_embedding_features.csv"
        ),
        help="Input Track D phage-distance feature CSV.",
    )
    parser.add_argument(
        "--track-e-rbp-compatibility-path",
        type=Path,
        default=train_v1_binary_classifier.TRACK_E_REQUIRED_BLOCKS[0][1],
        help="Input Track E RBP-receptor compatibility feature CSV.",
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=train_v1_binary_classifier.TRACK_E_REQUIRED_BLOCKS[1][1],
        help="Input Track E isolation-host distance feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg04_shap_explanations"),
        help="Directory for TG04 artifacts.",
    )
    parser.add_argument(
        "--recommendation-count",
        type=int,
        default=3,
        help="Number of top phages to explain for each strain.",
    )
    parser.add_argument(
        "--top-features-per-pair",
        type=int,
        default=3,
        help="Number of positive and negative SHAP features to surface for each recommended pair.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Fallback random state if TG01 summary is unavailable.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume TG01/TG02 outputs already exist instead of generating them when missing.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_prerequisite_outputs(args: argparse.Namespace) -> None:
    train_v1_args = train_v1_binary_classifier.parse_args(
        [
            "--st02-pair-table-path",
            str(args.st02_pair_table_path),
            "--st03-split-assignments-path",
            str(args.st03_split_assignments_path),
            "--track-c-pair-table-path",
            str(args.track_c_pair_table_path),
            "--track-d-genome-kmer-path",
            str(args.track_d_genome_kmer_path),
            "--track-d-distance-path",
            str(args.track_d_distance_path),
            "--track-e-rbp-compatibility-path",
            str(args.track_e_rbp_compatibility_path),
            "--track-e-isolation-distance-path",
            str(args.track_e_isolation_distance_path),
            "--skip-prerequisites",
        ]
    )
    train_v1_binary_classifier.ensure_prerequisite_outputs(train_v1_args)
    if args.skip_prerequisites:
        return
    if not args.tg01_model_summary_path.exists():
        train_v1_binary_classifier.main([])
    if not args.tg02_predictions_path.exists():
        calibrate_gbm_outputs.main([])


def load_best_lightgbm_params(path: Path) -> Tuple[Dict[str, object], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload["lightgbm"]["best_params"]
    random_state = int(payload.get("cv_protocol", {}).get("random_state", 42))
    return dict(params), random_state


def _dense_row(matrix: Any, row_index: int) -> np.ndarray:
    row = matrix[row_index]
    if sparse.issparse(row):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


def feature_block_for_vectorized_name(
    feature_name: str,
    feature_space: train_v1_binary_classifier.FeatureSpace,
) -> str:
    base_name = feature_name.split("=", 1)[0]
    if base_name in set(feature_space.track_d_columns) or base_name.startswith("phage_"):
        return "track_d_phage_genomic"
    if base_name in set(feature_space.track_e_columns):
        return "track_e_pairwise"
    if base_name in set(feature_space.track_c_additional_columns):
        return "track_c_host_genomic"
    return "st04_v0_baseline"


def top_feature_contributions(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int,
) -> Dict[str, List[Dict[str, object]]]:
    positives = sorted(
        (
            {
                "feature_name": feature_names[index],
                "feature_value": safe_round(float(feature_values[index])),
                "shap_value": safe_round(float(shap_values[index])),
            }
            for index in np.argsort(-shap_values)
            if float(shap_values[index]) > 0.0
        ),
        key=lambda item: float(item["shap_value"]),
        reverse=True,
    )[:top_k]
    negatives = sorted(
        (
            {
                "feature_name": feature_names[index],
                "feature_value": safe_round(float(feature_values[index])),
                "shap_value": safe_round(float(shap_values[index])),
            }
            for index in np.argsort(shap_values)
            if float(shap_values[index]) < 0.0
        ),
        key=lambda item: float(item["shap_value"]),
    )[:top_k]
    return {"positive": positives, "negative": negatives}


def format_contribution_summary(contributions: Sequence[Mapping[str, object]]) -> str:
    if not contributions:
        return ""
    return "; ".join(
        f"{item['feature_name']}={item['feature_value']} ({float(item['shap_value']):+.4f})" for item in contributions
    )


def build_global_feature_importance_rows(
    shap_matrix: Any,
    feature_names: Sequence[str],
    feature_space: train_v1_binary_classifier.FeatureSpace,
) -> List[Dict[str, object]]:
    if sparse.issparse(shap_matrix):
        abs_mean = np.asarray(np.abs(shap_matrix).mean(axis=0)).ravel()
        signed_mean = np.asarray(shap_matrix.mean(axis=0)).ravel()
        nonzero_fraction = np.asarray(shap_matrix.getnnz(axis=0) / shap_matrix.shape[0]).ravel()
    else:
        values = np.asarray(shap_matrix)
        abs_mean = np.mean(np.abs(values), axis=0)
        signed_mean = np.mean(values, axis=0)
        nonzero_fraction = np.count_nonzero(values, axis=0) / values.shape[0]

    rows = [
        {
            "feature_name": feature_name,
            "feature_block": feature_block_for_vectorized_name(feature_name, feature_space),
            "mean_abs_shap": safe_round(float(abs_mean[index])),
            "mean_shap": safe_round(float(signed_mean[index])),
            "nonzero_fraction": safe_round(float(nonzero_fraction[index])),
        }
        for index, feature_name in enumerate(feature_names)
    ]
    rows.sort(
        key=lambda row: (-float(row["mean_abs_shap"]), -float(abs(float(row["mean_shap"]))), str(row["feature_name"]))
    )
    return rows


def select_recommendation_rows(
    calibrated_rows: Sequence[Mapping[str, str]],
    *,
    recommendation_count: int,
) -> List[Dict[str, str]]:
    rows_by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in calibrated_rows:
        rows_by_bacteria[str(row["bacteria"])].append(dict(row))

    selected: List[Dict[str, str]] = []
    for bacteria in sorted(rows_by_bacteria):
        ranked = sorted(
            rows_by_bacteria[bacteria],
            key=lambda row: (-float(row["pred_lightgbm_isotonic"]), str(row["phage"])),
        )
        for rank, row in enumerate(ranked[:recommendation_count], start=1):
            enriched = dict(row)
            enriched["recommendation_rank"] = str(rank)
            selected.append(enriched)
    return selected


def classify_strain_difficulty(
    *,
    top_score: float,
    score_gap: float,
    mean_margin_from_half: float,
    top3_hit: bool,
) -> str:
    if (not top3_hit) or top_score < 0.45 or score_gap < 0.03 or mean_margin_from_half < 0.08:
        return "hard"
    if top3_hit and top_score >= 0.75 and score_gap >= 0.10 and mean_margin_from_half >= 0.18:
        return "easy"
    return "moderate"


def build_per_strain_summary_rows(
    calibrated_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, object]],
    *,
    recommendation_count: int,
) -> List[Dict[str, object]]:
    recommendation_rows_by_bacteria: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
        recommendation_rows_by_bacteria[str(row["bacteria"])].append(row)

    rows_by_bacteria: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in calibrated_rows:
        rows_by_bacteria[str(row["bacteria"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for bacteria in sorted(rows_by_bacteria):
        ranked = sorted(
            rows_by_bacteria[bacteria],
            key=lambda row: (-float(row["pred_lightgbm_isotonic"]), str(row["phage"])),
        )
        top_score = float(ranked[0]["pred_lightgbm_isotonic"])
        second_score = float(ranked[1]["pred_lightgbm_isotonic"]) if len(ranked) > 1 else 0.0
        score_gap = top_score - second_score
        top_rows = ranked[:recommendation_count]
        top3_hit = any(int(row["label_hard_any_lysis"]) == 1 for row in top_rows)
        positive_pair_count = sum(int(row["label_hard_any_lysis"]) for row in ranked)
        mean_margin = float(np.mean([abs(float(row["pred_lightgbm_raw"]) - 0.5) for row in ranked]))
        recommendation_details = recommendation_rows_by_bacteria.get(bacteria, [])

        positive_driver = ""
        negative_driver = ""
        if recommendation_details:
            positive_driver = str(recommendation_details[0].get("top_positive_feature_1", ""))
            negative_driver = str(recommendation_details[0].get("top_negative_feature_1", ""))

        difficulty = classify_strain_difficulty(
            top_score=top_score,
            score_gap=score_gap,
            mean_margin_from_half=mean_margin,
            top3_hit=top3_hit,
        )
        if difficulty == "easy":
            rationale = (
                f"Easy because {positive_driver or 'the leading signal'} separates the top recommendation "
                f"(score {top_score:.3f}, gap {score_gap:.3f})."
            )
        elif difficulty == "hard":
            rationale = (
                f"Hard because scores are compressed (top score {top_score:.3f}, gap {score_gap:.3f}) and "
                f"{negative_driver or 'suppressive host features'} keep likely matches uncertain."
            )
        else:
            rationale = (
                f"Moderate because the model has some separation (top score {top_score:.3f}, gap {score_gap:.3f}) "
                f"but not enough to treat the strain as clearly easy."
            )

        summary_rows.append(
            {
                "bacteria": bacteria,
                "difficulty_label": difficulty,
                "top_recommended_phage": top_rows[0]["phage"],
                "top_recommendation_score_isotonic": safe_round(top_score),
                "top1_top2_gap_isotonic": safe_round(score_gap),
                "mean_margin_from_raw_half": safe_round(mean_margin),
                "positive_pair_count": positive_pair_count,
                "top3_hit": int(top3_hit),
                "top_positive_driver": positive_driver,
                "top_negative_driver": negative_driver,
                "rationale": rationale,
            }
        )

    difficulty_order = {"hard": 0, "moderate": 1, "easy": 2}
    summary_rows.sort(key=lambda row: (difficulty_order[str(row["difficulty_label"])], str(row["bacteria"])))
    return summary_rows


def summarize_difficulty_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = {"hard": 0, "moderate": 0, "easy": 0}
    for row in rows:
        counts[str(row["difficulty_label"])] += 1
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("TG04 starting: compute SHAP explanations")
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)

    best_lightgbm_params, random_state = load_best_lightgbm_params(args.tg01_model_summary_path)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)
    calibrated_rows = read_csv_rows(args.tg02_predictions_path, required_columns=TG02_REQUIRED_COLUMNS)

    track_d_feature_columns = train_v1_binary_classifier._deduplicate_preserving_order(
        [column for column in track_d_genome_rows[0].keys() if column != "phage"]
        + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
    )
    track_e_feature_columns = train_v1_binary_classifier._deduplicate_preserving_order(
        [column for column in track_e_rbp_rows[0].keys() if column not in train_v1_binary_classifier.IDENTIFIER_COLUMNS]
        + [
            column
            for column in track_e_isolation_rows[0].keys()
            if column not in train_v1_binary_classifier.IDENTIFIER_COLUMNS
        ]
    )
    feature_space = train_v1_binary_classifier.build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )
    merged_rows = train_v1_binary_classifier.merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_isolation_rows),
    )

    lightgbm_factory = lambda params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=random_state or args.random_state,
    )
    estimator, vectorizer, _, _, _ = train_v1_binary_classifier.fit_final_estimator(
        merged_rows,
        feature_space,
        estimator_factory=lightgbm_factory,
        params=best_lightgbm_params,
    )

    explain_rows = [dict(row) for row in merged_rows if str(row["is_hard_trainable"]) == "1"]
    feature_matrix = vectorizer.transform(
        [
            train_v1_binary_classifier._build_feature_dict(
                row,
                categorical_columns=feature_space.categorical_columns,
                numeric_columns=feature_space.numeric_columns,
            )
            for row in explain_rows
        ]
    )
    model_probabilities = train_v1_binary_classifier._predict_probabilities(estimator, feature_matrix)

    import shap

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(feature_matrix)
    shap_matrix = explanation.values
    base_values = np.asarray(explanation.base_values).ravel()
    feature_names = list(vectorizer.get_feature_names_out())

    explain_index_by_pair_id = {row["pair_id"]: index for index, row in enumerate(explain_rows)}

    recommendation_source_rows = select_recommendation_rows(
        calibrated_rows,
        recommendation_count=args.recommendation_count,
    )
    recommendation_rows: List[Dict[str, object]] = []
    for row in recommendation_source_rows:
        pair_id = row["pair_id"]
        explain_index = explain_index_by_pair_id.get(pair_id)
        if explain_index is None:
            continue
        shap_row = _dense_row(shap_matrix, explain_index)
        feature_row = _dense_row(feature_matrix, explain_index)
        contributions = top_feature_contributions(
            shap_row,
            feature_row,
            feature_names,
            top_k=args.top_features_per_pair,
        )
        total_abs_shap = float(np.abs(shap_row).sum())
        explanation_text = (
            f"Recommended at rank {row['recommendation_rank']} because "
            f"{format_contribution_summary(contributions['positive']) or 'positive feature contributions'} "
            f"outweighed "
            f"{format_contribution_summary(contributions['negative']) or 'negative feature contributions'}."
        )

        pair_row: Dict[str, object] = {
            "pair_id": pair_id,
            "bacteria": row["bacteria"],
            "phage": row["phage"],
            "phage_family": row["phage_family"],
            "recommendation_rank": int(row["recommendation_rank"]),
            "split_holdout": row["split_holdout"],
            "prediction_context": row["prediction_context"],
            "label_hard_any_lysis": row["label_hard_any_lysis"],
            "pred_lightgbm_isotonic": row["pred_lightgbm_isotonic"],
            "pred_lightgbm_raw_oof_or_holdout": row["pred_lightgbm_raw"],
            "pred_lightgbm_platt": row["pred_lightgbm_platt"],
            "pred_lightgbm_raw_final_model": safe_round(model_probabilities[explain_index]),
            "shap_base_value": safe_round(float(base_values[explain_index])),
            "total_abs_shap": safe_round(total_abs_shap),
            "top_positive_feature_1": "",
            "top_positive_feature_2": "",
            "top_positive_feature_3": "",
            "top_positive_shap_1": "",
            "top_positive_shap_2": "",
            "top_positive_shap_3": "",
            "top_negative_feature_1": "",
            "top_negative_feature_2": "",
            "top_negative_feature_3": "",
            "top_negative_shap_1": "",
            "top_negative_shap_2": "",
            "top_negative_shap_3": "",
            "explanation_text": explanation_text,
        }
        for position, item in enumerate(contributions["positive"], start=1):
            pair_row[f"top_positive_feature_{position}"] = item["feature_name"]
            pair_row[f"top_positive_shap_{position}"] = item["shap_value"]
        for position, item in enumerate(contributions["negative"], start=1):
            pair_row[f"top_negative_feature_{position}"] = item["feature_name"]
            pair_row[f"top_negative_shap_{position}"] = item["shap_value"]
        recommendation_rows.append(pair_row)

    recommendation_rows.sort(key=lambda row: (str(row["bacteria"]), int(row["recommendation_rank"]), str(row["phage"])))
    global_rows = build_global_feature_importance_rows(shap_matrix, feature_names, feature_space)
    strain_summary_rows = build_per_strain_summary_rows(
        calibrated_rows,
        recommendation_rows,
        recommendation_count=args.recommendation_count,
    )
    difficulty_counts = summarize_difficulty_counts(strain_summary_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TG04",
        "recommendation_count": args.recommendation_count,
        "top_features_per_pair": args.top_features_per_pair,
        "explained_pair_count": len(recommendation_rows),
        "strain_count": len(strain_summary_rows),
        "difficulty_counts": difficulty_counts,
        "global_top_features": global_rows[:10],
        "inputs": {
            "tg01_model_summary": {
                "path": str(args.tg01_model_summary_path),
                "sha256": _sha256(args.tg01_model_summary_path),
            },
            "tg02_predictions": {
                "path": str(args.tg02_predictions_path),
                "sha256": _sha256(args.tg02_predictions_path),
            },
            "track_c_pair_table": {
                "path": str(args.track_c_pair_table_path),
                "sha256": _sha256(args.track_c_pair_table_path),
            },
            "track_d_genome_kmers": {
                "path": str(args.track_d_genome_kmer_path),
                "sha256": _sha256(args.track_d_genome_kmer_path),
            },
            "track_d_distance": {
                "path": str(args.track_d_distance_path),
                "sha256": _sha256(args.track_d_distance_path),
            },
            "track_e_rbp_receptor_compatibility": {
                "path": str(args.track_e_rbp_compatibility_path),
                "sha256": _sha256(args.track_e_rbp_compatibility_path),
            },
            "track_e_isolation_host_distance": {
                "path": str(args.track_e_isolation_distance_path),
                "sha256": _sha256(args.track_e_isolation_distance_path),
            },
        },
    }

    write_csv(
        args.output_dir / "tg04_recommendation_pair_explanations.csv",
        fieldnames=list(recommendation_rows[0].keys()),
        rows=recommendation_rows,
    )
    write_csv(
        args.output_dir / "tg04_global_feature_importance.csv",
        fieldnames=list(global_rows[0].keys()),
        rows=global_rows,
    )
    write_csv(
        args.output_dir / "tg04_per_strain_difficulty_summary.csv",
        fieldnames=list(strain_summary_rows[0].keys()),
        rows=strain_summary_rows,
    )
    write_json(args.output_dir / "tg04_shap_summary.json", summary)

    logger.info("TG04 completed.")
    logger.info("- Explained recommendation pairs: %d", len(recommendation_rows))
    logger.info("- Global features ranked: %d", len(global_rows))
    logger.info("- Per-strain difficulty rows: %d", len(strain_summary_rows))
    logger.info("- Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
