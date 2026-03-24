#!/usr/bin/env python3
"""TG03: Run holdout-locked feature-block ablations against the v0 reference feature set."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    CATEGORICAL_FEATURE_COLUMNS as V0_CATEGORICAL_FEATURE_COLUMNS,
)
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    NUMERIC_FEATURE_COLUMNS as V0_NUMERIC_FEATURE_COLUMNS,
)
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import DEFENSE_DERIVED_COLUMNS
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    IDENTIFIER_COLUMNS,
    LIGHTGBM_PARAMETER_GRID,
    FeatureSpace,
    build_feature_space,
    build_top3_ranking_rows,
    compute_binary_metrics,
    compute_top3_hit_rate,
    ensure_prerequisite_outputs,
    evaluate_candidate_grid,
    fit_final_estimator,
    flatten_candidate_rows,
    make_lightgbm_estimator,
    merge_expanded_feature_rows,
    prepare_fold_datasets,
    score_rows_with_cv_predictions,
    select_best_candidate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationArm:
    arm_id: str
    display_name: str
    categorical_columns: Tuple[str, ...]
    numeric_columns: Tuple[str, ...]
    included_blocks: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path(
            "lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/"
            "rbp_receptor_compatibility_features_v1.csv"
        ),
        help="Input Track E RBP-receptor compatibility feature CSV.",
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/isolation_host_distance_feature_block/"
            "isolation_host_distance_features_v1.csv"
        ),
        help="Input Track E isolation-host distance feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg03_feature_block_ablation_suite"),
        help="Directory for generated TG03 artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for tuned LightGBM ablation models.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume prerequisite Track C/D/E outputs already exist instead of generating missing artifacts.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_track_c_columns(track_c_columns: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    defense_columns: List[str] = []
    host_genomic_remainder: List[str] = []
    for column in track_c_columns:
        if column.startswith("host_defense_subtype_") or column in DEFENSE_DERIVED_COLUMNS:
            defense_columns.append(column)
            continue
        host_genomic_remainder.append(column)
    return {
        "defense_subtypes": tuple(defense_columns),
        "host_genomic_remainder": tuple(host_genomic_remainder),
    }


def build_ablation_arms(feature_space: FeatureSpace) -> List[AblationArm]:
    track_c_blocks = partition_track_c_columns(feature_space.track_c_additional_columns)
    v0_categorical = tuple(V0_CATEGORICAL_FEATURE_COLUMNS)
    v0_numeric = tuple(V0_NUMERIC_FEATURE_COLUMNS)

    return [
        AblationArm(
            arm_id="v0_features_only",
            display_name="v0 features only",
            categorical_columns=v0_categorical,
            numeric_columns=v0_numeric,
            included_blocks=("v0",),
        ),
        AblationArm(
            arm_id="plus_defense_subtypes",
            display_name="+defense subtypes",
            categorical_columns=v0_categorical,
            numeric_columns=v0_numeric + track_c_blocks["defense_subtypes"],
            included_blocks=("v0", "defense_subtypes"),
        ),
        AblationArm(
            arm_id="plus_omp_receptors",
            display_name="+OMP receptors",
            categorical_columns=tuple(
                column for column in feature_space.categorical_columns if column in set(v0_categorical)
            )
            + tuple(column for column in feature_space.categorical_columns if column not in set(v0_categorical)),
            numeric_columns=v0_numeric + track_c_blocks["host_genomic_remainder"],
            included_blocks=("v0", "omp_receptors_and_remaining_host_genomic"),
        ),
        AblationArm(
            arm_id="plus_phage_genomic",
            display_name="+phage genomic",
            categorical_columns=v0_categorical,
            numeric_columns=v0_numeric + feature_space.track_d_columns,
            included_blocks=("v0", "phage_genomic"),
        ),
        AblationArm(
            arm_id="plus_pairwise_compatibility",
            display_name="+pairwise compatibility",
            categorical_columns=v0_categorical,
            numeric_columns=v0_numeric + feature_space.track_e_columns,
            included_blocks=("v0", "pairwise_compatibility"),
        ),
        AblationArm(
            arm_id="all_features",
            display_name="all features",
            categorical_columns=feature_space.categorical_columns,
            numeric_columns=feature_space.numeric_columns,
            included_blocks=(
                "v0",
                "defense_subtypes",
                "omp_receptors_and_remaining_host_genomic",
                "phage_genomic",
                "pairwise_compatibility",
            ),
        ),
    ]


def _delta_vs_reference(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None:
        return None
    return safe_round(value - reference)


def _brier_improvement(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None:
        return None
    return safe_round(reference - value)


def summarize_arm_result(
    arm: AblationArm,
    best_result: Mapping[str, object],
    holdout_binary_metrics: Mapping[str, Optional[float]],
    holdout_top3_metrics: Mapping[str, object],
    *,
    reference_binary_metrics: Mapping[str, Optional[float]],
    reference_top3_metrics: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "display_name": arm.display_name,
        "included_blocks": list(arm.included_blocks),
        "feature_counts": {
            "categorical_feature_count": len(arm.categorical_columns),
            "numeric_feature_count": len(arm.numeric_columns),
        },
        "best_params": dict(best_result["params"]),
        "cv_summary": dict(best_result["summary"]),
        "holdout_binary_metrics": dict(holdout_binary_metrics),
        "holdout_top3_metrics": dict(holdout_top3_metrics),
        "lift_vs_v0": {
            "roc_auc_delta": _delta_vs_reference(
                holdout_binary_metrics.get("roc_auc"),
                reference_binary_metrics.get("roc_auc"),
            ),
            "brier_improvement": _brier_improvement(
                holdout_binary_metrics.get("brier_score"),
                reference_binary_metrics.get("brier_score"),
            ),
            "top3_hit_rate_all_strains_delta": _delta_vs_reference(
                holdout_top3_metrics.get("top3_hit_rate_all_strains"),
                reference_top3_metrics.get("top3_hit_rate_all_strains"),
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("TG03 starting: feature-block ablation suite")
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)

    track_d_feature_columns = tuple(
        column
        for column in list(track_d_genome_rows[0].keys()) + list(track_d_distance_rows[0].keys())
        if column != "phage"
    )
    track_d_feature_columns = tuple(dict.fromkeys(track_d_feature_columns))
    track_e_feature_columns = tuple(
        column
        for column in list(track_e_rbp_rows[0].keys()) + list(track_e_isolation_rows[0].keys())
        if column not in IDENTIFIER_COLUMNS
    )
    track_e_feature_columns = tuple(dict.fromkeys(track_e_feature_columns))

    full_feature_space = build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )
    merged_rows = merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_isolation_rows),
    )
    ablation_arms = build_ablation_arms(full_feature_space)
    lightgbm_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )

    arm_results: Dict[str, Dict[str, object]] = {}
    metrics_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []

    for arm in ablation_arms:
        feature_space = FeatureSpace(
            categorical_columns=arm.categorical_columns,
            numeric_columns=arm.numeric_columns,
            track_c_additional_columns=full_feature_space.track_c_additional_columns,
            track_d_columns=full_feature_space.track_d_columns,
            track_e_columns=full_feature_space.track_e_columns,
        )
        fold_datasets = prepare_fold_datasets(merged_rows, feature_space)
        candidate_results = evaluate_candidate_grid(
            fold_datasets,
            candidate_params=LIGHTGBM_PARAMETER_GRID,
            estimator_factory=lightgbm_factory,
            model_label=arm.arm_id,
        )
        best_result = select_best_candidate(candidate_results)
        cv_prediction_rows = score_rows_with_cv_predictions(
            fold_datasets,
            estimator_factory=lightgbm_factory,
            best_params=best_result["params"],
            probability_column="predicted_probability",
        )
        _, _, _, holdout_rows, holdout_probabilities = fit_final_estimator(
            merged_rows,
            feature_space,
            estimator_factory=lightgbm_factory,
            params=best_result["params"],
        )
        holdout_prediction_rows: List[Dict[str, object]] = []
        for row, probability in zip(holdout_rows, holdout_probabilities):
            scored = dict(row)
            scored["predicted_probability"] = probability
            scored["prediction_context"] = "holdout_final"
            holdout_prediction_rows.append(scored)

        holdout_y = [int(str(row["label_hard_any_lysis"])) for row in holdout_prediction_rows]
        holdout_binary_metrics = compute_binary_metrics(
            holdout_y,
            [float(row["predicted_probability"]) for row in holdout_prediction_rows],
        )
        holdout_top3_metrics = compute_top3_hit_rate(holdout_prediction_rows, probability_key="predicted_probability")

        arm_results[arm.arm_id] = {
            "arm": arm,
            "best_result": best_result,
            "holdout_binary_metrics": holdout_binary_metrics,
            "holdout_top3_metrics": holdout_top3_metrics,
        }

        candidate_rows.extend(flatten_candidate_rows(arm.arm_id, candidate_results))
        for row in cv_prediction_rows + holdout_prediction_rows:
            prediction_rows.append(
                {
                    "arm_id": arm.arm_id,
                    "arm_label": arm.display_name,
                    "pair_id": row["pair_id"],
                    "bacteria": row["bacteria"],
                    "phage": row["phage"],
                    "split_holdout": row["split_holdout"],
                    "split_cv5_fold": row["split_cv5_fold"],
                    "label_hard_any_lysis": row["label_hard_any_lysis"],
                    "prediction_context": row["prediction_context"],
                    "predicted_probability": safe_round(float(row["predicted_probability"])),
                }
            )
        ranking_rows.extend(
            build_top3_ranking_rows(
                holdout_prediction_rows,
                probability_key="predicted_probability",
                model_label=arm.arm_id,
            )
        )

    reference = arm_results["v0_features_only"]
    summary_arms: Dict[str, Dict[str, object]] = {}
    for arm in ablation_arms:
        result = arm_results[arm.arm_id]
        summarized = summarize_arm_result(
            arm,
            result["best_result"],
            result["holdout_binary_metrics"],
            result["holdout_top3_metrics"],
            reference_binary_metrics=reference["holdout_binary_metrics"],
            reference_top3_metrics=reference["holdout_top3_metrics"],
        )
        summary_arms[arm.arm_id] = summarized
        metrics_rows.append(
            {
                "arm_id": arm.arm_id,
                "arm_label": arm.display_name,
                "categorical_feature_count": summarized["feature_counts"]["categorical_feature_count"],
                "numeric_feature_count": summarized["feature_counts"]["numeric_feature_count"],
                "holdout_roc_auc": summarized["holdout_binary_metrics"]["roc_auc"],
                "holdout_brier_score": summarized["holdout_binary_metrics"]["brier_score"],
                "holdout_top3_hit_rate_all_strains": summarized["holdout_top3_metrics"]["top3_hit_rate_all_strains"],
                "holdout_top3_hit_rate_susceptible_only": summarized["holdout_top3_metrics"][
                    "top3_hit_rate_susceptible_only"
                ],
                "cv_mean_roc_auc": summarized["cv_summary"]["mean_roc_auc"],
                "cv_mean_brier_score": summarized["cv_summary"]["mean_brier_score"],
                "cv_mean_top3_hit_rate_all_strains": summarized["cv_summary"]["mean_top3_hit_rate_all_strains"],
                "roc_auc_delta_vs_v0": summarized["lift_vs_v0"]["roc_auc_delta"],
                "brier_improvement_vs_v0": summarized["lift_vs_v0"]["brier_improvement"],
                "top3_hit_rate_all_strains_delta_vs_v0": summarized["lift_vs_v0"]["top3_hit_rate_all_strains_delta"],
                "best_params_json": json.dumps(summarized["best_params"], sort_keys=True),
            }
        )

    feature_partitions = partition_track_c_columns(full_feature_space.track_c_additional_columns)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TG03",
        "reference_arm": "v0_features_only",
        "model_family": "lightgbm",
        "candidate_count_per_arm": len(LIGHTGBM_PARAMETER_GRID),
        "ablation_protocol": {
            "holdout_split": "ST0.3 holdout_test",
            "training_subset": "split_holdout=train_non_holdout and is_hard_trainable=1",
            "reference_policy": "All arms are compared directly against the v0-only LightGBM baseline.",
            "model_family_lock": "LightGBM is held fixed across arms so lift is attributable to feature blocks.",
            "host_genomic_arm_note": (
                "The '+OMP receptors' arm includes all non-defense Track C host-genomic additions: the OMP receptor "
                "one-hots plus the remaining capsule/LPS/phylogeny columns. TG03 does not define a separate arm for "
                "those residual host-genomic features."
            ),
        },
        "feature_blocks": {
            "v0_categorical_columns": list(V0_CATEGORICAL_FEATURE_COLUMNS),
            "v0_numeric_columns": list(V0_NUMERIC_FEATURE_COLUMNS),
            "defense_subtype_columns": list(feature_partitions["defense_subtypes"]),
            "omp_and_remaining_host_genomic_columns": list(feature_partitions["host_genomic_remainder"]),
            "phage_genomic_columns": list(full_feature_space.track_d_columns),
            "pairwise_compatibility_columns": list(full_feature_space.track_e_columns),
        },
        "arms": summary_arms,
        "inputs": {
            "st02_pair_table": {"path": str(args.st02_pair_table_path), "sha256": _sha256(args.st02_pair_table_path)},
            "st03_split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
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

    write_json(args.output_dir / "tg03_ablation_summary.json", summary)
    write_csv(
        args.output_dir / "tg03_ablation_metrics.csv",
        list(metrics_rows[0].keys()),
        metrics_rows,
    )
    write_csv(
        args.output_dir / "tg03_ablation_cv_candidate_results.csv",
        [
            "model_label",
            "params_json",
            "mean_average_precision",
            "mean_roc_auc",
            "mean_brier_score",
            "mean_log_loss",
            "mean_top3_hit_rate_all_strains",
            "mean_top3_hit_rate_susceptible_only",
        ],
        candidate_rows,
    )
    write_csv(
        args.output_dir / "tg03_ablation_pair_predictions.csv",
        [
            "arm_id",
            "arm_label",
            "pair_id",
            "bacteria",
            "phage",
            "split_holdout",
            "split_cv5_fold",
            "label_hard_any_lysis",
            "prediction_context",
            "predicted_probability",
        ],
        prediction_rows,
    )
    ranking_rows.sort(key=lambda row: (str(row["model_label"]), str(row["bacteria"]), int(row["rank"])))
    write_csv(
        args.output_dir / "tg03_ablation_holdout_top3_rankings.csv",
        [
            "model_label",
            "bacteria",
            "phage",
            "pair_id",
            "rank",
            "predicted_probability",
            "label_hard_any_lysis",
        ],
        ranking_rows,
    )

    logger.info("TG03 completed.")
    for arm in ablation_arms:
        arm_summary = summary_arms[arm.arm_id]
        logger.info(
            "- %s: holdout ROC-AUC %s, top-3 %s, Brier %s",
            arm.display_name,
            arm_summary["holdout_binary_metrics"]["roc_auc"],
            arm_summary["holdout_top3_metrics"]["top3_hit_rate_all_strains"],
            arm_summary["holdout_binary_metrics"]["brier_score"],
        )
    logger.info("- Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
