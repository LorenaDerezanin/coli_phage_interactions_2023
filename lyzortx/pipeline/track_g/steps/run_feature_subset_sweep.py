#!/usr/bin/env python3
"""TG05: Sweep 2-block and 3-block feature subsets with TG01-locked LightGBM hyperparameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
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
from lyzortx.pipeline.track_g.steps.run_feature_block_ablation_suite import partition_track_c_columns
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    IDENTIFIER_COLUMNS,
    FeatureSpace,
    build_feature_space,
    build_top3_ranking_rows,
    compute_binary_metrics,
    compute_top3_hit_rate,
    ensure_prerequisite_outputs,
    fit_final_estimator,
    make_lightgbm_estimator,
    merge_expanded_feature_rows,
    prepare_fold_datasets,
    project_rows_to_fields,
    score_rows_with_cv_predictions,
)
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier

LABEL_DERIVED_COLUMNS: Tuple[str, ...] = (
    "host_n_infections",
    "receptor_variant_training_positive_count",
)
BLOCK_ORDER: Tuple[str, ...] = ("defense", "omp", "phage_genomic", "pairwise")
BLOCK_DISPLAY_NAMES: Mapping[str, str] = {
    "defense": "defense",
    "omp": "OMP",
    "phage_genomic": "phage-genomic",
    "pairwise": "pairwise",
}


@dataclass(frozen=True)
class FeatureBlock:
    block_id: str
    display_name: str
    categorical_columns: Tuple[str, ...]
    numeric_columns: Tuple[str, ...]


@dataclass(frozen=True)
class SweepArm:
    arm_id: str
    display_name: str
    subset_blocks: Tuple[str, ...]
    evaluation_mode: str
    categorical_columns: Tuple[str, ...]
    numeric_columns: Tuple[str, ...]
    excluded_columns: Tuple[str, ...] = ()


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
        "--track-e-defense-evasion-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/defense_evasion_proxy_feature_block/"
            "defense_evasion_proxy_features_v1.csv"
        ),
        help="Input Track E defense-evasion proxy feature CSV.",
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
        "--tg01-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json"),
        help="TG01 model summary JSON used to lock the LightGBM hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep"),
        help="Directory for generated TG05 artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for fixed-parameter LightGBM refits.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume prerequisite Track C/D/E and TG01 outputs already exist.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(values: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def load_tg01_lightgbm_lock(summary_path: Path) -> Dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    best_params = summary["lightgbm"]["best_params"]
    holdout_metrics = summary["lightgbm"]["holdout_binary_metrics"]
    holdout_top3 = summary["lightgbm"]["holdout_top3_metrics"]
    feature_space = summary["feature_space"]
    return {
        "best_params": dict(best_params),
        "holdout_binary_metrics": dict(holdout_metrics),
        "holdout_top3_metrics": dict(holdout_top3),
        "feature_space": dict(feature_space),
    }


def build_feature_blocks(feature_space: FeatureSpace) -> Dict[str, FeatureBlock]:
    partitions = partition_track_c_columns(feature_space.track_c_additional_columns)
    track_c_categorical = set(feature_space.categorical_columns) - set(V0_CATEGORICAL_FEATURE_COLUMNS)
    omp_categorical = tuple(column for column in partitions["host_genomic_remainder"] if column in track_c_categorical)
    omp_numeric = tuple(column for column in partitions["host_genomic_remainder"] if column not in set(omp_categorical))
    return {
        "defense": FeatureBlock(
            block_id="defense",
            display_name=BLOCK_DISPLAY_NAMES["defense"],
            categorical_columns=(),
            numeric_columns=partitions["defense_subtypes"],
        ),
        "omp": FeatureBlock(
            block_id="omp",
            display_name=BLOCK_DISPLAY_NAMES["omp"],
            categorical_columns=omp_categorical,
            numeric_columns=omp_numeric,
        ),
        "phage_genomic": FeatureBlock(
            block_id="phage_genomic",
            display_name=BLOCK_DISPLAY_NAMES["phage_genomic"],
            categorical_columns=(),
            numeric_columns=feature_space.track_d_columns,
        ),
        "pairwise": FeatureBlock(
            block_id="pairwise",
            display_name=BLOCK_DISPLAY_NAMES["pairwise"],
            categorical_columns=(),
            numeric_columns=feature_space.track_e_columns,
        ),
    }


def build_subset_sweep_arms(feature_space: FeatureSpace) -> List[SweepArm]:
    blocks = build_feature_blocks(feature_space)
    arms: List[SweepArm] = []
    for subset_size in (2, 3):
        for subset in combinations(BLOCK_ORDER, subset_size):
            categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS)
            numeric_columns = list(V0_NUMERIC_FEATURE_COLUMNS)
            for block_id in subset:
                block = blocks[block_id]
                categorical_columns.extend(block.categorical_columns)
                numeric_columns.extend(block.numeric_columns)
            arms.append(
                SweepArm(
                    arm_id="subset_" + "__".join(subset),
                    display_name=" + ".join(BLOCK_DISPLAY_NAMES[block_id] for block_id in subset),
                    subset_blocks=subset,
                    evaluation_mode="panel_evaluation",
                    categorical_columns=_deduplicate(categorical_columns),
                    numeric_columns=_deduplicate(numeric_columns),
                )
            )
    return arms


def build_all_features_reference_arm(feature_space: FeatureSpace) -> SweepArm:
    return SweepArm(
        arm_id="tg01_all_features_reference",
        display_name="TG01 all-features reference",
        subset_blocks=BLOCK_ORDER,
        evaluation_mode="panel_evaluation",
        categorical_columns=feature_space.categorical_columns,
        numeric_columns=feature_space.numeric_columns,
    )


def build_deployment_realistic_arm(winning_arm: SweepArm) -> SweepArm:
    label_derived = tuple(
        column
        for column in LABEL_DERIVED_COLUMNS
        if column in set(winning_arm.categorical_columns + winning_arm.numeric_columns)
    )
    return SweepArm(
        arm_id=f"{winning_arm.arm_id}__deployment_realistic",
        display_name=f"{winning_arm.display_name} (deployment-realistic)",
        subset_blocks=winning_arm.subset_blocks,
        evaluation_mode="deployment_realistic",
        categorical_columns=tuple(
            column for column in winning_arm.categorical_columns if column not in set(label_derived)
        ),
        numeric_columns=tuple(column for column in winning_arm.numeric_columns if column not in set(label_derived)),
        excluded_columns=label_derived,
    )


def select_winning_subset(
    panel_rows: Sequence[Mapping[str, object]],
    *,
    all_features_auc: float,
) -> Dict[str, object]:
    eligible = [
        dict(row)
        for row in panel_rows
        if row["arm_id"] != "tg01_all_features_reference" and float(row["holdout_roc_auc"]) >= all_features_auc
    ]
    candidates = (
        eligible if eligible else [dict(row) for row in panel_rows if row["arm_id"] != "tg01_all_features_reference"]
    )
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["holdout_top3_hit_rate_all_strains"]),
            float(row["holdout_roc_auc"]),
            -float(row["holdout_brier_score"]),
            str(row["arm_id"]),
        ),
        reverse=True,
    )
    winner = dict(ranked[0])
    winner["auc_non_degrading_vs_tg01_all_features"] = float(winner["holdout_roc_auc"]) >= all_features_auc
    return winner


def ensure_tg01_summary(args: argparse.Namespace) -> None:
    if args.tg01_summary_path.exists():
        return
    train_args = [
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
        "--track-e-defense-evasion-path",
        str(args.track_e_defense_evasion_path),
        "--track-e-isolation-distance-path",
        str(args.track_e_isolation_distance_path),
        "--output-dir",
        str(args.tg01_summary_path.parent),
        "--random-state",
        str(args.random_state),
    ]
    if args.skip_prerequisites:
        train_args.append("--skip-prerequisites")
    train_v1_binary_classifier.main(train_args)


def summarize_arm(
    arm: SweepArm,
    holdout_binary_metrics: Mapping[str, Optional[float]],
    holdout_top3_metrics: Mapping[str, object],
    *,
    tg01_all_features_binary_metrics: Mapping[str, Optional[float]],
    tg01_all_features_top3_metrics: Mapping[str, object],
) -> Dict[str, object]:
    top3 = holdout_top3_metrics["top3_hit_rate_all_strains"]
    tg01_top3 = tg01_all_features_top3_metrics["top3_hit_rate_all_strains"]
    auc = holdout_binary_metrics["roc_auc"]
    tg01_auc = tg01_all_features_binary_metrics["roc_auc"]
    brier = holdout_binary_metrics["brier_score"]
    tg01_brier = tg01_all_features_binary_metrics["brier_score"]
    return {
        "arm_id": arm.arm_id,
        "arm_label": arm.display_name,
        "evaluation_mode": arm.evaluation_mode,
        "subset_blocks": list(arm.subset_blocks),
        "categorical_feature_count": len(arm.categorical_columns),
        "numeric_feature_count": len(arm.numeric_columns),
        "excluded_columns": list(arm.excluded_columns),
        "holdout_roc_auc": auc,
        "holdout_brier_score": brier,
        "holdout_top3_hit_rate_all_strains": top3,
        "holdout_top3_hit_rate_susceptible_only": holdout_top3_metrics["top3_hit_rate_susceptible_only"],
        "auc_delta_vs_tg01_all_features": safe_round(auc - tg01_auc)
        if auc is not None and tg01_auc is not None
        else None,
        "brier_improvement_vs_tg01_all_features": (
            safe_round(tg01_brier - brier) if brier is not None and tg01_brier is not None else None
        ),
        "top3_hit_rate_all_strains_delta_vs_tg01_all_features": (
            safe_round(top3 - tg01_top3) if top3 is not None and tg01_top3 is not None else None
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)
    ensure_tg01_summary(args)

    tg01_lock = load_tg01_lightgbm_lock(args.tg01_summary_path)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_defense_rows = read_csv_rows(args.track_e_defense_evasion_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)

    track_d_feature_columns = _deduplicate(
        [column for column in track_d_genome_rows[0].keys() if column != "phage"]
        + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
    )
    track_e_feature_columns = _deduplicate(
        [column for column in track_e_rbp_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
        + [column for column in track_e_defense_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
        + [column for column in track_e_isolation_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
    )
    feature_space = build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )
    merged_rows = merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_defense_rows, track_e_isolation_rows),
    )
    locked_params = tg01_lock["best_params"]
    lightgbm_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )

    panel_arms = build_subset_sweep_arms(feature_space)
    reference_arm = build_all_features_reference_arm(feature_space)
    arms_to_run = panel_arms + [reference_arm]

    metrics_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    summary_arms: Dict[str, Dict[str, object]] = {}

    for arm in arms_to_run:
        arm_feature_space = FeatureSpace(
            categorical_columns=arm.categorical_columns,
            numeric_columns=arm.numeric_columns,
            track_c_additional_columns=feature_space.track_c_additional_columns,
            track_d_columns=feature_space.track_d_columns,
            track_e_columns=feature_space.track_e_columns,
        )
        fold_datasets = prepare_fold_datasets(merged_rows, arm_feature_space)
        cv_prediction_rows = score_rows_with_cv_predictions(
            fold_datasets,
            estimator_factory=lightgbm_factory,
            best_params=locked_params,
            probability_column="predicted_probability",
        )
        _, _, _, holdout_rows, holdout_probabilities = fit_final_estimator(
            merged_rows,
            arm_feature_space,
            estimator_factory=lightgbm_factory,
            params=locked_params,
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
        summarized = summarize_arm(
            arm,
            holdout_binary_metrics,
            holdout_top3_metrics,
            tg01_all_features_binary_metrics=tg01_lock["holdout_binary_metrics"],
            tg01_all_features_top3_metrics=tg01_lock["holdout_top3_metrics"],
        )
        metrics_rows.append(summarized)
        summary_arms[arm.arm_id] = summarized

        for row in cv_prediction_rows + holdout_prediction_rows:
            prediction_rows.append(
                {
                    "arm_id": arm.arm_id,
                    "arm_label": arm.display_name,
                    "evaluation_mode": arm.evaluation_mode,
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

    winner = select_winning_subset(
        [row for row in metrics_rows if row["evaluation_mode"] == "panel_evaluation"],
        all_features_auc=float(tg01_lock["holdout_binary_metrics"]["roc_auc"]),
    )
    winning_arm = next(arm for arm in panel_arms if arm.arm_id == winner["arm_id"])
    deployment_arm = build_deployment_realistic_arm(winning_arm)
    deployment_feature_space = FeatureSpace(
        categorical_columns=deployment_arm.categorical_columns,
        numeric_columns=deployment_arm.numeric_columns,
        track_c_additional_columns=feature_space.track_c_additional_columns,
        track_d_columns=feature_space.track_d_columns,
        track_e_columns=feature_space.track_e_columns,
    )
    deployment_fold_datasets = prepare_fold_datasets(merged_rows, deployment_feature_space)
    deployment_cv_prediction_rows = score_rows_with_cv_predictions(
        deployment_fold_datasets,
        estimator_factory=lightgbm_factory,
        best_params=locked_params,
        probability_column="predicted_probability",
    )
    _, _, _, deployment_holdout_rows, deployment_holdout_probabilities = fit_final_estimator(
        merged_rows,
        deployment_feature_space,
        estimator_factory=lightgbm_factory,
        params=locked_params,
    )
    deployment_scored_rows: List[Dict[str, object]] = []
    for row, probability in zip(deployment_holdout_rows, deployment_holdout_probabilities):
        scored = dict(row)
        scored["predicted_probability"] = probability
        scored["prediction_context"] = "holdout_final"
        deployment_scored_rows.append(scored)
    deployment_holdout_y = [int(str(row["label_hard_any_lysis"])) for row in deployment_scored_rows]
    deployment_binary_metrics = compute_binary_metrics(
        deployment_holdout_y,
        [float(row["predicted_probability"]) for row in deployment_scored_rows],
    )
    deployment_top3_metrics = compute_top3_hit_rate(deployment_scored_rows, probability_key="predicted_probability")
    deployment_summary = summarize_arm(
        deployment_arm,
        deployment_binary_metrics,
        deployment_top3_metrics,
        tg01_all_features_binary_metrics=tg01_lock["holdout_binary_metrics"],
        tg01_all_features_top3_metrics=tg01_lock["holdout_top3_metrics"],
    )
    metrics_rows.append(deployment_summary)
    summary_arms[deployment_arm.arm_id] = deployment_summary
    for row in deployment_cv_prediction_rows + deployment_scored_rows:
        prediction_rows.append(
            {
                "arm_id": deployment_arm.arm_id,
                "arm_label": deployment_arm.display_name,
                "evaluation_mode": deployment_arm.evaluation_mode,
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
            deployment_scored_rows,
            probability_key="predicted_probability",
            model_label=deployment_arm.arm_id,
        )
    )

    for row in metrics_rows:
        row["is_winning_panel_subset"] = (
            row["arm_id"] == winner["arm_id"] and row["evaluation_mode"] == "panel_evaluation"
        )

    winning_panel_summary = summary_arms[winner["arm_id"]]
    final_feature_lock = {
        "task_id": "TG05",
        "locked_v1_feature_configuration": {
            "selection_policy": (
                "Among 2-block and 3-block panel-evaluation subsets, keep arms with holdout ROC-AUC >= TG01 "
                "all-features holdout ROC-AUC, then choose the highest holdout top-3 hit rate. Ties break on "
                "higher ROC-AUC, then lower Brier score, then arm_id."
            ),
            "winner_arm_id": winner["arm_id"],
            "winner_label": winner["arm_label"],
            "winner_subset_blocks": list(winning_arm.subset_blocks),
            "panel_evaluation_metrics": winning_panel_summary,
            "deployment_realistic_metrics": deployment_summary,
            "deployment_realistic_excluded_columns": list(deployment_arm.excluded_columns),
            "tg01_all_features_reference_metrics": {
                "holdout_roc_auc": tg01_lock["holdout_binary_metrics"]["roc_auc"],
                "holdout_brier_score": tg01_lock["holdout_binary_metrics"]["brier_score"],
                "holdout_top3_hit_rate_all_strains": tg01_lock["holdout_top3_metrics"]["top3_hit_rate_all_strains"],
                "holdout_top3_hit_rate_susceptible_only": tg01_lock["holdout_top3_metrics"][
                    "top3_hit_rate_susceptible_only"
                ],
            },
            "label_derived_columns": list(LABEL_DERIVED_COLUMNS),
        },
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TG05",
        "locked_lightgbm_hyperparameters": locked_params,
        "tg01_lightgbm_reference": tg01_lock,
        "feature_block_order": list(BLOCK_ORDER),
        "feature_blocks": {
            block_id: {
                "display_name": block.display_name,
                "categorical_columns": list(block.categorical_columns),
                "numeric_columns": list(block.numeric_columns),
            }
            for block_id, block in build_feature_blocks(feature_space).items()
        },
        "arms": summary_arms,
        "winning_panel_subset_arm_id": winner["arm_id"],
        "final_feature_lock": final_feature_lock["locked_v1_feature_configuration"],
        "inputs": {
            "tg01_summary": {"path": str(args.tg01_summary_path), "sha256": _sha256(args.tg01_summary_path)},
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
            "track_e_defense_evasion": {
                "path": str(args.track_e_defense_evasion_path),
                "sha256": _sha256(args.track_e_defense_evasion_path),
            },
            "track_e_isolation_host_distance": {
                "path": str(args.track_e_isolation_distance_path),
                "sha256": _sha256(args.track_e_isolation_distance_path),
            },
        },
    }

    write_json(args.output_dir / "tg05_feature_subset_summary.json", summary)
    write_json(args.output_dir / "tg05_locked_v1_feature_config.json", final_feature_lock)
    write_csv(
        args.output_dir / "tg05_feature_subset_metrics.csv",
        list(metrics_rows[0].keys()),
        metrics_rows,
    )
    write_csv(
        args.output_dir / "tg05_feature_subset_pair_predictions.csv",
        [
            "arm_id",
            "arm_label",
            "evaluation_mode",
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
        args.output_dir / "tg05_feature_subset_holdout_top3_rankings.csv",
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
    write_csv(
        args.output_dir / "tg05_locked_v1_feature_columns.csv",
        ["evaluation_mode", "feature_role", "column_name"],
        project_rows_to_fields(
            [
                {
                    "evaluation_mode": "panel_evaluation",
                    "feature_role": "categorical",
                    "column_name": column,
                }
                for column in winning_arm.categorical_columns
            ]
            + [
                {
                    "evaluation_mode": "panel_evaluation",
                    "feature_role": "numeric",
                    "column_name": column,
                }
                for column in winning_arm.numeric_columns
            ]
            + [
                {
                    "evaluation_mode": "deployment_realistic",
                    "feature_role": "categorical",
                    "column_name": column,
                }
                for column in deployment_arm.categorical_columns
            ]
            + [
                {
                    "evaluation_mode": "deployment_realistic",
                    "feature_role": "numeric",
                    "column_name": column,
                }
                for column in deployment_arm.numeric_columns
            ],
            ("evaluation_mode", "feature_role", "column_name"),
        ),
    )

    print("TG05 completed.")
    print(f"- Locked TG01 LightGBM params: {json.dumps(locked_params, sort_keys=True)}")
    print(
        f"- Winning panel subset: {winner['arm_label']} | holdout ROC-AUC {winner['holdout_roc_auc']} | "
        f"top-3 {winner['holdout_top3_hit_rate_all_strains']} | Brier {winner['holdout_brier_score']}"
    )
    print(
        f"- Deployment-realistic winner: holdout ROC-AUC {deployment_summary['holdout_roc_auc']} | "
        f"top-3 {deployment_summary['holdout_top3_hit_rate_all_strains']} | "
        f"Brier {deployment_summary['holdout_brier_score']}"
    )
    print(f"- Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
