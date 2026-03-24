#!/usr/bin/env python3
"""TK02: Measure the cumulative lift from adding BASEL supervision."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    IDENTIFIER_COLUMNS,
    FeatureSpace,
    build_feature_space,
    build_top3_ranking_rows,
    compute_binary_metrics,
    compute_top3_hit_rate,
    fit_final_estimator,
    make_lightgbm_estimator,
    merge_expanded_feature_rows,
)
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import (
    arm_name_for_source_systems,
    build_locked_feature_space,
    build_training_rows,
    classify_lift,
    load_locked_v1_feature_config,
    load_previous_best_source_systems,
    load_source_training_rows,
    load_tg01_best_params,
    sha256,
    source_systems_label,
    write_output_tables,
)

logger = logging.getLogger(__name__)

LOCKED_V1_FEATURE_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
TK01_MANIFEST_PATH = Path("lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement/tk01_vhrdb_lift_manifest.json")
TI08_TRAINING_COHORT_PATH = Path(
    "lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_k/tk02_basel_lift_measurement")
CURRENT_SOURCE_SYSTEM = "basel"
TRAIN_SPLIT = "train_non_holdout"
NEGLIGIBLE_DELTA_TOLERANCE = 0.001


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
    )
    parser.add_argument(
        "--track-c-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
    )
    parser.add_argument(
        "--track-d-genome-kmer-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv"),
    )
    parser.add_argument(
        "--track-d-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_d/phage_distance_embedding/phage_distance_embedding_features.csv"
        ),
    )
    parser.add_argument(
        "--track-e-rbp-compatibility-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/"
            "rbp_receptor_compatibility_features_v1.csv"
        ),
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/isolation_host_distance_feature_block/"
            "isolation_host_distance_features_v1.csv"
        ),
    )
    parser.add_argument("--v1-feature-config-path", type=Path, default=LOCKED_V1_FEATURE_CONFIG_PATH)
    parser.add_argument("--tg01-summary-path", type=Path, default=TG01_SUMMARY_PATH)
    parser.add_argument("--tk01-manifest-path", type=Path, default=TK01_MANIFEST_PATH)
    parser.add_argument("--ti08-training-cohort-path", type=Path, default=TI08_TRAINING_COHORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return safe_round(value - baseline)


def _trainable_rows(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [row for row in rows if row["split_holdout"] == TRAIN_SPLIT and str(row["is_hard_trainable"]) == "1"]


def _measure_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_space: FeatureSpace,
    estimator_factory,
    params: Mapping[str, object],
) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    _, _, _, eval_rows, probabilities = fit_final_estimator(
        rows,
        feature_space,
        estimator_factory=estimator_factory,
        params=params,
    )
    scored_rows: List[Dict[str, object]] = []
    for row, probability in zip(eval_rows, probabilities):
        scored = dict(row)
        scored["probability"] = probability
        scored_rows.append(scored)
    holdout_metrics = compute_binary_metrics(
        [int(str(row["label_hard_any_lysis"])) for row in scored_rows],
        [float(row["probability"]) for row in scored_rows],
    )
    top3 = compute_top3_hit_rate(scored_rows, probability_key="probability")
    return scored_rows, holdout_metrics, top3


def load_ti08_training_cohort_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing TI08 training cohort artifact: {path}")
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"TI08 training cohort is empty: {path}")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("TK02 starting: measure BASEL lift against the best-so-far Track K cohort")

    ensure_directory(args.output_dir)

    if not args.skip_prerequisites:
        from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import ensure_prerequisite_outputs

        ensure_prerequisite_outputs(args)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)

    locked_config = load_locked_v1_feature_config(args.v1_feature_config_path)
    locked_subset_blocks = tuple(str(block) for block in locked_config["winner_subset_blocks"])
    tg01_best_params = load_tg01_best_params(args.tg01_summary_path)
    baseline_params_source = "tg01_summary_lock"

    track_d_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_d_genome_rows[0].keys() if column != "phage"]
            + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
        )
    )
    track_e_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_e_rbp_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
            + [column for column in track_e_isolation_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
        )
    )
    full_feature_space = build_feature_space(
        st02_rows, track_c_pair_rows, track_d_feature_columns, track_e_feature_columns
    )
    locked_feature_space = build_locked_feature_space(full_feature_space, locked_subset_blocks)

    merged_rows = merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_isolation_rows),
    )
    cohort_rows = load_ti08_training_cohort_rows(args.ti08_training_cohort_path)

    source_rows_by_system: Dict[str, List[Dict[str, object]]] = {}
    current_source_rows, current_source_counts = load_source_training_rows(
        merged_rows,
        cohort_rows,
        CURRENT_SOURCE_SYSTEM,
    )
    if int(current_source_counts.get("cohort_rows", 0)) == 0:
        raise ValueError(f"TI08 cohort contains no BASEL rows: {args.ti08_training_cohort_path}")
    if int(current_source_counts.get("joined_rows", 0)) == 0:
        raise ValueError(
            "TI08 cohort contains no BASEL rows that join into the locked ST03 train split: "
            f"{args.ti08_training_cohort_path}"
        )
    source_rows_by_system[CURRENT_SOURCE_SYSTEM] = current_source_rows
    previous_best_source_systems = load_previous_best_source_systems(args.tk01_manifest_path)
    for source_system in previous_best_source_systems:
        if source_system not in source_rows_by_system:
            source_rows_by_system[source_system], _ = load_source_training_rows(merged_rows, cohort_rows, source_system)

    base_source_systems = ["internal", *previous_best_source_systems]
    augmented_source_systems = [*base_source_systems, CURRENT_SOURCE_SYSTEM]

    internal_training_rows = list(merged_rows)
    base_training_rows = build_training_rows(internal_training_rows, source_rows_by_system, base_source_systems)
    augmented_training_rows = build_training_rows(
        internal_training_rows,
        source_rows_by_system,
        augmented_source_systems,
    )

    estimator_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )

    baseline_rows, baseline_holdout_metrics, baseline_top3 = _measure_metrics(
        base_training_rows,
        feature_space=locked_feature_space,
        estimator_factory=estimator_factory,
        params=tg01_best_params,
    )
    augmented_rows, augmented_holdout_metrics, augmented_top3 = _measure_metrics(
        augmented_training_rows,
        feature_space=locked_feature_space,
        estimator_factory=estimator_factory,
        params=tg01_best_params,
    )

    metric_deltas = {
        "delta_roc_auc": _delta(augmented_holdout_metrics["roc_auc"], baseline_holdout_metrics["roc_auc"]),
        "delta_top3_hit_rate_all_strains": _delta(
            float(augmented_top3["top3_hit_rate_all_strains"]),
            float(baseline_top3["top3_hit_rate_all_strains"]),
        ),
        "delta_brier_score": _delta(augmented_holdout_metrics["brier_score"], baseline_holdout_metrics["brier_score"]),
    }
    lift_assessment = classify_lift(
        delta_roc_auc=metric_deltas["delta_roc_auc"],
        delta_top3=metric_deltas["delta_top3_hit_rate_all_strains"],
        delta_brier=metric_deltas["delta_brier_score"],
        tolerance=NEGLIGIBLE_DELTA_TOLERANCE,
    )

    summary_rows = [
        {
            "arm": arm_name_for_source_systems(base_source_systems),
            "source_systems": source_systems_label(base_source_systems),
            "training_row_count": len(_trainable_rows(base_training_rows)),
            "basel_row_count": 0,
            "holdout_roc_auc": baseline_holdout_metrics["roc_auc"],
            "holdout_top3_hit_rate_all_strains": baseline_top3["top3_hit_rate_all_strains"],
            "holdout_brier_score": baseline_holdout_metrics["brier_score"],
            "delta_roc_auc_vs_previous_best": 0.0,
            "delta_top3_vs_previous_best": 0.0,
            "delta_brier_vs_previous_best": 0.0,
        },
        {
            "arm": arm_name_for_source_systems(augmented_source_systems),
            "source_systems": source_systems_label(augmented_source_systems),
            "training_row_count": len(_trainable_rows(augmented_training_rows)),
            "basel_row_count": len(current_source_rows),
            "holdout_roc_auc": augmented_holdout_metrics["roc_auc"],
            "holdout_top3_hit_rate_all_strains": augmented_top3["top3_hit_rate_all_strains"],
            "holdout_brier_score": augmented_holdout_metrics["brier_score"],
            "delta_roc_auc_vs_previous_best": metric_deltas["delta_roc_auc"],
            "delta_top3_vs_previous_best": metric_deltas["delta_top3_hit_rate_all_strains"],
            "delta_brier_vs_previous_best": metric_deltas["delta_brier_score"],
        },
    ]

    summary_filename = "tk02_basel_lift_summary.csv"
    rankings_filename = "tk02_basel_holdout_top3_rankings.csv"
    manifest_path = args.output_dir / "tk02_basel_lift_manifest.json"

    ranking_rows = [
        *build_top3_ranking_rows(
            baseline_rows,
            probability_key="probability",
            model_label=arm_name_for_source_systems(base_source_systems),
        ),
        *build_top3_ranking_rows(
            augmented_rows,
            probability_key="probability",
            model_label=arm_name_for_source_systems(augmented_source_systems),
        ),
    ]

    write_output_tables(
        output_dir=args.output_dir,
        summary_rows=summary_rows,
        summary_fieldnames=[
            "arm",
            "source_systems",
            "training_row_count",
            "basel_row_count",
            "holdout_roc_auc",
            "holdout_top3_hit_rate_all_strains",
            "holdout_brier_score",
            "delta_roc_auc_vs_previous_best",
            "delta_top3_vs_previous_best",
            "delta_brier_vs_previous_best",
        ],
        ranking_rows=ranking_rows,
        ranking_fieldnames=[
            "model_label",
            "bacteria",
            "phage",
            "pair_id",
            "rank",
            "predicted_probability",
            "label_hard_any_lysis",
        ],
        manifest_path=manifest_path,
        manifest_payload={
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_basel_lift_report",
            "source_system_added": CURRENT_SOURCE_SYSTEM,
            "locked_feature_config_path": str(args.v1_feature_config_path),
            "locked_feature_config_sha256": sha256(args.v1_feature_config_path),
            "tg01_best_params_source": baseline_params_source,
            "tg01_best_params": tg01_best_params,
            "input_paths": {
                "st02_pair_table": str(args.st02_pair_table_path),
                "st03_split_assignments": str(args.st03_split_assignments_path),
                "track_c_pair_table": str(args.track_c_pair_table_path),
                "track_d_genome_kmers": str(args.track_d_genome_kmer_path),
                "track_d_distance": str(args.track_d_distance_path),
                "track_e_rbp_receptor_compatibility": str(args.track_e_rbp_compatibility_path),
                "track_e_isolation_host_distance": str(args.track_e_isolation_distance_path),
                "ti08_training_cohort_rows": str(args.ti08_training_cohort_path),
                "tk01_manifest": str(args.tk01_manifest_path),
            },
            "input_hashes_sha256": {
                "st02_pair_table": sha256(args.st02_pair_table_path),
                "st03_split_assignments": sha256(args.st03_split_assignments_path),
                "track_c_pair_table": sha256(args.track_c_pair_table_path),
                "track_d_genome_kmers": sha256(args.track_d_genome_kmer_path),
                "track_d_distance": sha256(args.track_d_distance_path),
                "track_e_rbp_receptor_compatibility": sha256(args.track_e_rbp_compatibility_path),
                "track_e_isolation_host_distance": sha256(args.track_e_isolation_distance_path),
                **(
                    {"ti08_training_cohort_rows": sha256(args.ti08_training_cohort_path)}
                    if args.ti08_training_cohort_path.exists()
                    else {}
                ),
                **({"tk01_manifest": sha256(args.tk01_manifest_path)} if args.tk01_manifest_path.exists() else {}),
            },
            "previous_best_source_systems": previous_best_source_systems,
            "best_source_systems": (
                [*previous_best_source_systems, CURRENT_SOURCE_SYSTEM]
                if lift_assessment == "adds"
                else previous_best_source_systems
            ),
            "base_source_systems": base_source_systems,
            "augmented_source_systems": augmented_source_systems,
            "source_rows_by_system": {
                source_system: len(rows) for source_system, rows in source_rows_by_system.items()
            },
            "current_source_counts": current_source_counts,
            "baseline_metrics": {
                "roc_auc": baseline_holdout_metrics["roc_auc"],
                "top3_hit_rate_all_strains": baseline_top3["top3_hit_rate_all_strains"],
                "brier_score": baseline_holdout_metrics["brier_score"],
            },
            "augmented_metrics": {
                "roc_auc": augmented_holdout_metrics["roc_auc"],
                "top3_hit_rate_all_strains": augmented_top3["top3_hit_rate_all_strains"],
                "brier_score": augmented_holdout_metrics["brier_score"],
            },
            "metric_deltas": metric_deltas,
            "lift_assessment": lift_assessment,
            "output_paths": {
                "summary": str(args.output_dir / summary_filename),
                "holdout_rankings": str(args.output_dir / rankings_filename),
            },
        },
        summary_filename=summary_filename,
        rankings_filename=rankings_filename,
    )

    logger.info("TK02 completed.")
    logger.info("- Previous best arm: %s", arm_name_for_source_systems(base_source_systems))
    logger.info("- Augmented arm: %s", arm_name_for_source_systems(augmented_source_systems))
    logger.info("- Delta ROC-AUC: %s", metric_deltas["delta_roc_auc"])
    logger.info("- Delta top-3: %s", metric_deltas["delta_top3_hit_rate_all_strains"])
    logger.info("- Delta Brier: %s", metric_deltas["delta_brier_score"])
    logger.info("- BASEL rows joined: %s", len(current_source_rows))
    logger.info("- Lift assessment: %s", lift_assessment)
