#!/usr/bin/env python3
"""TK01: Measure the lift from adding VHRdb supervision to the locked v1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
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
from lyzortx.pipeline.track_g.steps.run_feature_subset_sweep import build_feature_blocks
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

logger = logging.getLogger(__name__)

LOCKED_V1_FEATURE_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
TI08_TRAINING_COHORT_PATH = Path(
    "lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement")
VHRDB_SOURCE_SYSTEM = "vhrdb"
TRAIN_SPLIT = "train_non_holdout"
TRAIN_FOLD = -1
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
    parser.add_argument("--ti08-training-cohort-path", type=Path, default=TI08_TRAINING_COHORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prerequisites", action="store_true")
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


def load_locked_v1_feature_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tg01_best_params(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing locked TG01 summary artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return dict(summary["lightgbm"]["best_params"])


def build_locked_feature_space(feature_space: FeatureSpace, locked_subset_blocks: Sequence[str]) -> FeatureSpace:
    blocks = build_feature_blocks(feature_space)
    categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS)
    numeric_columns = list(V0_NUMERIC_FEATURE_COLUMNS)
    for block_id in locked_subset_blocks:
        block = blocks[block_id]
        categorical_columns.extend(block.categorical_columns)
        numeric_columns.extend(block.numeric_columns)
    return FeatureSpace(
        categorical_columns=_deduplicate(categorical_columns),
        numeric_columns=_deduplicate(numeric_columns),
        track_c_additional_columns=tuple(feature_space.track_c_additional_columns),
        track_d_columns=tuple(feature_space.track_d_columns),
        track_e_columns=tuple(feature_space.track_e_columns),
    )


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def load_vhrdb_training_rows(
    feature_rows: Sequence[Mapping[str, object]],
    cohort_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    feature_rows_by_pair = {str(row["pair_id"]): dict(row) for row in feature_rows}
    augmented_rows: List[Dict[str, object]] = []
    counts = Counter(
        {
            "cohort_rows": 0,
            "joined_rows": 0,
            "missing_feature_rows": 0,
            "excluded_rows": 0,
            "non_training_split_rows": 0,
        }
    )

    for row in cohort_rows:
        normalized = _normalize_row(row)
        if normalized.get("source_system") != VHRDB_SOURCE_SYSTEM:
            continue
        counts["cohort_rows"] += 1
        if normalized.get("external_label_include_in_training") != "1":
            counts["excluded_rows"] += 1
            continue

        feature_row = feature_rows_by_pair.get(normalized["pair_id"])
        if feature_row is None:
            counts["missing_feature_rows"] += 1
            continue
        if feature_row.get("split_holdout") != TRAIN_SPLIT or str(feature_row.get("is_hard_trainable", "")) != "1":
            counts["non_training_split_rows"] += 1
            continue

        merged = dict(feature_row)
        merged.update(
            {
                "source_system": VHRDB_SOURCE_SYSTEM,
                "source_family": "tier_a",
                "external_label_confidence_tier": normalized.get("external_label_confidence_tier", ""),
                "external_label_confidence_score": normalized.get("external_label_confidence_score", ""),
                "external_label_training_weight": normalized.get("external_label_training_weight", ""),
                "external_label_include_in_training": normalized.get("external_label_include_in_training", ""),
                "integration_status": "external_enhancer",
                "split_holdout": TRAIN_SPLIT,
                "split_cv5_fold": TRAIN_FOLD,
                "is_hard_trainable": "1",
                "training_origin": "vhrdb",
            }
        )
        if normalized.get("label_hard_any_lysis", ""):
            merged["label_hard_any_lysis"] = normalized["label_hard_any_lysis"]
        if normalized.get("label_strict_confidence_tier", ""):
            merged["label_strict_confidence_tier"] = normalized["label_strict_confidence_tier"]
        augmented_rows.append(merged)
        counts["joined_rows"] += 1

    return augmented_rows, dict(counts)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("TK01 starting: measure VHRdb lift against the locked v1 baseline")
    ensure_directory(args.output_dir)

    if not args.skip_prerequisites:
        # Deferred: avoid importing the prerequisite runner unless we need to materialize missing inputs.
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

    track_d_feature_columns = _deduplicate(
        [column for column in track_d_genome_rows[0].keys() if column != "phage"]
        + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
    )
    track_e_feature_columns = _deduplicate(
        [column for column in track_e_rbp_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
        + [column for column in track_e_isolation_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
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
    cohort_rows = read_csv_rows(args.ti08_training_cohort_path) if args.ti08_training_cohort_path.exists() else []
    vhrdb_rows, vhrdb_counts = load_vhrdb_training_rows(merged_rows, cohort_rows)

    internal_training_rows = list(merged_rows)
    augmented_training_rows = list(merged_rows) + vhrdb_rows
    internal_trainable_rows = [
        row for row in merged_rows if row["split_holdout"] == TRAIN_SPLIT and str(row["is_hard_trainable"]) == "1"
    ]
    augmented_trainable_rows = internal_trainable_rows + vhrdb_rows

    estimator_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )

    baseline_eval = fit_final_estimator(
        internal_training_rows,
        locked_feature_space,
        estimator_factory=estimator_factory,
        params=tg01_best_params,
    )
    augmented_eval = fit_final_estimator(
        augmented_training_rows,
        locked_feature_space,
        estimator_factory=estimator_factory,
        params=tg01_best_params,
    )

    _, _, _, baseline_eval_rows, baseline_holdout_probabilities = baseline_eval
    _, _, _, augmented_eval_rows, augmented_holdout_probabilities = augmented_eval

    baseline_holdout_rows = []
    for row, probability in zip(baseline_eval_rows, baseline_holdout_probabilities):
        scored = dict(row)
        scored["baseline_probability"] = probability
        baseline_holdout_rows.append(scored)
    augmented_holdout_rows = []
    for row, probability in zip(augmented_eval_rows, augmented_holdout_probabilities):
        scored = dict(row)
        scored["vhrdb_probability"] = probability
        augmented_holdout_rows.append(scored)

    baseline_holdout_metrics = compute_binary_metrics(
        [int(str(row["label_hard_any_lysis"])) for row in baseline_holdout_rows],
        [float(row["baseline_probability"]) for row in baseline_holdout_rows],
    )
    augmented_holdout_metrics = compute_binary_metrics(
        [int(str(row["label_hard_any_lysis"])) for row in augmented_holdout_rows],
        [float(row["vhrdb_probability"]) for row in augmented_holdout_rows],
    )
    baseline_top3 = compute_top3_hit_rate(baseline_holdout_rows, probability_key="baseline_probability")
    augmented_top3 = compute_top3_hit_rate(augmented_holdout_rows, probability_key="vhrdb_probability")

    def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
        if value is None or baseline is None:
            return None
        return safe_round(value - baseline)

    metric_deltas = {
        "delta_roc_auc": _delta(augmented_holdout_metrics["roc_auc"], baseline_holdout_metrics["roc_auc"]),
        "delta_top3_hit_rate_all_strains": _delta(
            float(augmented_top3["top3_hit_rate_all_strains"]),
            float(baseline_top3["top3_hit_rate_all_strains"]),
        ),
        "delta_brier_score": _delta(augmented_holdout_metrics["brier_score"], baseline_holdout_metrics["brier_score"]),
    }
    joined_vhrdb_rows = int(vhrdb_counts.get("joined_rows", 0))
    lift_is_negative_or_negligible = (
        metric_deltas["delta_roc_auc"] is not None
        and metric_deltas["delta_top3_hit_rate_all_strains"] is not None
        and metric_deltas["delta_brier_score"] is not None
        and metric_deltas["delta_roc_auc"] <= NEGLIGIBLE_DELTA_TOLERANCE
        and metric_deltas["delta_top3_hit_rate_all_strains"] <= NEGLIGIBLE_DELTA_TOLERANCE
        and metric_deltas["delta_brier_score"] >= -NEGLIGIBLE_DELTA_TOLERANCE
    )

    if not cohort_rows:
        lift_decision = "pending_external_artifact"
    elif joined_vhrdb_rows == 0:
        lift_decision = "no_joinable_vhrdb_rows"
    elif lift_is_negative_or_negligible:
        lift_decision = "do_not_include_vhrdb"
    else:
        lift_decision = "keep_vhrdb_for_followup_arms"

    summary_rows = [
        {
            "arm": "internal_only",
            "source_systems": "internal",
            "training_row_count": len(internal_trainable_rows),
            "vhrdb_row_count": 0,
            "holdout_roc_auc": baseline_holdout_metrics["roc_auc"],
            "holdout_top3_hit_rate_all_strains": baseline_top3["top3_hit_rate_all_strains"],
            "holdout_brier_score": baseline_holdout_metrics["brier_score"],
            "delta_roc_auc_vs_baseline": 0.0,
            "delta_top3_vs_baseline": 0.0,
            "delta_brier_vs_baseline": 0.0,
        },
        {
            "arm": "internal_plus_vhrdb",
            "source_systems": "internal|vhrdb",
            "training_row_count": len(augmented_trainable_rows),
            "vhrdb_row_count": len(vhrdb_rows),
            "holdout_roc_auc": augmented_holdout_metrics["roc_auc"],
            "holdout_top3_hit_rate_all_strains": augmented_top3["top3_hit_rate_all_strains"],
            "holdout_brier_score": augmented_holdout_metrics["brier_score"],
            "delta_roc_auc_vs_baseline": metric_deltas["delta_roc_auc"],
            "delta_top3_vs_baseline": metric_deltas["delta_top3_hit_rate_all_strains"],
            "delta_brier_vs_baseline": metric_deltas["delta_brier_score"],
        },
    ]

    output_summary_path = args.output_dir / "tk01_vhrdb_lift_summary.csv"
    output_rankings_path = args.output_dir / "tk01_vhrdb_holdout_top3_rankings.csv"
    output_manifest_path = args.output_dir / "tk01_vhrdb_lift_manifest.json"

    write_csv(
        output_summary_path,
        fieldnames=[
            "arm",
            "source_systems",
            "training_row_count",
            "vhrdb_row_count",
            "holdout_roc_auc",
            "holdout_top3_hit_rate_all_strains",
            "holdout_brier_score",
            "delta_roc_auc_vs_baseline",
            "delta_top3_vs_baseline",
            "delta_brier_vs_baseline",
        ],
        rows=summary_rows,
    )

    ranking_rows = [
        *build_top3_ranking_rows(
            baseline_holdout_rows,
            probability_key="baseline_probability",
            model_label="internal_only",
        ),
        *build_top3_ranking_rows(
            augmented_holdout_rows,
            probability_key="vhrdb_probability",
            model_label="internal_plus_vhrdb",
        ),
    ]
    write_csv(
        output_rankings_path,
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
    write_json(
        output_manifest_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_vhrdb_lift_report",
            "locked_feature_config_path": str(args.v1_feature_config_path),
            "locked_feature_config_sha256": _sha256(args.v1_feature_config_path),
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
            },
            "input_hashes_sha256": {
                "st02_pair_table": _sha256(args.st02_pair_table_path),
                "st03_split_assignments": _sha256(args.st03_split_assignments_path),
                "track_c_pair_table": _sha256(args.track_c_pair_table_path),
                "track_d_genome_kmers": _sha256(args.track_d_genome_kmer_path),
                "track_d_distance": _sha256(args.track_d_distance_path),
                "track_e_rbp_receptor_compatibility": _sha256(args.track_e_rbp_compatibility_path),
                "track_e_isolation_host_distance": _sha256(args.track_e_isolation_distance_path),
                **(
                    {"ti08_training_cohort_rows": _sha256(args.ti08_training_cohort_path)}
                    if args.ti08_training_cohort_path.exists()
                    else {}
                ),
            },
            "vhrdb_counts": vhrdb_counts,
            "joined_vhrdb_rows": joined_vhrdb_rows,
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
            "lift_decision": lift_decision,
            "locked_subset_blocks": list(locked_subset_blocks),
            "output_paths": {
                "summary": str(output_summary_path),
                "holdout_rankings": str(output_rankings_path),
            },
        },
    )

    logger.info("TK01 completed.")
    logger.info("- Baseline ROC-AUC: %s", baseline_holdout_metrics["roc_auc"])
    logger.info("- Augmented ROC-AUC: %s", augmented_holdout_metrics["roc_auc"])
    logger.info("- Delta ROC-AUC: %s", metric_deltas["delta_roc_auc"])
    logger.info("- VHRdb rows joined: %s", vhrdb_counts.get("joined_rows", 0))
    logger.info("- Lift decision: %s", lift_decision)
