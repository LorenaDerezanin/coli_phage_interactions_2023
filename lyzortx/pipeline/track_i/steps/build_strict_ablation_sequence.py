#!/usr/bin/env python3
"""TI09: Run the strict external-data ablation sequence in the planned source order."""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    FeatureSpace,
    build_feature_space,
    compute_binary_metrics,
    compute_top3_hit_rate,
    fit_final_estimator,
    make_lightgbm_estimator,
    merge_expanded_feature_rows,
)
from lyzortx.pipeline.track_i.steps.build_external_training_cohorts import TRAINING_ARM_INDEX, TRAINING_ARM_ORDER
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import (
    build_locked_feature_space,
    build_training_rows,
    load_locked_v1_feature_config,
    load_source_training_rows,
    load_tg01_best_params,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_COHORT_COLUMNS = (
    "pair_id",
    "source_system",
    "first_training_arm",
    "first_training_arm_index",
    "effective_training_weight",
)
REQUIRED_ST02_COLUMNS = ("pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier")

STRICT_ABLATION_SOURCE_ADDITIONS: Dict[str, Tuple[str, ...]] = {
    "internal_only": ("internal",),
    "plus_vhrdb": ("vhrdb",),
    "plus_basel": ("basel",),
    "plus_klebphacol": ("klebphacol",),
    "plus_gpb": ("gpb",),
    "plus_tier_b": ("virus_host_db", "ncbi_virus_biosample"),
}

STRICT_SOURCE_ORDER = (
    "internal",
    "vhrdb",
    "basel",
    "klebphacol",
    "gpb",
    "virus_host_db",
    "ncbi_virus_biosample",
)
STRICT_SOURCE_INDEX = {source: idx for idx, source in enumerate(STRICT_SOURCE_ORDER)}

LOCKED_V1_FEATURE_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
TI08_TRAINING_COHORT_PATH = Path(
    "lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_i/strict_ablation_sequence")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
    parser.add_argument("--training-cohort-path", type=Path, default=TI08_TRAINING_COHORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_sources(sources: Sequence[str]) -> str:
    return "|".join(sources)


def _sorted_unique_sources(sources: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for source in sources:
        if source not in STRICT_SOURCE_INDEX:
            raise ValueError(f"Unsupported source_system in strict ablation sequence: {source!r}")
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return sorted(ordered, key=lambda source: STRICT_SOURCE_INDEX[source])


def _planned_sources_for_arm(arm: str) -> Tuple[str, ...]:
    if arm not in STRICT_ABLATION_SOURCE_ADDITIONS:
        raise ValueError(f"Unsupported training arm in strict ablation sequence: {arm!r}")
    return STRICT_ABLATION_SOURCE_ADDITIONS[arm]


def _cumulative_planned_sources_for_arm(arm: str) -> Tuple[str, ...]:
    arm_index = TRAINING_ARM_INDEX[arm]
    cumulative: List[str] = []
    for prior_arm in TRAINING_ARM_ORDER[: arm_index + 1]:
        cumulative.extend(_planned_sources_for_arm(prior_arm))
    return tuple(cumulative)


def compute_strict_ablation_summary(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    previous_pair_ids: set[str] = set()
    previous_row_count = 0

    normalized_rows = [_normalize_row(row) for row in rows]
    for row in normalized_rows:
        source_system = row.get("source_system", "")
        if source_system not in STRICT_SOURCE_INDEX:
            raise ValueError(f"Unsupported source_system in strict ablation sequence: {source_system!r}")

    for arm in TRAINING_ARM_ORDER:
        arm_index = TRAINING_ARM_INDEX[arm]
        arm_rows = [row for row in normalized_rows if int(row["first_training_arm_index"]) <= arm_index]
        pair_ids = {str(row["pair_id"]) for row in arm_rows}
        observed_sources = _sorted_unique_sources([str(row["source_system"]) for row in arm_rows])
        current_rows = [row for row in normalized_rows if int(row["first_training_arm_index"]) == arm_index]
        current_pair_ids = {str(row["pair_id"]) for row in current_rows}
        current_sources = _sorted_unique_sources([str(row["source_system"]) for row in current_rows])
        external_rows = [row for row in arm_rows if str(row["source_system"]) != "internal"]
        external_pair_ids = {str(row["pair_id"]) for row in external_rows}

        summary_rows.append(
            {
                "arm": arm,
                "arm_index": arm_index,
                "planned_source_systems_added": _join_sources(_planned_sources_for_arm(arm)),
                "observed_source_systems_added": _join_sources(current_sources),
                "cumulative_source_systems": _join_sources(observed_sources),
                "cumulative_row_count": len(arm_rows),
                "cumulative_pair_count": len(pair_ids),
                "cumulative_external_row_count": len(external_rows),
                "cumulative_external_pair_count": len(external_pair_ids),
                "new_rows_vs_previous_arm": len(arm_rows) - previous_row_count,
                "new_pairs_vs_previous_arm": len(pair_ids - previous_pair_ids),
                "new_observed_pairs_vs_previous_arm": len(current_pair_ids - previous_pair_ids),
                "cumulative_training_weight": safe_round(
                    sum(float(row["effective_training_weight"]) for row in arm_rows)
                ),
                "cumulative_planned_source_count": len(_cumulative_planned_sources_for_arm(arm)),
            }
        )
        previous_pair_ids = pair_ids
        previous_row_count = len(arm_rows)

    return summary_rows


def _measure_holdout_metrics(
    training_rows: Sequence[Mapping[str, object]],
    *,
    feature_space: FeatureSpace,
    best_params: Mapping[str, object],
    random_state: int,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    estimator_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=random_state,
    )
    _, _, _, eval_rows, probabilities = fit_final_estimator(
        training_rows,
        feature_space,
        estimator_factory=estimator_factory,
        params=best_params,
        sample_weight_key="effective_training_weight",
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
    top3_metrics = compute_top3_hit_rate(scored_rows, probability_key="probability")
    return holdout_metrics, top3_metrics, scored_rows


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return safe_round(float(value) - float(baseline))


def _paired_count(rows: Sequence[Mapping[str, object]]) -> int:
    return len({str(row["pair_id"]) for row in rows})


def build_strict_ablation_report(
    *,
    merged_rows: Sequence[Mapping[str, object]],
    cohort_rows: Sequence[Mapping[str, str]],
    feature_space: FeatureSpace,
    best_params: Mapping[str, object],
    random_state: int,
) -> List[Dict[str, object]]:
    internal_rows = [dict(row, source_system="internal") for row in merged_rows]
    source_rows_by_system: Dict[str, List[Dict[str, object]]] = {}
    for source_system in STRICT_SOURCE_ORDER:
        if source_system == "internal":
            continue
        source_rows_by_system[source_system], _ = load_source_training_rows(
            merged_rows,
            cohort_rows,
            source_system,
        )

    report_rows: List[Dict[str, object]] = []
    previous_metrics: Optional[Dict[str, object]] = None
    previous_top3: Optional[Dict[str, object]] = None
    baseline_metrics: Optional[Dict[str, object]] = None
    baseline_top3: Optional[Dict[str, object]] = None
    previous_pair_ids: set[str] = set()
    previous_observed_pair_ids: set[str] = set()
    previous_row_count = 0

    for arm in TRAINING_ARM_ORDER:
        added_sources = _planned_sources_for_arm(arm)
        added_external_rows = sum(
            len(source_rows_by_system.get(source_system, []))
            for source_system in added_sources
            if source_system != "internal"
        )
        if arm != "internal_only" and added_external_rows == 0:
            raise ValueError(
                f"TI09 arm {arm} requires >0 external rows from the added source(s): {_join_sources(added_sources)}"
            )

        cumulative_sources = _cumulative_planned_sources_for_arm(arm)
        training_rows = build_training_rows(internal_rows, source_rows_by_system, cumulative_sources)
        holdout_metrics, top3_metrics, scored_rows = _measure_holdout_metrics(
            training_rows,
            feature_space=feature_space,
            best_params=best_params,
            random_state=random_state,
        )
        current_rows = [
            row for row in cohort_rows if int(str(row["first_training_arm_index"])) == TRAINING_ARM_INDEX[arm]
        ]
        current_sources = _sorted_unique_sources([str(row["source_system"]) for row in current_rows])
        external_rows = [row for row in training_rows if str(row.get("source_system", "internal")) != "internal"]
        cumulative_pair_ids = {str(row["pair_id"]) for row in training_rows}
        cumulative_external_pair_ids = {str(row["pair_id"]) for row in external_rows}
        current_pair_ids = {str(row["pair_id"]) for row in current_rows}
        added_external_pair_ids = set()
        for source_system in added_sources:
            if source_system == "internal":
                continue
            added_external_pair_ids.update(str(row["pair_id"]) for row in source_rows_by_system.get(source_system, []))

        if baseline_metrics is None:
            baseline_metrics = holdout_metrics
            baseline_top3 = top3_metrics

        row = {
            "arm": arm,
            "arm_index": TRAINING_ARM_INDEX[arm],
            "planned_source_systems_added": _join_sources(added_sources),
            "observed_source_systems_added": _join_sources(current_sources),
            "cumulative_source_systems": _join_sources(
                _sorted_unique_sources([str(row.get("source_system", "internal")) for row in training_rows])
            ),
            "cumulative_row_count": len(training_rows),
            "cumulative_pair_count": len(cumulative_pair_ids),
            "cumulative_external_row_count": len(external_rows),
            "cumulative_external_pair_count": len(cumulative_external_pair_ids),
            "new_rows_vs_previous_arm": len(training_rows) - previous_row_count,
            "new_pairs_vs_previous_arm": len(cumulative_pair_ids - previous_pair_ids),
            "new_observed_pairs_vs_previous_arm": len(current_pair_ids - previous_observed_pair_ids),
            "cumulative_training_weight": safe_round(
                sum(float(row.get("effective_training_weight", 0.0) or 0.0) for row in training_rows)
            ),
            "cumulative_planned_source_count": len(cumulative_sources),
            "added_external_row_count": added_external_rows,
            "added_external_pair_count": len(added_external_pair_ids),
            "holdout_roc_auc": holdout_metrics["roc_auc"],
            "holdout_top3_hit_rate_all_strains": top3_metrics["top3_hit_rate_all_strains"],
            "holdout_top3_hit_rate_susceptible_only": top3_metrics["top3_hit_rate_susceptible_only"],
            "holdout_brier_score": holdout_metrics["brier_score"],
            "delta_roc_auc_vs_internal_only": (
                0.0
                if baseline_metrics is None
                else _delta(holdout_metrics["roc_auc"], baseline_metrics["roc_auc"])
                if arm != "internal_only"
                else 0.0
            ),
            "delta_top3_vs_internal_only": (
                0.0
                if baseline_top3 is None
                else _delta(top3_metrics["top3_hit_rate_all_strains"], baseline_top3["top3_hit_rate_all_strains"])
                if arm != "internal_only"
                else 0.0
            ),
            "delta_brier_vs_internal_only": (
                0.0
                if baseline_metrics is None
                else _delta(holdout_metrics["brier_score"], baseline_metrics["brier_score"])
                if arm != "internal_only"
                else 0.0
            ),
            "delta_roc_auc_vs_previous_arm": 0.0
            if previous_metrics is None
            else _delta(holdout_metrics["roc_auc"], previous_metrics["roc_auc"]),
            "delta_top3_vs_previous_arm": 0.0
            if previous_top3 is None
            else _delta(top3_metrics["top3_hit_rate_all_strains"], previous_top3["top3_hit_rate_all_strains"]),
            "delta_brier_vs_previous_arm": 0.0
            if previous_metrics is None
            else _delta(holdout_metrics["brier_score"], previous_metrics["brier_score"]),
            "evaluation_rows": len(scored_rows),
            "training_rows": len(training_rows),
        }
        report_rows.append(row)
        previous_metrics = holdout_metrics
        previous_top3 = top3_metrics
        previous_pair_ids = cumulative_pair_ids
        previous_observed_pair_ids = current_pair_ids
        previous_row_count = len(training_rows)

    return report_rows


def ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    LOGGER.info("Starting TI09 strict ablation sequence")
    ensure_directory(args.output_dir)

    if not args.skip_prerequisites:
        from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import ensure_prerequisite_outputs

        ensure_prerequisite_outputs(args)

    st02_rows = read_csv_rows(args.st02_pair_table_path, REQUIRED_ST02_COLUMNS)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)
    cohort_rows = read_csv_rows(args.training_cohort_path, REQUIRED_COHORT_COLUMNS)

    locked_config = load_locked_v1_feature_config(args.v1_feature_config_path)
    locked_subset_blocks = tuple(str(block) for block in locked_config["winner_subset_blocks"])
    tg01_best_params = load_tg01_best_params(args.tg01_summary_path)

    track_d_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_d_genome_rows[0].keys() if column != "phage"]
            + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
        )
    )
    track_e_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_e_rbp_rows[0].keys() if column not in {"pair_id", "bacteria", "phage"}]
            + [column for column in track_e_isolation_rows[0].keys() if column not in {"pair_id", "bacteria", "phage"}]
        )
    )
    full_feature_space = build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )
    locked_feature_space = build_locked_feature_space(full_feature_space, locked_subset_blocks)

    merged_rows = merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_isolation_rows),
    )
    report_rows = build_strict_ablation_report(
        merged_rows=merged_rows,
        cohort_rows=cohort_rows,
        feature_space=locked_feature_space,
        best_params=tg01_best_params,
        random_state=args.random_state,
    )
    summary_rows = compute_strict_ablation_summary(cohort_rows)
    metrics_by_arm = {row["arm"]: row for row in report_rows}
    for row in summary_rows:
        metrics = metrics_by_arm[row["arm"]]
        row.update(
            {
                "added_external_row_count": metrics["added_external_row_count"],
                "added_external_pair_count": metrics["added_external_pair_count"],
                "holdout_roc_auc": metrics["holdout_roc_auc"],
                "holdout_top3_hit_rate_all_strains": metrics["holdout_top3_hit_rate_all_strains"],
                "holdout_top3_hit_rate_susceptible_only": metrics["holdout_top3_hit_rate_susceptible_only"],
                "holdout_brier_score": metrics["holdout_brier_score"],
                "delta_roc_auc_vs_internal_only": metrics["delta_roc_auc_vs_internal_only"],
                "delta_top3_vs_internal_only": metrics["delta_top3_vs_internal_only"],
                "delta_brier_vs_internal_only": metrics["delta_brier_vs_internal_only"],
                "delta_roc_auc_vs_previous_arm": metrics["delta_roc_auc_vs_previous_arm"],
                "delta_top3_vs_previous_arm": metrics["delta_top3_vs_previous_arm"],
                "delta_brier_vs_previous_arm": metrics["delta_brier_vs_previous_arm"],
            }
        )

    summary_output_path = args.output_dir / "ti09_strict_ablation_summary.csv"
    manifest_output_path = args.output_dir / "ti09_strict_ablation_manifest.json"

    write_csv(summary_output_path, fieldnames=ordered_fieldnames(summary_rows), rows=summary_rows)
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_strict_ablation_sequence",
            "strict_ablation_order": list(TRAINING_ARM_ORDER),
            "training_arm_source_systems": {
                arm: list(_cumulative_planned_sources_for_arm(arm)) for arm in TRAINING_ARM_ORDER
            },
            "input_paths": {
                "st02_pair_table": str(args.st02_pair_table_path),
                "st03_split_assignments": str(args.st03_split_assignments_path),
                "track_c_pair_table": str(args.track_c_pair_table_path),
                "track_d_genome_kmers": str(args.track_d_genome_kmer_path),
                "track_d_distance": str(args.track_d_distance_path),
                "track_e_rbp_receptor_compatibility": str(args.track_e_rbp_compatibility_path),
                "track_e_isolation_host_distance": str(args.track_e_isolation_distance_path),
                "training_cohort_rows": str(args.training_cohort_path),
            },
            "input_hashes_sha256": {
                "st02_pair_table": _hash_path(args.st02_pair_table_path),
                "st03_split_assignments": _hash_path(args.st03_split_assignments_path),
                "track_c_pair_table": _hash_path(args.track_c_pair_table_path),
                "track_d_genome_kmers": _hash_path(args.track_d_genome_kmer_path),
                "track_d_distance": _hash_path(args.track_d_distance_path),
                "track_e_rbp_receptor_compatibility": _hash_path(args.track_e_rbp_compatibility_path),
                "track_e_isolation_host_distance": _hash_path(args.track_e_isolation_distance_path),
                "training_cohort_rows": _hash_path(args.training_cohort_path),
                "v1_feature_config": _hash_path(args.v1_feature_config_path),
                "tg01_summary": _hash_path(args.tg01_summary_path),
            },
            "output_paths": {
                "summary": str(summary_output_path),
            },
            "arm_metrics": [
                {
                    "arm": row["arm"],
                    "holdout_roc_auc": row["holdout_roc_auc"],
                    "holdout_top3_hit_rate_all_strains": row["holdout_top3_hit_rate_all_strains"],
                    "holdout_brier_score": row["holdout_brier_score"],
                    "added_external_row_count": row["added_external_row_count"],
                }
                for row in summary_rows
            ],
        },
    )
    LOGGER.info("Finished TI09 strict ablation sequence with %d arms", len(summary_rows))


if __name__ == "__main__":
    main()
