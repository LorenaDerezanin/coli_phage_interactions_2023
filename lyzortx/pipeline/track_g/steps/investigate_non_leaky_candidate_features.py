#!/usr/bin/env python3
"""TG11: Evaluate non-leaky pairwise candidate features against the locked v1 baseline."""

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
from lyzortx.pipeline.track_g.steps.run_feature_subset_sweep import (
    build_feature_blocks,
    ensure_tg01_summary,
    load_tg01_lightgbm_lock,
)
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
)

logger = logging.getLogger(__name__)

OLD_LEAKED_REFERENCE_AUC = 0.910766
CLEAN_PAIRWISE_CANDIDATE_COLUMNS: Tuple[str, ...] = (
    "lookup_available",
    "target_receptor_present",
    "protein_target_present",
    "surface_target_present",
    "receptor_cluster_matches",
    "isolation_host_umap_euclidean_distance",
    "isolation_host_defense_jaccard_distance",
)
CANDIDATE_COLUMN_SOURCES: Mapping[str, str] = {
    "lookup_available": "TE01 curated lookup",
    "target_receptor_present": "TE01 curated lookup",
    "protein_target_present": "TE01 curated lookup",
    "surface_target_present": "TE01 curated lookup",
    "receptor_cluster_matches": "TE01 curated lookup",
    "isolation_host_umap_euclidean_distance": "TE03 isolation-host distance",
    "isolation_host_defense_jaccard_distance": "TE03 isolation-host distance",
}
TE01_BINARY_INDICATOR_CANDIDATE_COLUMNS: Tuple[str, ...] = (
    "lookup_available",
    "target_receptor_present",
    "protein_target_present",
    "surface_target_present",
)
LOCKED_BASELINE_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CandidateArm:
    arm_id: str
    display_name: str
    candidate_column: Optional[str]
    categorical_columns: Tuple[str, ...]
    numeric_columns: Tuple[str, ...]
    locked_subset_blocks: Tuple[str, ...]


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
        "--v1-feature-config-path",
        type=Path,
        default=Path("lyzortx/pipeline/track_g/v1_feature_configuration.json"),
        help="Locked v1 feature configuration JSON used as the baseline reference.",
    )
    parser.add_argument(
        "--leaked-reference-auc",
        type=float,
        default=OLD_LEAKED_REFERENCE_AUC,
        help="Reference ROC-AUC from the historical leaked model used to define the gap-recovery threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg11_non_leaky_candidate_features"),
        help="Directory for generated TG11 artifacts.",
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
        help="Assume prerequisite Track C/D/E outputs already exist.",
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


def load_locked_v1_feature_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_locked_baseline_arm(feature_space: FeatureSpace, locked_config: Mapping[str, object]) -> CandidateArm:
    locked_subset_blocks = tuple(str(block) for block in locked_config["winner_subset_blocks"])
    blocks = build_feature_blocks(feature_space)
    categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS)
    numeric_columns = list(V0_NUMERIC_FEATURE_COLUMNS)
    for block_id in locked_subset_blocks:
        block = blocks[block_id]
        categorical_columns.extend(block.categorical_columns)
        numeric_columns.extend(block.numeric_columns)
    return CandidateArm(
        arm_id=str(locked_config["winner_arm_id"]),
        display_name=str(locked_config["winner_label"]),
        candidate_column=None,
        categorical_columns=_deduplicate(categorical_columns),
        numeric_columns=_deduplicate(numeric_columns),
        locked_subset_blocks=locked_subset_blocks,
    )


def build_candidate_arms(feature_space: FeatureSpace, locked_config: Mapping[str, object]) -> List[CandidateArm]:
    baseline_arm = build_locked_baseline_arm(feature_space, locked_config)
    available_track_e_columns = set(feature_space.track_e_columns)
    missing = [column for column in CLEAN_PAIRWISE_CANDIDATE_COLUMNS if column not in available_track_e_columns]
    if missing:
        raise ValueError(f"Missing required clean pairwise candidate columns: {', '.join(missing)}")

    arms = [baseline_arm]
    for column in CLEAN_PAIRWISE_CANDIDATE_COLUMNS:
        # Keep TE01 binary indicators in numeric_columns so TG11 matches the LightGBM treatment used elsewhere.
        # LightGBM handles 0/1 indicators numerically, and mixing categorical encoding here would make the
        # single-feature arm comparisons harder to interpret against the locked v1 baseline.
        arms.append(
            CandidateArm(
                arm_id=f"{baseline_arm.arm_id}__plus__{column}",
                display_name=f"{baseline_arm.display_name} + {column}",
                candidate_column=column,
                categorical_columns=baseline_arm.categorical_columns,
                numeric_columns=_deduplicate([*baseline_arm.numeric_columns, column]),
                locked_subset_blocks=baseline_arm.locked_subset_blocks,
            )
        )
    return arms


def compute_gap_recovery_fraction(candidate_auc: float, baseline_auc: float, leaked_auc: float) -> float:
    auc_gap = leaked_auc - baseline_auc
    if auc_gap <= 0:
        raise ValueError("Leaked reference AUC must be greater than the locked baseline AUC.")
    return safe_round((candidate_auc - baseline_auc) / auc_gap)


def summarize_candidate_row(
    arm: CandidateArm,
    *,
    holdout_binary_metrics: Mapping[str, Optional[float]],
    holdout_top3_metrics: Mapping[str, object],
    locked_baseline_auc: float,
    locked_baseline_top3: float,
    leaked_reference_auc: float,
) -> Dict[str, object]:
    holdout_auc = float(holdout_binary_metrics["roc_auc"])
    holdout_top3 = float(holdout_top3_metrics["top3_hit_rate_all_strains"])
    threshold_auc = locked_baseline_auc + ((leaked_reference_auc - locked_baseline_auc) / 2.0)
    gap_recovery_fraction = compute_gap_recovery_fraction(holdout_auc, locked_baseline_auc, leaked_reference_auc)
    top3_non_degrading = holdout_top3 >= locked_baseline_top3
    return {
        "arm_id": arm.arm_id,
        "arm_label": arm.display_name,
        "candidate_column": arm.candidate_column,
        "candidate_source": (
            "locked_v1_baseline" if arm.candidate_column is None else CANDIDATE_COLUMN_SOURCES[arm.candidate_column]
        ),
        "locked_subset_blocks": list(arm.locked_subset_blocks),
        "categorical_feature_count": len(arm.categorical_columns),
        "numeric_feature_count": len(arm.numeric_columns),
        "holdout_roc_auc": holdout_auc,
        "holdout_brier_score": holdout_binary_metrics["brier_score"],
        "holdout_top3_hit_rate_all_strains": holdout_top3,
        "holdout_top3_hit_rate_susceptible_only": holdout_top3_metrics["top3_hit_rate_susceptible_only"],
        "auc_delta_vs_locked_v1": safe_round(holdout_auc - locked_baseline_auc),
        "top3_delta_vs_locked_v1": safe_round(holdout_top3 - locked_baseline_top3),
        "gap_recovery_fraction_vs_locked_v1": gap_recovery_fraction,
        "half_gap_target_auc": safe_round(threshold_auc),
        "top3_non_degrading_vs_locked_v1": top3_non_degrading,
        "recovers_gt_50pct_auc_gap_without_top3_degradation": (
            arm.candidate_column is not None and holdout_auc > threshold_auc and top3_non_degrading
        ),
    }


def assert_locked_baseline_matches_rerun(
    rerun_baseline: Mapping[str, object],
    locked_v1_config: Mapping[str, object],
    *,
    tolerance: float = LOCKED_BASELINE_MATCH_TOLERANCE,
) -> None:
    checks = (
        ("holdout_roc_auc", "holdout_roc_auc"),
        ("holdout_top3_hit_rate_all_strains", "holdout_top3_hit_rate_all_strains"),
    )
    mismatches = []
    for rerun_key, locked_key in checks:
        rerun_value = float(rerun_baseline[rerun_key])
        locked_value = float(locked_v1_config[locked_key])
        if abs(rerun_value - locked_value) > tolerance:
            mismatches.append(
                f"{rerun_key}: rerun={rerun_value:.6f}, locked_config={locked_value:.6f}, tolerance={tolerance:.1e}"
            )
    if mismatches:
        mismatch_message = "; ".join(mismatches)
        raise ValueError(
            "TG11 rerun baseline diverged from the locked v1 config. Update the locked reference or investigate "
            f"reproducibility drift before interpreting candidate deltas. {mismatch_message}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("TG11 starting: non-leaky candidate feature evaluation")
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)
    ensure_tg01_summary(args)

    tg01_lock = load_tg01_lightgbm_lock(args.tg01_summary_path)
    locked_v1_config = load_locked_v1_feature_config(args.v1_feature_config_path)

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

    locked_baseline_auc = float(locked_v1_config["holdout_roc_auc"])
    locked_baseline_top3 = float(locked_v1_config["holdout_top3_hit_rate_all_strains"])
    candidate_arms = build_candidate_arms(feature_space, locked_v1_config)

    metrics_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []

    for arm in candidate_arms:
        arm_feature_space = FeatureSpace(
            categorical_columns=arm.categorical_columns,
            numeric_columns=arm.numeric_columns,
            track_c_additional_columns=feature_space.track_c_additional_columns,
            track_d_columns=feature_space.track_d_columns,
            track_e_columns=feature_space.track_e_columns,
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
            holdout_prediction_rows.append(scored)
        holdout_y = [int(str(row["label_hard_any_lysis"])) for row in holdout_prediction_rows]
        holdout_binary_metrics = compute_binary_metrics(
            holdout_y,
            [float(row["predicted_probability"]) for row in holdout_prediction_rows],
        )
        holdout_top3_metrics = compute_top3_hit_rate(holdout_prediction_rows, probability_key="predicted_probability")
        metrics_rows.append(
            summarize_candidate_row(
                arm,
                holdout_binary_metrics=holdout_binary_metrics,
                holdout_top3_metrics=holdout_top3_metrics,
                locked_baseline_auc=locked_baseline_auc,
                locked_baseline_top3=locked_baseline_top3,
                leaked_reference_auc=args.leaked_reference_auc,
            )
        )
        ranking_rows.extend(
            build_top3_ranking_rows(
                holdout_prediction_rows,
                probability_key="predicted_probability",
                model_label=arm.arm_id,
            )
        )

    rerun_baseline = next(row for row in metrics_rows if row["candidate_column"] is None)
    assert_locked_baseline_matches_rerun(rerun_baseline, locked_v1_config)
    improving_candidates = [
        row for row in metrics_rows if row["recovers_gt_50pct_auc_gap_without_top3_degradation"] is True
    ]
    best_candidate = max(
        [row for row in metrics_rows if row["candidate_column"] is not None],
        key=lambda row: (
            bool(row["top3_non_degrading_vs_locked_v1"]),
            float(row["gap_recovery_fraction_vs_locked_v1"]),
            float(row["holdout_roc_auc"]),
            float(row["holdout_top3_hit_rate_all_strains"]),
            -float(row["holdout_brier_score"]),
            str(row["candidate_column"]),
        ),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TG11",
        "locked_lightgbm_hyperparameters": locked_params,
        "locked_v1_reference": locked_v1_config,
        "historical_leaked_reference_auc": args.leaked_reference_auc,
        "half_gap_target_auc": safe_round(
            locked_baseline_auc + ((args.leaked_reference_auc - locked_baseline_auc) / 2.0)
        ),
        "rerun_locked_v1_baseline": rerun_baseline,
        "best_candidate": best_candidate,
        "improving_candidates": improving_candidates,
        "candidate_columns": list(CLEAN_PAIRWISE_CANDIDATE_COLUMNS),
        "acceptance_outcome": (
            "No candidate recovered >50% of the AUC gap without degrading top-3; keep the 2-block calibration as the "
            "honest v1 baseline."
            if not improving_candidates
            else "At least one non-leaky candidate recovered >50% of the AUC gap without degrading top-3."
        ),
        "inputs": {
            "tg01_summary": {"path": str(args.tg01_summary_path), "sha256": _sha256(args.tg01_summary_path)},
            "v1_feature_config": {
                "path": str(args.v1_feature_config_path),
                "sha256": _sha256(args.v1_feature_config_path),
            },
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

    write_json(args.output_dir / "tg11_non_leaky_candidate_summary.json", summary)
    write_csv(
        args.output_dir / "tg11_non_leaky_candidate_metrics.csv",
        list(metrics_rows[0].keys()),
        metrics_rows,
    )
    ranking_rows.sort(key=lambda row: (str(row["model_label"]), str(row["bacteria"]), int(row["rank"])))
    write_csv(
        args.output_dir / "tg11_non_leaky_candidate_holdout_top3_rankings.csv",
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
    logger.info("TG11 completed: evaluated %d clean candidate features", len(CLEAN_PAIRWISE_CANDIDATE_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
