#!/usr/bin/env python3
"""TL05: Retrain the locked v1 model with TL03/TL04 mechanistic features."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    CATEGORICAL_FEATURE_COLUMNS as V0_CATEGORICAL_FEATURE_COLUMNS,
)
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    NUMERIC_FEATURE_COLUMNS as V0_NUMERIC_FEATURE_COLUMNS,
)
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier
from lyzortx.pipeline.track_g.steps.compute_shap_explanations import (
    format_contribution_summary,
    top_feature_contributions,
)
from lyzortx.pipeline.track_g.steps.run_feature_block_ablation_suite import partition_track_c_columns
from lyzortx.pipeline.track_l.steps import (
    build_mechanistic_defense_evasion_features,
    build_mechanistic_rbp_receptor_features,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import load_holdout_bacteria_ids, load_json

logger = logging.getLogger(__name__)

IDENTIFIER_COLUMNS: tuple[str, ...] = ("pair_id", "bacteria", "phage")
DEFAULT_TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
DEFAULT_V1_LOCK_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
DEFAULT_TL03_OUTPUT_PATH = Path(
    "lyzortx/generated_outputs/track_l/mechanistic_rbp_receptor_features/mechanistic_rbp_receptor_features_v1.csv"
)
DEFAULT_TL04_OUTPUT_PATH = Path(
    "lyzortx/generated_outputs/track_l/mechanistic_defense_evasion_features/mechanistic_defense_evasion_features_v1.csv"
)
DEFAULT_TL03_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_l/mechanistic_rbp_receptor_features/mechanistic_rbp_receptor_manifest_v1.json"
)
DEFAULT_TL04_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_l/mechanistic_defense_evasion_features/mechanistic_defense_evasion_manifest_v1.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/mechanistic_v1_lift")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
LOCKED_BASELINE_ARM_ID = "locked_baseline_defense_phage_genomic"
PRIMARY_LOCK_METRIC = "holdout_roc_auc"
SECONDARY_LOCK_METRICS = ("holdout_top3_hit_rate_all_strains", "holdout_brier_score")
BOOTSTRAP_METRIC_NAMES = ("holdout_roc_auc", "holdout_top3_hit_rate_all_strains", "holdout_brier_score")


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    display_name: str
    included_blocks: tuple[str, ...]
    tl03_columns: tuple[str, ...]
    tl04_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapMetricCI:
    point_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    bootstrap_samples_requested: int
    bootstrap_samples_used: int


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
        "--tl03-feature-path",
        type=Path,
        default=DEFAULT_TL03_OUTPUT_PATH,
        help="TL03 mechanistic RBP-receptor feature CSV.",
    )
    parser.add_argument(
        "--tl04-feature-path",
        type=Path,
        default=DEFAULT_TL04_OUTPUT_PATH,
        help="TL04 mechanistic defense-evasion feature CSV.",
    )
    parser.add_argument(
        "--tl03-manifest-path",
        type=Path,
        default=DEFAULT_TL03_MANIFEST_PATH,
        help="TL03 mechanistic RBP-receptor manifest JSON.",
    )
    parser.add_argument(
        "--tl04-manifest-path",
        type=Path,
        default=DEFAULT_TL04_MANIFEST_PATH,
        help="TL04 mechanistic defense-evasion manifest JSON.",
    )
    parser.add_argument(
        "--tg01-summary-path",
        type=Path,
        default=DEFAULT_TG01_SUMMARY_PATH,
        help="TG01 model summary JSON used to lock the LightGBM hyperparameters.",
    )
    parser.add_argument(
        "--v1-lock-path",
        type=Path,
        default=DEFAULT_V1_LOCK_PATH,
        help="Committed v1 feature lock artifact used as the current baseline reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated TL05 artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for deterministic LightGBM refits.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume prerequisite Track G/L outputs already exist instead of regenerating them.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of paired bootstrap resamples over holdout strains for metric uncertainty.",
    )
    parser.add_argument(
        "--bootstrap-random-state",
        type=int,
        default=42,
        help="Random seed for holdout-strain bootstrap uncertainty estimates.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _load_rows_and_columns(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, tuple(rows[0].keys())


def ensure_default_tg01_summary(path: Path) -> None:
    if path.exists():
        return
    if path != DEFAULT_TG01_SUMMARY_PATH:
        raise FileNotFoundError(f"Missing TG01 summary: {path}")
    logger.info("TG01 summary missing at %s; regenerating TG01", path)
    train_v1_binary_classifier.main([])
    if not path.exists():
        raise FileNotFoundError(f"TG01 rebuild did not produce expected summary: {path}")


def _validate_tl11_manifest(
    *,
    feature_path: Path,
    manifest_path: Path,
    expected_task_id: str,
    expected_split_assignments_path: Path,
) -> dict[str, object]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing TL11 feature file: {feature_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing TL11 manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    if manifest.get("task_id") != expected_task_id:
        raise ValueError(f"Unexpected TL11 task_id in {manifest_path}: {manifest.get('task_id')!r}")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"TL11 manifest missing provenance section: {manifest_path}")

    split_section = provenance.get("split_assignments")
    if not isinstance(split_section, dict):
        raise ValueError(f"TL11 manifest missing split_assignments section: {manifest_path}")
    if split_section.get("path") != str(expected_split_assignments_path):
        raise ValueError(f"TL11 manifest split assignments path mismatch: {manifest_path}")

    expected_holdout_ids = load_holdout_bacteria_ids(expected_split_assignments_path)
    manifest_holdout_ids = provenance.get("excluded_holdout_bacteria_ids")
    if sorted(str(value) for value in manifest_holdout_ids or []) != expected_holdout_ids:
        raise ValueError(f"TL11 manifest holdout IDs do not match ST03 split assignments: {manifest_path}")

    holdout_section = manifest.get("holdout_exclusion")
    if not isinstance(holdout_section, dict):
        raise ValueError(f"TL11 manifest missing holdout_exclusion section: {manifest_path}")
    if int(holdout_section.get("excluded_pair_rows", 0) or 0) <= 0:
        raise ValueError(f"TL11 manifest reports no excluded holdout pair rows: {manifest_path}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"TL11 manifest missing outputs section: {manifest_path}")
    output_entry = outputs.get("feature_csv")
    if not isinstance(output_entry, str):
        raise ValueError(f"TL11 manifest missing feature_csv output entry: {manifest_path}")
    if output_entry != str(feature_path):
        raise ValueError(f"TL11 manifest feature_csv path mismatch: {manifest_path}")
    expected_hash_key = "feature_csv_sha256"
    if outputs.get(expected_hash_key) != _sha256(feature_path):
        raise ValueError(f"TL11 manifest feature_csv hash mismatch: {manifest_path}")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "feature_path": str(feature_path),
        "feature_sha256": _sha256(feature_path),
        "holdout_bacteria_ids": expected_holdout_ids,
        "excluded_pair_rows": int(holdout_section["excluded_pair_rows"]),
    }


def _ensure_default_tl11_bundle(
    *,
    feature_path: Path,
    manifest_path: Path,
    expected_task_id: str,
    expected_split_assignments_path: Path,
    default_feature_path: Path,
    default_manifest_path: Path,
    rebuild_fn: Any,
    bundle_label: str,
) -> None:
    should_rebuild = False
    if not feature_path.exists() or not manifest_path.exists():
        if feature_path != default_feature_path or manifest_path != default_manifest_path:
            missing = feature_path if not feature_path.exists() else manifest_path
            raise FileNotFoundError(f"Missing {bundle_label} artifact: {missing}")
        should_rebuild = True
    else:
        try:
            _validate_tl11_manifest(
                feature_path=feature_path,
                manifest_path=manifest_path,
                expected_task_id=expected_task_id,
                expected_split_assignments_path=expected_split_assignments_path,
            )
        except (FileNotFoundError, ValueError):
            if feature_path != default_feature_path or manifest_path != default_manifest_path:
                raise
            should_rebuild = True

    if not should_rebuild:
        return

    logger.info("%s artifacts missing or stale; regenerating %s", bundle_label, expected_task_id)
    rebuild_fn([])
    _validate_tl11_manifest(
        feature_path=feature_path,
        manifest_path=manifest_path,
        expected_task_id=expected_task_id,
        expected_split_assignments_path=expected_split_assignments_path,
    )


def ensure_prerequisite_outputs(args: argparse.Namespace) -> None:
    if args.skip_prerequisites:
        return
    ensure_default_tg01_summary(args.tg01_summary_path)
    _ensure_default_tl11_bundle(
        feature_path=args.tl03_feature_path,
        manifest_path=args.tl03_manifest_path,
        expected_task_id="TL03",
        expected_split_assignments_path=args.st03_split_assignments_path,
        default_feature_path=DEFAULT_TL03_OUTPUT_PATH,
        default_manifest_path=DEFAULT_TL03_MANIFEST_PATH,
        rebuild_fn=build_mechanistic_rbp_receptor_features.main,
        bundle_label="TL03",
    )
    _ensure_default_tl11_bundle(
        feature_path=args.tl04_feature_path,
        manifest_path=args.tl04_manifest_path,
        expected_task_id="TL04",
        expected_split_assignments_path=args.st03_split_assignments_path,
        default_feature_path=DEFAULT_TL04_OUTPUT_PATH,
        default_manifest_path=DEFAULT_TL04_MANIFEST_PATH,
        rebuild_fn=build_mechanistic_defense_evasion_features.main,
        bundle_label="TL04",
    )


def load_tl11_feature_provenance(
    feature_path: Path,
    manifest_path: Path,
    expected_task_id: str,
    expected_split_assignments_path: Path,
) -> dict[str, object]:
    return _validate_tl11_manifest(
        feature_path=feature_path,
        manifest_path=manifest_path,
        expected_task_id=expected_task_id,
        expected_split_assignments_path=expected_split_assignments_path,
    )


def load_v1_lock(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    locked_blocks = list(payload.get("winner_subset_blocks", []))
    if locked_blocks != ["defense", "phage_genomic"]:
        raise ValueError(
            "TL05 expects the current locked v1 baseline to be defense + phage_genomic; "
            f"found {locked_blocks!r} in {path}"
        )
    return dict(payload)


def load_tg01_lock(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "best_params": dict(payload["lightgbm"]["best_params"]),
        "holdout_binary_metrics": dict(payload["lightgbm"]["holdout_binary_metrics"]),
        "holdout_top3_metrics": dict(payload["lightgbm"]["holdout_top3_metrics"]),
    }


def partition_track_c_defense_columns(track_c_columns: Sequence[str]) -> tuple[str, ...]:
    partitioned = train_v1_binary_classifier.deduplicate_preserving_order(track_c_columns)
    if not partitioned:
        raise ValueError("Track C feature table has no columns.")
    defense_columns = partition_track_c_columns(partitioned)["defense_subtypes"]
    if not defense_columns:
        raise ValueError("No defense subtype columns were found in Track C features.")
    return tuple(defense_columns)


def build_arm_specs(
    *,
    defense_columns: Sequence[str],
    track_d_columns: Sequence[str],
    tl03_columns: Sequence[str],
    tl04_columns: Sequence[str],
) -> list[ArmSpec]:
    baseline_numeric = _deduplicate([*V0_NUMERIC_FEATURE_COLUMNS, *defense_columns, *track_d_columns])
    tl03_numeric = _deduplicate([*baseline_numeric, *tl03_columns])
    tl04_numeric = _deduplicate([*baseline_numeric, *tl04_columns])
    combined_numeric = _deduplicate([*baseline_numeric, *tl03_columns, *tl04_columns])
    return [
        ArmSpec(
            arm_id=LOCKED_BASELINE_ARM_ID,
            display_name="Locked defense + phage-genomic baseline",
            included_blocks=("defense", "phage_genomic"),
            tl03_columns=(),
            tl04_columns=(),
            numeric_columns=baseline_numeric,
        ),
        ArmSpec(
            arm_id="locked_plus_tl03_rbp_receptor",
            display_name="Locked baseline + TL03 RBP-receptor",
            included_blocks=("defense", "phage_genomic", "tl03"),
            tl03_columns=tuple(tl03_columns),
            tl04_columns=(),
            numeric_columns=tl03_numeric,
        ),
        ArmSpec(
            arm_id="locked_plus_tl04_defense_evasion",
            display_name="Locked baseline + TL04 defense-evasion",
            included_blocks=("defense", "phage_genomic", "tl04"),
            tl03_columns=(),
            tl04_columns=tuple(tl04_columns),
            numeric_columns=tl04_numeric,
        ),
        ArmSpec(
            arm_id="locked_plus_tl03_tl04_combined",
            display_name="Locked baseline + TL03 + TL04",
            included_blocks=("defense", "phage_genomic", "tl03", "tl04"),
            tl03_columns=tuple(tl03_columns),
            tl04_columns=tuple(tl04_columns),
            numeric_columns=combined_numeric,
        ),
    ]


def build_arm_feature_space(
    arm: ArmSpec,
    *,
    defense_columns: Sequence[str],
    track_d_columns: Sequence[str],
) -> train_v1_binary_classifier.FeatureSpace:
    mechanistic_columns = _deduplicate([*arm.tl03_columns, *arm.tl04_columns])
    return train_v1_binary_classifier.FeatureSpace(
        categorical_columns=tuple(V0_CATEGORICAL_FEATURE_COLUMNS),
        numeric_columns=arm.numeric_columns,
        track_c_additional_columns=tuple(defense_columns),
        track_d_columns=tuple(track_d_columns),
        track_e_columns=mechanistic_columns,
    )


def classify_feature_block(
    feature_name: str,
    *,
    defense_columns: Sequence[str],
    track_d_columns: Sequence[str],
    tl03_columns: Sequence[str],
    tl04_columns: Sequence[str],
) -> str:
    base_name = feature_name.split("=", 1)[0]
    if base_name in set(tl03_columns):
        return "tl03_mechanistic"
    if base_name in set(tl04_columns):
        return "tl04_mechanistic"
    if base_name in set(track_d_columns) or base_name.startswith("phage_"):
        return "track_d_phage_genomic"
    if base_name in set(defense_columns):
        return "track_c_defense_baseline"
    return "st04_v0_baseline"


def build_global_feature_importance_rows(
    shap_matrix: Any,
    feature_names: Sequence[str],
    *,
    defense_columns: Sequence[str],
    track_d_columns: Sequence[str],
    tl03_columns: Sequence[str],
    tl04_columns: Sequence[str],
) -> list[dict[str, object]]:
    if hasattr(shap_matrix, "toarray"):
        values = np.asarray(shap_matrix.toarray())
    else:
        values = np.asarray(shap_matrix)

    abs_mean = np.mean(np.abs(values), axis=0)
    signed_mean = np.mean(values, axis=0)
    nonzero_fraction = np.count_nonzero(values, axis=0) / values.shape[0]
    rows = [
        {
            "feature_name": feature_name,
            "feature_block": classify_feature_block(
                feature_name,
                defense_columns=defense_columns,
                track_d_columns=track_d_columns,
                tl03_columns=tl03_columns,
                tl04_columns=tl04_columns,
            ),
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


def compute_fold_metrics(
    fold_datasets: Sequence[train_v1_binary_classifier.FoldDataset],
    *,
    estimator_factory: Any,
    locked_params: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fold_metrics: list[dict[str, object]] = []
    cv_prediction_rows: list[dict[str, object]] = []
    for dataset in fold_datasets:
        estimator = estimator_factory(locked_params, dataset.fold_id)
        estimator.fit(dataset.X_train, dataset.y_train, sample_weight=dataset.sample_weights)
        probabilities = train_v1_binary_classifier.predict_probabilities(estimator, dataset.X_valid)
        scored_rows: list[dict[str, object]] = []
        for row, probability in zip(dataset.valid_rows, probabilities):
            scored = dict(row)
            scored["predicted_probability"] = probability
            scored["prediction_context"] = "non_holdout_oof"
            scored_rows.append(scored)
            cv_prediction_rows.append(scored)

        fold_metrics.append(
            {
                "fold_id": dataset.fold_id,
                "train_rows": len(dataset.train_rows),
                "validation_rows": len(dataset.valid_rows),
                "binary_metrics": train_v1_binary_classifier.compute_binary_metrics(dataset.y_valid, probabilities),
                "top3_metrics": train_v1_binary_classifier.compute_top3_hit_rate(
                    scored_rows, probability_key="predicted_probability"
                ),
            }
        )
    cv_prediction_rows.sort(key=lambda row: (str(row["bacteria"]), str(row["phage"])))
    return fold_metrics, cv_prediction_rows


def summarize_fold_metrics(fold_metrics: Sequence[Mapping[str, object]]) -> dict[str, Optional[float]]:
    return train_v1_binary_classifier.summarize_fold_metrics(fold_metrics)


def evaluate_arm(
    arm: ArmSpec,
    merged_rows: Sequence[Mapping[str, object]],
    *,
    defense_columns: Sequence[str],
    track_d_columns: Sequence[str],
    locked_params: Mapping[str, object],
    estimator_factory: Any,
) -> dict[str, object]:
    logger.info("TL05 evaluating arm %s", arm.arm_id)
    arm_feature_space = build_arm_feature_space(
        arm,
        defense_columns=defense_columns,
        track_d_columns=track_d_columns,
    )
    fold_datasets = train_v1_binary_classifier.prepare_fold_datasets(merged_rows, arm_feature_space)
    fold_metrics, cv_prediction_rows = compute_fold_metrics(
        fold_datasets,
        estimator_factory=estimator_factory,
        locked_params=locked_params,
    )
    cv_summary = summarize_fold_metrics(fold_metrics)
    estimator, vectorizer, _, holdout_rows, holdout_probabilities = train_v1_binary_classifier.fit_final_estimator(
        merged_rows,
        arm_feature_space,
        estimator_factory=estimator_factory,
        params=locked_params,
        sample_weight_key="training_weight_v3",
    )
    holdout_prediction_rows: list[dict[str, object]] = []
    for row, probability in zip(holdout_rows, holdout_probabilities):
        scored = dict(row)
        scored["predicted_probability"] = probability
        scored["prediction_context"] = "holdout_final"
        holdout_prediction_rows.append(scored)

    holdout_y = [int(str(row["label_hard_any_lysis"])) for row in holdout_prediction_rows]
    holdout_binary_metrics = train_v1_binary_classifier.compute_binary_metrics(
        holdout_y,
        [float(row["predicted_probability"]) for row in holdout_prediction_rows],
    )
    holdout_top3_metrics = train_v1_binary_classifier.compute_top3_hit_rate(
        holdout_prediction_rows,
        probability_key="predicted_probability",
    )

    pair_prediction_rows = cv_prediction_rows + holdout_prediction_rows
    pair_prediction_rows.sort(key=lambda row: (str(row["prediction_context"]), str(row["bacteria"]), str(row["phage"])))
    holdout_top3_rows = train_v1_binary_classifier.build_top3_ranking_rows(
        holdout_prediction_rows,
        probability_key="predicted_probability",
        model_label=arm.arm_id,
    )

    logger.info(
        "TL05 arm %s complete: holdout ROC-AUC=%s top3=%s brier=%s",
        arm.arm_id,
        holdout_binary_metrics["roc_auc"],
        holdout_top3_metrics["top3_hit_rate_all_strains"],
        holdout_binary_metrics["brier_score"],
    )

    return {
        "arm": arm,
        "feature_space": arm_feature_space,
        "fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "cv_prediction_rows": cv_prediction_rows,
        "estimator": estimator,
        "vectorizer": vectorizer,
        "holdout_rows": holdout_rows,
        "holdout_prediction_rows": holdout_prediction_rows,
        "holdout_binary_metrics": holdout_binary_metrics,
        "holdout_top3_metrics": holdout_top3_metrics,
        "pair_prediction_rows": pair_prediction_rows,
        "holdout_top3_rows": holdout_top3_rows,
    }


def summarize_arm_metrics(
    arm_result: Mapping[str, object],
    *,
    baseline_binary_metrics: Mapping[str, Optional[float]],
    baseline_top3_metrics: Mapping[str, object],
    bootstrap_summary: Mapping[str, Mapping[str, BootstrapMetricCI]],
    baseline_arm_id: str,
) -> dict[str, object]:
    holdout_binary_metrics = arm_result["holdout_binary_metrics"]
    holdout_top3_metrics = arm_result["holdout_top3_metrics"]
    auc = holdout_binary_metrics["roc_auc"]
    baseline_auc = baseline_binary_metrics["roc_auc"]
    top3 = holdout_top3_metrics["top3_hit_rate_all_strains"]
    baseline_top3 = baseline_top3_metrics["top3_hit_rate_all_strains"]
    brier = holdout_binary_metrics["brier_score"]
    baseline_brier = baseline_binary_metrics["brier_score"]
    arm_id = str(arm_result["arm"].arm_id)
    arm_bootstrap = bootstrap_summary[arm_id]
    delta_bootstrap = bootstrap_summary.get(f"{arm_id}__delta_vs_{baseline_arm_id}")
    is_baseline = arm_id == baseline_arm_id
    return {
        "arm_id": arm_id,
        "arm_label": arm_result["arm"].display_name,
        "included_blocks": list(arm_result["arm"].included_blocks),
        "tl03_feature_count": len(arm_result["arm"].tl03_columns),
        "tl04_feature_count": len(arm_result["arm"].tl04_columns),
        "numeric_feature_count": len(arm_result["arm"].numeric_columns),
        "holdout_roc_auc": auc,
        "holdout_roc_auc_ci_low": arm_bootstrap["holdout_roc_auc"].ci_low,
        "holdout_roc_auc_ci_high": arm_bootstrap["holdout_roc_auc"].ci_high,
        "holdout_brier_score": brier,
        "holdout_brier_score_ci_low": arm_bootstrap["holdout_brier_score"].ci_low,
        "holdout_brier_score_ci_high": arm_bootstrap["holdout_brier_score"].ci_high,
        "holdout_top3_hit_rate_all_strains": top3,
        "holdout_top3_hit_rate_all_strains_ci_low": arm_bootstrap["holdout_top3_hit_rate_all_strains"].ci_low,
        "holdout_top3_hit_rate_all_strains_ci_high": arm_bootstrap["holdout_top3_hit_rate_all_strains"].ci_high,
        "holdout_top3_hit_rate_susceptible_only": holdout_top3_metrics["top3_hit_rate_susceptible_only"],
        "auc_delta_vs_locked_baseline": 0.0
        if is_baseline
        else safe_round(auc - baseline_auc)
        if auc is not None and baseline_auc is not None
        else None,
        "auc_delta_ci_low_vs_locked_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_roc_auc"].ci_low,
        "auc_delta_ci_high_vs_locked_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_roc_auc"].ci_high,
        "top3_delta_vs_locked_baseline": 0.0
        if is_baseline
        else safe_round(top3 - baseline_top3)
        if top3 is not None and baseline_top3 is not None
        else None,
        "top3_delta_ci_low_vs_locked_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_top3_hit_rate_all_strains"].ci_low,
        "top3_delta_ci_high_vs_locked_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_top3_hit_rate_all_strains"].ci_high,
        "brier_improvement_vs_locked_baseline": 0.0
        if is_baseline
        else safe_round(baseline_brier - brier)
        if brier is not None and baseline_brier is not None
        else None,
        "brier_improvement_ci_low_vs_locked_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_brier_score"].ci_low,
        "brier_improvement_ci_high_vs_locked_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_brier_score"].ci_high,
    }


def _evaluate_holdout_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    y_true = [int(str(row["label_hard_any_lysis"])) for row in rows]
    y_prob = [float(row["predicted_probability"]) for row in rows]
    return {
        "binary": train_v1_binary_classifier.compute_binary_metrics(y_true, y_prob),
        "top3": train_v1_binary_classifier.compute_top3_hit_rate(rows, probability_key="predicted_probability"),
    }


def bootstrap_holdout_metric_cis(
    holdout_rows_by_arm: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    bootstrap_samples: int,
    bootstrap_random_state: int,
    baseline_arm_id: str,
) -> dict[str, dict[str, BootstrapMetricCI]]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")

    if baseline_arm_id not in holdout_rows_by_arm:
        raise ValueError("Missing baseline arm for bootstrap evaluation.")

    holdout_by_bacteria: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in holdout_rows_by_arm[baseline_arm_id]:
        holdout_by_bacteria[str(row["bacteria"])].append(row)

    bacteria_ids = tuple(sorted(holdout_by_bacteria.keys()))
    if not bacteria_ids:
        raise ValueError("No holdout bacteria available for bootstrap evaluation.")

    arm_bacteria_sets: dict[str, dict[str, list[Mapping[str, object]]]] = {
        arm_id: {bacteria: [] for bacteria in bacteria_ids} for arm_id in holdout_rows_by_arm
    }
    for arm_id, rows in holdout_rows_by_arm.items():
        for row in rows:
            bacteria = str(row["bacteria"])
            if bacteria in arm_bacteria_sets[arm_id]:
                arm_bacteria_sets[arm_id][bacteria].append(row)
    for arm_id, bacteria_map in arm_bacteria_sets.items():
        missing = [bacteria for bacteria in bacteria_ids if not bacteria_map.get(bacteria)]
        if missing:
            raise ValueError(f"Missing holdout rows for arm {arm_id}: {', '.join(missing)}")

    rng = np.random.default_rng(bootstrap_random_state)
    metric_samples: dict[str, dict[str, list[float]]] = {
        arm_id: {metric_name: [] for metric_name in BOOTSTRAP_METRIC_NAMES} for arm_id in holdout_rows_by_arm
    }
    delta_samples: dict[str, dict[str, list[float]]] = {
        f"{arm_id}__delta_vs_{baseline_arm_id}": {metric_name: [] for metric_name in BOOTSTRAP_METRIC_NAMES}
        for arm_id in holdout_rows_by_arm
        if arm_id != baseline_arm_id
    }

    bacteria_count = len(bacteria_ids)
    progress_interval = max(1, bootstrap_samples // 5)
    for sample_index in range(bootstrap_samples):
        if sample_index == 0 or (sample_index + 1) % progress_interval == 0 or sample_index + 1 == bootstrap_samples:
            logger.info(
                "TL05 bootstrap progress: %d/%d paired holdout-strain resamples",
                sample_index + 1,
                bootstrap_samples,
            )
        sampled_bacteria_indices = rng.integers(0, bacteria_count, size=bacteria_count)
        sampled_rows_by_arm: dict[str, list[Mapping[str, object]]] = {}
        for arm_id, bacteria_map in arm_bacteria_sets.items():
            sampled_rows: list[Mapping[str, object]] = []
            for bacteria_index in sampled_bacteria_indices.tolist():
                sampled_rows.extend(bacteria_map[bacteria_ids[bacteria_index]])
            sampled_rows_by_arm[arm_id] = sampled_rows

        metrics_by_arm = {arm_id: _evaluate_holdout_rows(rows) for arm_id, rows in sampled_rows_by_arm.items()}
        baseline_metrics = metrics_by_arm[baseline_arm_id]
        for arm_id, metrics in metrics_by_arm.items():
            metric_samples[arm_id]["holdout_top3_hit_rate_all_strains"].append(
                float(metrics["top3"]["top3_hit_rate_all_strains"])
            )
            metric_samples[arm_id]["holdout_brier_score"].append(float(metrics["binary"]["brier_score"]))
            if metrics["binary"]["roc_auc"] is not None:
                metric_samples[arm_id]["holdout_roc_auc"].append(float(metrics["binary"]["roc_auc"]))

        for arm_id, metrics in metrics_by_arm.items():
            if arm_id == baseline_arm_id:
                continue
            delta_key = f"{arm_id}__delta_vs_{baseline_arm_id}"
            if baseline_metrics["binary"]["roc_auc"] is not None and metrics["binary"]["roc_auc"] is not None:
                delta_samples[delta_key]["holdout_roc_auc"].append(
                    float(metrics["binary"]["roc_auc"]) - float(baseline_metrics["binary"]["roc_auc"])
                )
            delta_samples[delta_key]["holdout_top3_hit_rate_all_strains"].append(
                float(metrics["top3"]["top3_hit_rate_all_strains"])
                - float(baseline_metrics["top3"]["top3_hit_rate_all_strains"])
            )
            delta_samples[delta_key]["holdout_brier_score"].append(
                float(baseline_metrics["binary"]["brier_score"]) - float(metrics["binary"]["brier_score"])
            )

    def _ci(values: Sequence[float]) -> tuple[Optional[float], Optional[float], int]:
        if not values:
            return None, None, 0
        low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
        return safe_round(float(low)), safe_round(float(high)), len(values)

    actual_metrics_by_arm = {arm_id: _evaluate_holdout_rows(rows) for arm_id, rows in holdout_rows_by_arm.items()}

    ci_summary: dict[str, dict[str, BootstrapMetricCI]] = {}
    for arm_id, samples in metric_samples.items():
        actual_metrics = actual_metrics_by_arm[arm_id]
        ci_summary[arm_id] = {}
        for metric_name, sample_values in samples.items():
            ci_low, ci_high, used = _ci(sample_values)
            ci_summary[arm_id][metric_name] = BootstrapMetricCI(
                point_estimate=(
                    float(actual_metrics["binary"]["roc_auc"])
                    if metric_name == "holdout_roc_auc"
                    else float(actual_metrics["top3"]["top3_hit_rate_all_strains"])
                    if metric_name == "holdout_top3_hit_rate_all_strains"
                    else float(actual_metrics["binary"]["brier_score"])
                ),
                ci_low=ci_low,
                ci_high=ci_high,
                bootstrap_samples_requested=bootstrap_samples,
                bootstrap_samples_used=used,
            )

    for delta_key, samples in delta_samples.items():
        ci_summary[delta_key] = {}
        for metric_name, sample_values in samples.items():
            ci_low, ci_high, used = _ci(sample_values)
            ci_summary[delta_key][metric_name] = BootstrapMetricCI(
                point_estimate=None,
                ci_low=ci_low,
                ci_high=ci_high,
                bootstrap_samples_requested=bootstrap_samples,
                bootstrap_samples_used=used,
            )

    return ci_summary


def select_locked_arm(
    arm_metrics: Sequence[Mapping[str, object]],
    *,
    baseline_arm_id: str,
) -> Optional[dict[str, object]]:
    eligible = []
    for row in arm_metrics:
        if row["arm_id"] == baseline_arm_id:
            continue
        if (
            float(row["auc_delta_ci_low_vs_locked_baseline"]) > 0.0
            and float(row["top3_delta_ci_high_vs_locked_baseline"]) >= 0.0
            and float(row["brier_improvement_ci_high_vs_locked_baseline"]) >= 0.0
        ):
            eligible.append(dict(row))
    if not eligible:
        return None

    eligible.sort(
        key=lambda row: (
            float(row["holdout_roc_auc"]),
            float(row["holdout_top3_hit_rate_all_strains"]),
            -float(row["holdout_brier_score"]),
            str(row["arm_id"]),
        ),
        reverse=True,
    )
    return eligible[0]


def select_best_mechanistic_arm(
    arm_metrics: Sequence[Mapping[str, object]],
    *,
    baseline_arm_id: str,
) -> dict[str, object]:
    candidates = [dict(row) for row in arm_metrics if row["arm_id"] != baseline_arm_id]
    candidates.sort(
        key=lambda row: (
            float(row["holdout_roc_auc"]),
            float(row["holdout_top3_hit_rate_all_strains"]),
            -float(row["holdout_brier_score"]),
            str(row["arm_id"]),
        ),
        reverse=True,
    )
    return candidates[0]


def build_lock_rejection_reason(row: Mapping[str, object]) -> str:
    reasons = []
    if float(row["auc_delta_ci_low_vs_locked_baseline"]) <= 0.0:
        reasons.append("AUC delta stays within bootstrap noise")
    if float(row["top3_delta_ci_high_vs_locked_baseline"]) < 0.0:
        reasons.append("top-3 hit rate materially degrades")
    if float(row["brier_improvement_ci_high_vs_locked_baseline"]) < 0.0:
        reasons.append("Brier score materially degrades")
    return "; ".join(reasons) if reasons else "meets lock rule"


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    logger.info("TL05 starting: retrain mechanistic v1 model")
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)
    try:
        shap_available = importlib.util.find_spec("shap") is not None
    except ValueError:
        shap_available = "shap" in sys.modules
    if not shap_available:
        raise ModuleNotFoundError("TL05 requires shap to compute SHAP explanations.")

    tg01_lock = load_tg01_lock(args.tg01_summary_path)
    current_v1_lock = load_v1_lock(args.v1_lock_path)

    logger.info("TL05 loading input tables")
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows, track_c_columns = _load_rows_and_columns(args.track_c_pair_table_path)
    track_d_genome_rows, track_d_genome_columns = _load_rows_and_columns(args.track_d_genome_kmer_path)
    track_d_distance_rows, track_d_distance_columns = _load_rows_and_columns(args.track_d_distance_path)
    tl03_rows, tl03_columns = _load_rows_and_columns(args.tl03_feature_path)
    tl04_rows, tl04_columns = _load_rows_and_columns(args.tl04_feature_path)
    tl03_provenance = load_tl11_feature_provenance(
        args.tl03_feature_path,
        args.tl03_manifest_path,
        "TL03",
        args.st03_split_assignments_path,
    )
    tl04_provenance = load_tl11_feature_provenance(
        args.tl04_feature_path,
        args.tl04_manifest_path,
        "TL04",
        args.st03_split_assignments_path,
    )

    track_d_feature_columns = _deduplicate(
        [column for column in track_d_genome_columns if column != "phage"]
        + [column for column in track_d_distance_columns if column != "phage"]
    )
    defense_columns = partition_track_c_defense_columns(track_c_columns)
    tl03_feature_columns = _deduplicate([column for column in tl03_columns if column not in IDENTIFIER_COLUMNS])
    tl04_feature_columns = _deduplicate([column for column in tl04_columns if column not in IDENTIFIER_COLUMNS])

    logger.info("TL05 merging feature blocks onto %d Track C rows", len(track_c_pair_rows))
    merged_rows = train_v1_binary_classifier.merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(tl03_rows, tl04_rows),
        allow_missing_pair_features=True,
    )
    lightgbm_factory = lambda params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )

    arm_specs = build_arm_specs(
        defense_columns=defense_columns,
        track_d_columns=track_d_feature_columns,
        tl03_columns=tl03_feature_columns,
        tl04_columns=tl04_feature_columns,
    )
    logger.info("TL05 evaluating %d arms", len(arm_specs))
    arm_results = []
    for arm in arm_specs:
        arm_results.append(
            evaluate_arm(
                arm,
                merged_rows,
                defense_columns=defense_columns,
                track_d_columns=track_d_feature_columns,
                locked_params=tg01_lock["best_params"],
                estimator_factory=lightgbm_factory,
            )
        )

    arm_holdout_rows = {result["arm"].arm_id: result["holdout_prediction_rows"] for result in arm_results}
    baseline_result = next(result for result in arm_results if result["arm"].arm_id == LOCKED_BASELINE_ARM_ID)
    holdout_strain_count = len({str(row["bacteria"]) for row in baseline_result["holdout_prediction_rows"]})
    logger.info(
        "TL05 bootstrapping holdout metrics: %d strains, %d samples",
        holdout_strain_count,
        args.bootstrap_samples,
    )
    bootstrap_summary = bootstrap_holdout_metric_cis(
        arm_holdout_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_random_state=args.bootstrap_random_state,
        baseline_arm_id=baseline_result["arm"].arm_id,
    )
    logger.info("TL05 selecting lock candidate against baseline %s", baseline_result["arm"].arm_id)
    arm_metrics = [
        summarize_arm_metrics(
            result,
            baseline_binary_metrics=baseline_result["holdout_binary_metrics"],
            baseline_top3_metrics=baseline_result["holdout_top3_metrics"],
            bootstrap_summary=bootstrap_summary,
            baseline_arm_id=baseline_result["arm"].arm_id,
        )
        for result in arm_results
    ]
    proposed_arm = select_locked_arm(arm_metrics, baseline_arm_id=LOCKED_BASELINE_ARM_ID)
    best_mechanistic_arm = select_best_mechanistic_arm(
        arm_metrics,
        baseline_arm_id=LOCKED_BASELINE_ARM_ID,
    )
    shap_arm_id = proposed_arm["arm_id"] if proposed_arm is not None else best_mechanistic_arm["arm_id"]
    shap_result = next(result for result in arm_results if result["arm"].arm_id == shap_arm_id)

    logger.info("TL05 preparing SHAP explanations for arm %s", shap_arm_id)
    shap_feature_matrix = shap_result["vectorizer"].transform(
        [
            train_v1_binary_classifier.build_feature_dict(
                row,
                categorical_columns=shap_result["feature_space"].categorical_columns,
                numeric_columns=shap_result["feature_space"].numeric_columns,
            )
            for row in shap_result["holdout_rows"]
        ]
    )
    import shap  # Heavy dependency; import only when TL05 runs.

    logger.info("TL05 running SHAP explainer for %d holdout rows", len(shap_result["holdout_rows"]))
    shap_explainer = shap.TreeExplainer(shap_result["estimator"])
    shap_explanation = shap_explainer(shap_feature_matrix)
    shap_values = shap_explanation.values
    shap_base_values = np.asarray(shap_explanation.base_values).ravel()
    feature_names = list(shap_result["vectorizer"].get_feature_names_out())

    shap_global_rows = build_global_feature_importance_rows(
        shap_values,
        feature_names,
        defense_columns=defense_columns,
        track_d_columns=track_d_feature_columns,
        tl03_columns=tl03_feature_columns,
        tl04_columns=tl04_feature_columns,
    )
    shap_block_totals: dict[str, float] = {}
    for row in shap_global_rows:
        block = str(row["feature_block"])
        shap_block_totals[block] = shap_block_totals.get(block, 0.0) + float(row["mean_abs_shap"])

    shap_prediction_rows = train_v1_binary_classifier.build_top3_ranking_rows(
        shap_result["holdout_prediction_rows"],
        probability_key="predicted_probability",
        model_label=shap_arm_id,
    )
    explain_index_by_pair_id = {row["pair_id"]: index for index, row in enumerate(shap_result["holdout_rows"])}
    shap_pair_rows: list[dict[str, object]] = []
    for row in shap_prediction_rows:
        explain_index = explain_index_by_pair_id.get(row["pair_id"])
        if explain_index is None:
            continue
        shap_row_value = shap_values[explain_index]
        if hasattr(shap_row_value, "toarray"):
            shap_row = np.asarray(shap_row_value.toarray()).ravel()
        else:
            shap_row = np.asarray(shap_row_value).ravel()
        feature_row = np.asarray(shap_feature_matrix[explain_index].toarray()).ravel()
        contributions = top_feature_contributions(
            shap_row,
            feature_row,
            feature_names,
            top_k=3,
        )
        shap_pair_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "prediction_rank": row["rank"],
                "predicted_probability": row["predicted_probability"],
                "shap_base_value": safe_round(float(shap_base_values[explain_index])),
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
                "top_positive_summary": format_contribution_summary(contributions["positive"]),
                "top_negative_summary": format_contribution_summary(contributions["negative"]),
            }
        )
        for position, item in enumerate(contributions["positive"], start=1):
            shap_pair_rows[-1][f"top_positive_feature_{position}"] = item["feature_name"]
            shap_pair_rows[-1][f"top_positive_shap_{position}"] = item["shap_value"]
        for position, item in enumerate(contributions["negative"], start=1):
            shap_pair_rows[-1][f"top_negative_feature_{position}"] = item["feature_name"]
            shap_pair_rows[-1][f"top_negative_shap_{position}"] = item["shap_value"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TL05",
        "current_locked_v1_reference": current_v1_lock,
        "tg01_lightgbm_reference": tg01_lock,
        "arm_metrics": arm_metrics,
        "baseline_reference_arm_id": baseline_result["arm"].arm_id,
        "best_mechanistic_arm": best_mechanistic_arm,
        "proposed_lock_arm": proposed_arm,
        "lock_rule": {
            "primary_metric": PRIMARY_LOCK_METRIC,
            "secondary_metrics": list(SECONDARY_LOCK_METRICS),
            "decision_rule": (
                "Lock only if the candidate's bootstrap CI for ROC-AUC delta vs baseline is entirely above zero "
                "and the top-3 hit-rate / Brier-improvement deltas are not materially negative."
            ),
        },
        "lock_decision": {
            "status": "proposed" if proposed_arm is not None else "no_honest_lift",
            "selected_arm_id": proposed_arm["arm_id"] if proposed_arm is not None else None,
            "selected_arm_label": proposed_arm["arm_label"] if proposed_arm is not None else None,
        },
        "shap_arm_id": shap_arm_id,
        "shap_block_importance_totals": shap_block_totals,
        "bootstrap_ci": {
            arm_id: {
                metric_name: {
                    "point_estimate": metric_ci.point_estimate,
                    "ci_low": metric_ci.ci_low,
                    "ci_high": metric_ci.ci_high,
                    "bootstrap_samples_requested": metric_ci.bootstrap_samples_requested,
                    "bootstrap_samples_used": metric_ci.bootstrap_samples_used,
                }
                for metric_name, metric_ci in bootstrap_summary[arm_id].items()
            }
            for arm_id in bootstrap_summary
            if "__delta_vs_" not in arm_id
        },
        "bootstrap_delta_ci_vs_baseline": {
            arm_id: {
                metric_name: {
                    "point_estimate": metric_ci.point_estimate,
                    "ci_low": metric_ci.ci_low,
                    "ci_high": metric_ci.ci_high,
                    "bootstrap_samples_requested": metric_ci.bootstrap_samples_requested,
                    "bootstrap_samples_used": metric_ci.bootstrap_samples_used,
                }
                for metric_name, metric_ci in bootstrap_summary[arm_id].items()
            }
            for arm_id in bootstrap_summary
            if "__delta_vs_" in arm_id
        },
        "shap_top_features": {
            "global": shap_global_rows[:15],
            "mechanistic_only": [
                row for row in shap_global_rows if row["feature_block"] in {"tl03_mechanistic", "tl04_mechanistic"}
            ][:15],
        },
        "inputs": {
            "v1_lock": {"path": str(args.v1_lock_path), "sha256": _sha256(args.v1_lock_path)},
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
            "tl03_features": {"path": str(args.tl03_feature_path), "sha256": _sha256(args.tl03_feature_path)},
            "tl04_features": {"path": str(args.tl04_feature_path), "sha256": _sha256(args.tl04_feature_path)},
            "tl03_manifest": tl03_provenance,
            "tl04_manifest": tl04_provenance,
        },
    }

    metrics_fieldnames = [
        "arm_id",
        "arm_label",
        "included_blocks",
        "tl03_feature_count",
        "tl04_feature_count",
        "numeric_feature_count",
        "holdout_roc_auc",
        "holdout_roc_auc_ci_low",
        "holdout_roc_auc_ci_high",
        "holdout_brier_score",
        "holdout_brier_score_ci_low",
        "holdout_brier_score_ci_high",
        "holdout_top3_hit_rate_all_strains",
        "holdout_top3_hit_rate_all_strains_ci_low",
        "holdout_top3_hit_rate_all_strains_ci_high",
        "holdout_top3_hit_rate_susceptible_only",
        "auc_delta_vs_locked_baseline",
        "auc_delta_ci_low_vs_locked_baseline",
        "auc_delta_ci_high_vs_locked_baseline",
        "top3_delta_vs_locked_baseline",
        "top3_delta_ci_low_vs_locked_baseline",
        "top3_delta_ci_high_vs_locked_baseline",
        "brier_improvement_vs_locked_baseline",
        "brier_improvement_ci_low_vs_locked_baseline",
        "brier_improvement_ci_high_vs_locked_baseline",
    ]

    logger.info("TL05 writing output artifacts to %s", args.output_dir)
    write_csv(args.output_dir / "tl05_mechanistic_lift_metrics.csv", metrics_fieldnames, arm_metrics)
    write_csv(
        args.output_dir / "tl05_mechanistic_pair_predictions.csv",
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
        [
            {
                "arm_id": result["arm"].arm_id,
                "arm_label": result["arm"].display_name,
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "split_holdout": row["split_holdout"],
                "split_cv5_fold": row["split_cv5_fold"],
                "label_hard_any_lysis": row["label_hard_any_lysis"],
                "prediction_context": row["prediction_context"],
                "predicted_probability": safe_round(float(row["predicted_probability"])),
            }
            for result in arm_results
            for row in result["pair_prediction_rows"]
        ],
    )
    write_csv(
        args.output_dir / "tl05_mechanistic_holdout_top3_rankings.csv",
        [
            "model_label",
            "bacteria",
            "phage",
            "pair_id",
            "rank",
            "predicted_probability",
            "label_hard_any_lysis",
        ],
        shap_result["holdout_top3_rows"],
    )
    write_csv(
        args.output_dir / "tl05_shap_global_feature_importance.csv",
        ["feature_name", "feature_block", "mean_abs_shap", "mean_shap", "nonzero_fraction"],
        shap_global_rows,
    )
    write_csv(
        args.output_dir / "tl05_shap_pair_explanations.csv",
        [
            "pair_id",
            "bacteria",
            "phage",
            "prediction_rank",
            "predicted_probability",
            "shap_base_value",
            "top_positive_feature_1",
            "top_positive_feature_2",
            "top_positive_feature_3",
            "top_positive_shap_1",
            "top_positive_shap_2",
            "top_positive_shap_3",
            "top_negative_feature_1",
            "top_negative_feature_2",
            "top_negative_feature_3",
            "top_negative_shap_1",
            "top_negative_shap_2",
            "top_negative_shap_3",
            "top_positive_summary",
            "top_negative_summary",
        ],
        shap_pair_rows,
    )
    write_json(args.output_dir / "tl05_mechanistic_lift_summary.json", summary)

    if proposed_arm is not None:
        proposed_config = {
            "task_id": "TL05",
            "source_lock_task_id": current_v1_lock.get("source_lock_task_id", "TG09"),
            "selection_policy": (
                "Select the mechanistic arm whose paired bootstrap ROC-AUC delta clears zero and whose top-3 / "
                "Brier deltas do not materially degrade relative to the current locked defense + phage-genomic "
                "baseline, then rank by ROC-AUC, top-3 hit rate, and inverse Brier."
            ),
            "baseline_arm_id": baseline_result["arm"].arm_id,
            "proposed_arm_id": proposed_arm["arm_id"],
            "proposed_label": proposed_arm["arm_label"],
            "proposed_subset_blocks": proposed_arm["included_blocks"],
            "holdout_roc_auc": proposed_arm["holdout_roc_auc"],
            "holdout_brier_score": proposed_arm["holdout_brier_score"],
            "holdout_top3_hit_rate_all_strains": proposed_arm["holdout_top3_hit_rate_all_strains"],
            "auc_delta_vs_locked_baseline": proposed_arm["auc_delta_vs_locked_baseline"],
            "top3_delta_vs_locked_baseline": proposed_arm["top3_delta_vs_locked_baseline"],
            "brier_improvement_vs_locked_baseline": proposed_arm["brier_improvement_vs_locked_baseline"],
            "locked_v1_reference": current_v1_lock,
        }
        write_json(args.output_dir / "tl05_proposed_v1_feature_config.json", proposed_config)
    else:
        logger.info("TL05 lock rule rejected all mechanistic arms: no honest lift.")
        rejected_arms = [
            {
                "arm_id": row["arm_id"],
                "arm_label": row["arm_label"],
                "reason": build_lock_rejection_reason(row),
            }
            for row in arm_metrics
            if row["arm_id"] != baseline_result["arm"].arm_id
        ]
        write_json(
            args.output_dir / "tl05_no_honest_lift_rejections.json",
            {
                "task_id": "TL05",
                "baseline_arm_id": baseline_result["arm"].arm_id,
                "rejected_arms": rejected_arms,
                "decision": "no_honest_lift",
            },
        )

    logger.info("TL05 completed.")
    logger.info("- Locked baseline ROC-AUC: %s", baseline_result["holdout_binary_metrics"]["roc_auc"])
    logger.info("- Best mechanistic arm: %s", best_mechanistic_arm["arm_label"])
    if proposed_arm is not None:
        logger.info("- Proposed lock: %s", proposed_arm["arm_label"])
    else:
        logger.info("- No mechanistic arm improved on the locked baseline.")
    logger.info("- Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
