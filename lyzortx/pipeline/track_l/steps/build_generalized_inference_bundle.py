#!/usr/bin/env python3
"""TL08 helper: train and persist a genome-only inference bundle."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import sha256_file
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import (
    build_defense_column_mask,
    build_defense_feature_rows,
)
from lyzortx.pipeline.track_d.steps import build_phage_genome_kmer_features
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier

logger = logging.getLogger(__name__)

DEFAULT_ST02_PAIR_TABLE_PATH = Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv")
DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH = Path(
    "lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"
)
DEFAULT_DEFENSE_SUBTYPES_PATH = Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv")
DEFAULT_PHAGE_KMER_FEATURE_PATH = Path(
    "lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv"
)
DEFAULT_PHAGE_KMER_SVD_PATH = Path(
    "lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_svd.joblib"
)
DEFAULT_TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/generalized_inference_bundle")
DEFAULT_CALIBRATION_FOLD = 0
BUNDLE_FILENAME = "tl08_generalized_inference_bundle.joblib"
PHAGE_SVD_FILENAME = "phage_genome_kmer_svd.joblib"
DEFENSE_MASK_FILENAME = "defense_subtype_column_mask.joblib"
PANEL_DEFENSE_SUBTYPES_FILENAME = "panel_defense_subtypes.csv"
DEFENSE_FINDER_MODELS_DIRNAME = "defense_finder_models"
PANEL_PREDICTIONS_FILENAME = "tl08_locked_panel_predictions.csv"
MANIFEST_FILENAME = "tl08_generalized_inference_manifest.json"
LOCKED_LIGHTGBM_KEYS = ("learning_rate", "min_child_samples", "n_estimators", "num_leaves")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=DEFAULT_ST02_PAIR_TABLE_PATH,
        help="ST0.2 pair table with labels and training weights.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH,
        help="ST0.3 split assignments used for final training and calibration.",
    )
    parser.add_argument(
        "--defense-subtypes-path",
        type=Path,
        default=DEFAULT_DEFENSE_SUBTYPES_PATH,
        help="Panel Defense Finder subtype CSV used to build the host-defense feature block and mask.",
    )
    parser.add_argument(
        "--phage-kmer-feature-path",
        type=Path,
        default=DEFAULT_PHAGE_KMER_FEATURE_PATH,
        help="TD02 phage genome k-mer feature CSV.",
    )
    parser.add_argument(
        "--phage-kmer-svd-path",
        type=Path,
        default=DEFAULT_PHAGE_KMER_SVD_PATH,
        help="TD02 phage genome k-mer SVD artifact.",
    )
    parser.add_argument(
        "--tg01-summary-path",
        type=Path,
        default=DEFAULT_TG01_SUMMARY_PATH,
        help="TG01 model summary JSON used to lock the LightGBM hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the saved TL08 inference bundle and reference predictions.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=DEFAULT_CALIBRATION_FOLD,
        help="Non-holdout fold used to fit isotonic calibration.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for the LightGBM estimator.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help=(
            "Assume prerequisite ST0.2/ST0.3/TD02/TG01 outputs already exist instead of regenerating missing defaults."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_prerequisite_outputs(args: argparse.Namespace) -> None:
    if args.skip_prerequisites:
        return

    st01_output_path = Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st01_pair_label_audit.csv")
    st01b_output_path = Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st01b_pair_confidence_audit.csv")
    if not st01_output_path.exists():
        st01_label_policy.main([])
    if not st01b_output_path.exists():
        st01b_confidence_tiers.main([])
    if not args.st02_pair_table_path.exists():
        st02_build_pair_table.main([])
    if not args.st03_split_assignments_path.exists():
        st03_build_splits.main([])
    if not args.phage_kmer_feature_path.exists() or not args.phage_kmer_svd_path.exists():
        build_phage_genome_kmer_features.main([])
    if not args.tg01_summary_path.exists():
        train_v1_binary_classifier.main([])


def load_locked_lightgbm_params(tg01_summary_path: Path) -> dict[str, Any]:
    payload = _load_json(tg01_summary_path)
    params = dict(payload["lightgbm"]["best_params"])
    missing = [key for key in LOCKED_LIGHTGBM_KEYS if key not in params]
    if missing:
        raise ValueError(f"TG01 summary at {tg01_summary_path} is missing locked LightGBM keys: {', '.join(missing)}")
    return params


def read_defense_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [{key: value for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No defense subtype rows found in {path}")
    return rows


def build_genome_only_feature_space(
    *,
    host_categorical_columns: Sequence[str] = (),
    host_feature_columns: Sequence[str],
    phage_categorical_columns: Sequence[str] = (),
    phage_feature_columns: Sequence[str],
    pairwise_categorical_columns: Sequence[str] = (),
    pairwise_feature_columns: Sequence[str] = (),
) -> train_v1_binary_classifier.FeatureSpace:
    categorical_columns = train_v1_binary_classifier.deduplicate_preserving_order(
        [*host_categorical_columns, *phage_categorical_columns, *pairwise_categorical_columns]
    )
    numeric_columns = train_v1_binary_classifier.deduplicate_preserving_order(
        [*host_feature_columns, *phage_feature_columns, *pairwise_feature_columns]
    )
    return train_v1_binary_classifier.FeatureSpace(
        categorical_columns=tuple(categorical_columns),
        numeric_columns=numeric_columns,
        track_c_additional_columns=tuple([*host_categorical_columns, *host_feature_columns]),
        track_d_columns=tuple([*phage_categorical_columns, *phage_feature_columns]),
        track_e_columns=tuple([*pairwise_categorical_columns, *pairwise_feature_columns]),
    )


def build_training_rows(
    *,
    st02_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
    host_rows: Sequence[Mapping[str, object]],
    phage_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    split_by_pair = {row["pair_id"]: dict(row) for row in split_rows}
    host_by_bacteria = {str(row["bacteria"]): dict(row) for row in host_rows}
    phage_by_name = {str(row["phage"]): dict(row) for row in phage_rows}

    merged_rows: list[dict[str, object]] = []
    for st02_row in st02_rows:
        pair_id = str(st02_row["pair_id"])
        bacteria = str(st02_row["bacteria"])
        phage = str(st02_row["phage"])
        split_row = split_by_pair.get(pair_id)
        host_row = host_by_bacteria.get(bacteria)
        phage_row = phage_by_name.get(phage)
        if split_row is None:
            raise KeyError(f"Missing ST0.3 split assignment for pair_id {pair_id}")
        if host_row is None:
            raise KeyError(f"Missing host-defense row for bacteria {bacteria}")
        if phage_row is None:
            raise KeyError(f"Missing phage genome feature row for phage {phage}")

        merged: dict[str, object] = dict(st02_row)
        merged.update(split_row)
        for column, value in host_row.items():
            if column != "bacteria":
                merged[column] = value
        for column, value in phage_row.items():
            if column != "phage":
                merged[column] = value
        merged_rows.append(merged)

    merged_rows.sort(key=lambda row: (str(row["bacteria"]), str(row["phage"])))
    return merged_rows


def augment_rows_with_pair_features(
    rows: Sequence[Mapping[str, object]],
    *,
    pair_feature_rows: Sequence[Mapping[str, object]],
    pair_feature_columns: Sequence[str],
) -> list[dict[str, object]]:
    if not pair_feature_columns:
        return [dict(row) for row in rows]
    pair_feature_index = {str(row["pair_id"]): dict(row) for row in pair_feature_rows}
    augmented_rows: list[dict[str, object]] = []
    for row in rows:
        augmented = dict(row)
        pair_row = pair_feature_index.get(str(row["pair_id"]))
        if pair_row is None:
            if str(row["split_holdout"]) != "holdout_test":
                raise KeyError(f"Missing deployable pair feature row for pair_id {row['pair_id']}")
            for column in pair_feature_columns:
                augmented[column] = 0.0
            augmented_rows.append(augmented)
            continue
        for column in pair_feature_columns:
            if column not in pair_row:
                raise KeyError(f"Missing deployable pair feature column {column} for pair_id {row['pair_id']}")
            augmented[column] = pair_row[column]
        augmented_rows.append(augmented)
    return augmented_rows


def augment_host_rows_with_features(
    host_rows: Sequence[Mapping[str, object]],
    *,
    extra_host_feature_rows: Sequence[Mapping[str, object]],
    extra_host_feature_columns: Sequence[str],
) -> list[dict[str, object]]:
    if not extra_host_feature_columns:
        return [dict(row) for row in host_rows]
    host_feature_index = {str(row["bacteria"]): dict(row) for row in extra_host_feature_rows}
    augmented_rows: list[dict[str, object]] = []
    for row in host_rows:
        augmented = dict(row)
        bacteria = str(row["bacteria"])
        extra_row = host_feature_index.get(bacteria)
        if extra_row is None:
            raise KeyError(f"Missing deployable host feature row for bacteria {bacteria}")
        for column in extra_host_feature_columns:
            if column not in extra_row:
                raise KeyError(f"Missing deployable host feature column {column} for bacteria {bacteria}")
            augmented[column] = extra_row[column]
        augmented_rows.append(augmented)
    return augmented_rows


def augment_phage_rows_with_features(
    phage_rows: Sequence[Mapping[str, str]],
    *,
    extra_phage_feature_rows: Sequence[Mapping[str, object]],
    extra_phage_feature_columns: Sequence[str],
) -> list[dict[str, object]]:
    if not extra_phage_feature_columns:
        return [dict(row) for row in phage_rows]
    phage_feature_index = {str(row["phage"]): dict(row) for row in extra_phage_feature_rows}
    augmented_rows: list[dict[str, object]] = []
    for row in phage_rows:
        augmented = dict(row)
        extra_row = phage_feature_index.get(str(row["phage"]))
        if extra_row is None:
            raise KeyError(f"Missing deployable phage feature row for phage {row['phage']}")
        for column in extra_phage_feature_columns:
            if column not in extra_row:
                raise KeyError(f"Missing deployable phage feature column {column} for phage {row['phage']}")
            augmented[column] = extra_row[column]
        augmented_rows.append(augmented)
    return augmented_rows


def fit_calibrator_from_cv_rows(
    rows: Sequence[Mapping[str, object]],
    feature_space: train_v1_binary_classifier.FeatureSpace,
    *,
    lightgbm_params: Mapping[str, object],
    random_state: int,
    calibration_fold: int,
) -> tuple[IsotonicRegression, int]:
    fold_datasets = train_v1_binary_classifier.prepare_fold_datasets(rows, feature_space)
    lightgbm_factory = lambda params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=random_state,
    )
    cv_rows = train_v1_binary_classifier.score_rows_with_cv_predictions(
        fold_datasets,
        estimator_factory=lightgbm_factory,
        best_params=lightgbm_params,
        probability_column="lightgbm_probability",
    )
    calibration_rows = [
        row
        for row in cv_rows
        if str(row["split_holdout"]) == "train_non_holdout" and int(str(row["split_cv5_fold"])) == calibration_fold
    ]
    if not calibration_rows:
        raise ValueError(f"No calibration rows found for fold {calibration_fold}")

    x_calib = [float(row["lightgbm_probability"]) for row in calibration_rows]
    y_calib = [int(str(row["label_hard_any_lysis"])) for row in calibration_rows]
    if len(set(y_calib)) < 2:
        raise ValueError(f"Calibration fold {calibration_fold} does not contain both classes.")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(x_calib, y_calib)
    return calibrator, len(calibration_rows)


def build_reference_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    estimator: Any,
    vectorizer: DictVectorizer,
    calibrator: IsotonicRegression,
    feature_space: train_v1_binary_classifier.FeatureSpace,
) -> list[dict[str, object]]:
    feature_dicts = [
        train_v1_binary_classifier.build_feature_dict(
            row,
            categorical_columns=feature_space.categorical_columns,
            numeric_columns=feature_space.numeric_columns,
        )
        for row in rows
    ]
    probabilities_raw = train_v1_binary_classifier.predict_probabilities(estimator, vectorizer.transform(feature_dicts))
    probabilities_calibrated = [float(value) for value in calibrator.predict(probabilities_raw)]

    enriched_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row, raw_probability, calibrated_probability in zip(rows, probabilities_raw, probabilities_calibrated):
        enriched = {
            "pair_id": row["pair_id"],
            "bacteria": row["bacteria"],
            "phage": row["phage"],
            "split_holdout": row["split_holdout"],
            "split_cv5_fold": row["split_cv5_fold"],
            "label_hard_any_lysis": row["label_hard_any_lysis"],
            "pred_lightgbm_raw": safe_round(float(raw_probability)),
            "pred_lightgbm_isotonic": safe_round(float(calibrated_probability)),
        }
        grouped.setdefault(str(row["bacteria"]), []).append(enriched)

    for bacteria in sorted(grouped):
        ranked = sorted(
            grouped[bacteria],
            key=lambda row: (-float(row["pred_lightgbm_isotonic"]), str(row["phage"])),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_lightgbm_isotonic"] = rank
            enriched_rows.append(row)
    return enriched_rows


def build_model_bundle(
    *,
    st02_pair_table_path: Path,
    st03_split_assignments_path: Path,
    defense_subtypes_path: Path,
    phage_kmer_feature_path: Path,
    phage_kmer_svd_path: Path,
    output_dir: Path,
    lightgbm_params: Mapping[str, object],
    random_state: int,
    calibration_fold: int,
    extra_host_feature_rows: Sequence[Mapping[str, object]] = (),
    extra_host_feature_columns: Sequence[str] = (),
    extra_host_categorical_columns: Sequence[str] = (),
    extra_phage_feature_rows: Sequence[Mapping[str, object]] = (),
    extra_phage_feature_columns: Sequence[str] = (),
    extra_phage_categorical_columns: Sequence[str] = (),
    pair_feature_rows: Sequence[Mapping[str, object]] = (),
    pair_feature_columns: Sequence[str] = (),
    pair_feature_categorical_columns: Sequence[str] = (),
    bundle_task_id: str = "TL08",
    bundle_format_version: str = "tl08_genome_only_inference_bundle_v2",
) -> dict[str, Any]:
    logger.info("Starting TL08 generalized inference bundle build")
    ensure_directory(output_dir)

    defense_rows = read_defense_rows(defense_subtypes_path)
    host_feature_rows, host_feature_columns, _ = build_defense_feature_rows(defense_rows)
    # Filter host rows to bacteria that appear in the pair table. Defense subtypes
    # cover 404 strains but only 369 appear in ST02. This filter is symmetric: both
    # baseline and candidate bundles use the same st02_pair_table_path.
    st02_rows = read_csv_rows(st02_pair_table_path)
    # Filter host-side feature rows to the bacteria that actually appear in ST02 pairs so every
    # caller (including the TL13 baseline bundle built inside TL18) trains and reports counts on
    # the same effective host cohort instead of padding the bundle with unused panel hosts.
    target_bacteria = {str(row["bacteria"]) for row in st02_rows}
    host_feature_rows = [row for row in host_feature_rows if str(row["bacteria"]) in target_bacteria]
    host_feature_rows = augment_host_rows_with_features(
        host_feature_rows,
        extra_host_feature_rows=extra_host_feature_rows,
        extra_host_feature_columns=[*extra_host_categorical_columns, *extra_host_feature_columns],
    )
    defense_mask = build_defense_column_mask(defense_rows)

    split_rows = read_csv_rows(st03_split_assignments_path)
    phage_rows = read_csv_rows(phage_kmer_feature_path)
    if not phage_rows:
        raise ValueError(f"No phage feature rows found in {phage_kmer_feature_path}")
    phage_rows = augment_phage_rows_with_features(
        phage_rows,
        extra_phage_feature_rows=extra_phage_feature_rows,
        extra_phage_feature_columns=[*extra_phage_categorical_columns, *extra_phage_feature_columns],
    )
    combined_host_feature_columns = [*host_feature_columns, *extra_host_feature_columns]
    phage_feature_columns = [
        column for column in phage_rows[0].keys() if column not in {"phage", *extra_phage_categorical_columns}
    ]

    merged_rows = build_training_rows(
        st02_rows=st02_rows,
        split_rows=split_rows,
        host_rows=host_feature_rows,
        phage_rows=phage_rows,
    )
    merged_rows = augment_rows_with_pair_features(
        merged_rows,
        pair_feature_rows=pair_feature_rows,
        pair_feature_columns=[*pair_feature_categorical_columns, *pair_feature_columns],
    )
    feature_space = build_genome_only_feature_space(
        host_categorical_columns=extra_host_categorical_columns,
        host_feature_columns=combined_host_feature_columns,
        phage_categorical_columns=extra_phage_categorical_columns,
        phage_feature_columns=phage_feature_columns,
        pairwise_categorical_columns=pair_feature_categorical_columns,
        pairwise_feature_columns=pair_feature_columns,
    )
    calibrator, calibration_row_count = fit_calibrator_from_cv_rows(
        merged_rows,
        feature_space,
        lightgbm_params=lightgbm_params,
        random_state=random_state,
        calibration_fold=calibration_fold,
    )
    lightgbm_factory = lambda params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=random_state,
    )
    estimator, vectorizer, _, holdout_rows, holdout_probabilities = train_v1_binary_classifier.fit_final_estimator(
        merged_rows,
        feature_space,
        estimator_factory=lightgbm_factory,
        params=lightgbm_params,
        sample_weight_key="training_weight_v3",
    )

    holdout_y = [int(str(row["label_hard_any_lysis"])) for row in holdout_rows]
    holdout_raw_metrics = train_v1_binary_classifier.compute_binary_metrics(holdout_y, holdout_probabilities)
    holdout_calibrated = [float(value) for value in calibrator.predict(holdout_probabilities)]
    holdout_isotonic_metrics = train_v1_binary_classifier.compute_binary_metrics(holdout_y, holdout_calibrated)

    reference_prediction_rows = build_reference_prediction_rows(
        merged_rows,
        estimator=estimator,
        vectorizer=vectorizer,
        calibrator=calibrator,
        feature_space=feature_space,
    )

    bundle_path = output_dir / BUNDLE_FILENAME
    phage_svd_copy_path = output_dir / PHAGE_SVD_FILENAME
    defense_mask_path = output_dir / DEFENSE_MASK_FILENAME
    panel_defense_subtypes_copy_path = output_dir / PANEL_DEFENSE_SUBTYPES_FILENAME
    predictions_path = output_dir / PANEL_PREDICTIONS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    shutil.copy2(phage_kmer_svd_path, phage_svd_copy_path)
    shutil.copy2(defense_subtypes_path, panel_defense_subtypes_copy_path)
    joblib.dump(defense_mask, defense_mask_path)

    bundle_payload = {
        "format_version": bundle_format_version,
        "task_id": bundle_task_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "lightgbm_estimator": estimator,
        "feature_vectorizer": vectorizer,
        "isotonic_calibrator": calibrator,
        "feature_space": {
            "categorical_columns": list(feature_space.categorical_columns),
            "numeric_columns": list(feature_space.numeric_columns),
            "host_feature_columns": list([*extra_host_categorical_columns, *combined_host_feature_columns]),
            "phage_feature_columns": list([*extra_phage_categorical_columns, *phage_feature_columns]),
            "pairwise_feature_columns": list([*pair_feature_categorical_columns, *pair_feature_columns]),
        },
        "artifacts": {
            "phage_svd_filename": PHAGE_SVD_FILENAME,
            "defense_mask_filename": DEFENSE_MASK_FILENAME,
            "panel_defense_subtypes_filename": PANEL_DEFENSE_SUBTYPES_FILENAME,
            "panel_predictions_filename": PANEL_PREDICTIONS_FILENAME,
        },
        "runtime": {
            "defense_finder_models_dirname": DEFENSE_FINDER_MODELS_DIRNAME,
        },
        "training": {
            "lightgbm_params": dict(lightgbm_params),
            "random_state": random_state,
            "calibration_fold": calibration_fold,
            "calibration_row_count": calibration_row_count,
            "host_count": len(host_feature_rows),
            "pair_count": len(merged_rows),
        },
        "inputs": {
            "st02_pair_table": {"path": str(st02_pair_table_path), "sha256": _sha256(st02_pair_table_path)},
            "st03_split_assignments": {
                "path": str(st03_split_assignments_path),
                "sha256": _sha256(st03_split_assignments_path),
            },
            "defense_subtypes": {"path": str(defense_subtypes_path), "sha256": _sha256(defense_subtypes_path)},
            "phage_kmer_features": {"path": str(phage_kmer_feature_path), "sha256": _sha256(phage_kmer_feature_path)},
            "phage_kmer_svd": {"path": str(phage_kmer_svd_path), "sha256": _sha256(phage_kmer_svd_path)},
        },
        "holdout_metrics": {
            "raw": holdout_raw_metrics,
            "isotonic": holdout_isotonic_metrics,
        },
    }
    joblib.dump(bundle_payload, bundle_path)
    write_csv(
        predictions_path,
        [
            "pair_id",
            "bacteria",
            "phage",
            "split_holdout",
            "split_cv5_fold",
            "label_hard_any_lysis",
            "pred_lightgbm_raw",
            "pred_lightgbm_isotonic",
            "rank_lightgbm_isotonic",
        ],
        reference_prediction_rows,
    )
    write_json(
        manifest_path,
        {
            "task_id": bundle_task_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_path": str(bundle_path),
            "panel_predictions_path": str(predictions_path),
            "holdout_metrics": {
                "raw": holdout_raw_metrics,
                "isotonic": holdout_isotonic_metrics,
            },
            "training": bundle_payload["training"],
            "artifact_hashes": {
                "bundle": _sha256(bundle_path),
                "phage_svd": _sha256(phage_svd_copy_path),
                "defense_mask": _sha256(defense_mask_path),
                "panel_defense_subtypes": _sha256(panel_defense_subtypes_copy_path),
                "panel_predictions": _sha256(predictions_path),
            },
        },
    )
    logger.info("Completed TL08 generalized inference bundle build")
    return {
        "bundle_path": bundle_path,
        "panel_predictions_path": predictions_path,
        "manifest_path": manifest_path,
        "holdout_metrics": {
            "raw": holdout_raw_metrics,
            "isotonic": holdout_isotonic_metrics,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_prerequisite_outputs(args)
    lightgbm_params = load_locked_lightgbm_params(args.tg01_summary_path)
    build_model_bundle(
        st02_pair_table_path=args.st02_pair_table_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
        defense_subtypes_path=args.defense_subtypes_path,
        phage_kmer_feature_path=args.phage_kmer_feature_path,
        phage_kmer_svd_path=args.phage_kmer_svd_path,
        output_dir=args.output_dir,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
