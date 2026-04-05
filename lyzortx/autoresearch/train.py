#!/usr/bin/env python3
"""User-facing entry point for short AUTORESEARCH experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch import build_contract
from lyzortx.pipeline.autoresearch.prepare_cache import (
    CACHE_CONTRACT_ID,
    CACHE_MANIFEST_FILENAME,
    DEFAULT_OUTPUT_ROOT,
    DISALLOWED_SEARCH_SPLITS,
    PROVENANCE_MANIFEST_FILENAME,
    SCHEMA_MANIFEST_FILENAME,
    SCHEMA_MANIFEST_ID,
    SEARCH_PAIR_TABLE_ID,
    SLOT_FEATURE_TABLE_FILENAME,
    SLOT_FEATURES_FILENAME,
    SLOT_SPEC_BY_NAME,
    SUPPORTED_SEARCH_SPLITS,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")
DEFAULT_TRAIN_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "train_runs" / "ar07_baseline"

BASELINE_ID = "ar07_host_phage_encoder_lightgbm_pair_scorer_v1"
PRIMARY_SEARCH_METRIC = "roc_auc"
SECONDARY_REPORT_ONLY_METRICS = ("top3_hit_rate", "brier_score")
FIXED_SINGLE_GPU_WALL_CLOCK_BUDGET_SECONDS = 1800
HOST_EMBEDDING_DIMENSION = 8
PHAGE_EMBEDDING_DIMENSION = 8
PAIR_SCORER_RANDOM_STATE = 7
PAIR_SCORER_PARAMS = {
    "n_estimators": 64,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}
REQUIRED_BASELINE_SLOTS = (
    "host_surface",
    "host_typing",
    "host_stats",
    "phage_projection",
    "phage_stats",
)
OPTIONAL_ADDITIVE_ABLATION_SLOTS = ("host_defense",)


@dataclass(frozen=True)
class SlotArtifact:
    slot_name: str
    entity_key: str
    feature_columns: tuple[str, ...]
    frame: pd.DataFrame


@dataclass(frozen=True)
class CacheContext:
    cache_dir: Path
    cache_manifest: dict[str, Any]
    schema_manifest: dict[str, Any]
    provenance_manifest: dict[str, Any]
    contract_manifest: dict[str, Any]
    split_frames: dict[str, pd.DataFrame]
    slot_artifacts: dict[str, SlotArtifact]


@dataclass(frozen=True)
class FittedEntityEncoder:
    entity_key: str
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    preprocessor: ColumnTransformer
    projector: TruncatedSVD | None
    embedding_columns: tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Prepared AUTORESEARCH search cache directory from prepare.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_DIR,
        help="Directory for baseline metrics and prediction outputs.",
    )
    parser.add_argument(
        "--device-type",
        choices=("cpu", "gpu", "cuda"),
        default="cuda",
        help="LightGBM device type for the fixed-budget pair scorer. RunPod search should use one GPU.",
    )
    parser.add_argument(
        "--include-host-defense",
        action="store_true",
        help="Add the reserved host_defense block as an additive ablation on top of the adsorption-first baseline.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV artifact: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def enforce_budget(start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    if elapsed > FIXED_SINGLE_GPU_WALL_CLOCK_BUDGET_SECONDS:
        raise TimeoutError(
            "AUTORESEARCH train.py exceeded the fixed single-GPU wall-clock budget of "
            f"{FIXED_SINGLE_GPU_WALL_CLOCK_BUDGET_SECONDS} seconds."
        )


def validate_cache_manifest(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_manifest_path = cache_dir / CACHE_MANIFEST_FILENAME
    schema_manifest_path = cache_dir / SCHEMA_MANIFEST_FILENAME
    provenance_manifest_path = cache_dir / PROVENANCE_MANIFEST_FILENAME
    for path in (cache_manifest_path, schema_manifest_path, provenance_manifest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Prepared search cache is incomplete at {cache_dir}. Missing required manifest: {path.name}"
            )

    cache_manifest = read_json(cache_manifest_path)
    schema_manifest = read_json(schema_manifest_path)
    provenance_manifest = read_json(provenance_manifest_path)

    if cache_manifest.get("cache_contract_id") != CACHE_CONTRACT_ID:
        raise ValueError(
            "AUTORESEARCH cache contract mismatch: "
            f"expected {CACHE_CONTRACT_ID}, got {cache_manifest.get('cache_contract_id')!r}"
        )
    if schema_manifest.get("schema_manifest_id") != SCHEMA_MANIFEST_ID:
        raise ValueError(
            "AUTORESEARCH schema manifest mismatch: "
            f"expected {SCHEMA_MANIFEST_ID}, got {schema_manifest.get('schema_manifest_id')!r}"
        )
    if schema_manifest.get("pair_table_id") != SEARCH_PAIR_TABLE_ID:
        raise ValueError(
            "AUTORESEARCH pair-table contract mismatch: "
            f"expected {SEARCH_PAIR_TABLE_ID}, got {schema_manifest.get('pair_table_id')!r}"
        )

    exported_splits = tuple(sorted(cache_manifest["pair_tables"].keys()))
    if exported_splits != tuple(sorted(SUPPORTED_SEARCH_SPLITS)):
        raise ValueError(
            "AUTORESEARCH cache does not match the frozen split contract: "
            f"expected {SUPPORTED_SEARCH_SPLITS}, got {exported_splits}"
        )
    if tuple(sorted(schema_manifest["supported_search_splits"])) != tuple(sorted(SUPPORTED_SEARCH_SPLITS)):
        raise ValueError("Top-level AUTORESEARCH schema manifest no longer advertises the frozen search splits.")
    if tuple(sorted(schema_manifest["disallowed_search_splits"])) != tuple(sorted(DISALLOWED_SEARCH_SPLITS)):
        raise ValueError("Top-level AUTORESEARCH schema manifest no longer seals the holdout split.")
    if not schema_manifest["pair_table_contract"].get("labels_read_only", False):
        raise ValueError("AUTORESEARCH labels must stay read-only inside the search sandbox.")

    holdout_named_paths = sorted(path.name for path in (cache_dir / "search_pairs").glob("*holdout*"))
    if holdout_named_paths:
        raise ValueError(
            "Sealed holdout labels leaked into the AUTORESEARCH search cache: " + ", ".join(holdout_named_paths)
        )
    if any(split in cache_manifest["pair_tables"] for split in DISALLOWED_SEARCH_SPLITS):
        raise ValueError("Sealed holdout split leaked into the AUTORESEARCH search cache.")

    return cache_manifest, schema_manifest, provenance_manifest


def load_contract_manifest(provenance_manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(provenance_manifest["source_contract"]["pair_contract_manifest_path"]))
    if not path.exists():
        raise FileNotFoundError(f"AUTORESEARCH pair-contract manifest not found: {path}")
    contract_manifest = read_json(path)
    split_hashes = contract_manifest.get("split_contract", {}).get("split_hashes", {})
    for split_name in (*SUPPORTED_SEARCH_SPLITS, *DISALLOWED_SEARCH_SPLITS):
        if split_name not in split_hashes:
            raise ValueError(f"AUTORESEARCH pair-contract manifest is missing split hash metadata for {split_name}.")
    return contract_manifest


def validate_split_pair_table(
    *,
    split_name: str,
    frame: pd.DataFrame,
    contract_manifest: dict[str, Any],
) -> None:
    if frame.empty:
        raise ValueError(f"AUTORESEARCH split {split_name} is empty.")
    if set(frame["split"]) != {split_name}:
        raise ValueError(f"AUTORESEARCH split {split_name} contains rows from a different split.")
    if set(frame["retained_for_autoresearch"]) - {"0", "1"}:
        raise ValueError(f"AUTORESEARCH split {split_name} has invalid retained_for_autoresearch values.")

    retained = frame.loc[frame["retained_for_autoresearch"] == "1"].copy()
    if retained.empty:
        raise ValueError(f"AUTORESEARCH split {split_name} has zero retained rows.")
    if any(label not in {"0", "1"} for label in retained["label_any_lysis"]):
        raise ValueError(f"AUTORESEARCH split {split_name} has non-binary retained labels.")

    expected_hash = contract_manifest["split_contract"]["split_hashes"][split_name]["retained_pair_ids_sha256"]
    actual_hash = build_contract.sha256_strings(sorted(str(value) for value in retained["pair_id"]))
    if actual_hash != expected_hash:
        raise ValueError(
            f"AUTORESEARCH split membership drift detected for {split_name}: retained pair IDs no longer match AR01."
        )


def load_slot_artifact(
    *,
    cache_dir: Path,
    schema_manifest: dict[str, Any],
    cache_manifest: dict[str, Any],
    slot_name: str,
    require_materialized_features: bool,
) -> SlotArtifact:
    if slot_name not in SLOT_SPEC_BY_NAME:
        raise ValueError(f"Unknown AUTORESEARCH slot requested by train.py: {slot_name}")

    slot_spec = SLOT_SPEC_BY_NAME[slot_name]
    top_level_slot = schema_manifest["feature_slots"].get(slot_name)
    if top_level_slot is None:
        raise ValueError(f"Top-level AUTORESEARCH schema manifest is missing slot {slot_name}.")
    if top_level_slot["join_keys"] != slot_spec.join_keys:
        raise ValueError(f"AUTORESEARCH slot {slot_name} changed join keys.")
    if top_level_slot["column_family_prefix"] != slot_spec.column_prefix:
        raise ValueError(f"AUTORESEARCH slot {slot_name} changed its reserved column prefix.")

    slot_summary = cache_manifest["feature_slots"].get(slot_name)
    if slot_summary is None:
        raise ValueError(f"AUTORESEARCH cache manifest is missing slot {slot_name}.")

    schema_path = Path(str(slot_summary["schema_manifest_path"]))
    if not schema_path.exists():
        raise FileNotFoundError(f"AUTORESEARCH slot schema manifest not found for {slot_name}: {schema_path}")
    slot_schema = read_json(schema_path)
    if slot_schema.get("schema_manifest_id") != SCHEMA_MANIFEST_ID:
        raise ValueError(f"AUTORESEARCH slot {slot_name} no longer points at the frozen schema manifest.")
    if slot_schema.get("slot_name") != slot_name:
        raise ValueError(
            f"AUTORESEARCH slot schema mismatch: expected {slot_name}, got {slot_schema.get('slot_name')}."
        )
    if slot_schema.get("join_keys") != slot_spec.join_keys:
        raise ValueError(f"AUTORESEARCH slot {slot_name} schema no longer matches the frozen join key.")

    feature_columns = tuple(str(column) for column in slot_schema["reserved_feature_columns"])
    if tuple(str(column) for column in top_level_slot["reserved_feature_columns"]) != feature_columns:
        raise ValueError(f"AUTORESEARCH slot {slot_name} bypassed the frozen top-level cache schema.")

    slot_dir = cache_dir / "feature_slots" / slot_name
    features_path = resolve_slot_features_path(slot_dir=slot_dir, slot_name=slot_name)
    if require_materialized_features and not feature_columns:
        raise ValueError(f"AUTORESEARCH baseline requires materialized columns for slot {slot_name}.")
    if not feature_columns:
        return SlotArtifact(
            slot_name=slot_name, entity_key=slot_spec.entity_key, feature_columns=(), frame=pd.DataFrame()
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"AUTORESEARCH slot {slot_name} is missing its materialized feature table at {features_path}."
        )

    frame = load_csv_frame(features_path)
    expected_header = [slot_spec.entity_key, *feature_columns]
    actual_header = list(frame.columns)
    if actual_header != expected_header:
        raise ValueError(
            f"AUTORESEARCH slot {slot_name} features no longer match the frozen cache schema: "
            f"expected {expected_header}, got {actual_header}"
        )
    if frame.empty:
        raise ValueError(f"AUTORESEARCH slot {slot_name} has zero feature rows.")

    return SlotArtifact(
        slot_name=slot_name,
        entity_key=slot_spec.entity_key,
        feature_columns=feature_columns,
        frame=frame,
    )


def resolve_slot_features_path(*, slot_dir: Path, slot_name: str) -> Path:
    features_path = slot_dir / SLOT_FEATURES_FILENAME
    if features_path.exists():
        return features_path
    if slot_name == "host_defense":
        legacy_features_path = slot_dir / SLOT_FEATURE_TABLE_FILENAME
        if legacy_features_path.exists():
            return legacy_features_path
    return features_path


def load_and_validate_cache(*, cache_dir: Path, include_host_defense: bool) -> CacheContext:
    cache_manifest, schema_manifest, provenance_manifest = validate_cache_manifest(cache_dir)
    contract_manifest = load_contract_manifest(provenance_manifest)

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, summary in cache_manifest["pair_tables"].items():
        frame = load_csv_frame(Path(str(summary["path"])))
        validate_split_pair_table(split_name=split_name, frame=frame, contract_manifest=contract_manifest)
        split_frames[split_name] = frame

    selected_slots = list(REQUIRED_BASELINE_SLOTS)
    if include_host_defense:
        selected_slots.append("host_defense")

    slot_artifacts = {
        slot_name: load_slot_artifact(
            cache_dir=cache_dir,
            schema_manifest=schema_manifest,
            cache_manifest=cache_manifest,
            slot_name=slot_name,
            require_materialized_features=True,
        )
        for slot_name in selected_slots
    }

    return CacheContext(
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        schema_manifest=schema_manifest,
        provenance_manifest=provenance_manifest,
        contract_manifest=contract_manifest,
        split_frames=split_frames,
        slot_artifacts=slot_artifacts,
    )


def detect_feature_types(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[list[str], list[str], pd.DataFrame]:
    typed = frame.copy()
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        numeric_candidate = pd.to_numeric(typed[column], errors="coerce")
        non_empty_mask = typed[column].astype(str) != ""
        if bool(non_empty_mask.any()) and bool(numeric_candidate[non_empty_mask].notna().all()):
            typed[column] = numeric_candidate.astype(float)
            numeric_columns.append(column)
        else:
            typed[column] = typed[column].astype(str)
            categorical_columns.append(column)
    if not numeric_columns and not categorical_columns:
        raise ValueError("AUTORESEARCH baseline requires at least one feature column.")
    return numeric_columns, categorical_columns, typed


def fit_entity_encoder(
    *,
    frame: pd.DataFrame,
    entity_key: str,
    embedding_dimension: int,
    random_state: int,
    encoder_label: str,
) -> FittedEntityEncoder:
    feature_columns = tuple(column for column in frame.columns if column != entity_key)
    numeric_columns, categorical_columns, typed = detect_feature_types(frame, feature_columns)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    transformed = preprocessor.fit_transform(typed.loc[:, feature_columns])
    if not sparse.issparse(transformed):
        transformed = np.asarray(transformed, dtype=float)

    max_components = min(embedding_dimension, transformed.shape[0] - 1, transformed.shape[1] - 1)
    projector: TruncatedSVD | None = None
    if max_components >= 1:
        projector = TruncatedSVD(n_components=max_components, random_state=random_state)
        projector.fit(transformed)
        embedding_width = max_components
    else:
        embedding_width = transformed.shape[1]
        LOGGER.info(
            "%s encoder has too little rank for SVD; using the preprocessed feature space directly (%d columns).",
            encoder_label,
            embedding_width,
        )

    embedding_columns = tuple(f"{encoder_label}_embedding_{index:02d}" for index in range(embedding_width))
    return FittedEntityEncoder(
        entity_key=entity_key,
        feature_columns=feature_columns,
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        preprocessor=preprocessor,
        projector=projector,
        embedding_columns=embedding_columns,
    )


def transform_entity_frame(frame: pd.DataFrame, *, encoder: FittedEntityEncoder) -> pd.DataFrame:
    typed = frame.copy()
    for column in encoder.numeric_columns:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype(float)
    for column in encoder.categorical_columns:
        typed[column] = typed[column].astype(str)

    transformed = encoder.preprocessor.transform(typed.loc[:, encoder.feature_columns])
    if encoder.projector is not None:
        embedding_matrix = encoder.projector.transform(transformed)
    else:
        embedding_matrix = (
            transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed, dtype=float)
        )
    embedding_frame = pd.DataFrame(embedding_matrix, columns=list(encoder.embedding_columns))
    embedding_frame.insert(0, encoder.entity_key, typed[encoder.entity_key].tolist())
    return embedding_frame


def build_entity_feature_table(
    slot_artifacts: dict[str, SlotArtifact],
    *,
    slot_names: Sequence[str],
    entity_key: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for slot_name in slot_names:
        artifact = slot_artifacts[slot_name]
        if artifact.entity_key != entity_key:
            raise ValueError(f"AUTORESEARCH slot {slot_name} does not join on {entity_key}.")
        frame = artifact.frame.copy()
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=entity_key, how="outer", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError(f"AUTORESEARCH entity table for {entity_key} is empty.")
    return merged


def build_pair_design_matrix(
    pair_frame: pd.DataFrame,
    *,
    host_embeddings: pd.DataFrame,
    phage_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    merged = pair_frame.merge(host_embeddings, on="bacteria", how="left", validate="many_to_one")
    merged = merged.merge(phage_embeddings, on="phage", how="left", validate="many_to_one")

    if merged.filter(regex=r"^(host_|phage_)embedding_").isna().any().any():
        raise ValueError("AUTORESEARCH pair table could not join all required host/phage embeddings.")

    host_columns = [column for column in merged.columns if column.startswith("host_embedding_")]
    phage_columns = [column for column in merged.columns if column.startswith("phage_embedding_")]
    interaction_width = min(len(host_columns), len(phage_columns))
    for index in range(interaction_width):
        host_column = host_columns[index]
        phage_column = phage_columns[index]
        merged[f"pair_abs_{index:02d}"] = (merged[host_column].astype(float) - merged[phage_column].astype(float)).abs()
        merged[f"pair_prod_{index:02d}"] = merged[host_column].astype(float) * merged[phage_column].astype(float)
    return merged


def build_pair_scorer(device_type: str) -> LGBMClassifier:
    estimator_params: dict[str, Any] = {
        **PAIR_SCORER_PARAMS,
        "objective": "binary",
        "random_state": PAIR_SCORER_RANDOM_STATE,
        "n_jobs": 1,
        "verbosity": -1,
        "device_type": device_type,
    }
    if device_type == "cpu":
        estimator_params["deterministic"] = True
        estimator_params["force_col_wise"] = True
    return LGBMClassifier(**estimator_params)


def compute_top3_hit_rate(scored_rows: pd.DataFrame) -> float:
    if scored_rows.empty:
        raise ValueError("Cannot compute top-3 hit rate on an empty score table.")

    hit_flags: list[float] = []
    for _, bacteria_rows in scored_rows.sort_values(
        ["bacteria", "prediction", "phage"], ascending=[True, False, True]
    ).groupby(
        "bacteria",
        sort=True,
    ):
        top_rows = bacteria_rows.head(3)
        hit_flags.append(float((top_rows["label_any_lysis"].astype(int) == 1).any()))
    return float(sum(hit_flags) / len(hit_flags))


def safe_float(value: float) -> float:
    return float(f"{value:.6f}")


def run_baseline(
    *,
    context: CacheContext,
    device_type: str,
    include_host_defense: bool,
    start_time: float,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    enforce_budget(start_time)

    host_slots = ["host_surface", "host_typing", "host_stats"]
    if include_host_defense:
        host_slots.append("host_defense")
    phage_slots = ["phage_projection", "phage_stats"]

    host_table = build_entity_feature_table(context.slot_artifacts, slot_names=host_slots, entity_key="bacteria")
    phage_table = build_entity_feature_table(context.slot_artifacts, slot_names=phage_slots, entity_key="phage")

    train_pairs = (
        context.split_frames[build_contract.TRAIN_SPLIT]
        .loc[lambda frame: frame["retained_for_autoresearch"] == "1"]
        .copy()
    )
    inner_val_pairs = (
        context.split_frames[build_contract.INNER_VAL_SPLIT]
        .loc[lambda frame: frame["retained_for_autoresearch"] == "1"]
        .copy()
    )

    train_bacteria = sorted(train_pairs["bacteria"].unique().tolist())
    train_phages = sorted(train_pairs["phage"].unique().tolist())
    host_encoder = fit_entity_encoder(
        frame=host_table.loc[host_table["bacteria"].isin(train_bacteria)].reset_index(drop=True),
        entity_key="bacteria",
        embedding_dimension=HOST_EMBEDDING_DIMENSION,
        random_state=PAIR_SCORER_RANDOM_STATE,
        encoder_label="host",
    )
    phage_encoder = fit_entity_encoder(
        frame=phage_table.loc[phage_table["phage"].isin(train_phages)].reset_index(drop=True),
        entity_key="phage",
        embedding_dimension=PHAGE_EMBEDDING_DIMENSION,
        random_state=PAIR_SCORER_RANDOM_STATE,
        encoder_label="phage",
    )

    host_embeddings = transform_entity_frame(host_table, encoder=host_encoder)
    phage_embeddings = transform_entity_frame(phage_table, encoder=phage_encoder)

    train_design = build_pair_design_matrix(
        train_pairs, host_embeddings=host_embeddings, phage_embeddings=phage_embeddings
    )
    inner_val_design = build_pair_design_matrix(
        inner_val_pairs,
        host_embeddings=host_embeddings,
        phage_embeddings=phage_embeddings,
    )

    feature_columns = [
        column
        for column in train_design.columns
        if column.startswith(("host_embedding_", "phage_embedding_", "pair_abs_", "pair_prod_"))
    ]
    if not feature_columns:
        raise ValueError("AUTORESEARCH baseline constructed zero pair features.")

    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    y_inner = inner_val_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    enforce_budget(start_time)
    estimator = build_pair_scorer(device_type=device_type)
    estimator.fit(train_design[feature_columns], y_train, sample_weight=sample_weight)
    enforce_budget(start_time)

    predictions = estimator.predict_proba(inner_val_design[feature_columns])[:, 1]
    scored_inner_val = inner_val_design.loc[:, ["pair_id", "bacteria", "phage", "label_any_lysis"]].copy()
    scored_inner_val["prediction"] = predictions
    scored_inner_val["rank_within_bacteria"] = (
        scored_inner_val.sort_values(["bacteria", "prediction", "phage"], ascending=[True, False, True])
        .groupby("bacteria")
        .cumcount()
        .add(1)
        .sort_index()
    )

    if len(np.unique(y_inner)) < 2:
        raise ValueError("AUTORESEARCH inner_val split must contain both classes to produce ROC-AUC.")

    metrics = {
        "roc_auc": safe_float(float(roc_auc_score(y_inner, predictions))),
        "top3_hit_rate": safe_float(compute_top3_hit_rate(scored_inner_val)),
        "brier_score": safe_float(float(brier_score_loss(y_inner, predictions))),
    }
    summary = {
        "baseline_id": BASELINE_ID,
        "host_encoder": {
            "type": "fitted_preprocessor_plus_truncated_svd",
            "slots": host_slots,
            "embedding_dimension": len(host_encoder.embedding_columns),
            "numeric_columns": list(host_encoder.numeric_columns),
            "categorical_columns": list(host_encoder.categorical_columns),
        },
        "phage_encoder": {
            "type": "fitted_preprocessor_plus_truncated_svd",
            "slots": phage_slots,
            "embedding_dimension": len(phage_encoder.embedding_columns),
            "numeric_columns": list(phage_encoder.numeric_columns),
            "categorical_columns": list(phage_encoder.categorical_columns),
        },
        "pair_scorer": {
            "type": "lightgbm_binary_classifier",
            "device_type": device_type,
            "params": dict(PAIR_SCORER_PARAMS),
        },
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    prediction_rows = [
        {
            "pair_id": str(row["pair_id"]),
            "bacteria": str(row["bacteria"]),
            "phage": str(row["phage"]),
            "label_any_lysis": int(row["label_any_lysis"]),
            "prediction": safe_float(float(row["prediction"])),
            "rank_within_bacteria": int(row["rank_within_bacteria"]),
        }
        for row in scored_inner_val.sort_values(["bacteria", "rank_within_bacteria", "phage"]).to_dict(orient="records")
    ]
    return summary, prediction_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    start_time = time.monotonic()

    context = load_and_validate_cache(cache_dir=args.cache_dir, include_host_defense=args.include_host_defense)
    LOGGER.info("AUTORESEARCH train sandbox validated cache at %s", args.cache_dir)
    LOGGER.info("Exported splits: %s", ", ".join(SUPPORTED_SEARCH_SPLITS))
    LOGGER.info(
        "Baseline feature slots: %s",
        ", ".join(REQUIRED_BASELINE_SLOTS + (() if not args.include_host_defense else ("host_defense",))),
    )
    LOGGER.info("Reserved-but-ignored baseline ablations: %s", ", ".join(OPTIONAL_ADDITIVE_ABLATION_SLOTS))
    LOGGER.info("Fixed single-GPU wall-clock budget: %d seconds", FIXED_SINGLE_GPU_WALL_CLOCK_BUDGET_SECONDS)
    LOGGER.info("train.py is the short-loop experiment surface; cache rebuilding belongs in prepare.py.")

    baseline_summary, prediction_rows = run_baseline(
        context=context,
        device_type=args.device_type,
        include_host_defense=args.include_host_defense,
        start_time=start_time,
    )
    elapsed_seconds = safe_float(time.monotonic() - start_time)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "ar07_inner_val_predictions.csv"
    summary_path = args.output_dir / "ar07_baseline_summary.json"
    write_rows(prediction_path, prediction_rows)
    output_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "cache_contract_id": CACHE_CONTRACT_ID,
        "schema_manifest_id": SCHEMA_MANIFEST_ID,
        "search_mutation_boundary": {
            "mutable_file": "lyzortx/autoresearch/train.py",
            "labels_out_of_bounds": True,
            "split_membership_out_of_bounds": True,
            "feature_extraction_out_of_bounds": True,
            "evaluation_contract_out_of_bounds": True,
        },
        "search_runtime_contract": {
            "device_type": args.device_type,
            "fixed_wall_clock_budget_seconds": FIXED_SINGLE_GPU_WALL_CLOCK_BUDGET_SECONDS,
            "cache_build_outside_budget": True,
            "cache_rebuild_allowed_in_train": False,
        },
        "baseline_contract": {
            "minimum_slots": list(REQUIRED_BASELINE_SLOTS),
            "host_defense_active": args.include_host_defense,
            "host_defense_role": "reserved_additive_ablation",
            "primary_metric": PRIMARY_SEARCH_METRIC,
            "secondary_report_only_metrics": list(SECONDARY_REPORT_ONLY_METRICS),
        },
        "artifacts": {
            "inner_val_predictions_path": str(prediction_path),
        },
        "inner_val_metrics": dict(baseline_summary["metrics"]),
        "search_metric": {
            "name": PRIMARY_SEARCH_METRIC,
            "value": baseline_summary["metrics"][PRIMARY_SEARCH_METRIC],
        },
        "elapsed_seconds": elapsed_seconds,
        **baseline_summary,
    }
    write_json(summary_path, output_summary)

    LOGGER.info(
        "Primary search metric: inner_val_%s=%.6f", PRIMARY_SEARCH_METRIC, output_summary["search_metric"]["value"]
    )
    LOGGER.info(
        "Secondary report-only metrics: top3_hit_rate=%.6f, brier_score=%.6f",
        *[output_summary["inner_val_metrics"][metric_name] for metric_name in SECONDARY_REPORT_ONLY_METRICS],
    )
    LOGGER.info("Wrote AUTORESEARCH baseline summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
