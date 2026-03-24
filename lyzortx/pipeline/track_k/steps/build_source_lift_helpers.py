#!/usr/bin/env python3
"""Shared helpers for Track K cumulative lift measurements."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    CATEGORICAL_FEATURE_COLUMNS as V0_CATEGORICAL_FEATURE_COLUMNS,
)
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    NUMERIC_FEATURE_COLUMNS as V0_NUMERIC_FEATURE_COLUMNS,
)
from lyzortx.pipeline.track_g.steps.run_feature_subset_sweep import build_feature_blocks
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import FeatureSpace
from lyzortx.pipeline.track_i.steps.build_external_training_cohorts import (
    TRAINING_ARM_INDEX,
    first_training_arm_for_source,
    source_family_for_source,
)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        categorical_columns=tuple(dict.fromkeys(categorical_columns)),
        numeric_columns=tuple(dict.fromkeys(numeric_columns)),
        track_c_additional_columns=tuple(feature_space.track_c_additional_columns),
        track_d_columns=tuple(feature_space.track_d_columns),
        track_e_columns=tuple(feature_space.track_e_columns),
    )


def load_source_training_rows(
    feature_rows: Sequence[Mapping[str, object]],
    cohort_rows: Sequence[Mapping[str, str]],
    source_system: str,
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
        if normalized.get("source_system") != source_system:
            continue
        counts["cohort_rows"] += 1
        if normalized.get("external_label_include_in_training") != "1":
            counts["excluded_rows"] += 1
            continue

        feature_row = feature_rows_by_pair.get(normalized["pair_id"])
        if feature_row is None:
            counts["missing_feature_rows"] += 1
            continue
        if (
            feature_row.get("split_holdout") != "train_non_holdout"
            or str(feature_row.get("is_hard_trainable", "")) != "1"
        ):
            counts["non_training_split_rows"] += 1
            continue

        merged = dict(feature_row)
        merged.update(
            {
                "source_system": source_system,
                "source_family": source_family_for_source(source_system),
                "external_label_confidence_tier": normalized.get("external_label_confidence_tier", ""),
                "external_label_confidence_score": normalized.get("external_label_confidence_score", ""),
                "external_label_training_weight": normalized.get("external_label_training_weight", ""),
                "external_label_include_in_training": normalized.get("external_label_include_in_training", ""),
                "integration_status": "external_enhancer",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": -1,
                "is_hard_trainable": "1",
                "training_origin": source_system,
                "first_training_arm": first_training_arm_for_source(source_system),
                "first_training_arm_index": TRAINING_ARM_INDEX[first_training_arm_for_source(source_system)],
                "effective_training_weight": float(normalized.get("external_label_training_weight", "0") or 0.0),
            }
        )
        if normalized.get("label_hard_any_lysis", ""):
            merged["label_hard_any_lysis"] = normalized["label_hard_any_lysis"]
        if normalized.get("label_strict_confidence_tier", ""):
            merged["label_strict_confidence_tier"] = normalized["label_strict_confidence_tier"]
        augmented_rows.append(merged)
        counts["joined_rows"] += 1

    return augmented_rows, dict(counts)


def load_source_training_rows_for_systems(
    feature_rows: Sequence[Mapping[str, object]],
    cohort_rows: Sequence[Mapping[str, str]],
    source_systems: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    combined_rows: List[Dict[str, object]] = []
    combined_counts: Dict[str, Dict[str, int]] = {}
    for source_system in canonical_source_systems(source_systems):
        rows, counts = load_source_training_rows(feature_rows, cohort_rows, source_system)
        combined_rows.extend(rows)
        combined_counts[source_system] = counts
    return combined_rows, combined_counts


def canonical_source_systems(source_systems: Sequence[str]) -> Tuple[str, ...]:
    ordered: List[str] = []
    for source_system in source_systems:
        if source_system not in ordered:
            ordered.append(source_system)
    return tuple(ordered)


def source_systems_label(source_systems: Sequence[str]) -> str:
    return "|".join(canonical_source_systems(source_systems))


def arm_name_for_source_systems(source_systems: Sequence[str]) -> str:
    canonical = canonical_source_systems(source_systems)
    if canonical == ("internal",):
        return "internal_only"
    if canonical and canonical[0] == "internal":
        return "internal_plus_" + "_plus_".join(canonical[1:])
    return "_plus_".join(canonical)


def build_training_rows(
    merged_rows: Sequence[Mapping[str, object]],
    source_rows_by_system: Mapping[str, Sequence[Mapping[str, object]]],
    source_systems: Sequence[str],
) -> List[Mapping[str, object]]:
    training_rows = list(merged_rows)
    for source_system in canonical_source_systems(source_systems):
        if source_system == "internal":
            continue
        training_rows.extend(source_rows_by_system.get(source_system, []))
    return training_rows


def classify_lift(
    *,
    delta_roc_auc: Optional[float],
    delta_top3: Optional[float],
    delta_brier: Optional[float],
    tolerance: float,
) -> str:
    if delta_roc_auc is None or delta_top3 is None or delta_brier is None:
        return "pending"

    improved = delta_roc_auc > tolerance or delta_top3 > tolerance or delta_brier < -tolerance
    worsened = delta_roc_auc < -tolerance or delta_top3 < -tolerance or delta_brier > tolerance
    if improved and not worsened:
        return "adds"
    if worsened and not improved:
        return "hurts"
    return "neutral"


def load_previous_best_source_systems(previous_manifest_path: Path) -> List[str]:
    if not previous_manifest_path.exists():
        return []

    with previous_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    best_source_systems = manifest.get("best_source_systems")
    if isinstance(best_source_systems, list) and best_source_systems:
        return [str(source_system) for source_system in best_source_systems if str(source_system) != "internal"]

    if manifest.get("lift_decision") == "keep_vhrdb_for_followup_arms":
        return ["vhrdb"]

    lift_assessment = manifest.get("lift_assessment")
    base_source_systems = manifest.get("base_source_systems")
    augmented_source_systems = manifest.get("augmented_source_systems")
    if isinstance(base_source_systems, list) and isinstance(augmented_source_systems, list):
        if lift_assessment == "adds":
            return [
                str(source_system) for source_system in augmented_source_systems if str(source_system) != "internal"
            ]
        if lift_assessment in {"hurts", "neutral"}:
            return [str(source_system) for source_system in base_source_systems if str(source_system) != "internal"]

    return []


def write_output_tables(
    *,
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, object]],
    summary_fieldnames: Sequence[str],
    ranking_rows: Sequence[Mapping[str, object]],
    ranking_fieldnames: Sequence[str],
    manifest_path: Path,
    manifest_payload: Mapping[str, object],
    summary_filename: str,
    rankings_filename: str,
) -> None:
    ensure_directory(output_dir)
    write_csv(output_dir / summary_filename, fieldnames=summary_fieldnames, rows=summary_rows)
    write_csv(output_dir / rankings_filename, fieldnames=ranking_fieldnames, rows=ranking_rows)
    write_json(manifest_path, manifest_payload)
