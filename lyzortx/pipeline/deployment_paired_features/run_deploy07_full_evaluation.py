#!/usr/bin/env python3
"""DEPLOY07: run full deployment-paired feature derivation, retraining, and parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.deployment_paired_features import derive_host_defense_features as deploy02
from lyzortx.pipeline.deployment_paired_features import derive_host_surface_features as deploy03
from lyzortx.pipeline.deployment_paired_features import derive_host_typing_features as deploy04
from lyzortx.pipeline.deployment_paired_features.download_picard_assemblies import (
    DEFAULT_ASSEMBLY_DIR,
    EXPECTED_ASSEMBLY_COUNT,
    download_picard_assemblies,
)
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import build_defense_feature_rows, slugify_token
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier as tg01
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle as tl08
from lyzortx.pipeline.track_l.steps import deployable_tl17_runtime as tl17_runtime
from lyzortx.pipeline.track_l.steps import retrain_mechanistic_v1_model as tl05
from lyzortx.pipeline.track_l.steps.build_tl17_phage_compatibility_preprocessor import (
    DEFAULT_CACHED_ANNOTATIONS_DIR,
    DEFAULT_EXPECTED_PANEL_COUNT,
    DEFAULT_OUTPUT_DIR as TL17_OUTPUT_DIR,
    DEFAULT_PHAGE_FASTA_DIR,
    DEFAULT_PHAGE_METADATA_PATH,
    FAMILY_METADATA_FILENAME,
    FASTA_INVENTORY_FILENAME,
    REFERENCE_FASTA_FILENAME,
    REFERENCE_METADATA_FILENAME,
)
from lyzortx.pipeline.track_l.steps.deployable_tl18_host_runtime import (
    TL15_CATEGORICAL_COLUMNS,
    TL15_NUMERIC_COLUMNS,
    TL16_CATEGORICAL_COLUMNS,
    TL16_NUMERIC_COLUMNS,
    build_tl15_panel_training_rows,
    build_tl15_runtime_payload,
    build_tl16_panel_training_rows,
    build_tl16_runtime_payload,
    project_tl15_host_features,
    project_tl16_host_features,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/deploy07_full_evaluation")
DEFAULT_DEPLOY_OUTPUT_ROOT = Path("lyzortx/generated_outputs/deployment_paired_features")
DEFAULT_DEFENSE_COUNTS_PATH = Path("lyzortx/data/deployment_paired_features/403_host_defense_gene_counts.csv")
DEFAULT_ST02_PAIR_TABLE_PATH = tl08.DEFAULT_ST02_PAIR_TABLE_PATH
DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH = tl08.DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH
DEFAULT_TG01_SUMMARY_PATH = tl08.DEFAULT_TG01_SUMMARY_PATH
DEFAULT_CALIBRATION_FOLD = tl08.DEFAULT_CALIBRATION_FOLD
DEFAULT_RANDOM_STATE = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_VALIDATION_FASTA_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_VALIDATION_HOSTS: tuple[str, ...] = ("55989", "EDL933", "LF82")
BASELINE_ARM_ID = "tl18_legacy_encoding_reference"
DEPLOYMENT_ARM_ID = "deploy07_deployment_paired"
BASELINE_ARM_TYPE = "baseline"
DEPLOYMENT_ARM_TYPE = "deployment"
VALIDATION_ARM_TYPES = frozenset({BASELINE_ARM_TYPE, DEPLOYMENT_ARM_TYPE})
KEY_COLUMNS = frozenset({"bacteria", "phage"})
PHAGE_BASELINE_FAMILY_COUNT_COLUMN = "tl17_rbp_family_count"
SCHEMA_VALIDATION_FILENAME = "schema_validation_summary.json"
DEFENSE_DISAGREEMENT_FILENAME = "defense_disagreement_summary.json"
METRIC_COMPARISON_FILENAME = "deploy07_metric_comparison.csv"
METRIC_SUMMARY_FILENAME = "deploy07_metric_summary.json"
PARITY_RESULTS_FILENAME = "deploy07_validation_parity.csv"
PARITY_SUMMARY_FILENAME = "deploy07_validation_parity_summary.json"
WINNING_BUNDLE_FILENAME = "deploy07_winning_model_bundle.joblib"
WINNING_BUNDLE_MANIFEST_FILENAME = "deploy07_winning_model_bundle_manifest.json"
WINNING_PREDICTIONS_FILENAME = "deploy07_validation_predictions.csv"
SURFACE_AGGREGATED_FILENAME = "403_host_surface_features.csv"
TYPING_AGGREGATED_FILENAME = "403_host_typing_features.csv"
PHAGE_AGGREGATED_FILENAME = "96_panel_phage_rbp_features.csv"
PHAGE_RUNTIME_FILENAME = "tl17_runtime_payload.joblib"
DEFENSE_DISAGREEMENT_GATE_MAX = 3.0


@dataclass(frozen=True)
class FeatureBlock:
    block_id: str
    key_column: str
    rows: list[dict[str, object]]
    schema: dict[str, Any]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    display_name: str
    feature_space: tg01.FeatureSpace
    estimator: Any
    vectorizer: Any
    calibrator: Any
    merged_rows: list[dict[str, object]]
    holdout_prediction_rows: list[dict[str, object]]
    holdout_binary_metrics: dict[str, Any]
    holdout_top3_metrics: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", type=Path, default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--deploy-output-root", type=Path, default=DEFAULT_DEPLOY_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--defense-counts-path", type=Path, default=DEFAULT_DEFENSE_COUNTS_PATH)
    parser.add_argument("--st02-pair-table-path", type=Path, default=DEFAULT_ST02_PAIR_TABLE_PATH)
    parser.add_argument("--st03-split-assignments-path", type=Path, default=DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH)
    parser.add_argument("--tg01-summary-path", type=Path, default=DEFAULT_TG01_SUMMARY_PATH)
    parser.add_argument("--validation-fasta-dir", type=Path, default=DEFAULT_VALIDATION_FASTA_DIR)
    parser.add_argument("--host-workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--calibration-fold", type=int, default=DEFAULT_CALIBRATION_FOLD)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_panel_defense_rows(path: Path = deploy02.DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH) -> list[dict[str, str]]:
    return deploy02._read_panel_defense_rows(path)


def _schema_column_names(schema: Mapping[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _load_aggregated_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    missing_mask = frame.isnull()
    if missing_mask.values.any():
        missing_columns = ", ".join(sorted(str(column) for column in frame.columns[missing_mask.any()]))
        raise ValueError(f"Unexpected NaN values in {path}: {missing_columns}")
    if frame.empty:
        raise ValueError(f"No rows found in {path}")
    return frame.to_dict("records")


def _build_numeric_columns(schema: Mapping[str, Any]) -> tuple[str, ...]:
    numeric_dtypes = {"float64", "int64"}
    return tuple(
        str(column["name"])
        for column in schema["columns"]
        if str(column["name"]) != str(schema["key_column"]) and str(column["dtype"]) in numeric_dtypes
    )


def _build_categorical_columns(schema: Mapping[str, Any]) -> tuple[str, ...]:
    if "categorical_columns" in schema:
        return tuple(str(column) for column in schema["categorical_columns"])
    return tuple(
        str(column["name"])
        for column in schema["columns"]
        if str(column["name"]) != str(schema["key_column"]) and str(column["dtype"]) == "string"
    )


def validate_schema_columns(
    *,
    block_id: str,
    key_column: str,
    schema: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    schema_columns = _schema_column_names(schema)
    actual_columns = list(rows[0].keys())
    missing_columns = sorted(column for column in schema_columns if column not in actual_columns)
    extra_columns = sorted(column for column in actual_columns if column not in schema_columns)
    if missing_columns:
        raise ValueError(f"{block_id} CSV is missing schema columns: {', '.join(missing_columns)}")
    return {
        "block_id": block_id,
        "key_column": key_column,
        "schema_column_count": len(schema_columns),
        "csv_column_count": len(actual_columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
    }


def validate_no_duplicate_block_columns(blocks: Sequence[FeatureBlock]) -> list[dict[str, object]]:
    seen: dict[str, str] = {}
    audit_rows: list[dict[str, object]] = []
    for block in blocks:
        duplicate_columns: list[str] = []
        for column in _schema_column_names(block.schema):
            if column == block.key_column:
                continue
            owner = seen.get(column)
            if owner is not None:
                duplicate_columns.append(column)
                continue
            seen[column] = block.block_id
        audit_rows.append(
            {
                "block_id": block.block_id,
                "key_column": block.key_column,
                "duplicate_columns": duplicate_columns,
                "non_key_schema_column_count": len(_schema_column_names(block.schema)) - 1,
            }
        )
        if duplicate_columns:
            raise ValueError(
                f"Duplicate feature columns detected across blocks for {block.block_id}: {', '.join(sorted(duplicate_columns))}"
            )
    return audit_rows


def select_winning_arm_id_from_auc_ci(
    arm_metrics: Sequence[Mapping[str, object]],
    *,
    baseline_arm_id: str,
    deployment_arm_id: str,
) -> str:
    deployment_row = next(row for row in arm_metrics if str(row["arm_id"]) == deployment_arm_id)
    if float(deployment_row["auc_delta_ci_low_vs_baseline"]) > 0.0:
        return deployment_arm_id
    return baseline_arm_id


def select_arm_type_for_winning_arm_id(winning_arm_id: str) -> str:
    if winning_arm_id == BASELINE_ARM_ID:
        return BASELINE_ARM_TYPE
    if winning_arm_id == DEPLOYMENT_ARM_ID:
        return DEPLOYMENT_ARM_TYPE
    raise ValueError(f"Unknown winning arm_id: {winning_arm_id!r}")


def build_baseline_phage_rows_from_continuous(
    rows: Sequence[Mapping[str, object]],
    *,
    family_score_columns: Sequence[str],
    hit_count_column: str,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    converted_rows: list[dict[str, object]] = []
    family_columns = tuple(str(column) for column in family_score_columns)
    for row in rows:
        converted: dict[str, object] = {"phage": str(row["phage"])}
        family_count = 0
        for column in family_columns:
            present = int(float(row.get(column, 0.0) or 0.0) > 0.0)
            converted[column] = present
            family_count += present
        converted[hit_count_column] = int(float(row.get(hit_count_column, 0) or 0))
        converted[PHAGE_BASELINE_FAMILY_COUNT_COLUMN] = family_count
        converted_rows.append(converted)
    return converted_rows, (*family_columns, hit_count_column, PHAGE_BASELINE_FAMILY_COUNT_COLUMN)


def _binarize_defense_row(count_row: Mapping[str, object], subtype_columns: Sequence[str]) -> dict[str, object]:
    encoded_row: dict[str, object] = {"bacteria": str(count_row["bacteria"])}
    defense_diversity = 0
    abi_burden = 0
    has_crispr = 0
    for subtype in subtype_columns:
        count = int(float(count_row.get(subtype, 0) or 0))
        value = 1 if count > 0 else 0
        encoded_row[f"host_defense_subtype_{_slugify_defense_token(subtype)}"] = value
        defense_diversity += value
        if subtype.startswith("Abi"):
            abi_burden += value
        if subtype.startswith("CAS_"):
            has_crispr = max(has_crispr, value)
    encoded_row["host_defense_diversity"] = defense_diversity
    encoded_row["host_defense_has_crispr"] = has_crispr
    encoded_row["host_defense_abi_burden"] = abi_burden
    return encoded_row


def _slugify_defense_token(value: str) -> str:
    return slugify_token(value)


def build_baseline_defense_rows_from_counts(
    count_rows: Sequence[Mapping[str, object]],
    *,
    subtype_columns: Sequence[str],
) -> list[dict[str, object]]:
    rows = [_binarize_defense_row(row, subtype_columns) for row in count_rows]
    rows.sort(key=lambda row: str(row["bacteria"]))
    return rows


def _merge_rows_by_key(
    blocks: Sequence[FeatureBlock],
    *,
    merged_block_id: str,
    key_column: str,
) -> FeatureBlock:
    if not blocks:
        raise ValueError(f"No blocks provided for {merged_block_id}")
    keyed_rows = [{str(row[key_column]): dict(row) for row in block.rows} for block in blocks]
    shared_keys = set(keyed_rows[0].keys())
    for row_map in keyed_rows[1:]:
        shared_keys &= set(row_map.keys())
    if not shared_keys:
        raise ValueError(f"No overlapping {key_column} values found while merging {merged_block_id}")

    merged_rows: list[dict[str, object]] = []
    for key in sorted(shared_keys):
        merged = {key_column: key}
        for block, row_map in zip(blocks, keyed_rows):
            source = row_map[key]
            for column in _schema_column_names(block.schema):
                if column == key_column:
                    continue
                if column in merged:
                    raise ValueError(f"Duplicate merged column {column} while building {merged_block_id}")
                merged[column] = source[column]
        merged_rows.append(merged)

    categorical_columns = tuple(
        column for block in blocks for column in block.categorical_columns if column != key_column
    )
    numeric_columns = tuple(column for block in blocks for column in block.numeric_columns if column != key_column)
    schema_columns = [{"name": key_column, "dtype": "string"}]
    for block in blocks:
        for column in block.schema["columns"]:
            if str(column["name"]) == key_column:
                continue
            schema_columns.append(dict(column))
    merged_schema = {
        "feature_block": merged_block_id,
        "key_column": key_column,
        "column_count": len(schema_columns),
        "columns": schema_columns,
        "categorical_columns": list(categorical_columns),
    }
    return FeatureBlock(
        block_id=merged_block_id,
        key_column=key_column,
        rows=merged_rows,
        schema=merged_schema,
        categorical_columns=tuple(categorical_columns),
        numeric_columns=tuple(numeric_columns),
    )


def _read_single_row_csv(path: Path) -> dict[str, object]:
    rows = _load_aggregated_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return dict(rows[0])


def _write_aggregated_block_csv(
    *,
    aggregated_path: Path,
    schema: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> None:
    columns = _schema_column_names(schema)
    write_csv(aggregated_path, columns, [{column: row.get(column, "") for column in columns} for row in rows])


def _process_host_surface(
    assembly_path: Path,
    *,
    output_root: Path,
    runtime_inputs: deploy03.SurfaceRuntimeInputs,
) -> None:
    bacteria = assembly_path.stem
    host_dir = output_root / bacteria
    feature_path = host_dir / deploy03.PER_HOST_FEATURES_FILENAME
    if feature_path.exists():
        return
    deploy03.derive_host_surface_features(
        assembly_path,
        bacteria_id=bacteria,
        output_dir=host_dir,
        runtime_inputs=runtime_inputs,
    )


def _process_host_typing(assembly_path: Path, *, output_root: Path) -> None:
    bacteria = assembly_path.stem
    host_dir = output_root / bacteria
    feature_path = host_dir / deploy04.PER_HOST_FEATURES_FILENAME
    if feature_path.exists():
        return
    deploy04.derive_host_typing_features(
        assembly_path,
        bacteria_id=bacteria,
        output_dir=host_dir,
    )


def _run_threaded_host_jobs(
    assemblies: Sequence[Path],
    *,
    worker_count: int,
    job_name: str,
    fn: Any,
) -> None:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    start = datetime.now(timezone.utc)
    LOGGER.info("Starting %s for %d hosts with %d workers", job_name, len(assemblies), worker_count)
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(fn, assembly_path): assembly_path.stem for assembly_path in assemblies}
        for future in as_completed(futures):
            bacteria = futures[future]
            future.result()
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == len(assemblies):
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                LOGGER.info(
                    "%s progress: %d/%d hosts complete (%.0fs elapsed)", job_name, completed, len(assemblies), elapsed
                )
                LOGGER.debug("%s last completed host: %s", job_name, bacteria)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    LOGGER.info("Completed %s in %.0fs", job_name, elapsed)


def run_all_host_surface_features(
    *,
    assembly_dir: Path,
    output_root: Path,
    worker_count: int,
) -> tuple[Path, Path]:
    ensure_directory(output_root)
    schema_path = output_root / deploy03.SCHEMA_MANIFEST_FILENAME
    aggregated_path = output_root / SURFACE_AGGREGATED_FILENAME
    assemblies = sorted(assembly_dir.glob("*.fasta"))
    if len(assemblies) != EXPECTED_ASSEMBLY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ASSEMBLY_COUNT} Picard assemblies in {assembly_dir}, found {len(assemblies)}"
        )
    runtime_inputs = deploy03.prepare_host_surface_runtime_inputs(
        assets_output_dir=output_root / "assets",
        picard_metadata_path=deploy03.tl15.DEFAULT_PICARD_METADATA_PATH,
        o_type_output_path=deploy03.tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
        o_type_allele_path=deploy03.tl15.DEFAULT_O_TYPE_ALLELE_PATH,
        o_antigen_override_path=deploy03.tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
        abc_capsule_profile_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
        omp_reference_path=deploy03.tl15.DEFAULT_OMP_REFERENCE_PATH,
    )
    schema = deploy03.build_host_surface_schema(runtime_inputs.capsule_profile_names)
    write_json(schema_path, schema)
    _run_threaded_host_jobs(
        assemblies,
        worker_count=worker_count,
        job_name="DEPLOY03 host surface derivation",
        fn=lambda assembly_path: _process_host_surface(
            assembly_path, output_root=output_root, runtime_inputs=runtime_inputs
        ),
    )
    rows = [
        _read_single_row_csv(output_root / assembly.stem / deploy03.PER_HOST_FEATURES_FILENAME)
        for assembly in assemblies
    ]
    rows.sort(key=lambda row: str(row["bacteria"]))
    _write_aggregated_block_csv(aggregated_path=aggregated_path, schema=schema, rows=rows)
    return aggregated_path, schema_path


def run_all_host_typing_features(
    *,
    assembly_dir: Path,
    output_root: Path,
    worker_count: int,
) -> tuple[Path, Path]:
    ensure_directory(output_root)
    schema = deploy04.build_host_typing_schema()
    schema_path = output_root / deploy04.SCHEMA_MANIFEST_FILENAME
    aggregated_path = output_root / TYPING_AGGREGATED_FILENAME
    write_json(schema_path, schema)
    assemblies = sorted(assembly_dir.glob("*.fasta"))
    if len(assemblies) != EXPECTED_ASSEMBLY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ASSEMBLY_COUNT} Picard assemblies in {assembly_dir}, found {len(assemblies)}"
        )
    _run_threaded_host_jobs(
        assemblies,
        worker_count=worker_count,
        job_name="DEPLOY04 host typing derivation",
        fn=lambda assembly_path: _process_host_typing(assembly_path, output_root=output_root),
    )
    rows = [
        _read_single_row_csv(output_root / assembly.stem / deploy04.PER_HOST_FEATURES_FILENAME)
        for assembly in assemblies
    ]
    rows.sort(key=lambda row: str(row["bacteria"]))
    _write_aggregated_block_csv(aggregated_path=aggregated_path, schema=schema, rows=rows)
    return aggregated_path, schema_path


def run_all_panel_phage_features(
    *,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    ensure_directory(output_root)
    reference_rows, family_rows = tl17_runtime.build_reference_proteins(
        phage_metadata_path=DEFAULT_PHAGE_METADATA_PATH,
        fna_dir=DEFAULT_PHAGE_FASTA_DIR,
        cached_annotations_dir=DEFAULT_CACHED_ANNOTATIONS_DIR,
        expected_panel_count=DEFAULT_EXPECTED_PANEL_COUNT,
    )
    schema = tl17_runtime.build_projection_schema(family_rows)
    schema_path = output_root / tl17_runtime.SCHEMA_MANIFEST_FILENAME
    write_json(schema_path, schema)
    reference_fasta_path = output_root / REFERENCE_FASTA_FILENAME
    tl17_runtime.write_reference_fasta(reference_rows, reference_fasta_path)
    tl17_runtime.write_reference_metadata_csv(reference_rows, output_root / REFERENCE_METADATA_FILENAME)
    tl17_runtime.write_family_metadata_csv(family_rows, output_root / FAMILY_METADATA_FILENAME)
    write_csv(
        output_root / FASTA_INVENTORY_FILENAME,
        ["phage", "fasta_path", "sha256"],
        tl17_runtime.build_fasta_inventory_rows(
            phage_metadata_path=DEFAULT_PHAGE_METADATA_PATH,
            fna_dir=DEFAULT_PHAGE_FASTA_DIR,
            expected_panel_count=DEFAULT_EXPECTED_PANEL_COUNT,
        ),
    )
    runtime_payload = tl17_runtime.build_runtime_payload(
        family_rows=family_rows,
        reference_rows=reference_rows,
        min_query_coverage=tl17_runtime.DEFAULT_MMSEQS_MIN_QUERY_COVERAGE,
    )
    runtime_path = output_root / PHAGE_RUNTIME_FILENAME
    joblib.dump(runtime_payload, runtime_path)
    rows = tl17_runtime.project_panel_feature_rows(
        phage_metadata_path=DEFAULT_PHAGE_METADATA_PATH,
        fna_dir=DEFAULT_PHAGE_FASTA_DIR,
        expected_panel_count=DEFAULT_EXPECTED_PANEL_COUNT,
        runtime_payload=runtime_payload,
        reference_fasta_path=reference_fasta_path,
        scratch_root=output_root / ".scratch_runtime",
    )
    rows.sort(key=lambda row: str(row["phage"]))
    aggregated_path = output_root / PHAGE_AGGREGATED_FILENAME
    _write_aggregated_block_csv(aggregated_path=aggregated_path, schema=schema, rows=rows)
    return aggregated_path, schema_path, runtime_path


def compute_full_defense_disagreement_summary(
    *,
    defense_count_rows: Sequence[Mapping[str, object]],
    panel_rows: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
) -> dict[str, object]:
    subtype_columns = tuple(str(column) for column in schema["retained_subtype_columns"])
    panel_by_bacteria = {str(row["bacteria"]): dict(row) for row in panel_rows}
    comparisons: list[dict[str, Any]] = []
    for row in defense_count_rows:
        bacteria = str(row["bacteria"])
        panel_row = panel_by_bacteria.get(bacteria)
        if panel_row is None:
            continue
        comparisons.append(
            deploy02.compare_host_defense_to_panel(
                row,
                panel_row,
                subtype_columns=subtype_columns,
            )
        )
    if not comparisons:
        raise ValueError("No overlapping bacteria found for defense disagreement comparison.")
    average_disagreement = sum(int(item["disagreement_subtype_count"]) for item in comparisons) / len(comparisons)
    top_hosts = sorted(comparisons, key=lambda item: int(item["disagreement_subtype_count"]), reverse=True)[:10]
    return {
        "host_count_compared": len(comparisons),
        "average_disagreement_subtype_count": safe_round(average_disagreement),
        "max_disagreement_subtype_count": max(int(item["disagreement_subtype_count"]) for item in comparisons),
        "top_disagreement_hosts": top_hosts,
    }


def _ensure_prerequisites(args: argparse.Namespace) -> None:
    if args.skip_prerequisites:
        return
    tl08.ensure_prerequisite_outputs(
        argparse.Namespace(
            skip_prerequisites=False,
            st02_pair_table_path=args.st02_pair_table_path,
            st03_split_assignments_path=args.st03_split_assignments_path,
            defense_subtypes_path=deploy02.DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
            phage_kmer_feature_path=tl08.DEFAULT_PHAGE_KMER_FEATURE_PATH,
            phage_kmer_svd_path=tl08.DEFAULT_PHAGE_KMER_SVD_PATH,
            tg01_summary_path=args.tg01_summary_path,
            output_dir=TL17_OUTPUT_DIR,
            calibration_fold=args.calibration_fold,
            random_state=args.random_state,
        )
    )


def _load_locked_lightgbm_params(path: Path) -> dict[str, Any]:
    return tl08.load_locked_lightgbm_params(path)


def _merge_pair_rows(
    *,
    st02_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
    host_rows: Sequence[Mapping[str, object]],
    phage_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return tl08.build_training_rows(
        st02_rows=st02_rows,
        split_rows=split_rows,
        host_rows=host_rows,
        phage_rows=phage_rows,
    )


def _build_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    estimator: Any,
    vectorizer: Any,
    calibrator: Any,
    feature_space: tg01.FeatureSpace,
) -> list[dict[str, object]]:
    feature_dicts = [
        tg01.build_feature_dict(
            row,
            categorical_columns=feature_space.categorical_columns,
            numeric_columns=feature_space.numeric_columns,
        )
        for row in rows
    ]
    raw_probabilities = tg01.predict_probabilities(estimator, vectorizer.transform(feature_dicts))
    calibrated = [float(value) for value in calibrator.predict(raw_probabilities)]
    prediction_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row, raw_probability, calibrated_probability in zip(rows, raw_probabilities, calibrated):
        scored = dict(row)
        scored["predicted_probability_raw"] = safe_round(float(raw_probability))
        scored["predicted_probability"] = safe_round(float(calibrated_probability))
        grouped.setdefault(str(row["bacteria"]), []).append(scored)
    for bacteria in sorted(grouped):
        ranked = sorted(grouped[bacteria], key=lambda item: (-float(item["predicted_probability"]), str(item["phage"])))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            prediction_rows.append(row)
    return prediction_rows


def evaluate_arm(
    *,
    arm_id: str,
    display_name: str,
    merged_rows: Sequence[Mapping[str, object]],
    host_categorical_columns: Sequence[str],
    host_numeric_columns: Sequence[str],
    phage_categorical_columns: Sequence[str],
    phage_numeric_columns: Sequence[str],
    lightgbm_params: Mapping[str, object],
    random_state: int,
    calibration_fold: int,
) -> ArmResult:
    feature_space = tl08.build_genome_only_feature_space(
        host_categorical_columns=host_categorical_columns,
        host_feature_columns=host_numeric_columns,
        phage_categorical_columns=phage_categorical_columns,
        phage_feature_columns=phage_numeric_columns,
    )
    calibrator, _ = tl08.fit_calibrator_from_cv_rows(
        merged_rows,
        feature_space,
        lightgbm_params=lightgbm_params,
        random_state=random_state,
        calibration_fold=calibration_fold,
    )
    lightgbm_factory = lambda params, seed_offset: tg01.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=random_state,
    )
    estimator, vectorizer, _, holdout_rows, _ = tg01.fit_final_estimator(
        merged_rows,
        feature_space,
        estimator_factory=lightgbm_factory,
        params=lightgbm_params,
        sample_weight_key="training_weight_v3",
    )
    holdout_prediction_rows = _build_prediction_rows(
        holdout_rows,
        estimator=estimator,
        vectorizer=vectorizer,
        calibrator=calibrator,
        feature_space=feature_space,
    )
    holdout_y = [int(str(row["label_hard_any_lysis"])) for row in holdout_prediction_rows]
    holdout_binary_metrics = tg01.compute_binary_metrics(
        holdout_y,
        [float(row["predicted_probability"]) for row in holdout_prediction_rows],
    )
    holdout_top3_metrics = tg01.compute_top3_hit_rate(
        holdout_prediction_rows,
        probability_key="predicted_probability",
    )
    LOGGER.info(
        "%s holdout metrics: ROC-AUC=%s top3=%s brier=%s",
        arm_id,
        holdout_binary_metrics["roc_auc"],
        holdout_top3_metrics["top3_hit_rate_all_strains"],
        holdout_binary_metrics["brier_score"],
    )
    return ArmResult(
        arm_id=arm_id,
        display_name=display_name,
        feature_space=feature_space,
        estimator=estimator,
        vectorizer=vectorizer,
        calibrator=calibrator,
        merged_rows=list(merged_rows),
        holdout_prediction_rows=holdout_prediction_rows,
        holdout_binary_metrics=holdout_binary_metrics,
        holdout_top3_metrics=holdout_top3_metrics,
    )


def summarize_arm_metrics(
    *,
    result: ArmResult,
    baseline_result: ArmResult,
    bootstrap_summary: Mapping[str, Mapping[str, tl05.BootstrapMetricCI]],
) -> dict[str, object]:
    row_bootstrap = bootstrap_summary[result.arm_id]
    delta_bootstrap = bootstrap_summary.get(f"{result.arm_id}__delta_vs_{baseline_result.arm_id}")
    is_baseline = result.arm_id == baseline_result.arm_id
    auc = result.holdout_binary_metrics["roc_auc"]
    baseline_auc = baseline_result.holdout_binary_metrics["roc_auc"]
    top3 = result.holdout_top3_metrics["top3_hit_rate_all_strains"]
    baseline_top3 = baseline_result.holdout_top3_metrics["top3_hit_rate_all_strains"]
    brier = result.holdout_binary_metrics["brier_score"]
    baseline_brier = baseline_result.holdout_binary_metrics["brier_score"]
    return {
        "arm_id": result.arm_id,
        "arm_label": result.display_name,
        "holdout_roc_auc": auc,
        "holdout_roc_auc_ci_low": row_bootstrap["holdout_roc_auc"].ci_low,
        "holdout_roc_auc_ci_high": row_bootstrap["holdout_roc_auc"].ci_high,
        "holdout_top3_hit_rate_all_strains": top3,
        "holdout_top3_hit_rate_all_strains_ci_low": row_bootstrap["holdout_top3_hit_rate_all_strains"].ci_low,
        "holdout_top3_hit_rate_all_strains_ci_high": row_bootstrap["holdout_top3_hit_rate_all_strains"].ci_high,
        "holdout_brier_score": brier,
        "holdout_brier_score_ci_low": row_bootstrap["holdout_brier_score"].ci_low,
        "holdout_brier_score_ci_high": row_bootstrap["holdout_brier_score"].ci_high,
        "auc_delta_vs_baseline": 0.0 if is_baseline else safe_round(float(auc) - float(baseline_auc)),
        "auc_delta_ci_low_vs_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_roc_auc"].ci_low,
        "auc_delta_ci_high_vs_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_roc_auc"].ci_high,
        "top3_delta_vs_baseline": 0.0 if is_baseline else safe_round(float(top3) - float(baseline_top3)),
        "top3_delta_ci_low_vs_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_top3_hit_rate_all_strains"].ci_low,
        "top3_delta_ci_high_vs_baseline": 0.0
        if is_baseline
        else delta_bootstrap["holdout_top3_hit_rate_all_strains"].ci_high,
        "brier_improvement_vs_baseline": 0.0 if is_baseline else safe_round(float(baseline_brier) - float(brier)),
        "brier_improvement_ci_low_vs_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_brier_score"].ci_low,
        "brier_improvement_ci_high_vs_baseline": 0.0 if is_baseline else delta_bootstrap["holdout_brier_score"].ci_high,
    }


def _make_feature_block(
    *,
    block_id: str,
    key_column: str,
    rows: Sequence[Mapping[str, object]],
    schema: Mapping[str, Any],
) -> FeatureBlock:
    return FeatureBlock(
        block_id=block_id,
        key_column=key_column,
        rows=[dict(row) for row in rows],
        schema=dict(schema),
        categorical_columns=_build_categorical_columns(schema),
        numeric_columns=_build_numeric_columns(schema),
    )


def _build_baseline_blocks(
    *,
    defense_count_rows: Sequence[Mapping[str, object]],
    defense_schema: Mapping[str, Any],
    phage_deployment_rows: Sequence[Mapping[str, object]],
    phage_schema: Mapping[str, Any],
) -> tuple[FeatureBlock, FeatureBlock]:
    subtype_columns = tuple(str(column) for column in defense_schema["retained_subtype_columns"])
    baseline_defense_rows = build_baseline_defense_rows_from_counts(defense_count_rows, subtype_columns=subtype_columns)
    defense_mask = build_defense_feature_rows(_load_panel_defense_rows())[2]["column_mask"]
    defense_schema_baseline = {
        "feature_block": "baseline_host_defense",
        "key_column": "bacteria",
        "column_count": len(["bacteria", *defense_mask["ordered_feature_columns"]]),
        "columns": [{"name": "bacteria", "dtype": "string"}]
        + [{"name": column, "dtype": "int64"} for column in defense_mask["ordered_feature_columns"]],
    }
    baseline_phage_rows, baseline_phage_numeric_columns = build_baseline_phage_rows_from_continuous(
        phage_deployment_rows,
        family_score_columns=tuple(str(column) for column in phage_schema["family_score_columns"]),
        hit_count_column=str(phage_schema["reference_hit_count_column"]),
    )
    baseline_phage_schema = {
        "feature_block": "baseline_phage_rbp",
        "key_column": "phage",
        "column_count": len(["phage", *baseline_phage_numeric_columns]),
        "columns": [{"name": "phage", "dtype": "string"}]
        + [{"name": column, "dtype": "int64"} for column in baseline_phage_numeric_columns],
    }
    return (
        _make_feature_block(
            block_id="baseline_host_defense",
            key_column="bacteria",
            rows=baseline_defense_rows,
            schema=defense_schema_baseline,
        ),
        _make_feature_block(
            block_id="baseline_phage_rbp",
            key_column="phage",
            rows=baseline_phage_rows,
            schema=baseline_phage_schema,
        ),
    )


def _build_baseline_host_surface_block(target_bacteria: Sequence[str]) -> FeatureBlock:
    rows = build_tl15_panel_training_rows(
        picard_metadata_path=deploy03.tl15.DEFAULT_PICARD_METADATA_PATH,
        receptor_cluster_path=deploy03.tl15.DEFAULT_RECEPTOR_CLUSTER_PATH,
        target_bacteria=target_bacteria,
    )
    schema = {
        "feature_block": "baseline_host_surface",
        "key_column": "bacteria",
        "column_count": len(["bacteria", *TL15_CATEGORICAL_COLUMNS, *TL15_NUMERIC_COLUMNS]),
        "columns": [{"name": "bacteria", "dtype": "string"}]
        + [{"name": column, "dtype": "string"} for column in TL15_CATEGORICAL_COLUMNS]
        + [{"name": column, "dtype": "float64"} for column in TL15_NUMERIC_COLUMNS],
        "categorical_columns": list(TL15_CATEGORICAL_COLUMNS),
    }
    return _make_feature_block(block_id="baseline_host_surface", key_column="bacteria", rows=rows, schema=schema)


def _build_baseline_host_typing_block(target_bacteria: Sequence[str]) -> FeatureBlock:
    rows = build_tl16_panel_training_rows(
        picard_metadata_path=deploy04.DEFAULT_PANEL_METADATA_PATH,
        target_bacteria=target_bacteria,
    )
    schema = {
        "feature_block": "baseline_host_typing",
        "key_column": "bacteria",
        "column_count": len(["bacteria", *TL16_CATEGORICAL_COLUMNS, *TL16_NUMERIC_COLUMNS]),
        "columns": [{"name": "bacteria", "dtype": "string"}]
        + [{"name": column, "dtype": "string"} for column in TL16_CATEGORICAL_COLUMNS]
        + [{"name": column, "dtype": "int64"} for column in TL16_NUMERIC_COLUMNS],
        "categorical_columns": list(TL16_CATEGORICAL_COLUMNS),
    }
    return _make_feature_block(block_id="baseline_host_typing", key_column="bacteria", rows=rows, schema=schema)


def _build_validation_pairs(hosts: Sequence[str], phages: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bacteria in hosts:
        for phage in phages:
            rows.append(
                {
                    "pair_id": f"validation::{bacteria}::{phage}",
                    "bacteria": bacteria,
                    "phage": phage,
                    "split_holdout": "validation_runtime",
                    "split_cv5_fold": -1,
                    "label_hard_any_lysis": 0,
                    "is_hard_trainable": 0,
                    "training_weight_v3": 1.0,
                }
            )
    return rows


def _derive_validation_host_block_rows(
    *,
    bundle_payload: Mapping[str, object],
    validation_fasta_dir: Path,
) -> FeatureBlock:
    arm_type = str(bundle_payload["arm_type"])
    if arm_type not in VALIDATION_ARM_TYPES:
        raise ValueError(f"Unknown arm_type in bundle: {arm_type!r}")
    hosts = tuple(str(host) for host in bundle_payload["validation_hosts"])
    rows: list[dict[str, object]] = []
    if arm_type == BASELINE_ARM_TYPE:
        tl15_payload = bundle_payload["baseline_runtime"]["tl15_payload"]
        tl16_payload = bundle_payload["baseline_runtime"]["tl16_payload"]
        bundle_dir = Path(bundle_payload["bundle_dir"])
        defense_lookup = {str(row["bacteria"]): dict(row) for row in bundle_payload["baseline_runtime"]["defense_rows"]}
        for host in hosts:
            assembly_path = validation_fasta_dir / f"{host}.fasta"
            if not assembly_path.exists():
                raise FileNotFoundError(f"Missing validation FASTA: {assembly_path}")
            defense_row = dict(defense_lookup[host])
            surface_row = project_tl15_host_features(
                assembly_path,
                bacteria=host,
                bundle_dir=bundle_dir,
                runtime_payload=tl15_payload,
                output_dir=bundle_dir / "runtime_validation" / host / "baseline_tl15",
            )
            typing_row = project_tl16_host_features(
                assembly_path,
                bacteria=host,
                bundle_dir=bundle_dir,
                runtime_payload=tl16_payload,
                output_dir=bundle_dir / "runtime_validation" / host / "baseline_tl16",
            )
            merged = {"bacteria": host}
            for source in (defense_row, surface_row, typing_row):
                for column, value in source.items():
                    if column == "bacteria":
                        continue
                    merged[column] = value
            rows.append(merged)
        training_block = bundle_payload["training_blocks"]["baseline_host"]
        schema = dict(training_block["schema"])
        return _make_feature_block(block_id="runtime_validation_host", key_column="bacteria", rows=rows, schema=schema)

    surface_runtime_output = Path(bundle_payload["bundle_dir"]) / "runtime_validation" / "deployment_surface"
    typing_runtime_output = Path(bundle_payload["bundle_dir"]) / "runtime_validation" / "deployment_typing"
    defense_lookup = {str(row["bacteria"]): dict(row) for row in bundle_payload["deployment_runtime"]["defense_rows"]}
    surface_runtime_inputs = deploy03.prepare_host_surface_runtime_inputs(
        assets_output_dir=Path(bundle_payload["bundle_dir"]) / "runtime_validation" / "deployment_surface_assets",
        picard_metadata_path=deploy03.tl15.DEFAULT_PICARD_METADATA_PATH,
        o_type_output_path=deploy03.tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
        o_type_allele_path=deploy03.tl15.DEFAULT_O_TYPE_ALLELE_PATH,
        o_antigen_override_path=deploy03.tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
        abc_capsule_profile_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
        omp_reference_path=deploy03.tl15.DEFAULT_OMP_REFERENCE_PATH,
    )
    for host in hosts:
        assembly_path = validation_fasta_dir / f"{host}.fasta"
        if not assembly_path.exists():
            raise FileNotFoundError(f"Missing validation FASTA: {assembly_path}")
        defense_row = dict(defense_lookup[host])
        surface_row = deploy03.derive_host_surface_features(
            assembly_path,
            bacteria_id=host,
            output_dir=surface_runtime_output / host,
            runtime_inputs=surface_runtime_inputs,
        )["feature_row"]
        typing_row = deploy04.derive_host_typing_features(
            assembly_path,
            bacteria_id=host,
            output_dir=typing_runtime_output / host,
        )["feature_row"]
        merged = {"bacteria": host}
        for source in (defense_row, surface_row, typing_row):
            for column, value in source.items():
                if column == "bacteria":
                    continue
                merged[column] = value
        rows.append(merged)
    training_block = bundle_payload["training_blocks"]["deployment_host"]
    schema = dict(training_block["schema"])
    return _make_feature_block(block_id="runtime_validation_host", key_column="bacteria", rows=rows, schema=schema)


def _derive_validation_phage_block_rows(bundle_payload: Mapping[str, object]) -> FeatureBlock:
    arm_type = str(bundle_payload["arm_type"])
    if arm_type not in VALIDATION_ARM_TYPES:
        raise ValueError(f"Unknown arm_type in bundle: {arm_type!r}")
    runtime_payload = joblib.load(Path(bundle_payload["phage_runtime"]["runtime_path"]))
    reference_fasta_path = Path(bundle_payload["phage_runtime"]["reference_fasta_path"])
    phage_rows = tl17_runtime.project_panel_feature_rows(
        phage_metadata_path=DEFAULT_PHAGE_METADATA_PATH,
        fna_dir=DEFAULT_PHAGE_FASTA_DIR,
        expected_panel_count=DEFAULT_EXPECTED_PANEL_COUNT,
        runtime_payload=runtime_payload,
        reference_fasta_path=reference_fasta_path,
        scratch_root=Path(bundle_payload["bundle_dir"]) / "runtime_validation" / "phage_runtime",
    )
    if arm_type == BASELINE_ARM_TYPE:
        training_schema = bundle_payload["training_blocks"]["baseline_phage"]["schema"]
        converted_rows, _ = build_baseline_phage_rows_from_continuous(
            phage_rows,
            family_score_columns=tuple(
                str(column) for column in bundle_payload["phage_runtime"]["family_score_columns"]
            ),
            hit_count_column=str(bundle_payload["phage_runtime"]["hit_count_column"]),
        )
        return _make_feature_block(
            block_id="runtime_validation_phage",
            key_column="phage",
            rows=converted_rows,
            schema=training_schema,
        )
    training_schema = bundle_payload["training_blocks"]["deployment_phage"]["schema"]
    return _make_feature_block(
        block_id="runtime_validation_phage", key_column="phage", rows=phage_rows, schema=training_schema
    )


def run_validation_parity_check(
    *,
    bundle_payload: Mapping[str, object],
    validation_fasta_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validation_hosts = tuple(str(host) for host in bundle_payload["validation_hosts"])
    runtime_host_block = _derive_validation_host_block_rows(
        bundle_payload=bundle_payload,
        validation_fasta_dir=validation_fasta_dir,
    )
    runtime_phage_block = _derive_validation_phage_block_rows(bundle_payload)
    phages = [str(row["phage"]) for row in runtime_phage_block.rows]
    validation_pairs = _build_validation_pairs(validation_hosts, phages)
    runtime_rows = _merge_pair_rows(
        st02_rows=validation_pairs,
        split_rows=validation_pairs,
        host_rows=runtime_host_block.rows,
        phage_rows=runtime_phage_block.rows,
    )
    expected_host_rows = bundle_payload["training_blocks"]["selected_host"]["rows"]
    expected_phage_rows = bundle_payload["training_blocks"]["selected_phage"]["rows"]
    expected_rows = _merge_pair_rows(
        st02_rows=validation_pairs,
        split_rows=validation_pairs,
        host_rows=expected_host_rows,
        phage_rows=expected_phage_rows,
    )
    feature_space = tg01.FeatureSpace(
        categorical_columns=tuple(bundle_payload["feature_space"]["categorical_columns"]),
        numeric_columns=tuple(bundle_payload["feature_space"]["numeric_columns"]),
        track_c_additional_columns=tuple(),
        track_d_columns=tuple(),
        track_e_columns=tuple(),
    )
    vectorizer = bundle_payload["feature_vectorizer"]
    expected_matrix = vectorizer.transform(
        [
            tg01.build_feature_dict(
                row,
                categorical_columns=feature_space.categorical_columns,
                numeric_columns=feature_space.numeric_columns,
            )
            for row in expected_rows
        ]
    )
    runtime_matrix = vectorizer.transform(
        [
            tg01.build_feature_dict(
                row,
                categorical_columns=feature_space.categorical_columns,
                numeric_columns=feature_space.numeric_columns,
            )
            for row in runtime_rows
        ]
    )
    delta = runtime_matrix != expected_matrix
    parity_rows: list[dict[str, object]] = []
    predictions = _build_prediction_rows(
        runtime_rows,
        estimator=bundle_payload["lightgbm_estimator"],
        vectorizer=bundle_payload["feature_vectorizer"],
        calibrator=bundle_payload["isotonic_calibrator"],
        feature_space=feature_space,
    )
    prediction_by_pair = {str(row["pair_id"]): dict(row) for row in predictions}
    for index, (expected_row, runtime_row) in enumerate(zip(expected_rows, runtime_rows)):
        pair_has_delta = delta.getrow(index).nnz > 0
        prediction = prediction_by_pair[str(runtime_row["pair_id"])]
        parity_rows.append(
            {
                "pair_id": runtime_row["pair_id"],
                "bacteria": runtime_row["bacteria"],
                "phage": runtime_row["phage"],
                "feature_vector_identical": int(not pair_has_delta),
                "nonzero_feature_delta_count": int(delta.getrow(index).nnz),
                "predicted_probability": prediction["predicted_probability"],
                "rank": prediction["rank"],
            }
        )
        if pair_has_delta:
            raise ValueError(
                "Validation parity failed: runtime feature vector diverged for "
                f"{runtime_row['bacteria']} x {runtime_row['phage']}"
            )
    return parity_rows, predictions


def _bundle_manifest(payload: Mapping[str, object], *, output_dir: Path) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_type": payload["arm_type"],
        "arm_id": payload["arm_id"],
        "validation_hosts": list(payload["validation_hosts"]),
        "bundle_path": str(output_dir / WINNING_BUNDLE_FILENAME),
        "training_feature_count": len(payload["feature_space"]["numeric_columns"])
        + len(payload["feature_space"]["categorical_columns"]),
        "phage_runtime": payload["phage_runtime"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_directory(args.deploy_output_root)
    _ensure_prerequisites(args)

    LOGGER.info("DEPLOY07 ensuring Picard assemblies are available")
    download_picard_assemblies(assembly_dir=args.assembly_dir)

    LOGGER.info("DEPLOY07 loading precomputed defense counts from %s", args.defense_counts_path)
    if not args.defense_counts_path.exists():
        raise FileNotFoundError(f"Missing DEPLOY06 defense counts CSV: {args.defense_counts_path}")
    defense_count_rows = _load_aggregated_rows(args.defense_counts_path)
    panel_defense_rows = _load_panel_defense_rows()
    defense_schema = deploy02.build_host_defense_schema(panel_defense_rows)
    defense_schema_path = args.deploy_output_root / "host_defense" / deploy02.SCHEMA_MANIFEST_FILENAME
    ensure_directory(defense_schema_path.parent)
    write_json(defense_schema_path, defense_schema)

    LOGGER.info("DEPLOY07 running full DEPLOY03 host surface derivation")
    surface_csv_path, surface_schema_path = run_all_host_surface_features(
        assembly_dir=args.assembly_dir,
        output_root=args.deploy_output_root / "host_surface",
        worker_count=args.host_workers,
    )
    LOGGER.info("DEPLOY07 running full DEPLOY04 host typing derivation")
    typing_csv_path, typing_schema_path = run_all_host_typing_features(
        assembly_dir=args.assembly_dir,
        output_root=args.deploy_output_root / "host_typing",
        worker_count=args.host_workers,
    )
    LOGGER.info("DEPLOY07 running full DEPLOY05 phage feature projection")
    phage_csv_path, phage_schema_path, phage_runtime_path = run_all_panel_phage_features(
        output_root=args.deploy_output_root / "phage_rbp",
    )

    LOGGER.info("DEPLOY07 loading feature blocks and validating schema contracts")
    defense_block = _make_feature_block(
        block_id="deploy02_host_defense",
        key_column="bacteria",
        rows=defense_count_rows,
        schema=defense_schema,
    )
    surface_block = _make_feature_block(
        block_id="deploy03_host_surface",
        key_column="bacteria",
        rows=_load_aggregated_rows(surface_csv_path),
        schema=_load_json(surface_schema_path),
    )
    typing_block = _make_feature_block(
        block_id="deploy04_host_typing",
        key_column="bacteria",
        rows=_load_aggregated_rows(typing_csv_path),
        schema=_load_json(typing_schema_path),
    )
    phage_block = _make_feature_block(
        block_id="deploy05_phage_rbp",
        key_column="phage",
        rows=_load_aggregated_rows(phage_csv_path),
        schema=_load_json(phage_schema_path),
    )
    schema_validation_rows = [
        validate_schema_columns(
            block_id=block.block_id,
            key_column=block.key_column,
            schema=block.schema,
            rows=block.rows,
        )
        for block in (defense_block, surface_block, typing_block, phage_block)
    ]
    duplicate_column_audit = validate_no_duplicate_block_columns(
        (defense_block, surface_block, typing_block, phage_block)
    )
    write_json(
        args.output_dir / SCHEMA_VALIDATION_FILENAME,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_validation": schema_validation_rows,
            "duplicate_column_audit": duplicate_column_audit,
            "inputs": {
                "defense_schema_path": str(defense_schema_path),
                "surface_schema_path": str(surface_schema_path),
                "typing_schema_path": str(typing_schema_path),
                "phage_schema_path": str(phage_schema_path),
            },
        },
    )

    LOGGER.info("DEPLOY07 auditing 403-host defense disagreement against the panel CSV")
    defense_disagreement = compute_full_defense_disagreement_summary(
        defense_count_rows=defense_count_rows,
        panel_rows=panel_defense_rows,
        schema=defense_schema,
    )
    write_json(args.output_dir / DEFENSE_DISAGREEMENT_FILENAME, defense_disagreement)
    if float(defense_disagreement["average_disagreement_subtype_count"]) > DEFENSE_DISAGREEMENT_GATE_MAX:
        raise ValueError(
            "Average defense disagreement exceeds the DEPLOY gate: "
            f"{defense_disagreement['average_disagreement_subtype_count']}"
        )

    target_bacteria = sorted(str(row["bacteria"]) for row in defense_block.rows)
    baseline_host_defense_block, baseline_phage_block = _build_baseline_blocks(
        defense_count_rows=defense_block.rows,
        defense_schema=defense_block.schema,
        phage_deployment_rows=phage_block.rows,
        phage_schema=phage_block.schema,
    )
    baseline_host_surface_block = _build_baseline_host_surface_block(target_bacteria)
    baseline_host_typing_block = _build_baseline_host_typing_block(target_bacteria)
    baseline_host_block = _merge_rows_by_key(
        (baseline_host_defense_block, baseline_host_surface_block, baseline_host_typing_block),
        merged_block_id="baseline_host_features",
        key_column="bacteria",
    )
    deployment_host_block = _merge_rows_by_key(
        (defense_block, surface_block, typing_block),
        merged_block_id="deployment_host_features",
        key_column="bacteria",
    )

    st02_rows = _read_csv(args.st02_pair_table_path)
    split_rows = _read_csv(args.st03_split_assignments_path)
    lightgbm_params = _load_locked_lightgbm_params(args.tg01_summary_path)
    baseline_merged_rows = _merge_pair_rows(
        st02_rows=st02_rows,
        split_rows=split_rows,
        host_rows=baseline_host_block.rows,
        phage_rows=baseline_phage_block.rows,
    )
    deployment_merged_rows = _merge_pair_rows(
        st02_rows=st02_rows,
        split_rows=split_rows,
        host_rows=deployment_host_block.rows,
        phage_rows=phage_block.rows,
    )

    LOGGER.info("DEPLOY07 training calibrated baseline and deployment-paired LightGBM arms")
    baseline_result = evaluate_arm(
        arm_id=BASELINE_ARM_ID,
        display_name="TL18 legacy-encoding reference",
        merged_rows=baseline_merged_rows,
        host_categorical_columns=baseline_host_block.categorical_columns,
        host_numeric_columns=baseline_host_block.numeric_columns,
        phage_categorical_columns=baseline_phage_block.categorical_columns,
        phage_numeric_columns=baseline_phage_block.numeric_columns,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
    )
    deployment_result = evaluate_arm(
        arm_id=DEPLOYMENT_ARM_ID,
        display_name="Deployment-paired model",
        merged_rows=deployment_merged_rows,
        host_categorical_columns=deployment_host_block.categorical_columns,
        host_numeric_columns=deployment_host_block.numeric_columns,
        phage_categorical_columns=phage_block.categorical_columns,
        phage_numeric_columns=phage_block.numeric_columns,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
    )

    LOGGER.info("DEPLOY07 bootstrapping holdout metrics (%d samples)", args.bootstrap_samples)
    bootstrap_summary = tl05.bootstrap_holdout_metric_cis(
        {
            baseline_result.arm_id: baseline_result.holdout_prediction_rows,
            deployment_result.arm_id: deployment_result.holdout_prediction_rows,
        },
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_random_state=args.bootstrap_random_state,
        baseline_arm_id=baseline_result.arm_id,
    )
    arm_metric_rows = [
        summarize_arm_metrics(
            result=baseline_result, baseline_result=baseline_result, bootstrap_summary=bootstrap_summary
        ),
        summarize_arm_metrics(
            result=deployment_result, baseline_result=baseline_result, bootstrap_summary=bootstrap_summary
        ),
    ]
    write_csv(args.output_dir / METRIC_COMPARISON_FILENAME, list(arm_metric_rows[0].keys()), arm_metric_rows)

    winning_arm_id = select_winning_arm_id_from_auc_ci(
        arm_metric_rows,
        baseline_arm_id=BASELINE_ARM_ID,
        deployment_arm_id=DEPLOYMENT_ARM_ID,
    )
    lock_decision = {
        "selected_arm_id": winning_arm_id,
        "selected_arm_label": next(row["arm_label"] for row in arm_metric_rows if row["arm_id"] == winning_arm_id),
        "decision_rule": "lock deployment arm only when the bootstrap 95% CI for AUC delta vs baseline excludes zero",
    }
    metric_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lightgbm_params": lightgbm_params,
        "calibration_fold": args.calibration_fold,
        "random_state": args.random_state,
        "bootstrap_samples": args.bootstrap_samples,
        "arm_metrics": arm_metric_rows,
        "lock_decision": lock_decision,
    }
    write_json(args.output_dir / METRIC_SUMMARY_FILENAME, metric_summary)

    LOGGER.info("DEPLOY07 building winning model bundle and validation parity payload")
    reference_fasta_path = args.deploy_output_root / "phage_rbp" / REFERENCE_FASTA_FILENAME
    phage_schema_payload = _load_json(phage_schema_path)
    bundle_payload: dict[str, Any] = {
        "arm_id": winning_arm_id,
        "bundle_dir": str(args.output_dir),
        "validation_hosts": list(DEFAULT_VALIDATION_HOSTS),
        "lightgbm_estimator": baseline_result.estimator
        if winning_arm_id == BASELINE_ARM_ID
        else deployment_result.estimator,
        "feature_vectorizer": baseline_result.vectorizer
        if winning_arm_id == BASELINE_ARM_ID
        else deployment_result.vectorizer,
        "isotonic_calibrator": baseline_result.calibrator
        if winning_arm_id == BASELINE_ARM_ID
        else deployment_result.calibrator,
        "feature_space": {
            "categorical_columns": list(
                (
                    baseline_result if winning_arm_id == BASELINE_ARM_ID else deployment_result
                ).feature_space.categorical_columns
            ),
            "numeric_columns": list(
                (
                    baseline_result if winning_arm_id == BASELINE_ARM_ID else deployment_result
                ).feature_space.numeric_columns
            ),
        },
        "phage_runtime": {
            "runtime_path": str(phage_runtime_path),
            "reference_fasta_path": str(reference_fasta_path),
            "family_score_columns": list(phage_schema_payload["family_score_columns"]),
            "hit_count_column": phage_schema_payload["reference_hit_count_column"],
        },
    }
    bundle_payload["training_blocks"] = {
        "baseline_host": {"rows": baseline_host_block.rows, "schema": baseline_host_block.schema},
        "deployment_host": {"rows": deployment_host_block.rows, "schema": deployment_host_block.schema},
        "baseline_phage": {"rows": baseline_phage_block.rows, "schema": baseline_phage_block.schema},
        "deployment_phage": {"rows": phage_block.rows, "schema": phage_block.schema},
    }
    bundle_payload["arm_type"] = select_arm_type_for_winning_arm_id(winning_arm_id)
    if winning_arm_id == BASELINE_ARM_ID:
        bundle_payload["baseline_runtime"] = {
            "defense_rows": baseline_host_defense_block.rows,
            "tl15_payload": build_tl15_runtime_payload(
                output_dir=args.output_dir,
                picard_metadata_path=deploy03.tl15.DEFAULT_PICARD_METADATA_PATH,
                o_type_output_path=deploy03.tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
                o_type_allele_path=deploy03.tl15.DEFAULT_O_TYPE_ALLELE_PATH,
                o_antigen_override_path=deploy03.tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
                abc_capsule_profile_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
                abc_capsule_definition_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_DEFINITION_DIR,
                omp_reference_path=deploy03.tl15.DEFAULT_OMP_REFERENCE_PATH,
            ),
            "tl16_payload": build_tl16_runtime_payload(
                output_dir=args.output_dir,
                capsule_definition_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_DEFINITION_DIR,
                capsule_profile_dir=deploy03.tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
            ),
        }
        bundle_payload["training_blocks"]["selected_host"] = {
            "rows": baseline_host_block.rows,
            "schema": baseline_host_block.schema,
        }
        bundle_payload["training_blocks"]["selected_phage"] = {
            "rows": baseline_phage_block.rows,
            "schema": baseline_phage_block.schema,
        }
    else:
        bundle_payload["deployment_runtime"] = {"defense_rows": defense_block.rows}
        bundle_payload["training_blocks"]["selected_host"] = {
            "rows": deployment_host_block.rows,
            "schema": deployment_host_block.schema,
        }
        bundle_payload["training_blocks"]["selected_phage"] = {"rows": phage_block.rows, "schema": phage_block.schema}
    joblib.dump(bundle_payload, args.output_dir / WINNING_BUNDLE_FILENAME)
    write_json(
        args.output_dir / WINNING_BUNDLE_MANIFEST_FILENAME, _bundle_manifest(bundle_payload, output_dir=args.output_dir)
    )

    LOGGER.info("DEPLOY07 running validation-host inference parity on the winning bundle")
    parity_rows, validation_predictions = run_validation_parity_check(
        bundle_payload=bundle_payload,
        validation_fasta_dir=args.validation_fasta_dir,
    )
    write_csv(args.output_dir / PARITY_RESULTS_FILENAME, list(parity_rows[0].keys()), parity_rows)
    write_csv(
        args.output_dir / WINNING_PREDICTIONS_FILENAME,
        list(validation_predictions[0].keys()),
        validation_predictions,
    )
    write_json(
        args.output_dir / PARITY_SUMMARY_FILENAME,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_arm_id": winning_arm_id,
            "validation_host_count": len(DEFAULT_VALIDATION_HOSTS),
            "panel_phage_count": len({row["phage"] for row in parity_rows}),
            "identical_pair_feature_vectors": sum(int(row["feature_vector_identical"]) for row in parity_rows),
            "total_pair_feature_vectors": len(parity_rows),
        },
    )
    LOGGER.info("DEPLOY07 completed successfully; winning arm is %s", winning_arm_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
