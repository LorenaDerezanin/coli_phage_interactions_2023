#!/usr/bin/env python3
"""AR02: scaffold the AUTORESEARCH sandbox and freeze the search-cache contract."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch import build_contract
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = build_contract.DEFAULT_OUTPUT_DIR
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "search_cache_v1"

CACHE_MANIFEST_FILENAME = "ar02_search_cache_manifest_v1.json"
SCHEMA_MANIFEST_FILENAME = "ar02_schema_manifest_v1.json"
PROVENANCE_MANIFEST_FILENAME = "ar02_provenance_manifest_v1.json"
TRAIN_PAIR_TABLE_FILENAME = "train_pairs.csv"
INNER_VAL_PAIR_TABLE_FILENAME = "inner_val_pairs.csv"
ENTITY_INDEX_FILENAME = "entity_index.csv"
SLOT_SCHEMA_FILENAME = "schema_manifest.json"

TASK_ID = "AR02"
CACHE_CONTRACT_ID = "autoresearch_search_cache_v1"
SCHEMA_MANIFEST_ID = "autoresearch_feature_schema_v1"
SEARCH_PAIR_TABLE_ID = "autoresearch_search_pair_tables_v1"
HOLDOUT_HANDLING_RULE = "sealed_holdout_outside_workspace"
SUPPORTED_SEARCH_SPLITS = (build_contract.TRAIN_SPLIT, build_contract.INNER_VAL_SPLIT)
DISALLOWED_SEARCH_SPLITS = (build_contract.HOLDOUT_SPLIT,)
PAIR_KEY = ("pair_id", "bacteria", "phage")


@dataclass(frozen=True)
class SlotSpec:
    slot_name: str
    entity_key: str
    column_prefix: str
    block_role: str
    description: str

    @property
    def join_keys(self) -> list[str]:
        return [self.entity_key]


SLOT_SPECS = (
    SlotSpec(
        slot_name="host_defense",
        entity_key="bacteria",
        column_prefix="host_defense__",
        block_role="host",
        description="Reserved host defense-system features derived from raw host assemblies.",
    ),
    SlotSpec(
        slot_name="host_surface",
        entity_key="bacteria",
        column_prefix="host_surface__",
        block_role="host",
        description="Reserved host surface and adsorption-related features derived from raw host assemblies.",
    ),
    SlotSpec(
        slot_name="host_typing",
        entity_key="bacteria",
        column_prefix="host_typing__",
        block_role="host",
        description="Reserved host typing calls derived from raw host assemblies.",
    ),
    SlotSpec(
        slot_name="host_stats",
        entity_key="bacteria",
        column_prefix="host_stats__",
        block_role="host",
        description="Reserved low-cost host sequence statistics derived from raw host assemblies.",
    ),
    SlotSpec(
        slot_name="phage_projection",
        entity_key="phage",
        column_prefix="phage_projection__",
        block_role="phage",
        description="Reserved phage projection features derived from raw phage genomes.",
    ),
    SlotSpec(
        slot_name="phage_stats",
        entity_key="phage",
        column_prefix="phage_stats__",
        block_role="phage",
        description="Reserved low-cost phage sequence statistics derived from raw phage genomes.",
    ),
)
SLOT_SPEC_BY_NAME = {spec.slot_name: spec for spec in SLOT_SPECS}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-interactions-path",
        type=Path,
        default=build_contract.DEFAULT_RAW_INTERACTIONS_PATH,
        help="Semicolon-delimited raw interaction table.",
    )
    parser.add_argument(
        "--host-assembly-dir",
        type=Path,
        default=build_contract.DEFAULT_ASSEMBLY_DIR,
        help="Directory containing Picard host FASTAs.",
    )
    parser.add_argument(
        "--phage-fasta-dir",
        type=Path,
        default=build_contract.DEFAULT_PHAGE_FASTA_DIR,
        help="Directory containing phage FASTA files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="AUTORESEARCH generated-output root directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory where the search cache should be written.",
    )
    parser.add_argument(
        "--warm-cache-manifest-path",
        type=Path,
        default=None,
        help="Optional manifest describing precomputed warm-cache accelerators.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=build_contract.DEFAULT_HOLDOUT_FRACTION,
        help="Fraction of bacteria routed to sealed holdout.",
    )
    parser.add_argument(
        "--inner-val-fraction",
        type=float,
        default=build_contract.DEFAULT_INNER_VAL_FRACTION,
        help="Fraction of bacteria routed to inner validation.",
    )
    parser.add_argument(
        "--split-salt",
        default=build_contract.DEFAULT_SPLIT_SALT,
        help="Deterministic salt for bacteria split assignment.",
    )
    parser.add_argument(
        "--skip-host-assembly-resolution",
        action="store_true",
        help="Skip download_picard_assemblies() and trust the provided host assembly directory as-is.",
    )
    return parser.parse_args(argv)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_ar01_contract(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    LOGGER.info("AR02 starting: refreshing AR01 raw-input contract under %s", args.output_root)
    exit_code = build_contract.main(
        [
            "--raw-interactions-path",
            str(args.raw_interactions_path),
            "--host-assembly-dir",
            str(args.host_assembly_dir),
            "--phage-fasta-dir",
            str(args.phage_fasta_dir),
            "--output-dir",
            str(args.output_root),
            "--holdout-fraction",
            str(args.holdout_fraction),
            "--inner-val-fraction",
            str(args.inner_val_fraction),
            "--split-salt",
            args.split_salt,
            *(["--skip-host-assembly-resolution"] if args.skip_host_assembly_resolution else []),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"AR01 contract build failed with exit code {exit_code}")

    pair_table_path = args.output_root / build_contract.PAIR_TABLE_FILENAME
    contract_manifest_path = args.output_root / build_contract.CONTRACT_MANIFEST_FILENAME
    input_checksums_path = args.output_root / build_contract.INPUT_CHECKSUMS_FILENAME
    for path in (pair_table_path, contract_manifest_path, input_checksums_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected AR01 artifact was not written: {path}")
    return pair_table_path, contract_manifest_path, input_checksums_path


def select_search_rows(pair_rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = {split: [] for split in SUPPORTED_SEARCH_SPLITS}
    seen_disallowed_splits: set[str] = set()

    for row in pair_rows:
        split_name = str(row["split"])
        if split_name in SUPPORTED_SEARCH_SPLITS:
            selected[split_name].append(dict(row))
        elif split_name in DISALLOWED_SEARCH_SPLITS:
            seen_disallowed_splits.add(split_name)
        else:
            raise ValueError(f"Unexpected split in AR01 pair table: {split_name}")

    for split_name in SUPPORTED_SEARCH_SPLITS:
        if not selected[split_name]:
            raise ValueError(f"AR02 search cache would have an empty required split: {split_name}")
    if not seen_disallowed_splits:
        raise ValueError("AR01 pair table did not contain the sealed holdout split expected by AR02.")
    return selected


def build_slot_index_rows(
    *,
    slot_spec: SlotSpec,
    selected_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[dict[str, str]]:
    values = {
        str(row[slot_spec.entity_key])
        for split_rows in selected_rows.values()
        for row in split_rows
        if str(row["retained_for_autoresearch"]) == "1"
    }
    if not values:
        raise ValueError(f"Reserved slot {slot_spec.slot_name} would have zero retained entities.")
    return [{slot_spec.entity_key: value} for value in sorted(values)]


def build_slot_schema_manifest(slot_spec: SlotSpec, row_count: int) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_manifest_id": SCHEMA_MANIFEST_ID,
        "cache_contract_id": CACHE_CONTRACT_ID,
        "slot_name": slot_spec.slot_name,
        "entity_key": slot_spec.entity_key,
        "join_keys": slot_spec.join_keys,
        "column_family_prefix": slot_spec.column_prefix,
        "block_role": slot_spec.block_role,
        "reserved_feature_columns": [],
        "reserved_feature_column_count": 0,
        "entity_index_row_count": row_count,
        "composability_contract": {
            "join_type": "left",
            "row_granularity": f"one_row_per_{slot_spec.entity_key}",
            "column_ownership": (
                f"Future columns for {slot_spec.slot_name} must start with {slot_spec.column_prefix} "
                f"and may only be added inside this slot."
            ),
        },
        "description": slot_spec.description,
    }


def build_top_level_schema_manifest() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_manifest_id": SCHEMA_MANIFEST_ID,
        "cache_contract_id": CACHE_CONTRACT_ID,
        "pair_keys": list(PAIR_KEY),
        "pair_table_id": SEARCH_PAIR_TABLE_ID,
        "supported_search_splits": list(SUPPORTED_SEARCH_SPLITS),
        "disallowed_search_splits": list(DISALLOWED_SEARCH_SPLITS),
        "holdout_handling_rule": HOLDOUT_HANDLING_RULE,
        "pair_table_contract": {
            "row_granularity": "one_row_per_bacteria_phage_pair",
            "pair_join_keys": list(PAIR_KEY),
            "labels_read_only": True,
            "required_columns": [
                "pair_id",
                "bacteria",
                "phage",
                "split",
                "label_any_lysis",
                "training_weight_v3",
                "retained_for_autoresearch",
                "host_fasta_path",
                "phage_fasta_path",
            ],
        },
        "slot_order": [spec.slot_name for spec in SLOT_SPECS],
        "feature_slots": {
            spec.slot_name: {
                "entity_key": spec.entity_key,
                "join_keys": spec.join_keys,
                "column_family_prefix": spec.column_prefix,
                "block_role": spec.block_role,
                "reserved_feature_columns": [],
                "reserved_feature_column_count": 0,
            }
            for spec in SLOT_SPECS
        },
        "composability_contract": {
            "training_code_mutation_boundary": "train.py may consume the cache but must not rewrite schema or manifests",
            "cache_building_boundary": "prepare.py is the only supported path from raw inputs to the search cache",
            "split_visibility_rule": (
                "Only train and inner_val pair tables are exported into the search workspace; sealed holdout rows are "
                "kept outside the cache entirely."
            ),
            "feature_block_composition_rule": (
                "Each slot is joined independently to the pair tables by its declared key; slot files may add columns "
                "later but may not change slot names, join keys, or prefixes."
            ),
            "warm_cache_rule": (
                "Optional warm-cache artifacts are accelerators only. They must declare the same schema_manifest_id and "
                "match the fixed slot contract exactly."
            ),
        },
    }


def parse_warm_cache_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if "slot_artifacts" not in manifest:
        raise ValueError(f"Warm-cache manifest is missing slot_artifacts: {path}")
    if not isinstance(manifest["slot_artifacts"], Mapping):
        raise ValueError(f"Warm-cache slot_artifacts must be a mapping: {path}")
    return manifest


def validate_warm_cache_manifest(path: Path, *, schema_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = parse_warm_cache_manifest(path)
    schema_manifest_id = schema_manifest["schema_manifest_id"]
    if manifest.get("schema_manifest_id") != schema_manifest_id:
        raise ValueError(
            "Warm-cache manifest schema mismatch: "
            f"expected {schema_manifest_id}, got {manifest.get('schema_manifest_id')}"
        )

    validated_slots: dict[str, Any] = {}
    for slot_name, descriptor in manifest["slot_artifacts"].items():
        if slot_name not in SLOT_SPEC_BY_NAME:
            raise ValueError(f"Warm-cache manifest declares unknown slot: {slot_name}")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Warm-cache descriptor must be a mapping for slot {slot_name}")

        slot_spec = SLOT_SPEC_BY_NAME[slot_name]
        join_keys = [str(value) for value in descriptor.get("join_keys", [])]
        if join_keys != slot_spec.join_keys:
            raise ValueError(
                f"Warm-cache join keys do not match frozen contract for {slot_name}: "
                f"expected {slot_spec.join_keys}, got {join_keys}"
            )

        prefix = str(descriptor.get("column_family_prefix", ""))
        if prefix != slot_spec.column_prefix:
            raise ValueError(
                f"Warm-cache column prefix does not match frozen contract for {slot_name}: "
                f"expected {slot_spec.column_prefix}, got {prefix}"
            )

        columns = [str(value) for value in descriptor.get("columns", [])]
        if any(not column.startswith(slot_spec.column_prefix) for column in columns):
            raise ValueError(f"Warm-cache columns must stay inside slot prefix {slot_spec.column_prefix}: {slot_name}")

        artifact_path = path.parent / str(descriptor.get("path", ""))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Warm-cache artifact not found for slot {slot_name}: {artifact_path}")

        header = load_csv_header(artifact_path)
        expected_header = slot_spec.join_keys + columns
        if header != expected_header:
            raise ValueError(
                f"Warm-cache artifact header mismatch for {slot_name}: expected {expected_header}, got {header}"
            )

        validated_slots[slot_name] = {
            "path": str(artifact_path),
            "join_keys": join_keys,
            "column_family_prefix": prefix,
            "columns": columns,
            "column_count": len(columns),
            "sha256": build_contract.sha256_file(artifact_path),
        }

    return {
        "warm_cache_manifest_path": str(path),
        "warm_cache_manifest_sha256": build_contract.sha256_file(path),
        "warm_cache_manifest_id": manifest.get("warm_cache_manifest_id", ""),
        "schema_manifest_id": manifest.get("schema_manifest_id"),
        "source_kind": manifest.get("source_kind", ""),
        "source_notes": manifest.get("source_notes", ""),
        "validated_slots": validated_slots,
    }


def load_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def write_split_pair_tables(cache_dir: Path, split_rows: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    pair_table_dir = cache_dir / "search_pairs"
    ensure_directory(pair_table_dir)
    pair_table_summaries: dict[str, Any] = {}

    for split_name, filename in (
        (build_contract.TRAIN_SPLIT, TRAIN_PAIR_TABLE_FILENAME),
        (build_contract.INNER_VAL_SPLIT, INNER_VAL_PAIR_TABLE_FILENAME),
    ):
        rows = [dict(row) for row in split_rows[split_name]]
        path = pair_table_dir / filename
        write_csv(path, fieldnames=list(rows[0].keys()), rows=rows)
        pair_table_summaries[split_name] = summarize_pair_table(path, rows)

    return pair_table_summaries


def summarize_pair_table(path: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    retained_rows = [row for row in rows if str(row["retained_for_autoresearch"]) == "1"]
    return {
        "path": str(path),
        "sha256": build_contract.sha256_file(path),
        "row_count": len(rows),
        "retained_row_count": len(retained_rows),
        "bacteria_count": len({str(row["bacteria"]) for row in retained_rows}),
        "phage_count": len({str(row["phage"]) for row in retained_rows}),
        "label_counts": dict(Counter(str(row["label_any_lysis"]) for row in retained_rows)),
    }


def write_slot_indexes(cache_dir: Path, split_rows: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    slot_root = cache_dir / "feature_slots"
    ensure_directory(slot_root)
    slot_summaries: dict[str, Any] = {}

    for slot_spec in SLOT_SPECS:
        slot_dir = slot_root / slot_spec.slot_name
        ensure_directory(slot_dir)
        rows = build_slot_index_rows(slot_spec=slot_spec, selected_rows=split_rows)
        index_path = slot_dir / ENTITY_INDEX_FILENAME
        write_csv(index_path, fieldnames=slot_spec.join_keys, rows=rows)

        schema_manifest = build_slot_schema_manifest(slot_spec, row_count=len(rows))
        schema_path = slot_dir / SLOT_SCHEMA_FILENAME
        write_json(schema_path, schema_manifest)

        slot_summaries[slot_spec.slot_name] = {
            "entity_key": slot_spec.entity_key,
            "index_path": str(index_path),
            "schema_manifest_path": str(schema_path),
            "entity_count": len(rows),
            "sha256": build_contract.sha256_file(index_path),
        }

    return slot_summaries


def build_provenance_manifest(
    *,
    output_root: Path,
    cache_dir: Path,
    contract_manifest_path: Path,
    input_checksums_path: Path,
    pair_rows: Sequence[Mapping[str, str]],
    split_pair_tables: Mapping[str, Any],
    slot_summaries: Mapping[str, Any],
    warm_cache_validation: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    holdout_rows = [row for row in pair_rows if str(row["split"]) == build_contract.HOLDOUT_SPLIT]
    holdout_retained_rows = [row for row in holdout_rows if str(row["retained_for_autoresearch"]) == "1"]

    return {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "cache_contract_id": CACHE_CONTRACT_ID,
        "schema_manifest_id": SCHEMA_MANIFEST_ID,
        "build_mode": "raw_inputs_plus_optional_warm_cache" if warm_cache_validation else "raw_inputs_only",
        "source_contract": {
            "pair_contract_manifest_path": str(contract_manifest_path),
            "pair_contract_manifest_sha256": build_contract.sha256_file(contract_manifest_path),
            "input_checksums_manifest_path": str(input_checksums_path),
            "input_checksums_manifest_sha256": build_contract.sha256_file(input_checksums_path),
            "output_root": str(output_root),
        },
        "search_workspace": {
            "cache_dir": str(cache_dir),
            "exported_splits": list(SUPPORTED_SEARCH_SPLITS),
            "disallowed_splits": list(DISALLOWED_SEARCH_SPLITS),
            "pair_tables": dict(split_pair_tables),
            "feature_slots": dict(slot_summaries),
        },
        "sealed_holdout": {
            "split_name": build_contract.HOLDOUT_SPLIT,
            "row_count": len(holdout_rows),
            "retained_row_count": len(holdout_retained_rows),
            "exported_to_search_cache": False,
            "rule": "Holdout labels and holdout-ready evaluation tables stay outside the RunPod workspace entirely.",
        },
        "warm_cache_validation": None if warm_cache_validation is None else dict(warm_cache_validation),
    }


def build_cache_manifest(
    *,
    cache_dir: Path,
    schema_manifest_path: Path,
    provenance_manifest_path: Path,
    split_pair_tables: Mapping[str, Any],
    slot_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "cache_contract_id": CACHE_CONTRACT_ID,
        "schema_manifest_id": SCHEMA_MANIFEST_ID,
        "cache_dir": str(cache_dir),
        "schema_manifest_path": str(schema_manifest_path),
        "provenance_manifest_path": str(provenance_manifest_path),
        "pair_tables": dict(split_pair_tables),
        "feature_slots": dict(slot_summaries),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_root)
    ensure_directory(args.cache_dir)

    pair_table_path, contract_manifest_path, input_checksums_path = run_ar01_contract(args)
    pair_rows = load_csv_rows(pair_table_path)
    split_rows = select_search_rows(pair_rows)

    schema_manifest = build_top_level_schema_manifest()
    schema_manifest_path = args.cache_dir / SCHEMA_MANIFEST_FILENAME
    write_json(schema_manifest_path, schema_manifest)

    split_pair_tables = write_split_pair_tables(args.cache_dir, split_rows)
    slot_summaries = write_slot_indexes(args.cache_dir, split_rows)

    warm_cache_validation = None
    if args.warm_cache_manifest_path is not None:
        LOGGER.info("Validating optional warm-cache manifest: %s", args.warm_cache_manifest_path)
        warm_cache_validation = validate_warm_cache_manifest(
            args.warm_cache_manifest_path,
            schema_manifest=schema_manifest,
        )

    provenance_manifest = build_provenance_manifest(
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        contract_manifest_path=contract_manifest_path,
        input_checksums_path=input_checksums_path,
        pair_rows=pair_rows,
        split_pair_tables=split_pair_tables,
        slot_summaries=slot_summaries,
        warm_cache_validation=warm_cache_validation,
    )
    provenance_manifest_path = args.cache_dir / PROVENANCE_MANIFEST_FILENAME
    write_json(provenance_manifest_path, provenance_manifest)

    cache_manifest = build_cache_manifest(
        cache_dir=args.cache_dir,
        schema_manifest_path=schema_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        split_pair_tables=split_pair_tables,
        slot_summaries=slot_summaries,
    )
    write_json(args.cache_dir / CACHE_MANIFEST_FILENAME, cache_manifest)

    LOGGER.info("AR02 completed: wrote search cache to %s", args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
