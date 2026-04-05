#!/usr/bin/env python3
"""Derive small host sequence-stat feature rows directly from raw assemblies."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_json
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import _gc_content
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_fasta_records

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_stats")
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
PER_HOST_FEATURES_FILENAME = "host_stats_features.csv"
MANIFEST_FILENAME = "manifest.json"
STRING_DTYPE = "string"
INT_DTYPE = "int64"
FLOAT_DTYPE = "float64"

HOST_STATS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("bacteria", STRING_DTYPE),
    ("host_sequence_record_count", INT_DTYPE),
    ("host_genome_length_nt", INT_DTYPE),
    ("host_gc_content", FLOAT_DTYPE),
    ("host_n50_contig_length_nt", INT_DTYPE),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_names_from_schema(schema: Mapping[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _write_single_row_csv(path: Path, row: Mapping[str, object], *, delimiter: str = ",") -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def _compute_n50(lengths: list[int]) -> int:
    if not lengths:
        raise ValueError("Cannot compute N50 for an empty assembly")
    total_length = sum(lengths)
    half_length = total_length / 2
    running_length = 0
    for length in sorted(lengths, reverse=True):
        running_length += length
        if running_length >= half_length:
            return length
    return lengths[0]


def build_host_stats_schema() -> dict[str, Any]:
    columns = [{"name": name, "dtype": dtype} for name, dtype in HOST_STATS_COLUMNS]
    numeric_columns = [name for name, dtype in HOST_STATS_COLUMNS if name != "bacteria" and dtype != STRING_DTYPE]
    return {
        "feature_block": "host_stats",
        "key_column": "bacteria",
        "column_count": len(columns),
        "columns": columns,
        "numeric_columns": numeric_columns,
    }


def build_host_stats_feature_row(assembly_path: Path, *, bacteria_id: str | None = None) -> dict[str, object]:
    if not assembly_path.exists():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_path}")

    records = read_fasta_records(assembly_path, protein=False)
    sequences = [record.sequence for record in records]
    lengths = [len(sequence) for sequence in sequences]
    genome_length_nt = sum(lengths)
    if genome_length_nt == 0:
        raise ValueError(f"Assembly {assembly_path} contains zero nucleotide bases")

    return {
        "bacteria": bacteria_id or assembly_path.stem,
        "host_sequence_record_count": len(records),
        "host_genome_length_nt": genome_length_nt,
        "host_gc_content": round(_gc_content(sequences), 6),
        "host_n50_contig_length_nt": _compute_n50(lengths),
    }


def derive_host_stats_features(
    assembly_path: Path,
    *,
    bacteria_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    schema = build_host_stats_schema()
    feature_row = build_host_stats_feature_row(assembly_path, bacteria_id=bacteria_id)

    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)
    feature_csv_path = output_dir / PER_HOST_FEATURES_FILENAME
    _write_single_row_csv(feature_csv_path, feature_row)

    manifest = {
        "step_name": "derive_host_stats_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assembly_path": str(assembly_path),
            "assembly_sha256": _sha256(assembly_path),
        },
        "outputs": {
            "feature_csv_path": str(feature_csv_path),
        },
        "guardrails": {
            "rebuildable_from_raw_fastas": True,
            "panel_metadata_used": False,
        },
        "summary_stats": {key: feature_row[key] for key in _column_names_from_schema(schema) if key != "bacteria"},
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    return {
        "schema": schema,
        "feature_row": feature_row,
        "manifest": manifest,
    }
