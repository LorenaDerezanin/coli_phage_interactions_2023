#!/usr/bin/env python3
"""Derive small phage sequence-stat feature rows directly from raw FASTAs."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_json
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import _gc_content
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_fasta_records

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/autoresearch/phage_stats")
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
PER_PHAGE_FEATURES_FILENAME = "phage_stats_features.csv"
MANIFEST_FILENAME = "manifest.json"
STRING_DTYPE = "string"
INT_DTYPE = "int64"
FLOAT_DTYPE = "float64"

PHAGE_STATS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("phage", STRING_DTYPE),
    ("phage_sequence_record_count", INT_DTYPE),
    ("phage_genome_length_nt", INT_DTYPE),
    ("phage_gc_content", FLOAT_DTYPE),
    ("phage_n50_contig_length_nt", INT_DTYPE),
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
        raise ValueError("Cannot compute N50 for an empty phage FASTA")
    total_length = sum(lengths)
    half_length = total_length / 2
    running_length = 0
    for length in sorted(lengths, reverse=True):
        running_length += length
        if running_length >= half_length:
            return length
    raise AssertionError("N50 computation should always return once lengths are non-empty")


def build_phage_stats_schema() -> dict[str, Any]:
    columns = [{"name": name, "dtype": dtype} for name, dtype in PHAGE_STATS_COLUMNS]
    numeric_columns = [name for name, dtype in PHAGE_STATS_COLUMNS if name != "phage" and dtype != STRING_DTYPE]
    return {
        "feature_block": "phage_stats",
        "key_column": "phage",
        "column_count": len(columns),
        "columns": columns,
        "numeric_columns": numeric_columns,
    }


def build_phage_stats_feature_row(phage_fasta_path: Path, *, phage_id: str | None = None) -> dict[str, object]:
    if not phage_fasta_path.exists():
        raise FileNotFoundError(f"Phage FASTA not found: {phage_fasta_path}")

    records = read_fasta_records(phage_fasta_path, protein=False)
    sequences = [record.sequence for record in records]
    lengths = [len(sequence) for sequence in sequences]
    genome_length_nt = sum(lengths)
    if genome_length_nt == 0:
        raise ValueError(f"Phage FASTA {phage_fasta_path} contains zero nucleotide bases")

    return {
        "phage": phage_id or phage_fasta_path.stem,
        "phage_sequence_record_count": len(records),
        "phage_genome_length_nt": genome_length_nt,
        "phage_gc_content": round(_gc_content(sequences), 6),
        "phage_n50_contig_length_nt": _compute_n50(lengths),
    }


def derive_phage_stats_features(
    phage_fasta_path: Path,
    *,
    phage_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    schema = build_phage_stats_schema()
    feature_row = build_phage_stats_feature_row(phage_fasta_path, phage_id=phage_id)

    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)
    feature_csv_path = output_dir / PER_PHAGE_FEATURES_FILENAME
    _write_single_row_csv(feature_csv_path, feature_row)

    manifest = {
        "step_name": "derive_phage_stats_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "phage_fasta_path": str(phage_fasta_path),
            "phage_fasta_sha256": _sha256(phage_fasta_path),
        },
        "outputs": {
            "feature_csv_path": str(feature_csv_path),
        },
        "guardrails": {
            "rebuildable_from_raw_fastas": True,
            "panel_metadata_used": False,
            "low_cost_baseline_feature_family": True,
        },
        "summary_stats": {key: feature_row[key] for key in _column_names_from_schema(schema) if key != "phage"},
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    return {
        "schema": schema,
        "feature_row": feature_row,
        "manifest": manifest,
    }
