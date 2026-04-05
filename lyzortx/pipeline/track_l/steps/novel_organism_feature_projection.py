"""Projection helpers for novel phage and host genomes."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import _parse_binary_flag
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import (
    _count_valid_kmer_windows,
    compute_kmer_frequency_vector,
    gc_content,
    read_fasta_records,
)

PHAGE_FEATURE_PREFIX = "phage_genome_tetra_svd_"
DEFAULT_DEFENSE_DELIMITERS: Tuple[str, ...] = (";", "\t", ",")
EXPECTED_DEFENSE_SOURCE_KEY = "retained_subtype_columns"
EXPECTED_DEFENSE_RETAINED_FEATURE_KEY = "retained_feature_columns"
EXPECTED_DEFENSE_FEATURE_KEY = "ordered_feature_columns"
EXPECTED_DEFENSE_DERIVED_KEY = "derived_columns"


def _infer_kmer_size_from_svd(svd: TruncatedSVD) -> int:
    n_features = getattr(svd, "n_features_in_", None)
    if n_features is None:
        raise ValueError("The saved TruncatedSVD object does not expose n_features_in_.")
    kmer_size = round(math.log(int(n_features), 4))
    if 4**kmer_size != int(n_features):
        raise ValueError(f"Cannot infer k-mer size from {n_features} input features.")
    return int(kmer_size)


def _load_phage_projection_artifact(path: Path) -> Tuple[TruncatedSVD, Dict[str, object]]:
    artifact = joblib.load(path)
    if isinstance(artifact, TruncatedSVD):
        return artifact, {}
    if isinstance(artifact, dict) and isinstance(artifact.get("svd"), TruncatedSVD):
        return artifact["svd"], artifact
    raise TypeError(f"Unsupported phage projection artifact in {path}: {type(artifact)!r}")


def _load_defense_mask(path: Path) -> Dict[str, object]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Unsupported defense column mask in {path}: {type(artifact)!r}")
    for key in (
        EXPECTED_DEFENSE_SOURCE_KEY,
        EXPECTED_DEFENSE_RETAINED_FEATURE_KEY,
        EXPECTED_DEFENSE_FEATURE_KEY,
        EXPECTED_DEFENSE_DERIVED_KEY,
    ):
        if key not in artifact:
            raise ValueError(f"Defense mask loaded from {path} is missing {key!r}")
    return artifact


def _detect_defense_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"No header found in {path}")
    delimiter = max(DEFAULT_DEFENSE_DELIMITERS, key=header.count)
    if header.count(delimiter) == 0:
        raise ValueError(f"Unable to detect a delimiter in {path}")
    return delimiter


def _read_single_defense_row(path: Path) -> Mapping[str, str]:
    delimiter = _detect_defense_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one Defense Finder row in {path}, found {len(rows)}")
    return rows[0]


def _compute_phage_kmer_profile(fna_path: Path, *, kmer_size: int) -> Tuple[np.ndarray, int, float]:
    records = read_fasta_records(fna_path, protein=False)
    sequences = [record.sequence for record in records]
    if not sequences:
        raise ValueError(f"No nucleotide sequences found in {fna_path}")

    total_windows = 0
    combined_vector = np.zeros(4**kmer_size, dtype=np.float64)
    for sequence in sequences:
        vector = compute_kmer_frequency_vector(sequence, k=kmer_size)
        valid_windows = _count_valid_kmer_windows(sequence, k=kmer_size)
        if valid_windows:
            combined_vector += vector * valid_windows
        total_windows += valid_windows

    if total_windows == 0:
        raise ValueError(f"No valid {kmer_size}-mer windows found in {fna_path}")
    combined_vector /= total_windows
    genome_length_nt = sum(len(sequence) for sequence in sequences)
    gc_fraction = round(gc_content(sequences), 6)
    return combined_vector, genome_length_nt, gc_fraction


def project_novel_phage(fna_path: Path, svd_path: Path) -> Dict[str, object]:
    """Project one novel phage genome into the TD02 feature space."""

    svd, artifact = _load_phage_projection_artifact(svd_path)
    kmer_size = int(artifact.get("kmer_size", _infer_kmer_size_from_svd(svd)))
    embedding_dim = int(artifact.get("embedding_dim", svd.n_components))

    vector, genome_length_nt, gc_content = _compute_phage_kmer_profile(fna_path, kmer_size=kmer_size)
    projected = svd.transform(vector.reshape(1, -1))[0]
    if embedding_dim > len(projected):
        padded = np.zeros(embedding_dim, dtype=np.float64)
        padded[: len(projected)] = projected
        projected = padded

    row: Dict[str, object] = {
        "phage": fna_path.stem,
        "phage_gc_content": gc_content,
        "phage_genome_length_nt": int(genome_length_nt),
    }
    for index, value in enumerate(projected[:embedding_dim]):
        row[f"{PHAGE_FEATURE_PREFIX}{index:02d}"] = round(float(value), 6)
    return row


def project_novel_host(defense_finder_output_path: Path, column_mask_path: Path) -> Dict[str, object]:
    """Project one novel Defense Finder output into the TC01 feature space."""

    mask = _load_defense_mask(column_mask_path)
    retained_source_columns = list(mask[EXPECTED_DEFENSE_SOURCE_KEY])
    retained_feature_columns = list(mask[EXPECTED_DEFENSE_RETAINED_FEATURE_KEY])
    ordered_feature_columns = list(mask[EXPECTED_DEFENSE_FEATURE_KEY])
    derived_columns = list(mask[EXPECTED_DEFENSE_DERIVED_KEY])
    if len(retained_source_columns) != len(retained_feature_columns):
        raise ValueError(f"Defense mask loaded from {column_mask_path} has mismatched source and feature columns.")
    if ordered_feature_columns != [*retained_feature_columns, *derived_columns]:
        raise ValueError(f"Defense mask loaded from {column_mask_path} has an unexpected ordered column list.")

    row = _read_single_defense_row(defense_finder_output_path)
    if "bacteria" in row and row["bacteria"]:
        output: Dict[str, object] = {"bacteria": row["bacteria"]}
    else:
        output = {}

    defense_diversity = 0
    defense_abi_burden = 0
    has_crispr = 0

    for source_column, feature_column in zip(retained_source_columns, retained_feature_columns, strict=True):
        if source_column not in row:
            raise KeyError(f"Missing retained Defense Finder column {source_column!r} in {defense_finder_output_path}")
        value = _parse_binary_flag(str(row[source_column]))
        output[feature_column] = value
        defense_diversity += value
        if source_column.startswith("Abi"):
            defense_abi_burden += value
        if source_column.startswith("CAS_"):
            has_crispr = max(has_crispr, value)

    derived_values = {
        "host_defense_diversity": defense_diversity,
        "host_defense_has_crispr": has_crispr,
        "host_defense_abi_burden": defense_abi_burden,
    }
    for column in derived_columns:
        if column not in derived_values:
            raise ValueError(f"Unexpected derived defense column {column!r} in mask {column_mask_path}")
        output[column] = derived_values[column]
    return output
