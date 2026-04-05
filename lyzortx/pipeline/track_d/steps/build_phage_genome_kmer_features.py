#!/usr/bin/env python3
"""TD02: Build phage genome tetranucleotide embedding features from FNA inputs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_fasta_records, read_panel_phages

GENOME_FASTA_SUFFIXES: Tuple[str, ...] = (".fna", ".fa", ".fasta")
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "phage",
    "input_path",
    "input_sha256",
    "sequence_record_count",
    "genome_length_nt",
    "gc_content",
    "valid_kmer_window_count",
    "included_in_panel_feature_csv",
)
METADATA_COLUMNS: Tuple[str, ...] = (
    "column_name",
    "feature_group",
    "feature_type",
    "source_path",
    "source_column",
    "transform",
    "provenance_note",
)
SVD_ARTIFACT_NAME = "phage_genome_kmer_svd.joblib"
NUCLEOTIDE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass(frozen=True)
class GenomeInput:
    """One discovered phage genome FASTA input."""

    phage: str
    path: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Semicolon-delimited phage panel metadata containing the canonical phage names.",
    )
    parser.add_argument(
        "--fna-dir",
        type=Path,
        default=Path("data/genomics/phages/FNA"),
        help="Directory containing phage genome FASTA inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_genome_kmer_features"),
        help="Directory for generated Track D genome k-mer feature artifacts.",
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=4,
        help="k-mer size used for the genome frequency vectors.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=24,
        help="Requested TruncatedSVD embedding dimension. Output is padded to this width if needed.",
    )
    parser.add_argument(
        "--expected-panel-count",
        type=int,
        default=96,
        help="Expected number of phages in the panel metadata.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for TruncatedSVD.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_genome_inputs(fna_dir: Path) -> List[GenomeInput]:
    discovered: Dict[str, Path] = {}
    for path in sorted(fna_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in GENOME_FASTA_SUFFIXES:
            continue
        phage = path.stem
        existing = discovered.get(phage)
        if existing is not None:
            raise ValueError(f"Duplicate genome FASTA discovered for {phage!r}: {existing} and {path}")
        discovered[phage] = path
    if not discovered:
        raise ValueError(f"No genome FASTA files found in {fna_dir}")
    return [GenomeInput(phage=phage, path=path) for phage, path in sorted(discovered.items())]


def compute_kmer_frequency_vector(sequence: str, *, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")

    vector = np.zeros(4**k, dtype=np.float64)
    if len(sequence) < k:
        return vector

    valid_windows = 0
    for start in range(len(sequence) - k + 1):
        index = 0
        is_valid = True
        for char in sequence[start : start + k]:
            nucleotide_index = NUCLEOTIDE_TO_INDEX.get(char)
            if nucleotide_index is None:
                is_valid = False
                break
            index = (index * 4) + nucleotide_index
        if not is_valid:
            continue
        vector[index] += 1.0
        valid_windows += 1

    if valid_windows:
        vector /= valid_windows
    return vector


def _count_valid_kmer_windows(sequence: str, *, k: int) -> int:
    if len(sequence) < k:
        return 0
    valid = 0
    for start in range(len(sequence) - k + 1):
        if all(char in NUCLEOTIDE_TO_INDEX for char in sequence[start : start + k]):
            valid += 1
    return valid


def gc_content(sequences: Sequence[str]) -> float:
    gc_count = 0
    valid_count = 0
    for sequence in sequences:
        for char in sequence:
            if char not in NUCLEOTIDE_TO_INDEX:
                continue
            valid_count += 1
            if char in {"G", "C"}:
                gc_count += 1
    if valid_count == 0:
        raise ValueError("Cannot compute GC content from sequences with no A/C/G/T bases")
    return gc_count / valid_count


def build_kmer_matrix(
    genome_inputs: Sequence[GenomeInput],
    *,
    k: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    if not genome_inputs:
        raise ValueError("genome_inputs must not be empty")

    summary_rows: List[Dict[str, object]] = []
    matrix = np.zeros((len(genome_inputs), 4**k), dtype=np.float64)

    for index, genome in enumerate(genome_inputs):
        records = read_fasta_records(genome.path, protein=False)
        sequences = [record.sequence for record in records]
        total_windows = 0
        combined_vector = np.zeros(4**k, dtype=np.float64)
        for sequence in sequences:
            vector = compute_kmer_frequency_vector(sequence, k=k)
            valid_windows = _count_valid_kmer_windows(sequence, k=k)
            if valid_windows:
                combined_vector += vector * valid_windows
            total_windows += valid_windows

        if total_windows == 0:
            raise ValueError(f"No valid {k}-mer windows found for {genome.phage}")
        combined_vector /= total_windows
        matrix[index, :] = combined_vector

        genome_length_nt = sum(len(sequence) for sequence in sequences)
        summary_rows.append(
            {
                "phage": genome.phage,
                "input_path": str(genome.path),
                "input_sha256": _sha256(genome.path),
                "sequence_record_count": len(records),
                "genome_length_nt": genome_length_nt,
                "gc_content": round(gc_content(sequences), 6),
                "valid_kmer_window_count": total_windows,
            }
        )

    return matrix, summary_rows


def reduce_kmer_matrix(
    matrix: np.ndarray,
    *,
    embedding_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, object], TruncatedSVD]:
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be two-dimensional and non-empty")

    effective_dim = min(embedding_dim, max(1, matrix.shape[0] - 1), matrix.shape[1])
    svd = TruncatedSVD(n_components=effective_dim, random_state=random_state)
    reduced = svd.fit_transform(matrix)

    if effective_dim < embedding_dim:
        padded = np.zeros((matrix.shape[0], embedding_dim), dtype=np.float64)
        padded[:, :effective_dim] = reduced
        reduced = padded

    metadata = {
        "requested_embedding_dim": embedding_dim,
        "effective_embedding_dim": effective_dim,
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "singular_values": [float(value) for value in svd.singular_values_],
    }
    return reduced, metadata, svd


def _feature_columns(embedding_dim: int) -> Tuple[str, ...]:
    return tuple(f"phage_genome_tetra_svd_{index:02d}" for index in range(embedding_dim))


def build_feature_rows(
    *,
    panel_phages: Sequence[str],
    genome_summary_rows: Sequence[Mapping[str, object]],
    reduced_matrix: np.ndarray,
    embedding_dim: int,
) -> List[Dict[str, object]]:
    summary_by_phage = {str(row["phage"]): row for row in genome_summary_rows}
    reduced_by_phage = {str(row["phage"]): reduced_matrix[index, :] for index, row in enumerate(genome_summary_rows)}
    missing = sorted(set(panel_phages) - set(summary_by_phage))
    if missing:
        raise ValueError("Missing genome FASTA files for panel phages: " + ", ".join(missing))

    feature_rows: List[Dict[str, object]] = []
    for phage in sorted(panel_phages):
        summary = summary_by_phage[phage]
        reduced = reduced_by_phage[phage]
        row: Dict[str, object] = {
            "phage": phage,
            "phage_gc_content": summary["gc_content"],
            "phage_genome_length_nt": int(summary["genome_length_nt"]),
        }
        for index, value in enumerate(reduced[:embedding_dim]):
            row[f"phage_genome_tetra_svd_{index:02d}"] = round(float(value), 6)
        feature_rows.append(row)
    return feature_rows


def build_metadata_rows(
    *,
    feature_columns: Sequence[str],
    fna_dir: Path,
    kmer_size: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for column_name in feature_columns:
        rows.append(
            {
                "column_name": column_name,
                "feature_group": "genome_embedding",
                "feature_type": "continuous",
                "source_path": str(fna_dir),
                "source_column": f"{kmer_size}-mer frequency vector",
                "transform": (
                    f"Compute per-genome normalized {kmer_size}-mer frequencies across all FASTA records and project "
                    "with TruncatedSVD."
                ),
                "provenance_note": "Embedding is fit across all discovered genome FASTA files, then panel phages are emitted.",
            }
        )

    rows.extend(
        [
            {
                "column_name": "phage_gc_content",
                "feature_group": "genome_summary",
                "feature_type": "continuous",
                "source_path": str(fna_dir),
                "source_column": "FASTA nucleotide sequence",
                "transform": "Compute fraction of valid A/C/G/T bases that are G or C across all records for each phage.",
                "provenance_note": "GC content is emitted directly alongside the embedding coordinates.",
            },
            {
                "column_name": "phage_genome_length_nt",
                "feature_group": "genome_summary",
                "feature_type": "continuous",
                "source_path": str(fna_dir),
                "source_column": "FASTA nucleotide sequence",
                "transform": "Sum nucleotide counts across all FASTA records for each phage genome.",
                "provenance_note": "Genome length stays on the original scale as a continuous feature.",
            },
        ]
    )
    return rows


def build_genome_kmer_feature_block(
    *,
    panel_phages: Sequence[str],
    fna_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    embedding_dim: int,
    kmer_size: int = 4,
    random_state: int = 0,
) -> Dict[str, object]:
    if kmer_size < 1:
        raise ValueError("kmer_size must be >= 1")

    genome_inputs = discover_genome_inputs(fna_dir)
    matrix, summary_rows = build_kmer_matrix(genome_inputs, k=kmer_size)
    reduced_matrix, svd_metadata, svd = reduce_kmer_matrix(
        matrix, embedding_dim=embedding_dim, random_state=random_state
    )

    panel_set = set(panel_phages)
    for row in summary_rows:
        row["included_in_panel_feature_csv"] = int(str(row["phage"]) in panel_set)
    summary_rows.sort(key=lambda row: str(row["phage"]))

    feature_rows = build_feature_rows(
        panel_phages=panel_phages,
        genome_summary_rows=summary_rows,
        reduced_matrix=reduced_matrix,
        embedding_dim=embedding_dim,
    )
    feature_columns = _feature_columns(embedding_dim)
    feature_fieldnames = ("phage", *feature_columns, "phage_gc_content", "phage_genome_length_nt")
    metadata_rows = build_metadata_rows(
        feature_columns=feature_columns,
        fna_dir=fna_dir,
        kmer_size=kmer_size,
    )

    ensure_directory(output_dir)
    features_path = output_dir / "phage_genome_kmer_features.csv"
    summary_path = output_dir / "phage_genome_kmer_source_summary.csv"
    metadata_csv_path = output_dir / "phage_genome_kmer_feature_metadata.csv"
    svd_path = output_dir / SVD_ARTIFACT_NAME
    write_csv(features_path, feature_fieldnames, feature_rows)
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    write_csv(metadata_csv_path, METADATA_COLUMNS, metadata_rows)
    joblib.dump(svd, svd_path)

    non_panel_genomes = sorted(str(row["phage"]) for row in summary_rows if str(row["phage"]) not in panel_set)
    manifest = {
        "step_name": "build_phage_genome_kmer_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "phage_metadata_path": str(metadata_path),
            "fna_dir": str(fna_dir),
            "kmer_size": kmer_size,
            "embedding_dim_requested": embedding_dim,
            "random_state": random_state,
        },
        "counts": {
            "panel_phage_count": len(panel_phages),
            "discovered_genome_count": len(summary_rows),
            "svd_source_genome_count": len(summary_rows),
            "output_row_count": len(feature_rows),
            "non_panel_genome_count": len(non_panel_genomes),
            "kmer_vector_dim": int(4**kmer_size),
            "embedding_dim_effective": int(svd_metadata["effective_embedding_dim"]),
        },
        "non_panel_genomes": non_panel_genomes,
        "output_format": {
            "feature_csv": str(features_path),
            "feature_metadata_csv": str(metadata_csv_path),
            "source_summary_csv": str(summary_path),
            "feature_columns": list(feature_fieldnames),
            "svd_joblib": str(svd_path),
        },
        "reproducibility": {
            "one_command": "python lyzortx/pipeline/track_d/run_track_d.py --step genome-kmers",
            "svd_explained_variance_ratio_sum": round(float(svd_metadata["explained_variance_ratio_sum"]), 6),
            "svd_singular_values": [round(float(value), 6) for value in svd_metadata["singular_values"]],
            "panel_output_policy": (
                "Fit the unsupervised embedding on all discovered genome FASTA files in FNA/, "
                "then emit rows only for canonical panel phages so the feature block joins cleanly on phage."
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    panel_phages = read_panel_phages(args.phage_metadata_path, expected_panel_count=args.expected_panel_count)
    manifest = build_genome_kmer_feature_block(
        panel_phages=panel_phages,
        fna_dir=args.fna_dir,
        output_dir=args.output_dir,
        metadata_path=args.phage_metadata_path,
        embedding_dim=args.embedding_dim,
        kmer_size=args.kmer_size,
        random_state=args.random_state,
    )
    print("Built phage genome k-mer embedding features.")
    print(f"- Panel rows written: {manifest['counts']['output_row_count']}")
    print(f"- Genome FASTA files used for SVD: {manifest['counts']['svd_source_genome_count']}")
    print(f"- Non-panel genomes retained in source summary only: {manifest['counts']['non_panel_genome_count']}")


if __name__ == "__main__":
    main()
