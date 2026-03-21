#!/usr/bin/env python3
"""Build reproducible per-phage protein FASTA sets from genome or protein inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pyrodigal

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

PROTEIN_SUFFIXES: Tuple[str, ...] = (".faa", ".aa", ".pep")
GENOME_FASTA_SUFFIXES: Tuple[str, ...] = (".fna", ".fa", ".fasta")
GENBANK_SUFFIXES: Tuple[str, ...] = (".gb", ".gbk", ".genbank")
SUPPORTED_SUFFIXES: Tuple[str, ...] = (*PROTEIN_SUFFIXES, *GENOME_FASTA_SUFFIXES, *GENBANK_SUFFIXES)
PANEL_METADATA_COLUMNS: Tuple[str, ...] = ("phage",)
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "phage",
    "input_type",
    "input_path",
    "input_sha256",
    "candidate_input_count",
    "sequence_record_count",
    "genome_nt_count",
    "protein_count",
    "total_protein_aa_count",
    "mean_protein_aa_length",
    "protein_source",
    "gene_finder_mode",
)
GENBANK_TRANSLATION_RE = re.compile(r'/translation="([^"]*)"')
MIN_SINGLE_GENOME_TRAINING_NT = 100_000


@dataclass(frozen=True)
class SequenceRecord:
    """One nucleotide or protein FASTA-style record."""

    identifier: str
    description: str
    sequence: str


@dataclass(frozen=True)
class PhageInput:
    """Chosen input artifact for one phage."""

    phage: str
    path: Path
    input_type: str
    candidate_count: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Semicolon-delimited phage panel metadata containing the canonical phage names.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/genomics/phages"),
        help="Root directory containing phage genome/protein inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_protein_sets"),
        help="Directory for per-phage protein FASTA files plus summary artifacts.",
    )
    parser.add_argument(
        "--expected-panel-count",
        type=int,
        default=96,
        help="Expected number of phages in the panel metadata.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_protein_sequence(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char.isalpha() or char == "*").rstrip("*")


def _clean_nucleotide_sequence(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char.isalpha())


def _path_suffix(path: Path) -> str:
    return path.suffix.lower()


def _input_type_for_path(path: Path) -> str:
    suffix = _path_suffix(path)
    if suffix in PROTEIN_SUFFIXES:
        return "protein_fasta"
    if suffix in GENBANK_SUFFIXES:
        return "genbank"
    if suffix in GENOME_FASTA_SUFFIXES:
        return "genome_fasta"
    raise ValueError(f"Unsupported phage input file type: {path}")


def _input_priority(path: Path) -> int:
    input_type = _input_type_for_path(path)
    if input_type == "protein_fasta":
        return 0
    if input_type == "genbank":
        return 1
    return 2


def read_panel_phages(path: Path, *, expected_panel_count: int) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [column for column in PANEL_METADATA_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        phages = [row["phage"].strip() for row in reader if row.get("phage", "").strip()]

    unique_phages = sorted(set(phages))
    if len(unique_phages) != len(phages):
        raise ValueError(f"Duplicate phage names found in panel metadata: {path}")
    if len(unique_phages) != expected_panel_count:
        raise ValueError(
            f"Unexpected phage panel size in {path}: expected={expected_panel_count}, actual={len(unique_phages)}"
        )
    return unique_phages


def discover_candidate_inputs(input_root: Path) -> Dict[str, List[Path]]:
    candidates: Dict[str, List[Path]] = {}
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        if _path_suffix(path) not in SUPPORTED_SUFFIXES:
            continue
        candidates.setdefault(path.stem, []).append(path)
    return candidates


def choose_inputs_for_panel(
    panel_phages: Sequence[str], candidate_inputs: Mapping[str, Sequence[Path]]
) -> List[PhageInput]:
    chosen: List[PhageInput] = []
    missing: List[str] = []
    for phage in panel_phages:
        candidates = list(candidate_inputs.get(phage, ()))
        if not candidates:
            missing.append(phage)
            continue
        candidates.sort(key=lambda path: (_input_priority(path), str(path)))
        selected = candidates[0]
        chosen.append(
            PhageInput(
                phage=phage,
                path=selected,
                input_type=_input_type_for_path(selected),
                candidate_count=len(candidates),
            )
        )

    if missing:
        raise ValueError("Missing phage input files for panel phages: " + ", ".join(sorted(missing)))
    return chosen


def read_fasta_records(path: Path, *, protein: bool) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    header: Optional[str] = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    records.append(_build_fasta_record(header, chunks, protein=protein))
                header = line[1:].strip()
                chunks = []
                continue
            chunks.append(line)

    if header is not None and chunks:
        records.append(_build_fasta_record(header, chunks, protein=protein))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _build_fasta_record(header: str, chunks: Sequence[str], *, protein: bool) -> SequenceRecord:
    if not header:
        raise ValueError("Encountered FASTA record with an empty header")
    parts = header.split(maxsplit=1)
    sequence = "".join(chunks)
    cleaned = _clean_protein_sequence(sequence) if protein else _clean_nucleotide_sequence(sequence)
    if not cleaned:
        raise ValueError(f"Encountered empty sequence for FASTA record {header!r}")
    return SequenceRecord(
        identifier=parts[0],
        description=header,
        sequence=cleaned,
    )


def extract_genbank_translations(path: Path) -> List[SequenceRecord]:
    text = path.read_text(encoding="utf-8")
    proteins: List[SequenceRecord] = []
    for index, match in enumerate(GENBANK_TRANSLATION_RE.finditer(text), start=1):
        sequence = _clean_protein_sequence(match.group(1))
        if not sequence:
            continue
        proteins.append(
            SequenceRecord(
                identifier=f"cds_{index:04d}",
                description=f"cds_{index:04d}",
                sequence=sequence,
            )
        )
    return proteins


def read_genbank_sequence_records(path: Path) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    locus_name = "record_0001"
    collecting = False
    chunks: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("LOCUS"):
                parts = line.split()
                if len(parts) >= 2:
                    locus_name = parts[1]
            elif line.startswith("ORIGIN"):
                collecting = True
                chunks = []
            elif line.startswith("//"):
                if collecting:
                    sequence = _clean_nucleotide_sequence("".join(chunks))
                    if sequence:
                        records.append(
                            SequenceRecord(
                                identifier=locus_name,
                                description=locus_name,
                                sequence=sequence,
                            )
                        )
                collecting = False
                chunks = []
            elif collecting:
                chunks.append("".join(char for char in line if char.isalpha()))

    if not records:
        raise ValueError(f"No ORIGIN sequence found in {path}")
    return records


def call_proteins_with_pyrodigal(
    phage: str, genome_records: Sequence[SequenceRecord]
) -> Tuple[List[SequenceRecord], str]:
    proteins: List[SequenceRecord] = []
    total_nt = sum(len(record.sequence) for record in genome_records)
    use_meta_mode = len(genome_records) > 1 or total_nt < MIN_SINGLE_GENOME_TRAINING_NT
    gene_finder_mode = "meta"

    if use_meta_mode:
        finder = pyrodigal.GeneFinder(meta=True)
        record_to_genes = [(record, finder.find_genes(record.sequence.encode("ascii"))) for record in genome_records]
    else:
        finder = pyrodigal.GeneFinder(meta=False)
        finder.train(genome_records[0].sequence.encode("ascii"))
        record_to_genes = [(genome_records[0], finder.find_genes(genome_records[0].sequence.encode("ascii")))]
        gene_finder_mode = "single"

    protein_index = 1
    for record, genes in record_to_genes:
        for gene in genes:
            translation = str(gene.translate()).rstrip("*")
            if not translation:
                continue
            proteins.append(
                SequenceRecord(
                    identifier=f"{phage}|prot_{protein_index:04d}",
                    description=(
                        f"{phage}|prot_{protein_index:04d} contig={record.identifier} "
                        f"start={gene.begin} end={gene.end} strand={gene.strand}"
                    ),
                    sequence=translation,
                )
            )
            protein_index += 1

    if not proteins:
        raise ValueError(f"Pyrodigal did not predict any proteins for {phage}")
    return proteins, gene_finder_mode


def write_fasta_records(path: Path, records: Sequence[SequenceRecord]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.description}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(f"{record.sequence[start : start + 80]}\n")


def build_protein_sets(
    *,
    panel_phages: Sequence[str],
    input_root: Path,
    output_dir: Path,
    metadata_path: Path,
) -> Dict[str, object]:
    candidate_inputs = discover_candidate_inputs(input_root)
    selected_inputs = choose_inputs_for_panel(panel_phages, candidate_inputs)

    protein_dir = output_dir / "protein_fastas"
    ensure_directory(protein_dir)
    summary_rows: List[Dict[str, object]] = []

    for item in selected_inputs:
        input_path = item.path
        protein_source = item.input_type
        gene_finder_mode = ""
        genome_nt_count = 0
        sequence_record_count = 0

        if item.input_type == "protein_fasta":
            proteins = read_fasta_records(input_path, protein=True)
            sequence_record_count = len(proteins)
        elif item.input_type == "genbank":
            proteins = extract_genbank_translations(input_path)
            if proteins:
                protein_source = "genbank_translation"
                sequence_record_count = len(proteins)
            else:
                genome_records = read_genbank_sequence_records(input_path)
                sequence_record_count = len(genome_records)
                genome_nt_count = sum(len(record.sequence) for record in genome_records)
                proteins, gene_finder_mode = call_proteins_with_pyrodigal(item.phage, genome_records)
                protein_source = "genbank_pyrodigal"
        else:
            genome_records = read_fasta_records(input_path, protein=False)
            sequence_record_count = len(genome_records)
            genome_nt_count = sum(len(record.sequence) for record in genome_records)
            proteins, gene_finder_mode = call_proteins_with_pyrodigal(item.phage, genome_records)
            protein_source = "genome_pyrodigal"

        output_path = protein_dir / f"{item.phage}.faa"
        write_fasta_records(output_path, proteins)
        total_aa = sum(len(record.sequence) for record in proteins)
        summary_rows.append(
            {
                "phage": item.phage,
                "input_type": item.input_type,
                "input_path": str(input_path),
                "input_sha256": _sha256(input_path),
                "candidate_input_count": item.candidate_count,
                "sequence_record_count": sequence_record_count,
                "genome_nt_count": genome_nt_count,
                "protein_count": len(proteins),
                "total_protein_aa_count": total_aa,
                "mean_protein_aa_length": round(total_aa / len(proteins), 3),
                "protein_source": protein_source,
                "gene_finder_mode": gene_finder_mode,
            }
        )

    summary_rows.sort(key=lambda row: str(row["phage"]))
    summary_path = output_dir / "phage_protein_summary.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)

    non_panel_inputs = sorted(set(candidate_inputs) - set(panel_phages))
    manifest = {
        "step_name": "build_phage_protein_sets",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "phage_metadata_path": str(metadata_path),
            "input_root": str(input_root),
        },
        "counts": {
            "panel_phage_count": len(panel_phages),
            "processed_phage_count": len(summary_rows),
            "ignored_non_panel_input_count": len(non_panel_inputs),
            "total_protein_count": sum(int(row["protein_count"]) for row in summary_rows),
        },
        "ignored_non_panel_inputs": non_panel_inputs,
        "output_format": {
            "per_phage_fasta_dir": str(protein_dir),
            "per_phage_fasta_pattern": "<phage>.faa",
            "per_phage_fasta_header": (
                "Predicted proteins use '<phage>|prot_0001 ... start=<n> end=<n> strand=<±1>'. "
                "Copied protein FASTA inputs preserve the original record description."
            ),
            "summary_csv": str(summary_path),
        },
        "reproducibility": {
            "one_command": "python lyzortx/pipeline/track_d/run_track_d.py",
            "pyrodigal_version": pyrodigal.__version__,
            "gene_finder_policy": (
                "Prefer supplied protein FASTA, then GenBank CDS translations, otherwise call proteins with "
                "pyrodigal (meta mode for multi-record or <100 kb genomes; trained single-genome mode otherwise)."
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    panel_phages = read_panel_phages(args.phage_metadata_path, expected_panel_count=args.expected_panel_count)
    manifest = build_protein_sets(
        panel_phages=panel_phages,
        input_root=args.input_root,
        output_dir=args.output_dir,
        metadata_path=args.phage_metadata_path,
    )
    print("Built per-phage protein FASTA sets.")
    print(f"- Panel phages processed: {manifest['counts']['processed_phage_count']}")
    print(f"- Ignored non-panel inputs: {manifest['counts']['ignored_non_panel_input_count']}")


if __name__ == "__main__":
    main()
