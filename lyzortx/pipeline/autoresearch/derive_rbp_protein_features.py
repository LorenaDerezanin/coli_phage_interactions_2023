"""Derive per-phage RBP protein descriptor features from genome FASTA + Pharokka annotations.

Extracts RBP CDS regions from phage genome FASTAs using Pharokka annotation coordinates,
translates to protein, and computes physicochemical descriptor vectors per RBP. Per-phage
features are the mean-pooled descriptor vectors across all annotated RBPs.

For phages without Pharokka-detected RBPs (16/97 in the current panel), all
descriptor features are zero and the `has_annotated_rbp` indicator is 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from lyzortx.pipeline.track_l.steps.parse_annotations import (
    CdsRecord,
    classify_rbp_genes,
    parse_merged_tsv,
)

LOGGER = logging.getLogger(__name__)

# Standard amino acids for composition vector.
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_COUNT = len(AMINO_ACIDS)

# Feature names for per-RBP protein descriptors (26 total).
AA_COMP_FEATURES = [f"aa_{aa}" for aa in AMINO_ACIDS]
PHYSICO_FEATURES = [
    "log_length",
    "molecular_weight",
    "gravy",
    "aromaticity",
    "isoelectric_point",
    "charge_ph7",
]
PER_RBP_FEATURE_NAMES = AA_COMP_FEATURES + PHYSICO_FEATURES

# Per-phage aggregated feature names: mean-pooled descriptors + metadata.
PHAGE_FEATURE_NAMES = ["has_annotated_rbp", "rbp_count"] + [f"rbp_mean_{name}" for name in PER_RBP_FEATURE_NAMES]


@dataclass(frozen=True)
class RbpProtein:
    """An extracted RBP protein sequence with metadata."""

    gene_id: str
    protein_seq: str
    annot: str


def read_genome_sequence(fasta_path: Path) -> dict[str, str]:
    """Read a genome FASTA and return {contig_id: sequence} mapping."""
    contigs: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    contigs[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
    if header is not None and chunks:
        contigs[header] = "".join(chunks)
    if not contigs:
        raise ValueError(f"No contigs found in {fasta_path}")
    return contigs


def extract_cds_protein(record: CdsRecord, contigs: dict[str, str]) -> str:
    """Extract CDS nucleotide sequence and translate to protein.

    Uses standard bacterial translation table (11). Handles forward and
    reverse complement strands.
    """
    contig_seq = contigs.get(record.contig)
    if contig_seq is None:
        raise ValueError(
            f"Contig {record.contig!r} for gene {record.gene!r} not found in genome. "
            f"Available contigs: {sorted(contigs)}"
        )
    # Pharokka uses 1-based coordinates.
    start = record.start - 1
    stop = record.stop
    if start < 0 or stop > len(contig_seq):
        raise ValueError(
            f"Gene {record.gene!r} coordinates ({record.start}-{record.stop}) "
            f"exceed contig {record.contig!r} length ({len(contig_seq)})"
        )
    nuc_seq = contig_seq[start:stop]
    if record.strand == "-":
        nuc_seq = str(Seq(nuc_seq).reverse_complement())

    # Translate with bacterial table 11, allowing alternative start codons.
    protein = str(Seq(nuc_seq).translate(table=11))
    # Strip trailing stop codon if present.
    if protein.endswith("*"):
        protein = protein[:-1]
    return protein


def compute_protein_descriptors(protein_seq: str) -> np.ndarray:
    """Compute a feature vector for a single protein sequence.

    Returns a 26-element vector: 20 amino acid frequencies + 6 physicochemical
    properties (log_length, molecular_weight, GRAVY, aromaticity, pI, charge_at_pH7).
    """
    # Filter to standard amino acids for ProteinAnalysis.
    clean_seq = "".join(aa for aa in protein_seq if aa in AMINO_ACIDS)
    if len(clean_seq) < 10:
        return np.zeros(len(PER_RBP_FEATURE_NAMES), dtype=np.float64)

    analysis = ProteinAnalysis(clean_seq)
    aa_percent = analysis.amino_acids_percent
    composition = np.array([aa_percent.get(aa, 0.0) for aa in AMINO_ACIDS], dtype=np.float64)

    physico = np.array(
        [
            np.log1p(len(clean_seq)),
            analysis.molecular_weight() / 1e5,  # Scale to ~0-1 range for typical RBPs.
            analysis.gravy(),
            analysis.aromaticity(),
            analysis.isoelectric_point(),
            analysis.charge_at_pH(7.0) / 100.0,  # Scale to ~0-1 range.
        ],
        dtype=np.float64,
    )

    return np.concatenate([composition, physico])


def extract_rbp_proteins_for_phage(
    phage_name: str,
    genome_fasta_path: Path,
    annotation_dir: Path,
) -> list[RbpProtein]:
    """Extract RBP protein sequences for a single phage.

    Returns empty list if no annotation file found or no RBPs annotated.
    """
    tsv_path = annotation_dir / f"{phage_name}_cds_final_merged_output.tsv"
    if not tsv_path.exists():
        LOGGER.debug("No Pharokka annotation for phage %s", phage_name)
        return []

    records = parse_merged_tsv(tsv_path)
    rbp_records = classify_rbp_genes(records)
    if not rbp_records:
        LOGGER.debug("No RBP genes annotated for phage %s", phage_name)
        return []

    contigs = read_genome_sequence(genome_fasta_path)
    proteins: list[RbpProtein] = []
    for rec in rbp_records:
        try:
            protein_seq = extract_cds_protein(rec, contigs)
            if len(protein_seq) >= 10:
                proteins.append(
                    RbpProtein(
                        gene_id=rec.gene,
                        protein_seq=protein_seq,
                        annot=rec.annot,
                    )
                )
        except (ValueError, KeyError) as exc:
            LOGGER.warning("Skipping RBP %s in phage %s: %s", rec.gene, phage_name, exc)
    return proteins


def build_phage_rbp_feature_row(
    phage_name: str,
    genome_fasta_path: Path,
    annotation_dir: Path,
) -> dict[str, object]:
    """Build a single feature row for one phage.

    Returns a dict with keys: phage, has_annotated_rbp, rbp_count, rbp_mean_aa_A, ..., etc.
    """
    proteins = extract_rbp_proteins_for_phage(phage_name, genome_fasta_path, annotation_dir)

    row: dict[str, object] = {"phage": phage_name}
    if not proteins:
        row["has_annotated_rbp"] = 0
        row["rbp_count"] = 0
        for name in PER_RBP_FEATURE_NAMES:
            row[f"rbp_mean_{name}"] = 0.0
        return row

    # Compute descriptor vectors for each RBP and mean-pool.
    descriptors = np.array([compute_protein_descriptors(p.protein_seq) for p in proteins], dtype=np.float64)
    mean_descriptors = descriptors.mean(axis=0)

    row["has_annotated_rbp"] = 1
    row["rbp_count"] = len(proteins)
    for i, name in enumerate(PER_RBP_FEATURE_NAMES):
        row[f"rbp_mean_{name}"] = round(float(mean_descriptors[i]), 6)
    return row


def build_phage_rbp_schema() -> list[str]:
    """Return the ordered list of feature column names (without entity key)."""
    return list(PHAGE_FEATURE_NAMES)
