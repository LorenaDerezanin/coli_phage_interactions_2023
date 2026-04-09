"""Tests for derive_rbp_protein_features: CDS extraction, protein descriptors, per-phage pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyzortx.pipeline.autoresearch.derive_rbp_protein_features import (
    AA_COUNT,
    PHAGE_FEATURE_NAMES,
    PER_RBP_FEATURE_NAMES,
    build_phage_rbp_feature_row,
    compute_protein_descriptors,
    extract_cds_protein,
    extract_rbp_proteins_for_phage,
    read_genome_sequence,
)
from lyzortx.pipeline.track_l.steps.parse_annotations import CdsRecord

# A minimal phage genome (200 bp) with a known ORF for testing translation.
# Encodes ATG + 30 codons + TAA stop = 96 nt CDS at position 11-106.
GENOME_SEQ = (
    "NNNNNNNNNN"  # 10 nt padding
    "ATGGCTAAAGCTGCTGCTAAAGCTGCTGCTAAAGCTGCTGCTAAAGCTGCTGCTAAAGCT"
    "GCTGCTAAAGCTGCTGCTAAAGCTGCTGCTAAAGCTGCTGCTTAA"  # 96 nt CDS
    "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"  # padding to 200 nt
    "NN"
)
# Expected protein: M + (AKA)*10 = 31 residues. Stop removed.
EXPECTED_PROTEIN = "M" + "AKAAKA" * 5 + "A"


def _write_genome_fasta(path: Path, contig_name: str = "test_phage", seq: str = GENOME_SEQ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{contig_name}\n{seq}\n", encoding="utf-8")


def _write_pharokka_tsv(path: Path, phage_name: str, genes: list[dict[str, str]]) -> None:
    """Write a minimal Pharokka merged TSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal columns needed by parse_merged_tsv.
    header = "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\tmmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\tcustom_hmm_bitscore\tcustom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\tvfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\tCARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\tProtein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table"
    lines = [header]
    for g in genes:
        # Fill required columns; most can be placeholders.
        row_values = [
            g["gene"],
            g["start"],
            g["stop"],
            g.get("strand", "+"),
            g.get("contig", phage_name),
            "-1",
            "No_PHROG",
            "0",
            "0",
            "1",
            "No_PHROGs_HMM",
            "0",
            "1",
            "No_custom_HMM",
            "No_custom_HMM",
            "No_custom_HMM",
            "No_PHROG",
            "PHANOTATE",
            "CDS",
            "None",
            g.get("annot", "hypothetical protein"),
            g.get("category", "unknown function"),
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "11",
        ]
        lines.append("\t".join(str(v) for v in row_values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestReadGenomeSequence:
    def test_single_contig(self, tmp_path: Path) -> None:
        fasta_path = tmp_path / "phage.fna"
        _write_genome_fasta(fasta_path, "contig1", "ATGCATGC")
        contigs = read_genome_sequence(fasta_path)
        assert contigs == {"contig1": "ATGCATGC"}

    def test_empty_fasta_raises(self, tmp_path: Path) -> None:
        fasta_path = tmp_path / "empty.fna"
        fasta_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No contigs"):
            read_genome_sequence(fasta_path)


class TestExtractCdsProtein:
    def test_forward_strand_translation(self) -> None:
        contigs = {"test_phage": GENOME_SEQ}
        record = CdsRecord(
            gene="CDS_001",
            start=11,
            stop=106,
            strand="+",
            contig="test_phage",
            phrog="1",
            annot="tail fiber",
            category="tail",
        )
        protein = extract_cds_protein(record, contigs)
        assert protein.startswith("M")
        assert "*" not in protein
        assert len(protein) >= 10

    def test_missing_contig_raises(self) -> None:
        contigs = {"other_contig": "ATGCATGC"}
        record = CdsRecord(
            gene="CDS_001",
            start=1,
            stop=96,
            strand="+",
            contig="missing",
            phrog="1",
            annot="test",
            category="test",
        )
        with pytest.raises(ValueError, match="not found in genome"):
            extract_cds_protein(record, contigs)


class TestComputeProteinDescriptors:
    def test_output_shape(self) -> None:
        protein = "MAKAKAAKAAKAAKAAKAAKAAKAAKAAKAAKA"
        descriptors = compute_protein_descriptors(protein)
        assert descriptors.shape == (len(PER_RBP_FEATURE_NAMES),)
        assert descriptors.dtype == np.float64

    def test_short_protein_returns_zeros(self) -> None:
        descriptors = compute_protein_descriptors("MAK")
        assert np.all(descriptors == 0.0)

    def test_amino_acid_composition_sums_to_100(self) -> None:
        protein = "MAKAKAAKAAKAAKAAKAAKAAKAAKAAKAAKA"
        descriptors = compute_protein_descriptors(protein)
        aa_comp = descriptors[:AA_COUNT]
        assert abs(aa_comp.sum() - 100.0) < 1e-4


class TestExtractRbpProteinsForPhage:
    def test_phage_with_rbp(self, tmp_path: Path) -> None:
        genome_path = tmp_path / "phage.fna"
        _write_genome_fasta(genome_path, "TEST_P1", GENOME_SEQ)
        annotation_dir = tmp_path / "annotations"
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [
                {
                    "gene": "CDS_001",
                    "start": "11",
                    "stop": "106",
                    "contig": "TEST_P1",
                    "annot": "tail fiber protein",
                    "category": "tail",
                }
            ],
        )
        proteins = extract_rbp_proteins_for_phage("TEST_P1", genome_path, annotation_dir)
        assert len(proteins) == 1
        assert proteins[0].gene_id == "CDS_001"
        assert len(proteins[0].protein_seq) >= 10

    def test_phage_without_annotation(self, tmp_path: Path) -> None:
        genome_path = tmp_path / "phage.fna"
        _write_genome_fasta(genome_path)
        proteins = extract_rbp_proteins_for_phage("MISSING_P1", genome_path, tmp_path)
        assert proteins == []


class TestBuildPhageRbpFeatureRow:
    def test_phage_with_rbps(self, tmp_path: Path) -> None:
        genome_path = tmp_path / "phage.fna"
        _write_genome_fasta(genome_path, "TEST_P1", GENOME_SEQ)
        annotation_dir = tmp_path / "annotations"
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [
                {
                    "gene": "CDS_001",
                    "start": "11",
                    "stop": "106",
                    "contig": "TEST_P1",
                    "annot": "tail fiber protein",
                    "category": "tail",
                }
            ],
        )
        row = build_phage_rbp_feature_row("TEST_P1", genome_path, annotation_dir)
        assert row["phage"] == "TEST_P1"
        assert row["has_annotated_rbp"] == 1
        assert row["rbp_count"] == 1
        # All mean features should be populated.
        for name in PER_RBP_FEATURE_NAMES:
            assert f"rbp_mean_{name}" in row

    def test_phage_without_rbps(self, tmp_path: Path) -> None:
        genome_path = tmp_path / "phage.fna"
        _write_genome_fasta(genome_path, "TEST_P1", GENOME_SEQ)
        annotation_dir = tmp_path / "annotations"
        # Write an annotation with no RBP genes.
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [
                {
                    "gene": "CDS_001",
                    "start": "11",
                    "stop": "106",
                    "contig": "TEST_P1",
                    "annot": "terminase large subunit",
                    "category": "head and packaging",
                }
            ],
        )
        row = build_phage_rbp_feature_row("TEST_P1", genome_path, annotation_dir)
        assert row["has_annotated_rbp"] == 0
        assert row["rbp_count"] == 0
        for name in PER_RBP_FEATURE_NAMES:
            assert row[f"rbp_mean_{name}"] == 0.0

    def test_feature_count(self, tmp_path: Path) -> None:
        genome_path = tmp_path / "phage.fna"
        _write_genome_fasta(genome_path, "TEST_P1", GENOME_SEQ)
        row = build_phage_rbp_feature_row("TEST_P1", genome_path, tmp_path)
        # 1 (phage) + len(PHAGE_FEATURE_NAMES) keys.
        assert len(row) == 1 + len(PHAGE_FEATURE_NAMES)
