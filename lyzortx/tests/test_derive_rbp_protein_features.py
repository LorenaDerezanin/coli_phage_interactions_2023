"""Tests for derive_rbp_protein_features: CDS extraction, PLM embedding loading, and PCA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyzortx.pipeline.autoresearch.derive_rbp_protein_features import (
    DEFAULT_N_COMPONENTS,
    METADATA_FEATURES,
    RAW_EMBEDDING_DIM,
    build_phage_rbp_plm_rows,
    build_phage_rbp_schema,
    extract_cds_protein,
    extract_rbp_proteins_for_phage,
    fit_pca_on_embeddings,
    load_cached_embeddings,
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


def _create_fake_embedding_cache(path: Path, n_phages: int = 10, n_with_rbp: int = 8) -> None:
    """Create a synthetic .npz embedding cache for testing."""
    rng = np.random.RandomState(42)
    phage_names = [f"PHAGE_P{i}" for i in range(1, n_phages + 1)]
    embeddings = rng.randn(n_phages, RAW_EMBEDDING_DIM).astype(np.float32)
    has_rbp = np.array([i < n_with_rbp for i in range(n_phages)])
    rbp_counts = np.array([2 if has_rbp[i] else 0 for i in range(n_phages)])

    # Zero out embeddings for phages without RBPs.
    embeddings[~has_rbp] = 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        phage_names=np.array(phage_names),
        embeddings=embeddings,
        has_rbp=has_rbp,
        rbp_counts=rbp_counts,
        model_backend=np.array("test"),
    )


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


class TestBuildPhageRbpSchema:
    def test_default_schema_length(self) -> None:
        schema = build_phage_rbp_schema()
        # 2 metadata + 32 PCA components.
        assert len(schema) == 2 + DEFAULT_N_COMPONENTS

    def test_schema_starts_with_metadata(self) -> None:
        schema = build_phage_rbp_schema()
        assert schema[:2] == METADATA_FEATURES

    def test_custom_n_components(self) -> None:
        schema = build_phage_rbp_schema(n_components=16)
        assert len(schema) == 2 + 16
        assert schema[-1] == "rbp_plm_pc15"


class TestLoadCachedEmbeddings:
    def test_loads_valid_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        _create_fake_embedding_cache(cache_path, n_phages=5, n_with_rbp=3)
        data = load_cached_embeddings(cache_path)
        assert len(data["phage_names"]) == 5
        assert data["embeddings"].shape == (5, RAW_EMBEDDING_DIM)
        assert data["model_backend"] == "test"

    def test_missing_cache_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="PLM embedding cache not found"):
            load_cached_embeddings(tmp_path / "nonexistent.npz")


class TestFitPcaOnEmbeddings:
    def test_fits_on_non_zero_only(self) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, RAW_EMBEDDING_DIM).astype(np.float32)
        has_rbp = np.array([True] * 8 + [False] * 2)
        embeddings[~has_rbp] = 0.0

        pca = fit_pca_on_embeddings(embeddings, has_rbp, n_components=4)
        assert pca.n_components_ == 4
        assert pca.explained_variance_ratio_.sum() > 0

    def test_clamps_components_to_available_samples(self) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(5, RAW_EMBEDDING_DIM).astype(np.float32)
        has_rbp = np.array([True] * 3 + [False] * 2)
        embeddings[~has_rbp] = 0.0

        # Request 32 components but only 3 non-zero samples → max 2 components.
        pca = fit_pca_on_embeddings(embeddings, has_rbp, n_components=32)
        assert pca.n_components_ == 2


class TestBuildPhageRbpPlmRows:
    def test_row_count_and_structure(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        _create_fake_embedding_cache(cache_path, n_phages=10, n_with_rbp=8)

        rows = build_phage_rbp_plm_rows(cache_path, n_components=4)
        assert len(rows) == 10

        # Check row keys.
        expected_keys = {"phage", "has_annotated_rbp", "rbp_count"} | {f"rbp_plm_pc{i}" for i in range(4)}
        assert set(rows[0].keys()) == expected_keys

    def test_zero_embedding_phages_have_zero_pca(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        _create_fake_embedding_cache(cache_path, n_phages=10, n_with_rbp=8)

        rows = build_phage_rbp_plm_rows(cache_path, n_components=4)

        # Last 2 phages have no RBPs — PCA features should be zero.
        for row in rows:
            if row["has_annotated_rbp"] == 0:
                for i in range(4):
                    assert row[f"rbp_plm_pc{i}"] == 0.0

    def test_non_zero_phages_have_non_zero_pca(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        _create_fake_embedding_cache(cache_path, n_phages=10, n_with_rbp=8)

        rows = build_phage_rbp_plm_rows(cache_path, n_components=4)

        # At least some phages with RBPs should have non-zero PCA features.
        has_nonzero = False
        for row in rows:
            if row["has_annotated_rbp"] == 1:
                pca_values = [row[f"rbp_plm_pc{i}"] for i in range(4)]
                if any(v != 0.0 for v in pca_values):
                    has_nonzero = True
                    break
        assert has_nonzero
