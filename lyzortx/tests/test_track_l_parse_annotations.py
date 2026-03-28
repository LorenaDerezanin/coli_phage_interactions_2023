"""Tests for Track L pharokka annotation parsing logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyzortx.pipeline.track_l.steps.parse_annotations import (
    CdsRecord,
    classify_anti_defense_genes,
    classify_rbp_genes,
    count_categories,
    matches_any_pattern,
    parse_merged_tsv,
    summarize_phage,
    ANTI_DEFENSE_PATTERNS,
    RBP_PATTERNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MERGED_TSV_HEADER = (
    "gene\tstart\tstop\tframe\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
    "mmseqs_seqIdentity\tmmseqs_eVal\tmmseqs_top_hit\tpyhmmer_phrog\t"
    "pyhmmer_bitscore\tpyhmmer_evalue\tphrog\tMethod\tRegion\tcolor\t"
    "annot\tcategory\tvfdb_hit\tvfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\t"
    "vfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\tCARD_alnScore\t"
    "CARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
    "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\t"
    "Resistance_Mechanism"
)

NONE_FIELDS = "\t".join(["None"] * 19)


def _make_row(gene: str, start: int, stop: int, phrog: str, annot: str, category: str) -> str:
    """Build a single TSV data row with the pharokka merged output columns."""
    return (
        f"{gene}\t{start}\t{stop}\t+\tcontig_1\t-10.0\t{phrog}\t50\t0.6\t1e-5\t"
        f"hit_1\tNo_PHROG\tNo_PHROG\tNo_PHROG\t{phrog}\tPHANOTATE\tCDS\t#838383\t"
        f"{annot}\t{category}\t{NONE_FIELDS}"
    )


def _write_tsv(path: Path, rows: list[str]) -> None:
    path.write_text(MERGED_TSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# parse_merged_tsv
# ---------------------------------------------------------------------------


def test_parse_merged_tsv_reads_records(tmp_path: Path) -> None:
    tsv = tmp_path / "test_cds_final_merged_output.tsv"
    _write_tsv(
        tsv,
        [
            _make_row("CDS_0001", 100, 200, "97", "UvsX-like recombinase", "other"),
            _make_row("CDS_0002", 300, 500, "1215", "hypothetical protein", "unknown function"),
        ],
    )
    records = parse_merged_tsv(tsv)
    assert len(records) == 2
    assert records[0].gene == "CDS_0001"
    assert records[0].phrog == "97"
    assert records[0].annot == "UvsX-like recombinase"
    assert records[0].category == "other"
    assert records[1].start == 300
    assert records[1].stop == 500


def test_parse_merged_tsv_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_merged_tsv(tmp_path / "nonexistent.tsv")


def test_parse_merged_tsv_raises_on_empty_file(tmp_path: Path) -> None:
    tsv = tmp_path / "empty.tsv"
    tsv.write_text(MERGED_TSV_HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Zero CDS"):
        parse_merged_tsv(tsv)


# ---------------------------------------------------------------------------
# matches_any_pattern
# ---------------------------------------------------------------------------


def test_rbp_patterns_match() -> None:
    assert matches_any_pattern("tail fiber protein", RBP_PATTERNS)
    assert matches_any_pattern("receptor binding protein Gp38", RBP_PATTERNS)
    assert matches_any_pattern("tail spike protein", RBP_PATTERNS)
    assert matches_any_pattern("side tail fiber", RBP_PATTERNS)
    assert matches_any_pattern("long tail fiber", RBP_PATTERNS)
    assert matches_any_pattern("host specificity protein", RBP_PATTERNS)


def test_rbp_patterns_do_not_match_unrelated() -> None:
    assert not matches_any_pattern("hypothetical protein", RBP_PATTERNS)
    assert not matches_any_pattern("DNA polymerase", RBP_PATTERNS)
    assert not matches_any_pattern("tail length tape measure protein", RBP_PATTERNS)


def test_anti_defense_patterns_match() -> None:
    assert matches_any_pattern("anti-CRISPR protein AcrIIA4", ANTI_DEFENSE_PATTERNS)
    assert matches_any_pattern("anti-restriction protein", ANTI_DEFENSE_PATTERNS)
    assert matches_any_pattern("Ocr anti-restriction", ANTI_DEFENSE_PATTERNS)
    assert matches_any_pattern("DNA methyltransferase", ANTI_DEFENSE_PATTERNS)
    assert matches_any_pattern("Dam methylase", ANTI_DEFENSE_PATTERNS)


def test_anti_defense_patterns_do_not_match_unrelated() -> None:
    assert not matches_any_pattern("hypothetical protein", ANTI_DEFENSE_PATTERNS)
    assert not matches_any_pattern("tail fiber protein", ANTI_DEFENSE_PATTERNS)


# ---------------------------------------------------------------------------
# classify_rbp_genes / classify_anti_defense_genes
# ---------------------------------------------------------------------------


def test_classify_rbp_genes() -> None:
    records = [
        CdsRecord("G1", 1, 100, "+", "c1", "10", "tail fiber protein", "tail"),
        CdsRecord("G2", 200, 400, "+", "c1", "20", "DNA polymerase", "DNA, RNA and nucleotide metabolism"),
        CdsRecord("G3", 500, 700, "+", "c1", "30", "receptor binding protein", "tail"),
    ]
    rbp = classify_rbp_genes(records)
    assert len(rbp) == 2
    assert {r.gene for r in rbp} == {"G1", "G3"}


def test_classify_anti_defense_genes() -> None:
    records = [
        CdsRecord(
            "G1", 1, 100, "+", "c1", "10", "anti-CRISPR protein", "moron, auxiliary metabolic gene and host takeover"
        ),
        CdsRecord("G2", 200, 400, "+", "c1", "20", "DNA polymerase", "DNA, RNA and nucleotide metabolism"),
        CdsRecord("G3", 500, 700, "+", "c1", "30", "DNA methyltransferase", "other"),
    ]
    anti = classify_anti_defense_genes(records)
    assert len(anti) == 2
    assert {r.gene for r in anti} == {"G1", "G3"}


# ---------------------------------------------------------------------------
# count_categories
# ---------------------------------------------------------------------------


def test_count_categories() -> None:
    records = [
        CdsRecord("G1", 1, 100, "+", "c1", "10", "tail fiber", "tail"),
        CdsRecord("G2", 200, 400, "+", "c1", "20", "lysozyme", "lysis"),
        CdsRecord("G3", 500, 700, "+", "c1", "30", "tail sheath", "tail"),
        CdsRecord("G4", 800, 900, "+", "c1", "40", "unknown", "unknown function"),
    ]
    counts = count_categories(records)
    assert counts["tail"] == 2
    assert counts["lysis"] == 1
    assert counts["unknown function"] == 1
    assert counts["head and packaging"] == 0


# ---------------------------------------------------------------------------
# summarize_phage
# ---------------------------------------------------------------------------


def test_summarize_phage() -> None:
    records = [
        CdsRecord("G1", 1, 100, "+", "c1", "10", "tail fiber protein", "tail"),
        CdsRecord(
            "G2", 200, 400, "+", "c1", "20", "anti-CRISPR protein", "moron, auxiliary metabolic gene and host takeover"
        ),
        CdsRecord("G3", 500, 700, "+", "c1", "30", "hypothetical protein", "unknown function"),
    ]
    summary = summarize_phage("test_phage", records)
    assert summary.phage_name == "test_phage"
    assert summary.total_cds == 3
    assert summary.category_counts["tail"] == 1
    assert summary.category_counts["unknown function"] == 1
    assert len(summary.rbp_genes) == 1
    assert summary.rbp_genes[0].gene == "G1"
    assert len(summary.anti_defense_genes) == 1
    assert summary.anti_defense_genes[0].gene == "G2"
