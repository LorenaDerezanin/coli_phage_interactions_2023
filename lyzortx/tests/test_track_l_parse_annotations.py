"""Tests for Track L pharokka annotation parsing logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyzortx.pipeline.track_l.steps.parse_annotations import (
    CdsRecord,
    PhageSummary,
    classify_anti_defense_genes,
    classify_rbp_genes,
    count_categories,
    discover_cached_phages,
    discover_phage_dirs,
    matches_any_pattern,
    parse_merged_tsv,
    summarize_phage,
    write_category_summary,
    write_rbp_gene_list,
    write_anti_defense_gene_list,
    ANTI_DEFENSE_PATTERNS,
    PHROG_CATEGORIES,
    RBP_PATTERNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MERGED_TSV_HEADER = (
    "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
    "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\t"
    "pyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\tcustom_hmm_bitscore\t"
    "custom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\t"
    "annot\tcategory\tvfdb_hit\tvfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\t"
    "vfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\tCARD_alnScore\t"
    "CARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
    "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\t"
    "Resistance_Mechanism\ttransl_table"
)

NONE_FIELDS = "\t".join(["None"] * 20)


def _make_row(gene: str, start: int, stop: int, phrog: str, annot: str, category: str) -> str:
    """Build a single TSV data row with the pharokka merged output columns."""
    return (
        f"{gene}\t{start}\t{stop}\t+\tcontig_1\t-10.0\t{phrog}\t50\t0.6\t1e-5\t"
        f"No_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\tNo_custom_hmm\tNo_custom_hmm\t"
        f"{phrog}\tPHANOTATE\tCDS\t#838383\t"
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


# ---------------------------------------------------------------------------
# discover_phage_dirs
# ---------------------------------------------------------------------------


def test_discover_phage_dirs(tmp_path: Path) -> None:
    for name in ("phageA", "phageB"):
        d = tmp_path / name
        d.mkdir()
        (d / f"{name}_cds_final_merged_output.tsv").write_text("header\n", encoding="utf-8")
    # Dir without the TSV should be ignored
    (tmp_path / "phageC").mkdir()

    dirs = discover_phage_dirs(tmp_path)
    assert len(dirs) == 2
    assert [d.name for d in dirs] == ["phageA", "phageB"]


def test_discover_cached_phages_extracts_names(tmp_path: Path) -> None:
    for name in ("phageA", "phageB"):
        (tmp_path / f"{name}_cds_final_merged_output.tsv").write_text("header\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("", encoding="utf-8")

    results = discover_cached_phages(tmp_path)
    assert len(results) == 2
    assert results[0][0] == "phageA"
    assert results[1][0] == "phageB"


# ---------------------------------------------------------------------------
# write_category_summary
# ---------------------------------------------------------------------------


def _make_summary(name: str, cds: int, tail: int, lysis: int) -> PhageSummary:
    counts = {cat: 0 for cat in PHROG_CATEGORIES}
    counts["tail"] = tail
    counts["lysis"] = lysis
    counts["unknown function"] = cds - tail - lysis
    return PhageSummary(phage_name=name, total_cds=cds, category_counts=counts)


def test_write_category_summary(tmp_path: Path) -> None:
    summaries = [_make_summary("P1", 10, 3, 2), _make_summary("P2", 5, 1, 0)]
    path = write_category_summary(summaries, tmp_path)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "tail" in lines[0]
    assert lines[1].startswith("P1,")


# ---------------------------------------------------------------------------
# write_rbp_gene_list / write_anti_defense_gene_list
# ---------------------------------------------------------------------------


def test_write_rbp_gene_list(tmp_path: Path) -> None:
    summaries = [
        PhageSummary(
            phage_name="P1",
            total_cds=3,
            rbp_genes=[CdsRecord("G1", 1, 100, "+", "c1", "10", "tail fiber", "tail")],
        ),
        PhageSummary(phage_name="P2", total_cds=2, rbp_genes=[]),
    ]
    path = write_rbp_gene_list(summaries, tmp_path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # header + 1 gene
    assert "P1" in lines[1]
    assert "tail fiber" in lines[1]


def test_write_anti_defense_gene_list(tmp_path: Path) -> None:
    summaries = [
        PhageSummary(
            phage_name="P1",
            total_cds=3,
            anti_defense_genes=[CdsRecord("G2", 200, 400, "+", "c1", "20", "anti-CRISPR", "other")],
        ),
    ]
    path = write_anti_defense_gene_list(summaries, tmp_path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert "anti-CRISPR" in lines[1]
