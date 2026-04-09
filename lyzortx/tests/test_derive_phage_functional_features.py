"""Tests for derive_phage_functional_features: PHROG categories, anti-defense, depolymerase."""

from __future__ import annotations

from pathlib import Path

from lyzortx.pipeline.autoresearch.derive_phage_functional_features import (
    PHAGE_FUNCTIONAL_FEATURE_NAMES,
    build_phage_functional_feature_row,
    build_phage_functional_schema,
)


def _write_pharokka_tsv(path: Path, phage_name: str, genes: list[dict[str, str]]) -> None:
    """Write a minimal Pharokka merged TSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
        "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\t"
        "pyhmmer_evalue\tcustom_hmm_id\tcustom_hmm_bitscore\tcustom_hmm_evalue\t"
        "phrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\tvfdb_alnScore\t"
        "vfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\t"
        "CARD_hit\tCARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\t"
        "ARO_Accession\tCARD_short_name\tProtein_Accession\tDNA_Accession\t"
        "AMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table"
    )
    lines = [header]
    none_fields = "\t".join(["None"] * 18)
    for g in genes:
        row = (
            f"{g['gene']}\t{g.get('start', '1')}\t{g.get('stop', '100')}\t"
            f"{g.get('strand', '+')}\t{g.get('contig', phage_name)}\t"
            f"-1\tNo_PHROG\t0\t0\t1\t"
            f"No_PHROGs_HMM\t0\t1\t"
            f"No_custom_HMM\tNo_custom_HMM\tNo_custom_HMM\t"
            f"No_PHROG\tPHANOTATE\tCDS\tNone\t"
            f"{g.get('annot', 'hypothetical protein')}\t"
            f"{g.get('category', 'unknown function')}\t"
            f"{none_fields}\t11"
        )
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestBuildPhageFunctionalFeatureRow:
    def test_phage_with_annotations(self, tmp_path: Path) -> None:
        annotation_dir = tmp_path / "annotations"
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [
                {"gene": "CDS_001", "annot": "terminase large subunit", "category": "head and packaging"},
                {"gene": "CDS_002", "annot": "tail fiber protein", "category": "tail"},
                {"gene": "CDS_003", "annot": "holin", "category": "lysis"},
                {
                    "gene": "CDS_004",
                    "annot": "anti-CRISPR protein",
                    "category": "moron, auxiliary metabolic gene and host takeover",
                },
                {"gene": "CDS_005", "annot": "depolymerase", "category": "tail"},
            ],
        )
        row = build_phage_functional_feature_row("TEST_P1", annotation_dir)
        assert row["phage"] == "TEST_P1"
        assert row["total_cds"] == 5
        assert row["phrog_count_head_packaging"] == 1
        assert row["phrog_count_tail"] == 2
        assert row["phrog_count_lysis"] == 1
        assert row["has_anti_defense"] == 1
        assert row["anti_defense_count"] == 1
        assert row["has_depolymerase"] == 1
        assert row["depolymerase_count"] == 1
        # Fractions should sum to 1.
        frac_sum = sum(
            row[f"phrog_frac_{slug}"]
            for slug in [
                "connector",
                "dna_rna_metabolism",
                "head_packaging",
                "integration_excision",
                "lysis",
                "moron_amg",
                "other",
                "tail",
                "transcription_reg",
                "unknown",
            ]
        )
        assert abs(frac_sum - 1.0) < 1e-4

    def test_phage_without_annotation(self, tmp_path: Path) -> None:
        row = build_phage_functional_feature_row("MISSING_P1", tmp_path)
        assert row["phage"] == "MISSING_P1"
        assert row["total_cds"] == 0
        assert row["has_anti_defense"] == 0

    def test_feature_count(self, tmp_path: Path) -> None:
        annotation_dir = tmp_path / "annotations"
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [{"gene": "CDS_001", "annot": "hypothetical protein", "category": "unknown function"}],
        )
        row = build_phage_functional_feature_row("TEST_P1", annotation_dir)
        # 1 (phage) + len(PHAGE_FUNCTIONAL_FEATURE_NAMES) keys.
        assert len(row) == 1 + len(PHAGE_FUNCTIONAL_FEATURE_NAMES)

    def test_no_anti_defense_no_depolymerase(self, tmp_path: Path) -> None:
        annotation_dir = tmp_path / "annotations"
        _write_pharokka_tsv(
            annotation_dir / "TEST_P1_cds_final_merged_output.tsv",
            "TEST_P1",
            [
                {"gene": "CDS_001", "annot": "major capsid protein", "category": "head and packaging"},
                {"gene": "CDS_002", "annot": "portal protein", "category": "head and packaging"},
            ],
        )
        row = build_phage_functional_feature_row("TEST_P1", annotation_dir)
        assert row["has_anti_defense"] == 0
        assert row["anti_defense_count"] == 0
        assert row["has_depolymerase"] == 0
        assert row["depolymerase_count"] == 0


class TestBuildPhageFunctionalSchema:
    def test_schema_length(self) -> None:
        schema = build_phage_functional_schema()
        # 1 total_cds + 10 counts + 10 fracs + 4 indicators = 25
        assert len(schema) == 25
        assert schema == list(PHAGE_FUNCTIONAL_FEATURE_NAMES)
