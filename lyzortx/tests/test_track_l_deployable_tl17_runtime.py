from pathlib import Path

from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import SequenceRecord
from lyzortx.pipeline.track_l.steps import deployable_tl17_runtime as tl17
from lyzortx.pipeline.track_l.steps.parse_annotations import CdsRecord


def test_build_reference_proteins_filters_singleton_families(tmp_path: Path, monkeypatch) -> None:
    phage_metadata_path = tmp_path / "panel.csv"
    phage_metadata_path.write_text("phage\nP1\nP2\nP3\n", encoding="utf-8")
    fna_dir = tmp_path / "FNA"
    annot_dir = tmp_path / "annot"
    fna_dir.mkdir()
    annot_dir.mkdir()
    for phage in ("P1", "P2", "P3"):
        (fna_dir / f"{phage}.fna").write_text(f">{phage}\nATGCGTATGCGTATGCGT\n", encoding="utf-8")
        (annot_dir / f"{phage}_cds_final_merged_output.tsv").write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(tl17, "read_panel_phages", lambda *_args, **_kwargs: ["P1", "P2", "P3"])
    monkeypatch.setattr(
        tl17,
        "read_fasta_records",
        lambda *_args, **_kwargs: [SequenceRecord(identifier="contig", description="contig", sequence="ATG")],
    )
    proteins_by_phage = {
        "P1": [SequenceRecord("p1", "p1", "MPEPTIDE"), SequenceRecord("p2", "p2", "MSECOND")],
        "P2": [SequenceRecord("p1", "p1", "MTHIRD"), SequenceRecord("p2", "p2", "MFOURTH")],
        "P3": [SequenceRecord("p1", "p1", "MFIFTH"), SequenceRecord("p2", "p2", "MSIXTH")],
    }
    monkeypatch.setattr(
        tl17,
        "call_proteins_with_pyrodigal",
        lambda phage, _records: (proteins_by_phage[phage], "meta"),
    )
    parsed_by_phage = {
        "P1": [CdsRecord("P1_CDS_0001", 1, 9, "+", "contig", "11", "tail fiber", "tail")],
        "P2": [CdsRecord("P2_CDS_0001", 1, 9, "+", "contig", "11", "tail fiber", "tail")],
        "P3": [CdsRecord("P3_CDS_0002", 1, 9, "+", "contig", "22", "tail spike", "tail")],
    }
    monkeypatch.setattr(tl17, "parse_merged_tsv", lambda path: parsed_by_phage[path.name.split("_")[0]])
    monkeypatch.setattr(tl17, "classify_rbp_genes", lambda records: records)

    reference_rows, family_rows = tl17.build_reference_proteins(
        phage_metadata_path=phage_metadata_path,
        fna_dir=fna_dir,
        cached_annotations_dir=annot_dir,
        expected_panel_count=3,
        min_family_phage_support=2,
    )

    assert [row.family_id for row in family_rows] == ["RBP_PHROG_11"]
    assert [row.phage for row in reference_rows] == ["P1", "P2"]


def test_resolve_reference_protein_sequence_prefers_coordinate_match_over_gene_suffix() -> None:
    proteins = [
        SequenceRecord(
            identifier="prot_0001",
            description="P1|prot_0001 contig=P1 start=100 end=220 strand=1",
            sequence="MFIRST",
        ),
        SequenceRecord(
            identifier="prot_0002",
            description="P1|prot_0002 contig=P1 start=37282 end=39813 strand=1",
            sequence="MCOORD",
        ),
    ]
    rbp_record = CdsRecord("P1_CDS_0059", 37282, 39813, "+", "contig", "14895", "tail spike", "tail")

    protein_index, sequence = tl17.resolve_reference_protein_sequence(proteins=proteins, rbp_record=rbp_record)

    assert protein_index == 2
    assert sequence == "MCOORD"


def test_read_mmseqs_matches_parses_query_coverage(tmp_path: Path) -> None:
    path = tmp_path / "hits.tsv"
    path.write_text("q1\tt1\t95.0\t80\t1\t80\t100\t1\t80\t80\t1e-20\t250\n", encoding="utf-8")

    matches = tl17.read_mmseqs_matches(path)

    assert len(matches) == 1
    assert matches[0].query_coverage == 0.8


def test_project_phage_feature_row_applies_identity_and_coverage_thresholds(tmp_path: Path, monkeypatch) -> None:
    phage_path = tmp_path / "P1.fna"
    phage_path.write_text(">P1\nATGC\n", encoding="utf-8")
    reference_fasta_path = tmp_path / "reference.faa"
    reference_fasta_path.write_text(">ref\nMPEPTIDE\n", encoding="utf-8")
    runtime_payload = {
        "family_rows": [
            {
                "family_id": "RBP_PHROG_11",
                "column_name": "tl17_phage_rbp_family_11_present",
                "supporting_phage_count": 2,
                "supporting_reference_count": 2,
            },
            {
                "family_id": "RBP_PHROG_22",
                "column_name": "tl17_phage_rbp_family_22_present",
                "supporting_phage_count": 2,
                "supporting_reference_count": 2,
            },
        ],
        "reference_rows": [
            {
                "reference_id": "ref_11",
                "phage": "P1",
                "family_id": "RBP_PHROG_11",
                "gene_name": "P1_CDS_0001",
                "protein_index": 1,
                "annotation": "tail fiber",
                "phrog": "11",
                "protein_sequence": "MPEPTIDE",
            },
            {
                "reference_id": "ref_22",
                "phage": "P2",
                "family_id": "RBP_PHROG_22",
                "gene_name": "P2_CDS_0001",
                "protein_index": 1,
                "annotation": "tail spike",
                "phrog": "22",
                "protein_sequence": "MOTHER",
            },
        ],
        "matching_policy": {
            "min_percent_identity": 30.0,
            "min_query_coverage": 0.70,
            "mmseqs_command": ["mmseqs"],
        },
    }

    monkeypatch.setattr(
        tl17,
        "write_query_fasta",
        lambda _phage_path, output_path: (output_path, ["P1|query_prot_0001"]),
    )

    def fake_run_mmseqs_search(*, output_tsv_path: Path, **_kwargs) -> Path:
        output_tsv_path.write_text(
            (
                "P1|query_prot_0001\tref_11\t88.0\t80\t1\t80\t100\t1\t80\t80\t1e-30\t300\n"
                "P1|query_prot_0001\tref_22\t45.0\t40\t1\t40\t100\t1\t60\t60\t1e-10\t110\n"
            ),
            encoding="utf-8",
        )
        return output_tsv_path

    monkeypatch.setattr(tl17, "run_mmseqs_search", fake_run_mmseqs_search)

    observed = tl17.project_phage_feature_row(
        phage_path,
        runtime_payload=runtime_payload,
        reference_fasta_path=reference_fasta_path,
        scratch_root=tmp_path / "scratch",
    )

    assert observed == {
        "phage": "P1",
        "tl17_phage_rbp_family_11_present": 1,
        "tl17_phage_rbp_family_22_present": 0,
        tl17.SUMMARY_HIT_COUNT_COLUMN: 1,
        tl17.SUMMARY_FAMILY_COUNT_COLUMN: 1,
    }


def test_build_fasta_inventory_rows_hashes_panel_fastas(tmp_path: Path) -> None:
    phage_metadata_path = tmp_path / "panel.csv"
    phage_metadata_path.write_text("phage;\nP1;\nP2;\n", encoding="utf-8")
    fna_dir = tmp_path / "FNA"
    fna_dir.mkdir()
    (fna_dir / "P1.fna").write_text(">P1\nATGC\n", encoding="utf-8")
    (fna_dir / "P2.fna").write_text(">P2\nATGCGC\n", encoding="utf-8")

    rows = tl17.build_fasta_inventory_rows(
        phage_metadata_path=phage_metadata_path,
        fna_dir=fna_dir,
        expected_panel_count=2,
    )

    assert [row["phage"] for row in rows] == ["P1", "P2"]
    assert all(len(str(row["sha256"])) == 64 for row in rows)
