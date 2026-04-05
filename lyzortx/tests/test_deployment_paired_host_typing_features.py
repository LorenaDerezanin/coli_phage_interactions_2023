import csv
import json
from pathlib import Path

from lyzortx.pipeline.deployment_paired_features import derive_host_typing_features


def test_build_host_typing_schema_marks_every_feature_as_categorical() -> None:
    schema = derive_host_typing_features.build_host_typing_schema()

    assert schema["columns"] == [
        {"name": "bacteria", "dtype": "string"},
        {"name": "host_clermont_phylo", "dtype": "string"},
        {"name": "host_st_warwick", "dtype": "string"},
        {"name": "host_o_type", "dtype": "string"},
        {"name": "host_h_type", "dtype": "string"},
        {"name": "host_serotype", "dtype": "string"},
    ]
    assert schema["categorical_columns"] == [
        "host_clermont_phylo",
        "host_st_warwick",
        "host_o_type",
        "host_h_type",
        "host_serotype",
    ]


def test_build_host_typing_feature_row_projects_direct_calls() -> None:
    row = derive_host_typing_features.build_host_typing_feature_row(
        bacteria="LF82",
        phylogroup_call={"phylogroup": "B2"},
        serotype_call={"o_type": "O83", "h_type": "H1"},
        mlst_call={"st_warwick": "135"},
    )

    assert row == {
        "bacteria": "LF82",
        "host_clermont_phylo": "B2",
        "host_st_warwick": "135",
        "host_o_type": "O83",
        "host_h_type": "H1",
        "host_serotype": "O83:H1",
    }


def test_compare_host_typing_to_panel_reports_field_matches() -> None:
    derived_row = {
        "bacteria": "LF82",
        "host_clermont_phylo": "B2",
        "host_st_warwick": "135",
        "host_o_type": "O83",
        "host_h_type": "H1",
        "host_serotype": "O83:H1",
    }
    panel_row = {
        "Clermont_Phylo": "B2",
        "ST_Warwick": "135",
        "O-type": "O83",
        "H-type": "H1",
    }

    comparison = derive_host_typing_features.compare_host_typing_to_panel(derived_row, panel_row)

    assert comparison["field_matches"] == {
        "phylogroup": True,
        "o_type": True,
        "h_type": True,
        "st_warwick": True,
        "serotype": True,
    }
    assert comparison["exact_match_field_count"] == 5
    assert comparison["resolved_field_count"] == 4


def test_derive_host_typing_features_writes_schema_and_feature_outputs(monkeypatch, tmp_path: Path) -> None:
    assembly_path = tmp_path / "LF82.fasta"
    assembly_path.write_text(">contig\nATGC\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    metadata_path = tmp_path / "picard_collection.csv"
    metadata_path.write_text(
        "bacteria;Clermont_Phylo;ST_Warwick;O-type;H-type\nLF82;B2;135;O83;H1\n",
        encoding="utf-8",
    )

    phylogroup_report = tmp_path / "typing_phylogroups.txt"
    phylogroup_report.write_text("LF82.fasta\t['trpA']\t['+']\t['trpAgpC']\tB2\tB2\n", encoding="utf-8")
    serotype_output = tmp_path / "output.tsv"
    serotype_output.write_text(
        "Name\tSpecies\tO-type\tH-type\tSerotype\tQC\tWarnings\nLF82\tEscherichia coli\tO83\tH1\tO83:H1\t-\t-\n",
        encoding="utf-8",
    )
    mlst_output = tmp_path / "mlst_legacy.tsv"
    mlst_output.write_text(
        f"This is mlst 2.32.2 running on linux\nDone.\nFILE\tSCHEME\tST\n{assembly_path}\tecoli_achtman_4\t135\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(derive_host_typing_features.tl16, "run_phylogroup_caller", lambda **_: phylogroup_report)
    monkeypatch.setattr(
        derive_host_typing_features.tl16,
        "run_serotype_caller",
        lambda **_: (serotype_output, tmp_path / "blastn_output_alleles.txt"),
    )
    monkeypatch.setattr(derive_host_typing_features.tl16, "run_sequence_type_caller", lambda **_: mlst_output)

    result = derive_host_typing_features.derive_host_typing_features(
        assembly_path,
        output_dir=output_dir,
        picard_metadata_path=metadata_path,
    )

    assert result["feature_row"]["host_clermont_phylo"] == "B2"
    assert result["feature_row"]["host_serotype"] == "O83:H1"
    schema = json.loads((output_dir / "schema_manifest.json").read_text(encoding="utf-8"))
    assert schema["categorical_columns"] == [
        "host_clermont_phylo",
        "host_st_warwick",
        "host_o_type",
        "host_h_type",
        "host_serotype",
    ]
    feature_csv = output_dir / "host_typing_features.csv"
    with feature_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "bacteria": "LF82",
            "host_clermont_phylo": "B2",
            "host_st_warwick": "135",
            "host_o_type": "O83",
            "host_h_type": "H1",
            "host_serotype": "O83:H1",
        }
    ]
    assert result["comparison"]["field_matches"]["serotype"] is True


def test_derive_host_typing_features_can_skip_panel_metadata_and_records_unresolved_mlst(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assembly_path = tmp_path / "EDL933.fasta"
    assembly_path.write_text(">contig\nATGC\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    phylogroup_report = tmp_path / "typing_phylogroups.txt"
    phylogroup_report.write_text("EDL933.fasta\t['arpA']\t['+']\t['arpAgpE']\tE\tE\n", encoding="utf-8")
    serotype_output = tmp_path / "output.tsv"
    serotype_output.write_text(
        "Name\tSpecies\tO-type\tH-type\tSerotype\tQC\tWarnings\nEDL933\tEscherichia coli\tO157\tH7\tO157:H7\t-\t-\n",
        encoding="utf-8",
    )
    mlst_output = tmp_path / "mlst_legacy.tsv"
    mlst_output.write_text(
        f"This is mlst 2.32.2 running on linux\nDone.\nFILE\tSCHEME\tST\n{assembly_path}\tecoli_achtman_4\t-\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(derive_host_typing_features.tl16, "run_phylogroup_caller", lambda **_: phylogroup_report)
    monkeypatch.setattr(
        derive_host_typing_features.tl16,
        "run_serotype_caller",
        lambda **_: (serotype_output, tmp_path / "blastn_output_alleles.txt"),
    )
    monkeypatch.setattr(derive_host_typing_features.tl16, "run_sequence_type_caller", lambda **_: mlst_output)
    monkeypatch.setattr(
        derive_host_typing_features.tl16,
        "load_panel_metadata",
        lambda *_: (_ for _ in ()).throw(AssertionError("panel metadata should not load for runtime-only typing")),
    )

    result = derive_host_typing_features.derive_host_typing_features(
        assembly_path,
        output_dir=output_dir,
        picard_metadata_path=None,
    )

    assert result["feature_row"]["host_st_warwick"] == ""
    assert result["comparison"] is None
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["guardrails"]["panel_metadata_used_for_feature_construction"] is False
    assert manifest["comparison"] is None
    assert manifest["runtime_caveats"] == [
        {
            "bacteria": "EDL933",
            "caller": "sequence_type",
            "field": "host_st_warwick",
            "raw_value": "-",
            "normalized_value": "",
            "message": "MLST returned an unresolved sequence type; the exported ST remains blank instead of using a placeholder.",
        }
    ]


def test_run_validation_subset_writes_combined_validation_outputs(monkeypatch, tmp_path: Path) -> None:
    validation_dir = tmp_path / "fastas"
    validation_dir.mkdir()
    for host in derive_host_typing_features.VALIDATION_HOSTS:
        (validation_dir / f"{host}.fasta").write_text(">contig\nATGC\n", encoding="utf-8")

    metadata = {
        "55989": {"Clermont_Phylo": "B1", "ST_Warwick": "678", "O-type": "O104", "H-type": "H4"},
        "EDL933": {"Clermont_Phylo": "E", "ST_Warwick": "11", "O-type": "O157", "H-type": "H7"},
        "LF82": {"Clermont_Phylo": "B2", "ST_Warwick": "135", "O-type": "O83", "H-type": "H1"},
    }
    monkeypatch.setattr(derive_host_typing_features.tl16, "load_panel_metadata", lambda *_: metadata)

    rows_by_host = {
        "55989": {
            "bacteria": "55989",
            "host_clermont_phylo": "B1",
            "host_st_warwick": "678",
            "host_o_type": "O104",
            "host_h_type": "H4",
            "host_serotype": "O104:H4",
        },
        "EDL933": {
            "bacteria": "EDL933",
            "host_clermont_phylo": "E",
            "host_st_warwick": "11",
            "host_o_type": "O157",
            "host_h_type": "H7",
            "host_serotype": "O157:H7",
        },
        "LF82": {
            "bacteria": "LF82",
            "host_clermont_phylo": "B2",
            "host_st_warwick": "135",
            "host_o_type": "O83",
            "host_h_type": "H1",
            "host_serotype": "O83:H1",
        },
    }

    def fake_derive_host_typing_features(
        assembly_path: Path,
        *,
        bacteria_id: str | None = None,
        output_dir: Path,
        picard_metadata_path: Path,
    ) -> dict[str, object]:
        bacteria = bacteria_id or assembly_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "schema": derive_host_typing_features.build_host_typing_schema(),
            "feature_row": rows_by_host[bacteria],
            "comparison": {},
            "manifest": {},
        }

    monkeypatch.setattr(derive_host_typing_features, "derive_host_typing_features", fake_derive_host_typing_features)

    summary = derive_host_typing_features.run_validation_subset(
        validation_fastas_dir=validation_dir,
        output_dir=tmp_path / "out",
        picard_metadata_path=tmp_path / "picard.csv",
    )

    assert summary["host_count"] == 3
    assert summary["field_exact_match_counts"] == {
        "phylogroup": 3,
        "o_type": 3,
        "h_type": 3,
        "st_warwick": 3,
        "serotype": 3,
    }

    with (tmp_path / "out" / "validation_host_typing_features.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["bacteria"] for row in rows] == ["55989", "EDL933", "LF82"]

    report = json.loads((tmp_path / "out" / "validation_report.json").read_text(encoding="utf-8"))
    assert report["host_reports"][0]["field_matches"]["o_type"] is True
