import csv
import json
from pathlib import Path

from lyzortx.pipeline.deployment_paired_features import derive_host_defense_features


def test_build_host_defense_schema_uses_panel_subtype_names_and_drops_derived_columns() -> None:
    panel_rows = [
        {"bacteria": "B1", "AbiD": "1", "CAS_Class1-Subtype-I-E": "0", "RareThing": "0"},
        {"bacteria": "B2", "AbiD": "0", "CAS_Class1-Subtype-I-E": "1", "RareThing": "0"},
        {"bacteria": "B3", "AbiD": "1", "CAS_Class1-Subtype-I-E": "1", "RareThing": "0"},
        {"bacteria": "B4", "AbiD": "1", "CAS_Class1-Subtype-I-E": "0", "RareThing": "0"},
        {"bacteria": "B5", "AbiD": "0", "CAS_Class1-Subtype-I-E": "1", "RareThing": "0"},
    ]

    schema = derive_host_defense_features.build_host_defense_schema(
        panel_rows,
        min_present_count=2,
        max_present_count=4,
    )

    assert schema["retained_subtype_columns"] == ["AbiD", "CAS_Class1-Subtype-I-E"]
    assert schema["derived_columns_dropped"] == [
        "host_defense_has_crispr",
        "host_defense_diversity",
        "host_defense_abi_burden",
    ]
    assert schema["columns"] == [
        {"name": "bacteria", "dtype": "string"},
        {"name": "AbiD", "dtype": "int64"},
        {"name": "CAS_Class1-Subtype-I-E", "dtype": "int64"},
    ]


def test_compare_host_defense_to_panel_separates_gains_losses_and_count_changes() -> None:
    comparison = derive_host_defense_features.compare_host_defense_to_panel(
        {
            "bacteria": "LF82",
            "AbiD": 1,
            "CAS_Class1-Subtype-I-E": 0,
            "MazEF": 2,
            "RM_Type_I": 2,
        },
        {
            "bacteria": "LF82",
            "AbiD": "0",
            "CAS_Class1-Subtype-I-E": "1",
            "MazEF": "2",
            "RM_Type_I": "1",
        },
        subtype_columns=["AbiD", "CAS_Class1-Subtype-I-E", "MazEF", "RM_Type_I"],
    )

    assert comparison["systems_gained"] == [{"subtype": "AbiD", "panel_count": 0, "derived_count": 1}]
    assert comparison["systems_lost"] == [{"subtype": "CAS_Class1-Subtype-I-E", "panel_count": 1, "derived_count": 0}]
    assert comparison["count_changes"] == [{"subtype": "RM_Type_I", "panel_count": 1, "derived_count": 2, "delta": 1}]
    assert comparison["exact_match_subtype_count"] == 1
    assert comparison["disagreement_subtype_count"] == 3


def test_derive_host_defense_features_outputs_integer_counts_without_summary_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assembly_path = tmp_path / "host.fasta"
    assembly_path.write_text(">chromosome\nATGCGTATGCGTATGCGTATGCGT\n", encoding="utf-8")

    monkeypatch.setattr(
        derive_host_defense_features,
        "ensure_defense_finder_models",
        lambda *, models_dir, force_update: "existing",
    )

    def fake_run_defense_finder_on_assembly(
        assembly_path: Path,
        *,
        output_dir: Path,
        models_dir: Path,
        workers: int,
        preserve_raw: bool,
        force_run: bool,
    ) -> tuple[Path, dict[str, object]]:
        systems_path = output_dir / f"{assembly_path.stem}_defense_finder_systems.tsv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with systems_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sys_id", "type", "subtype", "activity"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({"sys_id": "sys_1", "type": "AbiD", "subtype": "", "activity": "defense"})
            writer.writerow({"sys_id": "sys_2", "type": "MazEF", "subtype": "", "activity": "defense"})
            writer.writerow({"sys_id": "sys_3", "type": "MazEF", "subtype": "", "activity": "defense"})
        return (
            systems_path,
            {
                "protein_fasta_path": str(output_dir / f"{assembly_path.stem}.prt"),
                "replicon_count": 1,
                "genome_nt_count": 24,
                "predicted_cds_count": 12,
                "gene_finder_modes": ["meta"],
                "used_cached_systems": False,
            },
        )

    monkeypatch.setattr(
        derive_host_defense_features,
        "run_defense_finder_on_assembly",
        fake_run_defense_finder_on_assembly,
    )

    panel_path = tmp_path / "panel.csv"
    panel_path.write_text(
        "bacteria;AbiD;MazEF;RM_Type_I\nB1;1;1;1\nB2;1;1;1\nB3;1;1;1\nB4;1;1;1\nB5;1;1;1\n",
        encoding="utf-8",
    )

    result = derive_host_defense_features.derive_host_defense_features(
        assembly_path,
        bacteria_id="host_1",
        output_dir=tmp_path / "output",
        panel_defense_subtypes_path=panel_path,
        models_dir=tmp_path / "models",
        force_run=False,
    )

    assert result["feature_row"] == {"bacteria": "host_1", "AbiD": 1, "MazEF": 2, "RM_Type_I": 0}
    assert "host_defense_has_crispr" not in result["feature_row"]
    assert "host_defense_diversity" not in result["feature_row"]
    assert "host_defense_abi_burden" not in result["feature_row"]
    assert result["manifest"]["counts"]["matched_panel_subtype_system_count"] == 3

    counts_path = tmp_path / "output" / "host_defense_gene_counts.csv"
    with counts_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["bacteria", "AbiD", "MazEF", "RM_Type_I"]
    assert rows == [{"bacteria": "host_1", "AbiD": "1", "MazEF": "2", "RM_Type_I": "0"}]

    schema = json.loads((tmp_path / "output" / "schema_manifest.json").read_text(encoding="utf-8"))
    assert schema["columns"] == [
        {"dtype": "string", "name": "bacteria"},
        {"dtype": "int64", "name": "AbiD"},
        {"dtype": "int64", "name": "MazEF"},
        {"dtype": "int64", "name": "RM_Type_I"},
    ]


def test_run_validation_subset_writes_combined_counts_and_report(monkeypatch, tmp_path: Path) -> None:
    validation_dir = tmp_path / "fastas"
    validation_dir.mkdir()
    for host in derive_host_defense_features.VALIDATION_HOSTS:
        (validation_dir / f"{host}.fasta").write_text(">contig\nATGC\n", encoding="utf-8")

    panel_path = tmp_path / "panel.csv"
    panel_path.write_text(
        "bacteria;AbiD;MazEF;RM_Type_I;CAS_Class1-Subtype-I-E\n"
        "55989;1;0;0;0\n"
        "EDL933;0;1;0;0\n"
        "LF82;0;0;1;0\n"
        "B4;1;1;1;1\n"
        "B5;1;1;1;1\n"
        "B6;1;1;1;1\n"
        "B7;1;1;1;1\n"
        "B8;1;1;1;1\n",
        encoding="utf-8",
    )

    rows_by_host = {
        "55989": {"bacteria": "55989", "AbiD": 1, "MazEF": 0, "RM_Type_I": 0, "CAS_Class1-Subtype-I-E": 0},
        "EDL933": {"bacteria": "EDL933", "AbiD": 1, "MazEF": 0, "RM_Type_I": 0, "CAS_Class1-Subtype-I-E": 0},
        "LF82": {"bacteria": "LF82", "AbiD": 0, "MazEF": 0, "RM_Type_I": 2, "CAS_Class1-Subtype-I-E": 0},
    }

    def fake_derive_host_defense_features(
        assembly_path: Path,
        *,
        bacteria_id: str | None = None,
        output_dir: Path,
        panel_defense_subtypes_path: Path,
        models_dir: Path,
        workers: int,
        force_model_update: bool,
        force_run: bool,
        preserve_raw: bool,
    ) -> dict[str, object]:
        bacteria = bacteria_id or assembly_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "schema": {},
            "feature_row": rows_by_host[bacteria],
            "manifest": {
                "matched_detected_subtypes": {},
                "unmatched_detected_subtypes": {},
            },
        }

    monkeypatch.setattr(
        derive_host_defense_features,
        "derive_host_defense_features",
        fake_derive_host_defense_features,
    )

    summary = derive_host_defense_features.run_validation_subset(
        validation_fastas_dir=validation_dir,
        output_dir=tmp_path / "output",
        panel_defense_subtypes_path=panel_path,
        models_dir=tmp_path / "models",
    )

    assert summary["average_disagreement_systems_per_host"] == 1.0

    counts_path = tmp_path / "output" / "validation_host_defense_gene_counts.csv"
    with counts_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert [row["bacteria"] for row in rows] == ["55989", "EDL933", "LF82"]

    report = json.loads((tmp_path / "output" / "validation_disagreement_report.json").read_text(encoding="utf-8"))
    host_report_by_name = {entry["bacteria"]: entry for entry in report["host_reports"]}
    assert host_report_by_name["55989"]["disagreement_subtype_count"] == 0
    assert host_report_by_name["EDL933"]["systems_gained"] == [
        {"subtype": "AbiD", "panel_count": 0, "derived_count": 1}
    ]
    assert host_report_by_name["LF82"]["count_changes"] == [
        {"subtype": "RM_Type_I", "panel_count": 1, "derived_count": 2, "delta": 1}
    ]
