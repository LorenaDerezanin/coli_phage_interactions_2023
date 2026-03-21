import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_c.steps.build_receptor_surface_feature_block import (
    _parse_integral_value,
    build_column_metadata,
    build_feature_rows,
    main,
    resolve_k_antigen_type,
    target_host_set,
)


def test_target_host_set_uses_intersection() -> None:
    interaction_rows = [{"bacteria": "B1"}, {"bacteria": "B2"}, {"bacteria": "B3"}]
    lps_rows = [{"bacteria": "B1"}, {"bacteria": "B3"}, {"bacteria": "B4"}]
    receptor_rows = [{"bacteria": "B1"}, {"bacteria": "B2"}, {"bacteria": "B4"}]

    assert target_host_set(interaction_rows, lps_rows, receptor_rows) == ["B1"]


def test_build_feature_rows_keeps_variant_columns_and_tonb_gap() -> None:
    rows = build_feature_rows(
        hosts=["B1", "B2"],
        host_metadata_by_bacteria={
            "B1": {
                "O-type": "O25",
                "Klebs_capsule_type": "",
                "ABC_serotype": "K1",
                "Capsule_ABC": "1.0",
                "Capsule_GroupIV_e": "0.0",
                "Capsule_GroupIV_e_stricte": "0.0",
                "Capsule_GroupIV_s": "2.0",
                "Capsule_Wzy_stricte": "1.0",
            },
            "B2": {
                "O-type": "-",
                "Klebs_capsule_type": "K63",
                "ABC_serotype": "Unknown",
                "Capsule_ABC": "0.0",
                "Capsule_GroupIV_e": "0.0",
                "Capsule_GroupIV_e_stricte": "0.0",
                "Capsule_GroupIV_s": "0.0",
                "Capsule_Wzy_stricte": "0.0",
            },
        },
        lps_by_bacteria={"B1": {"LPS_type": "R1"}, "B2": {"LPS_type": "R3"}},
        receptor_by_bacteria={
            "B1": {"BTUB": "99_1", "FHUA": "99_2", "LAMB": "99_3", "OMPA": "99_4", "OMPC": "99_5"},
            "B2": {"BTUB": "", "FHUA": "99_8", "LAMB": "", "OMPA": "99_9", "OMPC": "99_10"},
        },
    )

    assert rows[0]["host_o_antigen_type"] == "O25"
    assert rows[0]["host_k_antigen_type"] == "K1"
    assert rows[0]["host_k_antigen_type_source"] == "ABC_serotype"
    assert rows[0]["host_k_antigen_proxy_present"] == 1
    assert rows[0]["host_capsule_groupiv_s"] == 2
    assert rows[0]["host_receptor_btub_variant"] == "99_1"
    assert rows[0]["host_receptor_tonB_present"] == ""
    assert rows[1]["host_o_antigen_present"] == 0
    assert rows[1]["host_k_antigen_type"] == "K63"
    assert rows[1]["host_receptor_btub_present"] == 0


def test_resolve_k_antigen_type_prefers_klebsiella_call() -> None:
    value, source = resolve_k_antigen_type({"Klebs_capsule_type": "K54", "ABC_serotype": "K1"})

    assert value == "K54"
    assert source == "Klebs_capsule_type"


def test_parse_integral_value_rejects_fractional_input() -> None:
    try:
        _parse_integral_value("1.7")
    except ValueError as exc:
        assert "integer-like value" in str(exc)
    else:
        raise AssertionError("Expected fractional capsule metadata to raise ValueError")


def test_build_column_metadata_reports_missingness() -> None:
    rows = [
        {
            "bacteria": "B1",
            "host_o_antigen_present": 1,
            "host_o_antigen_type": "O1",
            "host_k_antigen_present": 0,
            "host_k_antigen_type": "",
            "host_k_antigen_type_source": "",
            "host_k_antigen_proxy_present": 0,
            "host_lps_core_present": 1,
            "host_lps_core_type": "R1",
            "host_capsule_abc_present": 0,
            "host_capsule_groupiv_e_present": 0,
            "host_capsule_groupiv_e_stricte_present": 0,
            "host_capsule_groupiv_s": 0,
            "host_capsule_wzy_stricte_present": 0,
            "host_receptor_btub_present": 1,
            "host_receptor_btub_variant": "99_1",
            "host_receptor_fhua_present": 1,
            "host_receptor_fhua_variant": "99_2",
            "host_receptor_lamB_present": 1,
            "host_receptor_lamB_variant": "99_3",
            "host_receptor_ompA_present": 1,
            "host_receptor_ompA_variant": "99_4",
            "host_receptor_ompC_present": 1,
            "host_receptor_ompC_variant": "99_5",
            "host_receptor_tonB_present": "",
            "host_receptor_tonB_variant": "",
        }
    ]

    metadata = build_column_metadata(rows)
    tonb_present = next(row for row in metadata if row["column_name"] == "host_receptor_tonB_present")
    k_type = next(row for row in metadata if row["column_name"] == "host_k_antigen_type")

    assert tonb_present["missing_count"] == 1
    assert tonb_present["missing_rate"] == 1.0
    assert "TonB locus table" in tonb_present["provenance_note"]
    assert k_type["missing_count"] == 1


def test_main_writes_matrix_metadata_and_manifest(tmp_path: Path) -> None:
    interaction_path = tmp_path / "interaction_matrix.csv"
    host_metadata_path = tmp_path / "picard_collection.csv"
    lps_path = tmp_path / "lps.tsv"
    receptors_path = tmp_path / "receptors.tsv"
    output_dir = tmp_path / "out"

    interaction_path.write_text("bacteria;P1\nB1;1\nB2;0\n", encoding="utf-8")
    host_metadata_path.write_text(
        (
            "bacteria;O-type;Klebs_capsule_type;ABC_serotype;Capsule_ABC;Capsule_GroupIV_e;"
            "Capsule_GroupIV_e_stricte;Capsule_GroupIV_s;Capsule_Wzy_stricte\n"
            "B1;O1;;K1;1.0;0.0;0.0;1.0;1.0\n"
            "B2;O2;K63;;0.0;0.0;0.0;0.0;0.0\n"
        ),
        encoding="utf-8",
    )
    lps_path.write_text("bacteria\tLPS_type\nB1\tR1\nB2\tR3\n", encoding="utf-8")
    receptors_path.write_text(
        "bacteria\tBTUB\tFHUA\tLAMB\tOMPA\tOMPC\nB1\t99_1\t99_2\t99_3\t99_4\t99_5\nB2\t\t99_8\t\t99_9\t99_10\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--interaction-matrix-path",
            str(interaction_path),
            "--host-metadata-path",
            str(host_metadata_path),
            "--lps-core-path",
            str(lps_path),
            "--receptor-clusters-path",
            str(receptors_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
            "--expected-host-count",
            "2",
        ]
    )

    assert exit_code == 0

    matrix_path = output_dir / "host_receptor_surface_features_test.csv"
    metadata_path = output_dir / "host_receptor_surface_feature_metadata_test.csv"
    manifest_path = output_dir / "host_receptor_surface_feature_manifest_test.json"

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(matrix_rows) == 2
    assert matrix_rows[0]["host_receptor_btub_variant"] == "99_1"
    assert matrix_rows[1]["host_receptor_tonB_variant"] == ""
    assert any(row["column_name"] == "host_lps_core_type" for row in metadata_rows)
    assert manifest["host_count"] == 2
    assert manifest["host_set_definition"]["rule"] == "interaction_hosts ∩ lps_core_hosts ∩ receptor_cluster_hosts"
