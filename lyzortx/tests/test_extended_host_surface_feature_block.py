import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_c.steps.build_extended_host_surface_feature_block import (
    build_feature_rows,
    build_lps_core_index,
    main,
)


def test_build_lps_core_index_merges_primary_and_supplemental_sources() -> None:
    primary_rows = [
        {"bacteria": "B1", "LPS_type": "R1"},
        {"bacteria": "B2", "LPS_type": "R3"},
    ]
    supplemental_rows = [
        {"Strain": "B3", "LPS_type": "K12"},
    ]

    merged = build_lps_core_index(primary_rows, supplemental_rows)

    assert merged == {
        "B1": {"lps_core_type": "R1", "source_table": "primary"},
        "B2": {"lps_core_type": "R3", "source_table": "primary"},
        "B3": {"lps_core_type": "K12", "source_table": "supplemental"},
    }


def test_build_lps_core_index_rejects_conflicting_annotations() -> None:
    primary_rows = [{"bacteria": "B1", "LPS_type": "R1"}]
    supplemental_rows = [{"Strain": "B1", "LPS_type": "R3"}]

    try:
        build_lps_core_index(primary_rows, supplemental_rows)
    except ValueError as exc:
        assert "Conflicting LPS core annotations for 'B1'" in str(exc)
    else:
        raise AssertionError("Expected conflicting LPS annotations to raise ValueError")


def test_build_feature_rows_uses_umap_panel_and_adds_capsule_missingness() -> None:
    umap_rows = [
        {
            "bacteria": "B2",
            "UMAP0": "1.5",
            "UMAP1": "2.5",
            "UMAP2": "3.5",
            "UMAP3": "4.5",
            "UMAP4": "5.5",
            "UMAP5": "6.5",
            "UMAP6": "7.5",
            "UMAP7": "8.5",
        },
        {
            "bacteria": "B1",
            "UMAP0": "0.5",
            "UMAP1": "1.5",
            "UMAP2": "2.5",
            "UMAP3": "3.5",
            "UMAP4": "4.5",
            "UMAP5": "5.5",
            "UMAP6": "6.5",
            "UMAP7": "7.5",
        },
    ]
    capsule_index = {"B1": {"klebsiella_capsule_type": "K2"}}
    lps_index = {
        "B1": {"lps_core_type": "R1", "source_table": "primary"},
        "B2": {"lps_core_type": "K12", "source_table": "supplemental"},
    }

    feature_rows = build_feature_rows(
        umap_rows=umap_rows,
        capsule_index=capsule_index,
        lps_index=lps_index,
    )

    assert [row["bacteria"] for row in feature_rows] == ["B1", "B2"]
    assert feature_rows[0]["host_surface_klebsiella_capsule_type"] == "K2"
    assert feature_rows[0]["host_surface_klebsiella_capsule_type_missing"] == 0
    assert feature_rows[1]["host_surface_klebsiella_capsule_type"] == ""
    assert feature_rows[1]["host_surface_klebsiella_capsule_type_missing"] == 1
    assert feature_rows[1]["host_surface_lps_core_type"] == "K12"
    assert feature_rows[0]["host_phylogeny_umap_00"] == 0.5
    assert feature_rows[1]["host_phylogeny_umap_07"] == 8.5


def test_main_writes_matrix_metadata_and_manifest(tmp_path: Path) -> None:
    umap_path = tmp_path / "umap.tsv"
    capsule_path = tmp_path / "capsule.tsv"
    lps_primary_path = tmp_path / "lps_primary.tsv"
    lps_supplemental_path = tmp_path / "lps_supplemental.tsv"
    output_dir = tmp_path / "out"

    umap_path.write_text(
        (
            "bacteria\tUMAP0\tUMAP1\tUMAP2\tUMAP3\tUMAP4\tUMAP5\tUMAP6\tUMAP7\n"
            "B2\t1.1\t2.1\t3.1\t4.1\t5.1\t6.1\t7.1\t8.1\n"
            "B1\t0.1\t1.1\t2.1\t3.1\t4.1\t5.1\t6.1\t7.1\n"
        ),
        encoding="utf-8",
    )
    capsule_path.write_text(
        "bacteria\tKlebs_capsule_type\tlocus\nB1\tK54\tGood\n",
        encoding="utf-8",
    )
    lps_primary_path.write_text(
        "bacteria\tgembase\tLPS_type\nB1\tGB1\tR1\n",
        encoding="utf-8",
    )
    lps_supplemental_path.write_text(
        "Strain\tLPS_type\nB2\tK12\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--umap-path",
            str(umap_path),
            "--capsule-path",
            str(capsule_path),
            "--lps-primary-path",
            str(lps_primary_path),
            "--lps-supplemental-path",
            str(lps_supplemental_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
            "--expected-host-count",
            "2",
        ]
    )

    assert exit_code == 0

    matrix_path = output_dir / "host_extended_surface_features_test.csv"
    metadata_path = output_dir / "host_extended_surface_feature_metadata_test.csv"
    manifest_path = output_dir / "host_extended_surface_feature_manifest_test.json"

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(matrix_rows) == 2
    assert matrix_rows[0]["bacteria"] == "B1"
    assert matrix_rows[1]["host_surface_klebsiella_capsule_type_missing"] == "1"
    assert matrix_rows[0]["host_phylogeny_umap_00"] == "0.1"
    capsule_metadata = {
        row["column_name"]: row for row in metadata_rows if row["column_name"] == "host_surface_klebsiella_capsule_type"
    }
    assert capsule_metadata["host_surface_klebsiella_capsule_type"]["missing_count"] == "1"
    assert manifest["host_count"] == 2
    assert manifest["feature_count"] == 11
    assert manifest["coverage"]["klebsiella_capsule_type"]["observed_hosts"] == 1
    assert manifest["coverage"]["lps_core_type"]["source_breakdown"] == {"primary": 1, "supplemental": 1}
