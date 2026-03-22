import csv
import json
from pathlib import Path

import pytest

from lyzortx.pipeline.track_e.steps.build_isolation_host_distance_feature_block import (
    build_feature_rows,
    build_isolation_host_feature_index,
    compute_jaccard_distance,
    main,
)


def test_compute_jaccard_distance_handles_overlap_and_empty_union() -> None:
    assert compute_jaccard_distance((1, 0, 1), (1, 1, 0)) == pytest.approx(2 / 3)
    assert compute_jaccard_distance((0, 0), (0, 0)) == 0.0


def test_build_feature_rows_emits_distances_and_missing_host_flag() -> None:
    pair_rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "phage_host": "ISO1",
            "host_phylogeny_umap_00": "0.0",
            "host_phylogeny_umap_01": "0.0",
            "host_defense_subtype_abi_d": "1",
            "host_defense_subtype_cas_type_i": "0",
        },
        {
            "pair_id": "B2__P2",
            "bacteria": "B2",
            "phage": "P2",
            "phage_host": "LF110",
            "host_phylogeny_umap_00": "1.0",
            "host_phylogeny_umap_01": "1.0",
            "host_defense_subtype_abi_d": "0",
            "host_defense_subtype_cas_type_i": "1",
        },
    ]
    umap_rows = [
        {"bacteria": "ISO1", "UMAP0": "3.0", "UMAP1": "4.0"},
    ]
    defense_rows = [
        {"bacteria": "ISO1", "AbiD": "1", "CAS_Type_I": "1"},
    ]

    isolation_host_index, _ = build_isolation_host_feature_index(
        umap_rows=umap_rows,
        defense_rows=defense_rows,
        target_umap_columns=["host_phylogeny_umap_00", "host_phylogeny_umap_01"],
        target_defense_columns=["host_defense_subtype_abi_d", "host_defense_subtype_cas_type_i"],
        requested_hosts={"ISO1", "LF110"},
    )
    feature_rows, phage_summary_rows = build_feature_rows(
        pair_rows,
        target_umap_columns=["host_phylogeny_umap_00", "host_phylogeny_umap_01"],
        target_defense_columns=["host_defense_subtype_abi_d", "host_defense_subtype_cas_type_i"],
        isolation_host_index=isolation_host_index,
    )

    by_pair_id = {row["pair_id"]: row for row in feature_rows}

    assert by_pair_id["B1__P1"]["isolation_host_umap_euclidean_distance"] == 5.0
    assert by_pair_id["B1__P1"]["isolation_host_defense_jaccard_distance"] == 0.5
    assert by_pair_id["B1__P1"]["isolation_host_feature_available"] == 1

    assert by_pair_id["B2__P2"]["isolation_host_umap_euclidean_distance"] == 0.0
    assert by_pair_id["B2__P2"]["isolation_host_defense_jaccard_distance"] == 0.0
    assert by_pair_id["B2__P2"]["isolation_host_feature_available"] == 0

    assert phage_summary_rows == [
        {"phage": "P1", "isolation_host": "ISO1", "isolation_host_feature_available": 1},
        {"phage": "P2", "isolation_host": "LF110", "isolation_host_feature_available": 0},
    ]


def test_main_writes_feature_matrix_metadata_coverage_and_manifest(tmp_path: Path) -> None:
    pair_table_path = tmp_path / "pair_table.csv"
    umap_path = tmp_path / "umap.tsv"
    defense_path = tmp_path / "defense.csv"
    output_dir = tmp_path / "out"

    pair_table_path.write_text(
        (
            "pair_id,bacteria,phage,phage_host,host_phylogeny_umap_00,host_phylogeny_umap_01,"
            "host_defense_subtype_abi_d,host_defense_subtype_cas_type_i\n"
            "B1__P1,B1,P1,ISO1,0.0,0.0,1,0\n"
            "B2__P2,B2,P2,LF110,1.0,1.0,0,1\n"
        ),
        encoding="utf-8",
    )
    umap_path.write_text(
        ("bacteria\tUMAP0\tUMAP1\nISO1\t3.0\t4.0\n"),
        encoding="utf-8",
    )
    defense_path.write_text(
        ("bacteria;AbiD;CAS_Type_I\nISO1;1;1\n"),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--pair-table-path",
            str(pair_table_path),
            "--umap-path",
            str(umap_path),
            "--defense-subtypes-path",
            str(defense_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
        ]
    )

    assert exit_code == 0

    feature_path = output_dir / "isolation_host_distance_features_test.csv"
    metadata_path = output_dir / "isolation_host_distance_feature_metadata_test.csv"
    phage_coverage_path = output_dir / "phage_isolation_host_coverage_test.csv"
    host_coverage_path = output_dir / "isolation_host_feature_coverage_test.csv"
    manifest_path = output_dir / "isolation_host_distance_manifest_test.json"

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with phage_coverage_path.open("r", encoding="utf-8", newline="") as handle:
        phage_coverage_rows = list(csv.DictReader(handle))
    with host_coverage_path.open("r", encoding="utf-8", newline="") as handle:
        host_coverage_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(feature_rows) == 2
    by_pair_id = {row["pair_id"]: row for row in feature_rows}
    assert by_pair_id["B1__P1"]["isolation_host_umap_euclidean_distance"] == "5.0"
    assert by_pair_id["B1__P1"]["isolation_host_defense_jaccard_distance"] == "0.5"
    assert by_pair_id["B2__P2"]["isolation_host_feature_available"] == "0"
    assert any(row["column_name"] == "isolation_host_feature_available" for row in metadata_rows)
    assert phage_coverage_rows == [
        {"phage": "P1", "isolation_host": "ISO1", "isolation_host_feature_available": "1"},
        {"phage": "P2", "isolation_host": "LF110", "isolation_host_feature_available": "0"},
    ]
    assert {
        "isolation_host": "ISO1",
        "has_umap_profile": "1",
        "has_defense_profile": "1",
        "isolation_host_feature_available": "1",
    } in host_coverage_rows
    assert {
        "isolation_host": "LF110",
        "has_umap_profile": "0",
        "has_defense_profile": "0",
        "isolation_host_feature_available": "0",
    } in host_coverage_rows
    assert manifest["pair_count"] == 2
    assert manifest["feature_count"] == 3
    assert manifest["coverage"]["unavailable_phages"] == ["P2"]
