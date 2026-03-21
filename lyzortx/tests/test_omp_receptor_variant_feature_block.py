import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_c.steps.build_omp_receptor_variant_feature_block import (
    CategoryStat,
    build_feature_rows,
    main,
    select_feature_categories,
    summarize_receptor_categories,
)


def test_summarize_receptor_categories_groups_low_support_clusters_to_rare() -> None:
    rows = [
        {"bacteria": "B1", "BTUB": "99_1", "FADL": "99_7"},
        {"bacteria": "B2", "BTUB": "99_1", "FADL": "99_7"},
        {"bacteria": "B3", "BTUB": "99_2", "FADL": "99_8"},
        {"bacteria": "B4", "BTUB": "99_3", "FADL": ""},
    ]

    summaries = summarize_receptor_categories(rows, receptor_columns=("BTUB", "FADL"), min_cluster_count=2)

    assert summaries["BTUB"]["kept_cluster_counts"] == {"99_1": 2}
    assert summaries["BTUB"]["rare_clusters"] == ("99_2", "99_3")
    btub_stats = {stat.category: stat for stat in summaries["BTUB"]["category_stats"]}
    assert btub_stats["rare"].count == 2
    assert summaries["FADL"]["missing_count"] == 1


def test_select_feature_categories_keeps_one_per_receptor_and_respects_budget() -> None:
    summaries = {
        "BTUB": {
            "category_stats": [
                CategoryStat("BTUB", "99_1", 6, 0.5, 0.25, ("99_1",)),
                CategoryStat("BTUB", "rare", 3, 0.25, 0.1875, ("99_7", "99_8")),
                CategoryStat("BTUB", "missing", 3, 0.25, 0.1875, ()),
            ]
        },
        "FADL": {
            "category_stats": [
                CategoryStat("FADL", "99_9", 5, 0.416667, 0.243056, ("99_9",)),
                CategoryStat("FADL", "rare", 4, 0.333333, 0.222222, ("99_10",)),
                CategoryStat("FADL", "missing", 3, 0.25, 0.1875, ()),
            ]
        },
    }

    selected = select_feature_categories(
        summaries,
        receptor_columns=("BTUB", "FADL"),
        max_feature_count=3,
    )

    assert [(stat.receptor, stat.category) for stat in selected] == [
        ("BTUB", "99_1"),
        ("FADL", "99_9"),
        ("FADL", "rare"),
    ]


def test_build_feature_rows_emits_selected_indicator_columns() -> None:
    rows = [
        {"bacteria": "B2", "BTUB": "99_1", "FADL": ""},
        {"bacteria": "B1", "BTUB": "99_7", "FADL": "99_9"},
    ]
    summaries = summarize_receptor_categories(rows, receptor_columns=("BTUB", "FADL"), min_cluster_count=2)
    selected = [
        CategoryStat("BTUB", "rare", 1, 0.5, 0.25, ("99_1", "99_7")),
        CategoryStat("FADL", "missing", 1, 0.5, 0.25, ()),
    ]

    feature_rows = build_feature_rows(rows, summaries, selected)

    assert [row["bacteria"] for row in feature_rows] == ["B1", "B2"]
    assert feature_rows[0]["host_omp_receptor_btub_cluster_rare"] == 1
    assert feature_rows[0]["host_omp_receptor_fadl_cluster_missing"] == 0
    assert feature_rows[1]["host_omp_receptor_fadl_cluster_missing"] == 1


def test_main_writes_matrix_metadata_and_manifest(tmp_path: Path) -> None:
    receptor_path = tmp_path / "receptors.tsv"
    output_dir = tmp_path / "out"

    receptor_path.write_text(
        (
            "bacteria\tBTUB\tFADL\tFHUA\tLAMB\tLPTD\tNFRA\tOMPA\tOMPC\tOMPF\tTOLC\tTSX\tYNCD\n"
            "B1\t99_1\t99_7\t99_4\t99_5\t99_6\t99_7\t99_8\t99_9\t99_10\t99_11\t99_12\t99_13\n"
            "B2\t99_1\t99_7\t99_4\t99_5\t99_6\t99_7\t99_8\t99_9\t99_10\t99_11\t99_12\t99_13\n"
            "B3\t99_2\t99_8\t99_4\t99_5\t99_6\t99_7\t99_8\t99_9\t99_10\t99_11\t99_14\t99_15\n"
            "B4\t\t99_8\t99_16\t99_17\t99_18\t99_19\t99_20\t99_21\t99_22\t99_23\t\t99_24\n"
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--receptor-clusters-path",
            str(receptor_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
            "--min-cluster-count",
            "2",
            "--max-feature-count",
            "12",
            "--expected-host-count",
            "4",
        ]
    )

    assert exit_code == 0

    matrix_path = output_dir / "host_omp_receptor_variant_features_test.csv"
    metadata_path = output_dir / "host_omp_receptor_variant_feature_metadata_test.csv"
    manifest_path = output_dir / "host_omp_receptor_variant_feature_manifest_test.json"

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(matrix_rows) == 4
    assert matrix_rows[0]["bacteria"] == "B1"
    assert len(matrix_rows[0]) == 13
    assert any(row["receptor"] == "BTUB" for row in metadata_rows)
    rare_rows = [row for row in metadata_rows if row["grouped_category"] == "rare"]
    assert all("support < 2" in row["transform"] for row in rare_rows)
    assert manifest["host_count"] == 4
    assert manifest["feature_count"] == 12
    assert manifest["encoding_policy"]["rare_cluster_min_count"] == 2
    assert manifest["receptors"]["BTUB"]["retained_clusters"] == ["99_1"]
