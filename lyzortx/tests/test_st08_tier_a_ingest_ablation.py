"""Unit tests for ST0.8 Tier A ingest and ablation summaries."""

import csv
import json

from lyzortx.pipeline.steel_thread_v0.steps.st08_tier_a_ingest_ablation import (
    build_internal_rows,
    build_tier_a_source_specs,
    compute_ablation_summary,
    compute_lift_failure_rows,
    main,
    normalize_generic_tier_a_row,
    normalize_vhrdb_row,
    read_source_registry,
)


def test_normalize_vhrdb_row_preserves_source_fidelity_fields() -> None:
    row = {
        "source_native_record_id": "VH123",
        "bacteria": "b1",
        "phage": "p1",
        "global_response": "Resistant",
        "datasource_response": "Sensitive",
        "uncertainty": "A",
    }
    normalized = normalize_vhrdb_row(row)
    assert normalized["source_native_record_id"] == "VH123"
    assert normalized["global_response"] == "Resistant"
    assert normalized["datasource_response"] == "Sensitive"
    assert normalized["source_disagreement_flag"] == "1"
    assert normalized["source_uncertainty"] == "A"


def test_normalize_vhrdb_row_defaults_datasource_id_to_vhrdb() -> None:
    row = {
        "source_native_record_id": "VH999",
        "bacteria": "b9",
        "phage": "p9",
        "global_response": "1",
        "datasource_response": "1",
        "uncertainty": "A",
    }
    normalized = normalize_vhrdb_row(row)
    assert normalized["source_datasource_id"] == "vhrdb"


def test_normalize_generic_tier_a_row_uses_source_defaults_and_strength_label() -> None:
    row = {
        "source_native_record_id": "BAS123",
        "bacteria": "b2",
        "phage": "p2",
        "label_hard_any_lysis": "Sensitive",
        "label_strict_confidence_tier": "A",
        "potency_label": "high",
    }
    normalized = normalize_generic_tier_a_row(row, source_id="basel")
    assert normalized["source_system"] == "basel"
    assert normalized["label_hard_any_lysis"] == "sensitive"
    assert normalized["source_datasource_id"] == "basel"
    assert normalized["source_strength_label"] == "high"


def test_ablation_arm_sizing_and_sequential_novel_pair_counts() -> None:
    merged_rows = [
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "source_system": "internal",
        },
        {
            "pair_id": "b2__p2",
            "bacteria": "b2",
            "phage": "p2",
            "source_system": "vhrdb",
        },
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "source_system": "vhrdb",
        },
        {
            "pair_id": "b3__p3",
            "bacteria": "b3",
            "phage": "p3",
            "source_system": "basel",
        },
    ]
    rows = compute_ablation_summary(merged_rows, tier_a_priority=["vhrdb", "basel"])
    assert rows[0]["pair_count"] == 1
    assert rows[1]["pair_count"] == 2
    assert rows[1]["new_pairs_vs_internal"] == 1
    assert rows[1]["new_pairs_vs_previous_arm"] == 1
    assert rows[2]["pair_count"] == 3
    assert rows[2]["new_pairs_vs_internal"] == 2
    assert rows[2]["new_pairs_vs_previous_arm"] == 1


def test_ablation_summary_preserves_available_zero_row_sources() -> None:
    merged_rows = [
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "source_system": "internal",
        },
        {
            "pair_id": "b2__p2",
            "bacteria": "b2",
            "phage": "p2",
            "source_system": "vhrdb",
        },
    ]
    rows = compute_ablation_summary(merged_rows, tier_a_priority=["vhrdb", "basel"])
    assert [row["arm"] for row in rows] == ["internal_only", "plus_vhrdb", "plus_basel"]
    assert rows[2]["pair_count"] == 2
    assert rows[2]["new_pairs_vs_internal"] == 1
    assert rows[2]["new_pairs_vs_previous_arm"] == 0


def test_lift_failure_counts_by_source_datasource_tier_and_quality() -> None:
    merged_rows = [
        {
            "source_system": "vhrdb",
            "source_datasource_id": "source_a",
            "source_uncertainty": "A",
            "source_disagreement_flag": "1",
            "source_strength_label": "",
        },
        {
            "source_system": "vhrdb",
            "source_datasource_id": "source_a",
            "source_uncertainty": "B",
            "source_disagreement_flag": "0",
            "source_strength_label": "",
        },
        {
            "source_system": "gpb",
            "source_datasource_id": "gpb",
            "source_uncertainty": "A",
            "source_disagreement_flag": "0",
            "source_strength_label": "medium",
        },
    ]
    rows = compute_lift_failure_rows(merged_rows)
    assert {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "source_system"} == {
        "gpb": 1,
        "vhrdb": 2,
    }
    assert {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "datasource"} == {
        "gpb": 1,
        "source_a": 2,
    }
    assert {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "confidence_tier"} == {
        "A": 2,
        "B": 1,
    }
    quality = {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "quality"}
    assert quality["datasource_disagreement"] == 1
    assert quality["strength_labels_present"] == 1


def test_build_internal_rows_sets_source_fields() -> None:
    internal = [
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "A",
        }
    ]
    out = build_internal_rows(internal)
    assert out[0]["source_system"] == "internal"
    assert out[0]["source_native_record_id"] == ""
    assert out[0]["global_response"] == ""
    assert out[0]["datasource_response"] == ""
    assert out[0]["source_strength_label"] == ""


def test_source_registry_drives_tier_a_priority(tmp_path) -> None:
    registry_path = tmp_path / "source_registry.csv"
    registry_path.write_text(
        "\n".join(
            [
                "source_id,confidence_tier",
                "vhrdb,A",
                "basel,A",
                "klebphacol,A",
                "gpb,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vhrdb_path = tmp_path / "vhrdb.csv"
    vhrdb_path.write_text(
        "\n".join(
            [
                "source_native_record_id,bacteria,phage,global_response,datasource_response,uncertainty",
                "VH123,b1,p1,Resistant,Sensitive,B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    basel_path = tmp_path / "basel.csv"
    basel_path.write_text(
        "\n".join(
            [
                "source_native_record_id,bacteria,phage,label_hard_any_lysis,label_strict_confidence_tier",
                "BAS123,b2,p2,Sensitive,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "vhrdb_path": vhrdb_path,
            "basel_path": basel_path,
            "klebphacol_path": tmp_path / "missing_klebphacol.csv",
            "gpb_path": tmp_path / "missing_gpb.csv",
        },
    )()

    registry_rows = read_source_registry(registry_path)
    specs = build_tier_a_source_specs(args, registry_rows)

    assert [spec.source_id for spec in specs] == ["vhrdb", "basel"]


def test_main_preserves_vhrdb_source_fidelity_and_orders_tier_a_ablation(tmp_path) -> None:
    internal_path = tmp_path / "internal.csv"
    internal_path.write_text(
        "\n".join(
            [
                "pair_id,bacteria,phage,label_hard_any_lysis,label_strict_confidence_tier",
                "b0__p0,b0,p0,1,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "source_registry.csv"
    registry_path.write_text(
        "\n".join(
            [
                "source_id,confidence_tier",
                "vhrdb,A",
                "basel,A",
                "klebphacol,A",
                "gpb,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vhrdb_path = tmp_path / "vhrdb.csv"
    vhrdb_path.write_text(
        "\n".join(
            [
                "source_native_record_id,bacteria,phage,global_response,datasource_response,uncertainty,datasource_id",
                "VH123,b1,p1,Resistant,Sensitive,B,upstream_a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    basel_path = tmp_path / "basel.csv"
    basel_path.write_text(
        "\n".join(
            [
                "source_native_record_id,bacteria,phage,label_hard_any_lysis,label_strict_confidence_tier,potency_label",
                "BAS123,b2,p2,Sensitive,A,high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--internal-pair-table-path",
            str(internal_path),
            "--source-registry-path",
            str(registry_path),
            "--vhrdb-path",
            str(vhrdb_path),
            "--basel-path",
            str(basel_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "st08_tier_a_ingested_pairs.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    vhrdb_row = next(row for row in rows if row["source_system"] == "vhrdb")
    basel_row = next(row for row in rows if row["source_system"] == "basel")
    assert "global_response" in vhrdb_row
    assert "datasource_response" in vhrdb_row
    assert vhrdb_row["global_response"] == "Resistant"
    assert vhrdb_row["datasource_response"] == "Sensitive"
    assert vhrdb_row["label_hard_any_lysis"] == "resistant"
    assert vhrdb_row["source_datasource_id"] == "upstream_a"
    assert vhrdb_row["source_native_record_id"] == "VH123"
    assert vhrdb_row["source_disagreement_flag"] == "1"
    assert basel_row["source_strength_label"] == "high"

    with (output_dir / "st08_ablation_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        ablation_rows = list(csv.DictReader(handle))
    assert [row["arm"] for row in ablation_rows] == ["internal_only", "plus_vhrdb", "plus_basel"]
    assert ablation_rows[1]["new_pairs_vs_previous_arm"] == "1"
    assert ablation_rows[2]["new_pairs_vs_previous_arm"] == "1"
    assert ablation_rows[2]["new_pairs_vs_internal"] == "2"


def test_main_keeps_zero_row_tier_a_sources_in_ablation_summary_and_manifest(tmp_path) -> None:
    internal_path = tmp_path / "internal.csv"
    internal_path.write_text(
        "\n".join(
            [
                "pair_id,bacteria,phage,label_hard_any_lysis,label_strict_confidence_tier",
                "b0__p0,b0,p0,1,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "source_registry.csv"
    registry_path.write_text(
        "\n".join(
            [
                "source_id,confidence_tier",
                "vhrdb,A",
                "basel,A",
                "klebphacol,A",
                "gpb,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vhrdb_path = tmp_path / "vhrdb.csv"
    vhrdb_path.write_text(
        "\n".join(
            [
                "source_native_record_id,bacteria,phage,global_response,datasource_response,uncertainty",
                "VH123,b1,p1,Resistant,Sensitive,B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    basel_path = tmp_path / "basel.csv"
    basel_path.write_text(
        "source_native_record_id,bacteria,phage,label_hard_any_lysis,label_strict_confidence_tier\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--internal-pair-table-path",
            str(internal_path),
            "--source-registry-path",
            str(registry_path),
            "--vhrdb-path",
            str(vhrdb_path),
            "--basel-path",
            str(basel_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "st08_ablation_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        ablation_rows = list(csv.DictReader(handle))
    assert [row["arm"] for row in ablation_rows] == ["internal_only", "plus_vhrdb", "plus_basel"]
    assert ablation_rows[2]["pair_count"] == "2"
    assert ablation_rows[2]["new_pairs_vs_internal"] == "1"
    assert ablation_rows[2]["new_pairs_vs_previous_arm"] == "0"

    manifest = json.loads((output_dir / "st08_tier_a_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_tier_a_sources"] == ["vhrdb", "basel"]
