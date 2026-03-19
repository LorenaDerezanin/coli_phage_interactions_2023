"""Unit tests for ST0.8 VHRdb ingest and ablation summaries."""

import csv

from lyzortx.pipeline.steel_thread_v0.steps.st08_vhrdb_ingest_ablation import (
    build_internal_rows,
    compute_ablation_summary,
    compute_lift_failure_rows,
    main,
    normalize_vhrdb_row,
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


def test_ablation_arm_sizing_and_novel_pair_count() -> None:
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
    ]
    rows = compute_ablation_summary(merged_rows)
    assert rows[0]["pair_count"] == 1
    assert rows[1]["pair_count"] == 2
    assert rows[1]["new_pairs_vs_internal"] == 1


def test_lift_failure_counts_by_datasource_tier_and_disagreement() -> None:
    merged_rows = [
        {
            "source_system": "vhrdb",
            "source_datasource_id": "source_a",
            "source_uncertainty": "A",
            "source_disagreement_flag": "1",
        },
        {
            "source_system": "vhrdb",
            "source_datasource_id": "source_a",
            "source_uncertainty": "B",
            "source_disagreement_flag": "0",
        },
        {
            "source_system": "vhrdb",
            "source_datasource_id": "source_b",
            "source_uncertainty": "A",
            "source_disagreement_flag": "0",
        },
    ]
    rows = compute_lift_failure_rows(merged_rows)
    assert {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "datasource"} == {
        "source_a": 2,
        "source_b": 1,
    }
    assert {r["slice_value"]: r["row_count"] for r in rows if r["slice_type"] == "confidence_tier"} == {
        "A": 2,
        "B": 1,
    }
    quality = [r for r in rows if r["slice_type"] == "quality"][0]
    assert quality["row_count"] == 1


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


def test_main_preserves_raw_vhrdb_response_columns(tmp_path) -> None:
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
    output_dir = tmp_path / "out"

    main(
        [
            "--internal-pair-table-path",
            str(internal_path),
            "--vhrdb-path",
            str(vhrdb_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "st08_vhrdb_ingested_pairs.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    vhrdb_row = next(row for row in rows if row["source_system"] == "vhrdb")
    assert "global_response" in vhrdb_row
    assert "datasource_response" in vhrdb_row
    assert vhrdb_row["global_response"] == "Resistant"
    assert vhrdb_row["datasource_response"] == "Sensitive"
    assert vhrdb_row["label_hard_any_lysis"] == "resistant"
    assert vhrdb_row["source_datasource_id"] == "upstream_a"
    assert vhrdb_row["source_native_record_id"] == "VH123"
    assert vhrdb_row["source_disagreement_flag"] == "1"
