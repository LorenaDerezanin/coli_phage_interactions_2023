"""Unit tests for ST0.8 VHRdb ingest and ablation summaries."""

from lyzortx.pipeline.steel_thread_v0.steps.st08_vhrdb_ingest_ablation import (
    build_internal_rows,
    compute_ablation_summary,
    compute_lift_failure_rows,
    normalize_vhrdb_row,
)


def test_normalize_vhrdb_row_preserves_source_fidelity_fields() -> None:
    row = {
        "source_native_record_id": "VH123",
        "bacteria": "b1",
        "phage": "p1",
        "global_response": "1",
        "datasource_response": "0",
        "uncertainty": "A",
    }
    normalized = normalize_vhrdb_row(row)
    assert normalized["source_native_record_id"] == "VH123"
    assert normalized["source_global_response"] == "1"
    assert normalized["source_datasource_response"] == "0"
    assert normalized["source_disagreement_flag"] == "1"
    assert normalized["source_uncertainty"] == "A"


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
            "source_datasource_response": "source_a",
            "source_uncertainty": "A",
            "source_disagreement_flag": "1",
        },
        {
            "source_system": "vhrdb",
            "source_datasource_response": "source_a",
            "source_uncertainty": "B",
            "source_disagreement_flag": "0",
        },
        {
            "source_system": "vhrdb",
            "source_datasource_response": "source_b",
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
