"""Unit tests for TI08 external training cohort integration."""

from __future__ import annotations

import csv
import json

from lyzortx.pipeline.track_i import run_track_i
from lyzortx.pipeline.track_i.steps.build_external_training_cohorts import (
    build_integrated_training_rows,
    compute_training_arm_summary,
    main,
)


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_integrated_training_rows_assigns_planned_training_arms() -> None:
    rows = build_integrated_training_rows(
        internal_rows=[
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            }
        ],
        external_rows=[
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "vhrdb",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "external_label_include_in_training": "1",
            },
            {
                "pair_id": "b2__p2",
                "bacteria": "b2",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "",
                "source_system": "virus_host_db",
                "external_label_confidence_tier": "exclude",
                "external_label_confidence_score": "0",
                "external_label_training_weight": "0.0",
                "external_label_include_in_training": "0",
            },
        ],
    )

    by_source = {row["source_system"]: row for row in rows}
    assert by_source["internal"]["first_training_arm"] == "internal_only"
    assert by_source["internal"]["effective_training_weight"] == 1.0
    assert by_source["vhrdb"]["first_training_arm"] == "plus_vhrdb"
    assert by_source["vhrdb"]["integration_status"] == "external_enhancer"
    assert by_source["vhrdb"]["confidence_tier"] == "high"
    assert by_source["vhrdb"]["training_weight"] == "1.0"
    assert by_source["virus_host_db"]["first_training_arm"] == "excluded"
    assert by_source["virus_host_db"]["first_training_arm_index"] == -1
    assert by_source["virus_host_db"]["integration_status"] == "excluded_by_confidence"


def test_compute_training_arm_summary_keeps_internal_only_runnable() -> None:
    rows = build_integrated_training_rows(
        internal_rows=[
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b2__p2",
                "bacteria": "b2",
                "phage": "p2",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
            },
        ],
        external_rows=[],
    )

    summary = compute_training_arm_summary(rows)

    assert summary[0]["arm"] == "internal_only"
    assert summary[0]["cumulative_row_count"] == 2
    assert summary[0]["cumulative_external_row_count"] == 0
    assert all(row["new_rows_vs_previous_arm"] == 0 for row in summary[1:])
    assert all(row["cumulative_pair_count"] == 2 for row in summary)


def test_main_emits_internal_and_external_outputs(tmp_path) -> None:
    internal_pair_table = tmp_path / "st02_pair_table.csv"
    external_confidence = tmp_path / "ti07_external_label_confidence_pairs.csv"
    _write_csv(
        internal_pair_table,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier"],
        [
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            }
        ],
    )
    _write_csv(
        external_confidence,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "confidence_tier",
            "training_weight",
            "external_label_confidence_tier",
            "external_label_confidence_score",
            "external_label_training_weight",
            "external_label_include_in_training",
        ],
        [
            {
                "pair_id": "b2__p2",
                "bacteria": "b2",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "vhrdb",
                "confidence_tier": "high",
                "training_weight": "1.0",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "external_label_include_in_training": "1",
            }
        ],
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--internal-pair-table-path",
            str(internal_pair_table),
            "--external-confidence-path",
            str(external_confidence),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti08_training_cohort_rows.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    internal_row = next(row for row in rows if row["source_system"] == "internal")
    external_row = next(row for row in rows if row["source_system"] == "vhrdb")
    assert internal_row["first_training_arm"] == "internal_only"
    assert external_row["first_training_arm"] == "plus_vhrdb"
    assert external_row["confidence_tier"] == "high"
    assert external_row["training_weight"] == "1.0"
    assert external_row["external_label_confidence_tier"] == "high"
    assert external_row["external_label_training_weight"] == "1.0"

    manifest = json.loads((output_dir / "ti08_training_cohort_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_external_training_cohorts"
    assert manifest["external_confidence_input_present"] is True


def test_main_raises_when_all_external_rows_are_excluded(tmp_path) -> None:
    internal_pair_table = tmp_path / "st02_pair_table.csv"
    external_confidence = tmp_path / "ti07_external_label_confidence_pairs.csv"
    _write_csv(
        internal_pair_table,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier"],
        [
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            }
        ],
    )
    _write_csv(
        external_confidence,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "confidence_tier",
            "training_weight",
            "external_label_confidence_tier",
            "external_label_confidence_score",
            "external_label_training_weight",
            "external_label_include_in_training",
        ],
        [
            {
                "pair_id": "b2__p2",
                "bacteria": "b2",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "vhrdb",
                "confidence_tier": "high",
                "training_weight": "1.0",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "external_label_include_in_training": "0",
            }
        ],
    )
    output_dir = tmp_path / "out"

    try:
        main(
            [
                "--internal-pair-table-path",
                str(internal_pair_table),
                "--external-confidence-path",
                str(external_confidence),
                "--output-dir",
                str(output_dir),
            ]
        )
    except ValueError as exc:
        assert "requires >0 external rows" in str(exc)
    else:
        raise AssertionError("TI08 should fail when every external row is excluded")


def test_run_track_i_dispatches_ti08_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_i.build_tier_a_vhrdb_ingest,
        "main",
        lambda argv: calls.append("tier-a-vhrdb-ingest"),
    )
    monkeypatch.setattr(
        run_track_i.build_tier_a_additional_source_ingests,
        "main",
        lambda argv: calls.append("tier-a-ti04-ingest"),
    )
    monkeypatch.setattr(
        run_track_i.build_tier_a_harmonized_pairs,
        "main",
        lambda argv: calls.append("tier-a-harmonization"),
    )
    monkeypatch.setattr(
        run_track_i.build_tier_b_weak_label_ingest,
        "main",
        lambda argv: calls.append("weak-label-ingest"),
    )
    monkeypatch.setattr(
        run_track_i.build_external_label_confidence_tiers,
        "main",
        lambda argv: calls.append("external-confidence-tiers"),
    )
    monkeypatch.setattr(
        run_track_i.build_external_training_cohorts,
        "main",
        lambda argv: calls.append("training-cohorts"),
    )
    monkeypatch.setattr(
        run_track_i.build_strict_ablation_sequence,
        "main",
        lambda argv: calls.append("strict-ablation-sequence"),
    )
    monkeypatch.setattr(
        run_track_i.build_incremental_lift_failure_analysis,
        "main",
        lambda argv: calls.append("incremental-lift-failure-analysis"),
    )

    run_track_i.main(["--step", "training-cohorts"])
    assert calls == ["training-cohorts"]

    calls.clear()
    run_track_i.main(["--step", "all"])
    assert calls == [
        "tier-a-vhrdb-ingest",
        "tier-a-ti04-ingest",
        "tier-a-harmonization",
        "weak-label-ingest",
        "external-confidence-tiers",
        "training-cohorts",
        "strict-ablation-sequence",
        "incremental-lift-failure-analysis",
    ]
