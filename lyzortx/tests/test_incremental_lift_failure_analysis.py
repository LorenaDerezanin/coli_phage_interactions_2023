"""Unit tests for TI10 incremental lift and failure-mode analysis."""

from __future__ import annotations

import csv
import json

from lyzortx.pipeline.track_i import run_track_i
from lyzortx.pipeline.track_i.steps.build_incremental_lift_failure_analysis import (
    compute_failure_mode_rows,
    compute_source_tier_lift_rows,
    main,
)


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_source_tier_lift_rows_tracks_included_and_excluded_external_rows() -> None:
    cohort_rows = [
        {
            "pair_id": "b1__p1",
            "source_system": "internal",
            "first_training_arm": "internal_only",
            "first_training_arm_index": "0",
            "effective_training_weight": "1.0",
            "integration_status": "baseline_internal",
            "external_label_include_in_training": "",
            "external_label_confidence_tier": "",
            "source_resolution_status": "",
            "source_disagreement_flag": "0",
            "source_qc_flag": "",
        },
        {
            "pair_id": "b2__p2",
            "source_system": "vhrdb",
            "first_training_arm": "plus_vhrdb",
            "first_training_arm_index": "1",
            "effective_training_weight": "1.0",
            "integration_status": "external_enhancer",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "high",
            "source_resolution_status": "resolved",
            "source_disagreement_flag": "0",
            "source_qc_flag": "ok",
        },
        {
            "pair_id": "b3__p3",
            "source_system": "vhrdb",
            "first_training_arm": "plus_vhrdb",
            "first_training_arm_index": "1",
            "effective_training_weight": "1.0",
            "integration_status": "external_enhancer",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "low",
            "source_resolution_status": "resolved",
            "source_disagreement_flag": "1",
            "source_qc_flag": "ok",
        },
        {
            "pair_id": "b4__p4",
            "source_system": "virus_host_db",
            "first_training_arm": "excluded",
            "first_training_arm_index": "-1",
            "effective_training_weight": "0.0",
            "integration_status": "excluded_by_confidence",
            "external_label_include_in_training": "0",
            "external_label_confidence_tier": "medium",
            "source_resolution_status": "unresolved|resolved",
            "source_disagreement_flag": "0",
            "source_qc_flag": "biosample_missing",
        },
    ]
    ablation_rows = [
        {
            "arm": "internal_only",
            "arm_index": "0",
            "source_system_added": "internal",
            "cumulative_row_count": "1",
            "cumulative_pair_count": "1",
            "cumulative_external_row_count": "0",
            "cumulative_external_pair_count": "0",
            "new_rows_vs_previous_arm": "0",
            "new_pairs_vs_previous_arm": "0",
            "cumulative_training_weight": "1.0",
        },
        {
            "arm": "plus_vhrdb",
            "arm_index": "1",
            "source_system_added": "vhrdb",
            "cumulative_row_count": "3",
            "cumulative_pair_count": "3",
            "cumulative_external_row_count": "2",
            "cumulative_external_pair_count": "2",
            "new_rows_vs_previous_arm": "2",
            "new_pairs_vs_previous_arm": "2",
            "cumulative_training_weight": "3.0",
        },
        {
            "arm": "plus_tier_b",
            "arm_index": "5",
            "source_system_added": "tier_b",
            "cumulative_row_count": "4",
            "cumulative_pair_count": "4",
            "cumulative_external_row_count": "3",
            "cumulative_external_pair_count": "3",
            "new_rows_vs_previous_arm": "1",
            "new_pairs_vs_previous_arm": "1",
            "cumulative_training_weight": "3.0",
        },
    ]

    source_tier_rows = compute_source_tier_lift_rows(cohort_rows, ablation_rows)
    vhrdb_high = next(
        row for row in source_tier_rows if row["source_system"] == "vhrdb" and row["confidence_tier"] == "high"
    )
    virus_host_db_medium = next(
        row
        for row in source_tier_rows
        if row["source_system"] == "virus_host_db" and row["confidence_tier"] == "medium"
    )

    assert vhrdb_high["row_count"] == 1
    assert vhrdb_high["included_row_count"] == 1
    assert vhrdb_high["excluded_row_count"] == 0
    assert vhrdb_high["first_training_arm"] == "plus_vhrdb"
    assert vhrdb_high["row_share_of_arm"] == 0.333333
    assert virus_host_db_medium["included_row_count"] == 0
    assert virus_host_db_medium["excluded_row_count"] == 1
    assert virus_host_db_medium["first_training_arm"] == "excluded"


def test_compute_failure_mode_rows_groups_clean_and_failure_rows() -> None:
    cohort_rows = [
        {
            "pair_id": "b2__p2",
            "source_system": "vhrdb",
            "first_training_arm": "plus_vhrdb",
            "first_training_arm_index": "1",
            "effective_training_weight": "1.0",
            "integration_status": "external_enhancer",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "high",
            "source_resolution_status": "resolved",
            "source_disagreement_flag": "0",
            "source_qc_flag": "ok",
        },
        {
            "pair_id": "b3__p3",
            "source_system": "vhrdb",
            "first_training_arm": "plus_vhrdb",
            "first_training_arm_index": "1",
            "effective_training_weight": "1.0",
            "integration_status": "external_enhancer",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "low",
            "source_resolution_status": "resolved",
            "source_disagreement_flag": "1",
            "source_qc_flag": "ok",
        },
        {
            "pair_id": "b4__p4",
            "source_system": "virus_host_db",
            "first_training_arm": "excluded",
            "first_training_arm_index": "-1",
            "effective_training_weight": "0.0",
            "integration_status": "excluded_by_confidence",
            "external_label_include_in_training": "0",
            "external_label_confidence_tier": "medium",
            "source_resolution_status": "unresolved|resolved",
            "source_disagreement_flag": "0",
            "source_qc_flag": "biosample_missing",
        },
    ]

    failure_rows = compute_failure_mode_rows(cohort_rows)
    assert {row["failure_mode"] for row in failure_rows} == {
        "clean",
        "source_disagreement",
        "excluded_by_confidence",
        "non_ok_qc",
        "unresolved_entity_mapping",
    }
    clean = next(row for row in failure_rows if row["failure_mode"] == "clean")
    disagreement = next(row for row in failure_rows if row["failure_mode"] == "source_disagreement")
    excluded = next(row for row in failure_rows if row["failure_mode"] == "excluded_by_confidence")
    assert clean["row_count"] == 1
    assert disagreement["row_count"] == 1
    assert excluded["row_count"] == 1
    assert excluded["source_tier"] == "virus_host_db:medium"


def test_main_emits_ti10_outputs(tmp_path) -> None:
    cohort_rows = tmp_path / "ti08_training_cohort_rows.csv"
    _write_csv(
        cohort_rows,
        [
            "pair_id",
            "source_system",
            "first_training_arm",
            "first_training_arm_index",
            "effective_training_weight",
            "integration_status",
            "external_label_include_in_training",
            "external_label_confidence_tier",
            "source_resolution_status",
            "source_disagreement_flag",
            "source_qc_flag",
        ],
        [
            {
                "pair_id": "b1__p1",
                "source_system": "internal",
                "first_training_arm": "internal_only",
                "first_training_arm_index": "0",
                "effective_training_weight": "1.0",
                "integration_status": "baseline_internal",
                "external_label_include_in_training": "",
                "external_label_confidence_tier": "",
                "source_resolution_status": "",
                "source_disagreement_flag": "0",
                "source_qc_flag": "",
            },
            {
                "pair_id": "b2__p2",
                "source_system": "vhrdb",
                "first_training_arm": "plus_vhrdb",
                "first_training_arm_index": "1",
                "effective_training_weight": "1.0",
                "integration_status": "external_enhancer",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "source_resolution_status": "resolved",
                "source_disagreement_flag": "0",
                "source_qc_flag": "ok",
            },
            {
                "pair_id": "b3__p3",
                "source_system": "vhrdb",
                "first_training_arm": "plus_vhrdb",
                "first_training_arm_index": "1",
                "effective_training_weight": "1.0",
                "integration_status": "external_enhancer",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "low",
                "source_resolution_status": "resolved",
                "source_disagreement_flag": "1",
                "source_qc_flag": "ok",
            },
        ],
    )
    strict_ablation = tmp_path / "ti09_strict_ablation_summary.csv"
    _write_csv(
        strict_ablation,
        [
            "arm",
            "arm_index",
            "source_system_added",
            "cumulative_row_count",
            "cumulative_pair_count",
            "cumulative_external_row_count",
            "cumulative_external_pair_count",
            "new_rows_vs_previous_arm",
            "new_pairs_vs_previous_arm",
            "cumulative_training_weight",
        ],
        [
            {
                "arm": "internal_only",
                "arm_index": "0",
                "source_system_added": "internal",
                "cumulative_row_count": "1",
                "cumulative_pair_count": "1",
                "cumulative_external_row_count": "0",
                "cumulative_external_pair_count": "0",
                "new_rows_vs_previous_arm": "0",
                "new_pairs_vs_previous_arm": "0",
                "cumulative_training_weight": "1.0",
            },
            {
                "arm": "plus_vhrdb",
                "arm_index": "1",
                "source_system_added": "vhrdb",
                "cumulative_row_count": "3",
                "cumulative_pair_count": "3",
                "cumulative_external_row_count": "2",
                "cumulative_external_pair_count": "2",
                "new_rows_vs_previous_arm": "2",
                "new_pairs_vs_previous_arm": "2",
                "cumulative_training_weight": "3.0",
            },
        ],
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--training-cohort-path",
            str(cohort_rows),
            "--strict-ablation-summary-path",
            str(strict_ablation),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti10_incremental_lift_summary.csv").open("r", encoding="utf-8") as handle:
        arm_rows = list(csv.DictReader(handle))
    assert arm_rows[1]["row_lift_vs_internal"] == "2"
    assert arm_rows[1]["training_weight_lift_vs_internal"] == "2.0"

    with (output_dir / "ti10_source_tier_lift_summary.csv").open("r", encoding="utf-8") as handle:
        source_tier_rows = list(csv.DictReader(handle))
    assert {row["source_system"] for row in source_tier_rows} == {"vhrdb"}

    manifest = json.loads((output_dir / "ti10_incremental_lift_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_incremental_lift_failure_analysis"
    assert manifest["failure_modes"]["clean"] == 1


def test_run_track_i_dispatches_ti10_step(monkeypatch) -> None:
    calls: list[str] = []

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

    run_track_i.main(["--step", "incremental-lift-failure-analysis"])
    assert calls == ["incremental-lift-failure-analysis"]

    calls.clear()
    run_track_i.main(["--step", "all"])
    assert calls == [
        "weak-label-ingest",
        "external-confidence-tiers",
        "training-cohorts",
        "strict-ablation-sequence",
        "incremental-lift-failure-analysis",
    ]
