"""Unit tests for TI09 strict ablation sequencing."""

from __future__ import annotations

import csv
import json

from lyzortx.pipeline.track_i import run_track_i
from lyzortx.pipeline.track_i.steps.build_strict_ablation_sequence import (
    compute_strict_ablation_summary,
    main,
)


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_strict_ablation_summary_keeps_the_planned_source_order() -> None:
    rows = [
        {
            "pair_id": "b1__p1",
            "source_system": "internal",
            "first_training_arm": "internal_only",
            "first_training_arm_index": "0",
            "effective_training_weight": "1.0",
        },
        {
            "pair_id": "b2__p2",
            "source_system": "vhrdb",
            "first_training_arm": "plus_vhrdb",
            "first_training_arm_index": "1",
            "effective_training_weight": "1.0",
        },
        {
            "pair_id": "b3__p3",
            "source_system": "basel",
            "first_training_arm": "plus_basel",
            "first_training_arm_index": "2",
            "effective_training_weight": "0.8",
        },
        {
            "pair_id": "b4__p4",
            "source_system": "klebphacol",
            "first_training_arm": "plus_klebphacol",
            "first_training_arm_index": "3",
            "effective_training_weight": "0.6",
        },
        {
            "pair_id": "b5__p5",
            "source_system": "gpb",
            "first_training_arm": "plus_gpb",
            "first_training_arm_index": "4",
            "effective_training_weight": "0.5",
        },
        {
            "pair_id": "b6__p6",
            "source_system": "virus_host_db",
            "first_training_arm": "plus_tier_b",
            "first_training_arm_index": "5",
            "effective_training_weight": "0.2",
        },
        {
            "pair_id": "b7__p7",
            "source_system": "ncbi_virus_biosample",
            "first_training_arm": "plus_tier_b",
            "first_training_arm_index": "5",
            "effective_training_weight": "0.2",
        },
    ]

    summary = compute_strict_ablation_summary(rows)

    assert [row["arm"] for row in summary] == [
        "internal_only",
        "plus_vhrdb",
        "plus_basel",
        "plus_klebphacol",
        "plus_gpb",
        "plus_tier_b",
    ]
    assert summary[0]["cumulative_source_systems"] == "internal"
    assert summary[1]["cumulative_source_systems"] == "internal|vhrdb"
    assert summary[-1]["planned_source_systems_added"] == "virus_host_db|ncbi_virus_biosample"
    assert summary[-1]["observed_source_systems_added"] == "virus_host_db|ncbi_virus_biosample"
    assert summary[-1]["cumulative_source_systems"] == (
        "internal|vhrdb|basel|klebphacol|gpb|virus_host_db|ncbi_virus_biosample"
    )
    assert summary[-1]["new_pairs_vs_previous_arm"] == 2


def test_main_emits_strict_ablation_outputs(tmp_path) -> None:
    cohort_rows = tmp_path / "ti08_training_cohort_rows.csv"
    _write_csv(
        cohort_rows,
        [
            "pair_id",
            "source_system",
            "first_training_arm",
            "first_training_arm_index",
            "effective_training_weight",
        ],
        [
            {
                "pair_id": "b1__p1",
                "source_system": "internal",
                "first_training_arm": "internal_only",
                "first_training_arm_index": "0",
                "effective_training_weight": "1.0",
            },
            {
                "pair_id": "b2__p2",
                "source_system": "vhrdb",
                "first_training_arm": "plus_vhrdb",
                "first_training_arm_index": "1",
                "effective_training_weight": "1.0",
            },
        ],
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--training-cohort-path",
            str(cohort_rows),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti09_strict_ablation_summary.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["arm"] for row in rows] == [
        "internal_only",
        "plus_vhrdb",
        "plus_basel",
        "plus_klebphacol",
        "plus_gpb",
        "plus_tier_b",
    ]
    assert rows[1]["planned_source_systems_added"] == "vhrdb"
    assert rows[2]["new_rows_vs_previous_arm"] == "0"

    manifest = json.loads((output_dir / "ti09_strict_ablation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_strict_ablation_sequence"
    assert manifest["strict_ablation_order"][-1] == "plus_tier_b"


def test_run_track_i_dispatches_ti09_step(monkeypatch) -> None:
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

    run_track_i.main(["--step", "strict-ablation-sequence"])
    assert calls == ["strict-ablation-sequence"]

    calls.clear()
    run_track_i.main(["--step", "all"])
    assert calls == [
        "weak-label-ingest",
        "external-confidence-tiers",
        "training-cohorts",
        "strict-ablation-sequence",
    ]
