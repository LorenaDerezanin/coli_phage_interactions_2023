"""Unit tests for TI09 strict ablation sequencing."""

from __future__ import annotations

import csv
import json

import pytest

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


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_strict_ablation_inputs(tmp_path, *, include_external_rows: bool) -> dict[str, object]:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    cohort = tmp_path / "ti08_training_cohort_rows.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"
    tg01_summary = tmp_path / "tg01_model_summary.json"

    _write_csv(
        st02,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier", "host_pathotype"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
        ],
    )
    _write_csv(
        st03,
        ["pair_id", "bacteria", "phage", "split_holdout", "split_cv5_fold", "is_hard_trainable"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "1",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "is_hard_trainable": "1",
            },
        ],
    )
    _write_csv(
        track_c,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier", "host_pathotype"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
        ],
    )
    _write_csv(
        track_d_genome,
        ["phage", "phage_gc_content"],
        [
            {"phage": "p0", "phage_gc_content": "0.40"},
            {"phage": "p1", "phage_gc_content": "0.50"},
            {"phage": "p2", "phage_gc_content": "0.60"},
            {"phage": "p3", "phage_gc_content": "0.70"},
        ],
    )
    _write_csv(
        track_d_distance,
        ["phage", "phage_distance_umap_00"],
        [
            {"phage": "p0", "phage_distance_umap_00": "0.05"},
            {"phage": "p1", "phage_distance_umap_00": "0.10"},
            {"phage": "p2", "phage_distance_umap_00": "0.20"},
            {"phage": "p3", "phage_distance_umap_00": "0.30"},
        ],
    )
    _write_csv(
        track_e_rbp,
        ["pair_id", "bacteria", "phage", "lookup_available"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "lookup_available": "1"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "lookup_available": "1"},
            {"pair_id": "b1__p2", "bacteria": "b1", "phage": "p2", "lookup_available": "0"},
            {"pair_id": "b1__p3", "bacteria": "b1", "phage": "p3", "lookup_available": "1"},
        ],
    )
    _write_csv(
        track_e_isolation,
        ["pair_id", "bacteria", "phage", "isolation_host_distance"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "isolation_host_distance": "0.25"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "isolation_host_distance": "0.30"},
            {"pair_id": "b1__p2", "bacteria": "b1", "phage": "p2", "isolation_host_distance": "0.35"},
            {"pair_id": "b1__p3", "bacteria": "b1", "phage": "p3", "isolation_host_distance": "0.40"},
        ],
    )
    _write_json(v1_config, {"winner_subset_blocks": ["defense", "phage_genomic"]})
    _write_json(
        tg01_summary,
        {
            "lightgbm": {
                "best_params": {
                    "n_estimators": 10,
                    "learning_rate": 0.1,
                    "num_leaves": 7,
                    "min_child_samples": 1,
                }
            }
        },
    )

    external_rows = []
    if include_external_rows:
        external_rows = [
            {
                "pair_id": "b1__p0",
                "source_system": "vhrdb",
                "first_training_arm": "plus_vhrdb",
                "first_training_arm_index": "1",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p1",
                "source_system": "basel",
                "first_training_arm": "plus_basel",
                "first_training_arm_index": "2",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p0",
                "source_system": "klebphacol",
                "first_training_arm": "plus_klebphacol",
                "first_training_arm_index": "3",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p1",
                "source_system": "gpb",
                "first_training_arm": "plus_gpb",
                "first_training_arm_index": "4",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p0",
                "source_system": "virus_host_db",
                "first_training_arm": "plus_tier_b",
                "first_training_arm_index": "5",
                "effective_training_weight": "0.5",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "medium",
                "external_label_confidence_score": "2",
                "external_label_training_weight": "0.5",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p1",
                "source_system": "ncbi_virus_biosample",
                "first_training_arm": "plus_tier_b",
                "first_training_arm_index": "5",
                "effective_training_weight": "0.5",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "medium",
                "external_label_confidence_score": "2",
                "external_label_training_weight": "0.5",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
            },
        ]

    _write_csv(
        cohort,
        [
            "pair_id",
            "source_system",
            "first_training_arm",
            "first_training_arm_index",
            "effective_training_weight",
            "external_label_include_in_training",
            "external_label_confidence_tier",
            "external_label_confidence_score",
            "external_label_training_weight",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
        ],
        [
            {
                "pair_id": "b1__p0",
                "source_system": "internal",
                "first_training_arm": "internal_only",
                "first_training_arm_index": "0",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "",
                "external_label_confidence_tier": "",
                "external_label_confidence_score": "",
                "external_label_training_weight": "",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            },
            {
                "pair_id": "b1__p1",
                "source_system": "internal",
                "first_training_arm": "internal_only",
                "first_training_arm_index": "0",
                "effective_training_weight": "1.0",
                "external_label_include_in_training": "",
                "external_label_confidence_tier": "",
                "external_label_confidence_score": "",
                "external_label_training_weight": "",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
            },
            *external_rows,
        ],
    )

    return {
        "st02": st02,
        "st03": st03,
        "track_c": track_c,
        "track_d_genome": track_d_genome,
        "track_d_distance": track_d_distance,
        "track_e_rbp": track_e_rbp,
        "track_e_isolation": track_e_isolation,
        "cohort": cohort,
        "v1_config": v1_config,
        "tg01_summary": tg01_summary,
    }


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
    paths = _write_strict_ablation_inputs(tmp_path, include_external_rows=True)
    output_dir = tmp_path / "out"

    main(
        [
            "--skip-prerequisites",
            "--st02-pair-table-path",
            str(paths["st02"]),
            "--st03-split-assignments-path",
            str(paths["st03"]),
            "--track-c-pair-table-path",
            str(paths["track_c"]),
            "--track-d-genome-kmer-path",
            str(paths["track_d_genome"]),
            "--track-d-distance-path",
            str(paths["track_d_distance"]),
            "--track-e-rbp-compatibility-path",
            str(paths["track_e_rbp"]),
            "--track-e-isolation-distance-path",
            str(paths["track_e_isolation"]),
            "--v1-feature-config-path",
            str(paths["v1_config"]),
            "--tg01-summary-path",
            str(paths["tg01_summary"]),
            "--training-cohort-path",
            str(paths["cohort"]),
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
    assert rows[1]["holdout_roc_auc"] != ""
    assert rows[-1]["added_external_row_count"] == "2"
    assert rows[-1]["holdout_top3_hit_rate_all_strains"] != ""
    assert rows[-1]["holdout_brier_score"] != ""

    manifest = json.loads((output_dir / "ti09_strict_ablation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_strict_ablation_sequence"
    assert manifest["strict_ablation_order"][-1] == "plus_tier_b"
    assert manifest["training_arm_source_systems"]["plus_tier_b"][-1] == "ncbi_virus_biosample"


def test_main_raises_when_an_added_source_has_zero_rows(tmp_path) -> None:
    paths = _write_strict_ablation_inputs(tmp_path, include_external_rows=False)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="requires >0 external rows"):
        main(
            [
                "--skip-prerequisites",
                "--st02-pair-table-path",
                str(paths["st02"]),
                "--st03-split-assignments-path",
                str(paths["st03"]),
                "--track-c-pair-table-path",
                str(paths["track_c"]),
                "--track-d-genome-kmer-path",
                str(paths["track_d_genome"]),
                "--track-d-distance-path",
                str(paths["track_d_distance"]),
                "--track-e-rbp-compatibility-path",
                str(paths["track_e_rbp"]),
                "--track-e-isolation-distance-path",
                str(paths["track_e_isolation"]),
                "--v1-feature-config-path",
                str(paths["v1_config"]),
                "--tg01-summary-path",
                str(paths["tg01_summary"]),
                "--training-cohort-path",
                str(paths["cohort"]),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_run_track_i_dispatches_ti09_step(monkeypatch) -> None:
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

    run_track_i.main(["--step", "strict-ablation-sequence"])
    assert calls == ["strict-ablation-sequence"]

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
