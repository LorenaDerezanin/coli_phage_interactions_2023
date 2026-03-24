"""Unit tests for TK02 BASEL lift measurement helpers."""

from __future__ import annotations

import csv
import json

import pytest

from lyzortx.pipeline.track_k.steps.build_basel_lift_report import main
from lyzortx.pipeline.track_k.steps.build_basel_lift_report import load_ti08_training_cohort_rows
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import load_source_training_rows


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_source_training_rows_keeps_joinable_basel_rows() -> None:
    feature_rows = [
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "split_holdout": "train_non_holdout",
            "split_cv5_fold": "0",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "A",
        },
        {
            "pair_id": "b2__p2",
            "bacteria": "b2",
            "phage": "p2",
            "split_holdout": "holdout_test",
            "split_cv5_fold": "-1",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "0",
            "label_strict_confidence_tier": "A",
        },
    ]
    cohort_rows = [
        {
            "pair_id": "b1__p1",
            "bacteria": "b1",
            "phage": "p1",
            "source_system": "basel",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "high",
            "external_label_confidence_score": "3",
            "external_label_training_weight": "1.0",
            "label_hard_any_lysis": "0",
            "label_strict_confidence_tier": "B",
        },
        {
            "pair_id": "b2__p2",
            "bacteria": "b2",
            "phage": "p2",
            "source_system": "basel",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "high",
            "external_label_confidence_score": "3",
            "external_label_training_weight": "1.0",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "B",
        },
        {
            "pair_id": "b3__p3",
            "bacteria": "b3",
            "phage": "p3",
            "source_system": "basel",
            "external_label_include_in_training": "0",
            "external_label_confidence_tier": "exclude",
            "external_label_confidence_score": "0",
            "external_label_training_weight": "0.0",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "B",
        },
    ]

    augmented_rows, counts = load_source_training_rows(feature_rows, cohort_rows, "basel")

    assert len(augmented_rows) == 1
    assert augmented_rows[0]["pair_id"] == "b1__p1"
    assert augmented_rows[0]["training_origin"] == "basel"
    assert counts["joined_rows"] == 1
    assert counts["non_training_split_rows"] == 1
    assert counts["excluded_rows"] == 1


def test_load_ti08_training_cohort_rows_requires_a_real_file(tmp_path) -> None:
    cohort_path = tmp_path / "ti08_training_cohort_rows.csv"

    with pytest.raises(FileNotFoundError):
        load_ti08_training_cohort_rows(cohort_path)


def test_load_ti08_training_cohort_rows_rejects_empty_file(tmp_path) -> None:
    cohort_path = tmp_path / "ti08_training_cohort_rows.csv"
    cohort_path.write_text("pair_id,source_system\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_ti08_training_cohort_rows(cohort_path)


def test_main_carries_forward_vhrdb_when_tk01_kept_it(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"
    tg01_summary = tmp_path / "tg01_model_summary.json"
    tk01_manifest = tmp_path / "tk01_vhrdb_lift_manifest.json"

    _write_csv(
        st02,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier", "host_pathotype"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
        ],
    )
    _write_csv(
        st03,
        ["pair_id", "bacteria", "phage", "cv_group", "split_holdout", "split_cv5_fold", "is_hard_trainable"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "cv_group": "g0",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "1",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "cv_group": "g1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "cv_group": "g2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "cv_group": "g3",
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
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            },
        ],
    )
    _write_csv(
        track_d_genome,
        ["phage", "phage_gc_content"],
        [
            {"phage": "p0", "phage_gc_content": "0.4"},
            {"phage": "p1", "phage_gc_content": "0.5"},
            {"phage": "p2", "phage_gc_content": "0.6"},
            {"phage": "p3", "phage_gc_content": "0.7"},
        ],
    )
    _write_csv(
        track_d_distance,
        ["phage", "phage_distance_umap_00"],
        [
            {"phage": "p0", "phage_distance_umap_00": "0.05"},
            {"phage": "p1", "phage_distance_umap_00": "0.1"},
            {"phage": "p2", "phage_distance_umap_00": "0.2"},
            {"phage": "p3", "phage_distance_umap_00": "0.3"},
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
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "isolation_host_distance": "0.3"},
            {"pair_id": "b1__p2", "bacteria": "b1", "phage": "p2", "isolation_host_distance": "0.4"},
            {"pair_id": "b1__p3", "bacteria": "b1", "phage": "p3", "isolation_host_distance": "0.2"},
        ],
    )
    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    _write_json(
        tg01_summary,
        {
            "lightgbm": {
                "best_params": {
                    "n_estimators": 150,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_child_samples": 10,
                }
            }
        },
    )
    _write_json(tk01_manifest, {"lift_decision": "keep_vhrdb_for_followup_arms"})
    ti08_cohort = tmp_path / "ti08_training_cohort_rows.csv"
    _write_csv(
        ti08_cohort,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "external_label_include_in_training",
            "external_label_confidence_tier",
            "external_label_confidence_score",
            "external_label_training_weight",
        ],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "B",
                "source_system": "vhrdb",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
            },
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "B",
                "source_system": "basel",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "B",
                "source_system": "basel",
                "external_label_include_in_training": "0",
                "external_label_confidence_tier": "exclude",
                "external_label_confidence_score": "0",
                "external_label_training_weight": "0.0",
            },
        ],
    )

    output_dir = tmp_path / "out"

    main(
        [
            "--st02-pair-table-path",
            str(st02),
            "--st03-split-assignments-path",
            str(st03),
            "--track-c-pair-table-path",
            str(track_c),
            "--track-d-genome-kmer-path",
            str(track_d_genome),
            "--track-d-distance-path",
            str(track_d_distance),
            "--track-e-rbp-compatibility-path",
            str(track_e_rbp),
            "--track-e-isolation-distance-path",
            str(track_e_isolation),
            "--v1-feature-config-path",
            str(v1_config),
            "--tg01-summary-path",
            str(tg01_summary),
            "--tk01-manifest-path",
            str(tk01_manifest),
            "--ti08-training-cohort-path",
            str(ti08_cohort),
            "--output-dir",
            str(output_dir),
            "--skip-prerequisites",
        ]
    )

    summary = list(csv.DictReader((output_dir / "tk02_basel_lift_summary.csv").open("r", encoding="utf-8")))
    assert summary[0]["arm"] == "internal_plus_vhrdb"
    assert summary[1]["arm"] == "internal_plus_vhrdb_plus_basel"
    assert summary[1]["basel_row_count"] == "1"

    manifest = json.loads((output_dir / "tk02_basel_lift_manifest.json").read_text(encoding="utf-8"))
    assert manifest["previous_best_source_systems"] == ["vhrdb"]
    assert manifest["lift_assessment"] in {"adds", "hurts", "neutral"}


def test_main_rejects_basel_rows_that_do_not_join_into_training(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"
    tg01_summary = tmp_path / "tg01_model_summary.json"
    tk01_manifest = tmp_path / "tk01_vhrdb_lift_manifest.json"
    ti08_cohort = tmp_path / "ti08_training_cohort_rows.csv"

    _write_csv(
        st02,
        ["pair_id", "bacteria", "phage", "label_hard_any_lysis", "label_strict_confidence_tier", "host_pathotype"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            }
        ],
    )
    _write_csv(
        st03,
        ["pair_id", "bacteria", "phage", "cv_group", "split_holdout", "split_cv5_fold", "is_hard_trainable"],
        [
            {
                "pair_id": "b1__p0",
                "bacteria": "b1",
                "phage": "p0",
                "cv_group": "g0",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            }
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
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "host_pathotype": "pt",
            }
        ],
    )
    _write_csv(track_d_genome, ["phage", "phage_gc_content"], [{"phage": "p0", "phage_gc_content": "0.4"}])
    _write_csv(
        track_d_distance, ["phage", "phage_distance_umap_00"], [{"phage": "p0", "phage_distance_umap_00": "0.05"}]
    )
    _write_csv(
        track_e_rbp,
        ["pair_id", "bacteria", "phage", "lookup_available"],
        [{"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "lookup_available": "1"}],
    )
    _write_csv(
        track_e_isolation,
        ["pair_id", "bacteria", "phage", "isolation_host_distance"],
        [{"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "isolation_host_distance": "0.25"}],
    )
    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    _write_json(
        tg01_summary,
        {
            "lightgbm": {
                "best_params": {
                    "n_estimators": 150,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_child_samples": 10,
                }
            }
        },
    )
    _write_json(tk01_manifest, {"lift_decision": "keep_vhrdb_for_followup_arms"})
    _write_csv(
        ti08_cohort,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "external_label_include_in_training",
            "external_label_confidence_tier",
            "external_label_confidence_score",
            "external_label_training_weight",
        ],
        [
            {
                "pair_id": "b1__p999",
                "bacteria": "b1",
                "phage": "p999",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "B",
                "source_system": "basel",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
            }
        ],
    )

    with pytest.raises(ValueError, match="no BASEL rows that join"):
        main(
            [
                "--st02-pair-table-path",
                str(st02),
                "--st03-split-assignments-path",
                str(st03),
                "--track-c-pair-table-path",
                str(track_c),
                "--track-d-genome-kmer-path",
                str(track_d_genome),
                "--track-d-distance-path",
                str(track_d_distance),
                "--track-e-rbp-compatibility-path",
                str(track_e_rbp),
                "--track-e-isolation-distance-path",
                str(track_e_isolation),
                "--v1-feature-config-path",
                str(v1_config),
                "--tg01-summary-path",
                str(tg01_summary),
                "--tk01-manifest-path",
                str(tk01_manifest),
                "--ti08-training-cohort-path",
                str(ti08_cohort),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-prerequisites",
            ]
        )
