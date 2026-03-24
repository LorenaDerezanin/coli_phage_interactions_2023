"""Unit tests for TK01 VHRdb lift measurement helpers."""

from __future__ import annotations

import csv
import json

import pytest

from lyzortx.pipeline.track_k.steps.build_vhrdb_lift_report import load_vhrdb_training_rows
from lyzortx.pipeline.track_k.steps.build_vhrdb_lift_report import main


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_vhrdb_training_rows_only_keeps_joinable_train_split_rows() -> None:
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
            "source_system": "vhrdb",
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
            "source_system": "vhrdb",
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
            "source_system": "vhrdb",
            "external_label_include_in_training": "1",
            "external_label_confidence_tier": "high",
            "external_label_confidence_score": "3",
            "external_label_training_weight": "1.0",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "B",
        },
    ]

    augmented_rows, counts = load_vhrdb_training_rows(feature_rows, cohort_rows)

    assert len(augmented_rows) == 1
    assert augmented_rows[0]["pair_id"] == "b1__p1"
    assert augmented_rows[0]["label_hard_any_lysis"] == "0"
    assert augmented_rows[0]["training_origin"] == "vhrdb"
    assert counts["joined_rows"] == 1
    assert counts["non_training_split_rows"] == 1
    assert counts["missing_feature_rows"] == 1


def test_main_emits_vhrdb_augmented_summary_when_ti08_rows_join(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"

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
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "host_pathotype",
        ],
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
    v1_config.write_text(
        json.dumps(
            {
                "winner_subset_blocks": ["defense", "phage_genomic"],
            }
        ),
        encoding="utf-8",
    )
    tg01_summary = tmp_path / "tg01_model_summary.json"
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
    ti08 = tmp_path / "ti08_training_cohort_rows.csv"
    _write_csv(
        ti08,
        [
            "pair_id",
            "bacteria",
            "phage",
            "source_system",
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
                "bacteria": "b1",
                "phage": "p0",
                "source_system": "vhrdb",
                "external_label_include_in_training": "1",
                "external_label_confidence_tier": "high",
                "external_label_confidence_score": "3",
                "external_label_training_weight": "1.0",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
            }
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
            "--ti08-training-cohort-path",
            str(ti08),
            "--output-dir",
            str(output_dir),
            "--skip-prerequisites",
        ]
    )

    summary = list(csv.DictReader((output_dir / "tk01_vhrdb_lift_summary.csv").open("r", encoding="utf-8")))
    assert summary[0]["arm"] == "internal_only"
    assert summary[1]["arm"] == "internal_plus_vhrdb"
    assert summary[1]["vhrdb_row_count"] == "1"

    manifest = json.loads((output_dir / "tk01_vhrdb_lift_manifest.json").read_text(encoding="utf-8"))
    assert manifest["vhrdb_counts"]["cohort_rows"] == 1
    assert manifest["vhrdb_counts"]["joined_rows"] == 1
    assert manifest["lift_decision"] in {"do_not_include_vhrdb", "keep_vhrdb_for_followup_arms"}
    assert manifest["lift_reason"]


def test_main_requires_ti08_training_cohort_file(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"

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
        ],
    )
    _write_csv(
        track_d_genome,
        ["phage", "phage_gc_content"],
        [{"phage": "p0", "phage_gc_content": "0.4"}, {"phage": "p1", "phage_gc_content": "0.5"}],
    )
    _write_csv(
        track_d_distance,
        ["phage", "phage_distance_umap_00"],
        [{"phage": "p0", "phage_distance_umap_00": "0.05"}, {"phage": "p1", "phage_distance_umap_00": "0.1"}],
    )
    _write_csv(
        track_e_rbp,
        ["pair_id", "bacteria", "phage", "lookup_available"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "lookup_available": "1"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "lookup_available": "1"},
        ],
    )
    _write_csv(
        track_e_isolation,
        ["pair_id", "bacteria", "phage", "isolation_host_distance"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "isolation_host_distance": "0.25"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "isolation_host_distance": "0.3"},
        ],
    )
    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    tg01_summary = tmp_path / "tg01_model_summary.json"
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

    with pytest.raises(FileNotFoundError, match="Missing TI08 training cohort artifact"):
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
                "--ti08-training-cohort-path",
                str(tmp_path / "missing_ti08.csv"),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-prerequisites",
            ]
        )


def test_main_rejects_empty_ti08_training_cohort_file(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"
    ti08 = tmp_path / "ti08_training_cohort_rows.csv"

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
        ],
    )
    _write_csv(
        track_d_genome,
        ["phage", "phage_gc_content"],
        [{"phage": "p0", "phage_gc_content": "0.4"}, {"phage": "p1", "phage_gc_content": "0.5"}],
    )
    _write_csv(
        track_d_distance,
        ["phage", "phage_distance_umap_00"],
        [{"phage": "p0", "phage_distance_umap_00": "0.05"}, {"phage": "p1", "phage_distance_umap_00": "0.1"}],
    )
    _write_csv(
        track_e_rbp,
        ["pair_id", "bacteria", "phage", "lookup_available"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "lookup_available": "1"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "lookup_available": "1"},
        ],
    )
    _write_csv(
        track_e_isolation,
        ["pair_id", "bacteria", "phage", "isolation_host_distance"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "isolation_host_distance": "0.25"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "isolation_host_distance": "0.3"},
        ],
    )
    ti08.write_text(
        "pair_id,bacteria,phage,source_system,external_label_include_in_training,external_label_confidence_tier,"
        "external_label_confidence_score,external_label_training_weight,label_hard_any_lysis,label_strict_confidence_tier\n",
        encoding="utf-8",
    )
    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    tg01_summary = tmp_path / "tg01_model_summary.json"
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

    with pytest.raises(ValueError, match="TI08 training cohort is empty"):
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
                "--ti08-training-cohort-path",
                str(ti08),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-prerequisites",
            ]
        )


def test_main_requires_locked_tg01_summary_when_skip_prerequisites_is_used(tmp_path) -> None:
    st02 = tmp_path / "st02_pair_table.csv"
    st03 = tmp_path / "st03_split_assignments.csv"
    track_c = tmp_path / "pair_table_v1.csv"
    track_d_genome = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance = tmp_path / "phage_distance_embedding_features.csv"
    track_e_rbp = tmp_path / "rbp_receptor_compatibility_features_v1.csv"
    track_e_isolation = tmp_path / "isolation_host_distance_features_v1.csv"
    v1_config = tmp_path / "v1_feature_configuration.json"

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
        ],
    )
    _write_csv(
        track_d_genome,
        ["phage", "phage_gc_content"],
        [{"phage": "p0", "phage_gc_content": "0.4"}, {"phage": "p1", "phage_gc_content": "0.5"}],
    )
    _write_csv(
        track_d_distance,
        ["phage", "phage_distance_umap_00"],
        [{"phage": "p0", "phage_distance_umap_00": "0.05"}, {"phage": "p1", "phage_distance_umap_00": "0.1"}],
    )
    _write_csv(
        track_e_rbp,
        ["pair_id", "bacteria", "phage", "lookup_available"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "lookup_available": "1"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "lookup_available": "1"},
        ],
    )
    _write_csv(
        track_e_isolation,
        ["pair_id", "bacteria", "phage", "isolation_host_distance"],
        [
            {"pair_id": "b1__p0", "bacteria": "b1", "phage": "p0", "isolation_host_distance": "0.25"},
            {"pair_id": "b1__p1", "bacteria": "b1", "phage": "p1", "isolation_host_distance": "0.3"},
        ],
    )
    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")

    try:
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
                str(tmp_path / "missing_tg01.json"),
                "--ti08-training-cohort-path",
                str(tmp_path / "missing_ti08.csv"),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-prerequisites",
            ]
        )
    except FileNotFoundError as exc:
        assert "Missing locked TG01 summary artifact" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing TG01 summary artifact")
