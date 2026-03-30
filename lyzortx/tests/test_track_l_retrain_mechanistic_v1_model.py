import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from lyzortx.pipeline.track_l.steps import retrain_mechanistic_v1_model as tl05


def test_build_arm_specs_keeps_tl03_and_tl04_separate() -> None:
    arms = tl05.build_arm_specs(
        defense_columns=("host_defense_subtype_a",),
        track_d_columns=("phage_gc_content",),
        tl03_columns=("tl03_pair_001", "tl03_direct_001"),
        tl04_columns=("tl04_pair_001",),
    )

    assert [arm.arm_id for arm in arms] == [
        "locked_baseline_defense_phage_genomic",
        "locked_plus_tl03_rbp_receptor",
        "locked_plus_tl04_defense_evasion",
        "locked_plus_tl03_tl04_combined",
    ]
    assert arms[0].tl03_columns == ()
    assert arms[1].tl03_columns == ("tl03_pair_001", "tl03_direct_001")
    assert arms[2].tl04_columns == ("tl04_pair_001",)
    assert "tl03_pair_001" not in arms[2].numeric_columns
    assert "tl04_pair_001" in arms[3].numeric_columns


def test_classify_feature_block_distinguishes_mechanistic_columns() -> None:
    block = tl05.classify_feature_block(
        "tl03_pair_profile_001_x_host_weight",
        defense_columns=("host_defense_subtype_a",),
        track_d_columns=("phage_gc_content",),
        tl03_columns=("tl03_pair_profile_001_x_host_weight",),
        tl04_columns=("tl04_pair_profile_001_x_defense_weight",),
    )

    assert block == "tl03_mechanistic"
    assert (
        tl05.classify_feature_block(
            "tl04_pair_profile_001_x_defense_weight",
            defense_columns=("host_defense_subtype_a",),
            track_d_columns=("phage_gc_content",),
            tl03_columns=("tl03_pair_profile_001_x_host_weight",),
            tl04_columns=("tl04_pair_profile_001_x_defense_weight",),
        )
        == "tl04_mechanistic"
    )


def test_build_global_feature_importance_rows_labels_blocks() -> None:
    rows = tl05.build_global_feature_importance_rows(
        np.array([[0.8, 0.1, 0.2, -0.4], [0.7, 0.0, 0.3, -0.2]], dtype=float),
        [
            "host_defense_subtype_a",
            "phage_gc_content",
            "tl03_pair_profile_001_x_host_weight",
            "tl04_pair_profile_001_x_defense_weight",
        ],
        defense_columns=("host_defense_subtype_a",),
        track_d_columns=("phage_gc_content",),
        tl03_columns=("tl03_pair_profile_001_x_host_weight",),
        tl04_columns=("tl04_pair_profile_001_x_defense_weight",),
    )

    assert [row["feature_block"] for row in rows] == [
        "track_c_defense_baseline",
        "tl04_mechanistic",
        "tl03_mechanistic",
        "track_d_phage_genomic",
    ]


def test_select_proposed_arm_prefers_highest_auc_mechanistic_gain() -> None:
    proposal = tl05.select_locked_arm(
        [
            {
                "arm_id": "locked_baseline_defense_phage_genomic",
                "holdout_roc_auc": 0.80,
                "holdout_top3_hit_rate_all_strains": 0.50,
                "holdout_brier_score": 0.20,
            },
            {
                "arm_id": "locked_plus_tl03_rbp_receptor",
                "holdout_roc_auc": 0.82,
                "holdout_top3_hit_rate_all_strains": 0.49,
                "holdout_brier_score": 0.19,
                "auc_delta_ci_low_vs_locked_baseline": 0.01,
                "top3_delta_ci_high_vs_locked_baseline": 0.02,
                "brier_improvement_ci_high_vs_locked_baseline": 0.03,
            },
            {
                "arm_id": "locked_plus_tl04_defense_evasion",
                "holdout_roc_auc": 0.79,
                "holdout_top3_hit_rate_all_strains": 0.55,
                "holdout_brier_score": 0.18,
                "auc_delta_ci_low_vs_locked_baseline": -0.01,
                "top3_delta_ci_high_vs_locked_baseline": 0.01,
                "brier_improvement_ci_high_vs_locked_baseline": 0.01,
            },
        ],
        baseline_arm_id="locked_baseline_defense_phage_genomic",
    )

    assert proposal is not None
    assert proposal["arm_id"] == "locked_plus_tl03_rbp_receptor"


def test_bootstrap_holdout_metric_cis_reports_arm_and_delta_intervals() -> None:
    bootstrap_summary = tl05.bootstrap_holdout_metric_cis(
        {
            "locked_baseline_defense_phage_genomic": [
                {"bacteria": "B1", "phage": "P1", "label_hard_any_lysis": "1", "predicted_probability": 0.9},
                {"bacteria": "B1", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.2},
                {"bacteria": "B1", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.1},
                {"bacteria": "B2", "phage": "P1", "label_hard_any_lysis": "0", "predicted_probability": 0.8},
                {"bacteria": "B2", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.2},
                {"bacteria": "B2", "phage": "P3", "label_hard_any_lysis": "1", "predicted_probability": 0.7},
            ],
            "locked_plus_tl03_rbp_receptor": [
                {"bacteria": "B1", "phage": "P1", "label_hard_any_lysis": "1", "predicted_probability": 0.95},
                {"bacteria": "B1", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.15},
                {"bacteria": "B1", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.05},
                {"bacteria": "B2", "phage": "P1", "label_hard_any_lysis": "0", "predicted_probability": 0.7},
                {"bacteria": "B2", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.1},
                {"bacteria": "B2", "phage": "P3", "label_hard_any_lysis": "1", "predicted_probability": 0.85},
            ],
        },
        bootstrap_samples=32,
        bootstrap_random_state=7,
    )

    assert bootstrap_summary["locked_baseline_defense_phage_genomic"]["holdout_roc_auc"].bootstrap_samples_used == 32
    assert bootstrap_summary["locked_plus_tl03_rbp_receptor"]["holdout_roc_auc"].ci_low is not None
    assert (
        bootstrap_summary["locked_plus_tl03_rbp_receptor__delta_vs_locked_baseline_defense_phage_genomic"][
            "holdout_top3_hit_rate_all_strains"
        ].ci_high
        is not None
    )


def test_main_writes_lift_and_shap_outputs_with_mocked_training(tmp_path: Path, monkeypatch) -> None:
    st03_path = tmp_path / "st03_split_assignments.csv"
    st02_path = tmp_path / "st02_pair_table.csv"
    track_c_path = tmp_path / "pair_table_v1.csv"
    track_d_genome_path = tmp_path / "phage_genome_kmer_features.csv"
    track_d_distance_path = tmp_path / "phage_distance_embedding_features.csv"
    tl03_path = tmp_path / "tl03.csv"
    tl04_path = tmp_path / "tl04.csv"
    tg01_summary_path = tmp_path / "tg01_model_summary.json"
    lock_path = tmp_path / "v1_feature_configuration.json"
    output_dir = tmp_path / "out"

    st03_path.write_text(
        "pair_id,bacteria,phage,cv_group,split_holdout,split_cv5_fold,is_hard_trainable\n"
        "B1__P1,B1,P1,G1,train_non_holdout,0,1\n"
        "B1__P2,B1,P2,G1,holdout_test,-1,1\n"
        "B2__P1,B2,P1,G2,train_non_holdout,1,1\n"
        "B2__P2,B2,P2,G2,holdout_test,-1,1\n",
        encoding="utf-8",
    )
    st02_path.write_text(
        "pair_id,bacteria,phage,label_hard_any_lysis\nB1__P1,B1,P1,1\nB1__P2,B1,P2,0\nB2__P1,B2,P1,1\nB2__P2,B2,P2,0\n",
        encoding="utf-8",
    )
    track_c_path.write_text(
        "pair_id,bacteria,phage,host_defense_subtype_a,label_hard_any_lysis,training_weight_v3\n"
        "B1__P1,B1,P1,1,1,1.0\n"
        "B1__P2,B1,P2,0,0,1.0\n"
        "B2__P1,B2,P1,1,1,1.0\n"
        "B2__P2,B2,P2,0,0,1.0\n",
        encoding="utf-8",
    )
    track_d_genome_path.write_text(
        "phage,phage_gc_content\nP1,0.40\nP2,0.60\n",
        encoding="utf-8",
    )
    track_d_distance_path.write_text(
        "phage,phage_viridic_mds_00\nP1,0.10\nP2,0.20\n",
        encoding="utf-8",
    )
    tl03_path.write_text(
        "pair_id,bacteria,phage,tl03_direct_profile_001_present,tl03_pair_profile_001_x_host_weight\n"
        "B1__P1,B1,P1,1,0.7\n"
        "B1__P2,B1,P2,0,0.0\n"
        "B2__P1,B2,P1,1,0.7\n"
        "B2__P2,B2,P2,0,0.0\n",
        encoding="utf-8",
    )
    tl04_path.write_text(
        "pair_id,bacteria,phage,tl04_direct_profile_001_present,tl04_pair_profile_001_x_defense_weight\n"
        "B1__P1,B1,P1,1,0.3\n"
        "B1__P2,B1,P2,0,0.0\n"
        "B2__P1,B2,P1,1,0.3\n"
        "B2__P2,B2,P2,0,0.0\n",
        encoding="utf-8",
    )
    tg01_summary_path.write_text(
        json.dumps(
            {
                "lightgbm": {
                    "best_params": {"num_leaves": 15},
                    "holdout_binary_metrics": {"roc_auc": 0.80, "brier_score": 0.20},
                    "holdout_top3_metrics": {
                        "top3_hit_rate_all_strains": 0.50,
                        "top3_hit_rate_susceptible_only": 0.50,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps(
            {
                "winner_subset_blocks": ["defense", "phage_genomic"],
                "source_lock_task_id": "TG09",
            }
        ),
        encoding="utf-8",
    )

    def fake_evaluate_arm(arm, merged_rows, *, defense_columns, track_d_columns, locked_params, estimator_factory):
        rows = [
            {
                "pair_id": "B1__P2",
                "bacteria": "B1",
                "phage": "P2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "label_hard_any_lysis": "1",
                "host_pathotype": "pt",
                "host_defense_subtype_a": 1.0,
                "phage_gc_content": 0.6,
                "tl03_direct_profile_001_present": 1.0 if arm.tl03_columns else 0.0,
                "tl03_pair_profile_001_x_host_weight": 0.7 if arm.tl03_columns else 0.0,
                "tl04_direct_profile_001_present": 1.0 if arm.tl04_columns else 0.0,
                "tl04_pair_profile_001_x_defense_weight": 0.3 if arm.tl04_columns else 0.0,
            },
            {
                "pair_id": "B2__P2",
                "bacteria": "B2",
                "phage": "P2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "label_hard_any_lysis": "0",
                "host_pathotype": "pt",
                "host_defense_subtype_a": 0.0,
                "phage_gc_content": 0.6,
                "tl03_direct_profile_001_present": 0.0,
                "tl03_pair_profile_001_x_host_weight": 0.0,
                "tl04_direct_profile_001_present": 0.0,
                "tl04_pair_profile_001_x_defense_weight": 0.0,
            },
        ]
        vectorizer = DictVectorizer(sparse=True, sort=True)
        vectorizer.fit(
            [
                {
                    "host_pathotype": "pt",
                    "host_defense_subtype_a": 1.0,
                    "phage_gc_content": 0.6,
                    "tl03_direct_profile_001_present": 1.0 if arm.tl03_columns else 0.0,
                    "tl03_pair_profile_001_x_host_weight": 0.7 if arm.tl03_columns else 0.0,
                    "tl04_direct_profile_001_present": 1.0 if arm.tl04_columns else 0.0,
                    "tl04_pair_profile_001_x_defense_weight": 0.3 if arm.tl04_columns else 0.0,
                }
            ]
        )
        estimator = types.SimpleNamespace(feature_names_=tuple(vectorizer.get_feature_names_out()))
        metrics = {
            "locked_baseline_defense_phage_genomic": (0.80, 0.50, 0.20),
            "locked_plus_tl03_rbp_receptor": (0.83, 0.55, 0.18),
            "locked_plus_tl04_defense_evasion": (0.78, 0.49, 0.21),
            "locked_plus_tl03_tl04_combined": (0.81, 0.52, 0.19),
        }
        auc, top3, brier = metrics[arm.arm_id]
        holdout_prediction_rows = [
            {
                **row,
                "predicted_probability": 0.9 if row["pair_id"] == "B1__P2" else 0.1,
                "prediction_context": "holdout_final",
            }
            for row in rows
        ]
        return {
            "arm": arm,
            "feature_space": tl05.build_arm_feature_space(
                arm,
                defense_columns=("host_defense_subtype_a",),
                track_d_columns=("phage_gc_content",),
            ),
            "fold_metrics": [],
            "cv_summary": {"mean_roc_auc": auc},
            "cv_prediction_rows": [],
            "estimator": estimator,
            "vectorizer": vectorizer,
            "holdout_rows": rows,
            "holdout_prediction_rows": holdout_prediction_rows,
            "holdout_binary_metrics": {"roc_auc": auc, "brier_score": brier},
            "holdout_top3_metrics": {
                "top3_hit_rate_all_strains": top3,
                "top3_hit_rate_susceptible_only": top3,
            },
            "pair_prediction_rows": holdout_prediction_rows,
            "holdout_top3_rows": [
                {
                    "model_label": arm.arm_id,
                    "bacteria": "B1",
                    "phage": "P2",
                    "pair_id": "B1__P2",
                    "rank": 1,
                    "predicted_probability": 0.9,
                    "label_hard_any_lysis": "1",
                }
            ],
        }

    class FakeExplanation:
        def __init__(self, feature_names, n_rows):
            values = np.zeros((n_rows, len(feature_names)), dtype=float)
            tl03_index = next((index for index, name in enumerate(feature_names) if "tl03_" in name), 0)
            values[:, tl03_index] = 0.7
            values[:, 0] = 0.1
            self.values = values
            self.base_values = np.full(n_rows, 0.5)

    class FakeExplainer:
        def __init__(self, estimator):
            self.feature_names = estimator.feature_names_

        def __call__(self, feature_matrix):
            return FakeExplanation(self.feature_names, feature_matrix.shape[0])

    monkeypatch.setattr(tl05, "ensure_prerequisite_outputs", lambda args: None)
    monkeypatch.setattr(tl05, "load_tg01_lock", lambda path: {"best_params": {"num_leaves": 15}})
    monkeypatch.setattr(
        tl05,
        "load_v1_lock",
        lambda path: {"winner_subset_blocks": ["defense", "phage_genomic"], "source_lock_task_id": "TG09"},
    )
    monkeypatch.setattr(
        tl05,
        "load_tl11_feature_provenance",
        lambda *args, **kwargs: {
            "manifest_path": str(tmp_path / "fake_manifest.json"),
            "manifest_sha256": "fake",
            "feature_path": str(args[0]),
            "feature_sha256": "fake",
            "holdout_bacteria_ids": ["B2"],
            "excluded_pair_rows": 1,
        },
    )
    monkeypatch.setattr(tl05, "evaluate_arm", fake_evaluate_arm)
    fake_shap_module = types.SimpleNamespace(TreeExplainer=FakeExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap_module)

    exit_code = tl05.main(
        [
            "--st03-split-assignments-path",
            str(st03_path),
            "--st02-pair-table-path",
            str(st02_path),
            "--track-c-pair-table-path",
            str(track_c_path),
            "--track-d-genome-kmer-path",
            str(track_d_genome_path),
            "--track-d-distance-path",
            str(track_d_distance_path),
            "--tl03-feature-path",
            str(tl03_path),
            "--tl04-feature-path",
            str(tl04_path),
            "--tg01-summary-path",
            str(tg01_summary_path),
            "--v1-lock-path",
            str(lock_path),
            "--output-dir",
            str(output_dir),
            "--skip-prerequisites",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "tl05_mechanistic_lift_summary.json").read_text(encoding="utf-8"))
    metrics_rows = list(csv.DictReader((output_dir / "tl05_mechanistic_lift_metrics.csv").open(encoding="utf-8")))
    shap_rows = list(csv.DictReader((output_dir / "tl05_shap_global_feature_importance.csv").open(encoding="utf-8")))

    assert len(metrics_rows) == 4
    assert summary["lock_decision"]["status"] == "no_honest_lift"
    assert summary["proposed_lock_arm"] is None
    assert (output_dir / "tl05_no_honest_lift_rejections.json").exists()
    assert "holdout_roc_auc_ci_low" in metrics_rows[0]
    assert shap_rows
