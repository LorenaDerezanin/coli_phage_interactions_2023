import numpy as np

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
        baseline_arm_id=tl05.LOCKED_BASELINE_ARM_ID,
    )

    assert bootstrap_summary["locked_baseline_defense_phage_genomic"]["holdout_roc_auc"].bootstrap_samples_used == 32
    assert bootstrap_summary["locked_plus_tl03_rbp_receptor"]["holdout_roc_auc"].ci_low is not None
    assert (
        bootstrap_summary["locked_plus_tl03_rbp_receptor__delta_vs_locked_baseline_defense_phage_genomic"][
            "holdout_top3_hit_rate_all_strains"
        ].ci_high
        is not None
    )
