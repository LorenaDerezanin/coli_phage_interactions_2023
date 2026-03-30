import csv
import json
import sys
import types

import numpy as np
import pytest

from lyzortx.pipeline.track_g import run_track_g
from lyzortx.pipeline.track_g.steps import calibrate_gbm_outputs
from lyzortx.pipeline.track_g.steps.compute_shap_explanations import (
    build_global_feature_importance_rows,
    classify_strain_difficulty,
    select_recommendation_rows,
    top_feature_contributions,
)
from lyzortx.pipeline.track_g.steps.run_feature_block_ablation_suite import (
    build_ablation_arms,
    partition_track_c_columns,
)
from lyzortx.pipeline.track_g.steps.run_feature_subset_sweep import (
    build_subset_sweep_arms,
    select_winning_subset,
)
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    FeatureSpace,
    build_feature_space,
    compute_top3_hit_rate,
    fit_final_estimator,
    merge_expanded_feature_rows,
    make_lightgbm_estimator,
    select_best_candidate,
)


def test_merge_expanded_feature_rows_adds_split_phage_and_pair_features() -> None:
    merged = merge_expanded_feature_rows(
        track_c_pair_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": "1",
                "host_surface_lps_core_type": "R1",
            }
        ],
        split_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "cv_group": "G1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            }
        ],
        phage_feature_blocks=[
            [{"phage": "P1", "phage_gc_content": "0.5"}],
            [{"phage": "P1", "phage_viridic_mds_00": "0.2"}],
        ],
        pair_feature_blocks=[
            [{"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "lookup_available": "1"}],
            [
                {
                    "pair_id": "B1__P1",
                    "bacteria": "B1",
                    "phage": "P1",
                    "isolation_host_distance": "0.3",
                }
            ],
        ],
    )

    assert merged[0]["cv_group"] == "G1"
    assert merged[0]["phage_gc_content"] == "0.5"
    assert merged[0]["lookup_available"] == "1"
    assert merged[0]["isolation_host_distance"] == "0.3"


def test_merge_expanded_feature_rows_can_zero_fill_missing_pair_features() -> None:
    merged = merge_expanded_feature_rows(
        track_c_pair_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": "1",
            },
            {
                "pair_id": "B2__P1",
                "bacteria": "B2",
                "phage": "P1",
                "label_hard_any_lysis": "0",
            },
        ],
        split_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "cv_group": "G1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            },
            {
                "pair_id": "B2__P1",
                "bacteria": "B2",
                "phage": "P1",
                "cv_group": "G2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "-1",
                "is_hard_trainable": "1",
            },
        ],
        phage_feature_blocks=[[{"phage": "P1", "phage_gc_content": "0.5"}]],
        pair_feature_blocks=[[{"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "lookup_available": "1"}]],
        allow_missing_pair_features=True,
    )

    assert merged[1]["lookup_available"] == 0.0


def test_merge_expanded_feature_rows_still_raises_on_non_holdout_miss_with_zero_fill_enabled() -> None:
    with pytest.raises(KeyError, match="Missing pair-level feature row for pair_id B1__P1"):
        merge_expanded_feature_rows(
            track_c_pair_rows=[
                {
                    "pair_id": "B1__P1",
                    "bacteria": "B1",
                    "phage": "P1",
                    "label_hard_any_lysis": "1",
                }
            ],
            split_rows=[
                {
                    "pair_id": "B1__P1",
                    "bacteria": "B1",
                    "phage": "P1",
                    "cv_group": "G1",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "is_hard_trainable": "1",
                }
            ],
            phage_feature_blocks=[[{"phage": "P1", "phage_gc_content": "0.5"}]],
            pair_feature_blocks=[[{"pair_id": "B2__P1", "bacteria": "B2", "phage": "P1", "lookup_available": "1"}]],
            allow_missing_pair_features=True,
        )


def test_build_feature_space_keeps_v0_columns_and_adds_track_specific_blocks() -> None:
    feature_space = build_feature_space(
        st02_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "host_pathotype": "pt",
                "host_mouse_killed_10": "0",
            }
        ],
        track_c_pair_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "host_pathotype": "pt",
                "host_mouse_killed_10": "0",
                "host_surface_lps_core_type": "R1",
                "host_phylogeny_umap_00": "0.1",
            }
        ],
        track_d_feature_columns=["phage_gc_content", "phage_viridic_mds_00"],
        track_e_feature_columns=["lookup_available", "isolation_host_distance"],
    )

    assert "host_pathotype" in feature_space.categorical_columns
    assert "host_surface_lps_core_type" in feature_space.categorical_columns
    assert "host_phylogeny_umap_00" in feature_space.numeric_columns
    assert "phage_gc_content" in feature_space.numeric_columns
    assert "lookup_available" in feature_space.numeric_columns


def test_compute_top3_hit_rate_reports_all_and_susceptible_denominators() -> None:
    metrics = compute_top3_hit_rate(
        [
            {"bacteria": "B1", "phage": "P1", "label_hard_any_lysis": "1", "predicted_probability": 0.9},
            {"bacteria": "B1", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.8},
            {"bacteria": "B1", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.7},
            {"bacteria": "B2", "phage": "P1", "label_hard_any_lysis": "0", "predicted_probability": 0.9},
            {"bacteria": "B2", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.8},
            {"bacteria": "B2", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.7},
        ],
        probability_key="predicted_probability",
    )

    assert metrics["strain_count"] == 2
    assert metrics["hit_count"] == 1
    assert metrics["top3_hit_rate_all_strains"] == 0.5
    assert metrics["susceptible_strain_count"] == 1
    assert metrics["top3_hit_rate_susceptible_only"] == 1.0


def test_fit_final_estimator_forwards_sample_weights_when_requested() -> None:
    captured: dict[str, object] = {}

    class FakeEstimator:
        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = list(sample_weight) if sample_weight is not None else None
            captured["train_rows"] = len(y)
            return self

        def predict_proba(self, X):
            return np.array([[0.25, 0.75]] * X.shape[0])

    rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "1",
            "feature_a": "1.0",
            "effective_training_weight": "0.2",
        },
        {
            "pair_id": "B1__P2",
            "bacteria": "B1",
            "phage": "P2",
            "split_holdout": "holdout_test",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "0",
            "feature_a": "2.0",
        },
    ]
    feature_space = FeatureSpace(
        categorical_columns=(),
        numeric_columns=("feature_a",),
        track_c_additional_columns=(),
        track_d_columns=(),
        track_e_columns=(),
    )

    fit_final_estimator(
        rows,
        feature_space,
        estimator_factory=lambda params, seed_offset: FakeEstimator(),
        params={},
        sample_weight_key="effective_training_weight",
    )

    assert captured["train_rows"] == 1
    assert captured["sample_weight"] == [0.2]


def test_select_best_candidate_prefers_auc_then_top3_then_brier() -> None:
    best = select_best_candidate(
        [
            {
                "params": {"name": "a"},
                "summary": {
                    "mean_roc_auc": 0.81,
                    "mean_top3_hit_rate_all_strains": 0.80,
                    "mean_brier_score": 0.20,
                },
            },
            {
                "params": {"name": "b"},
                "summary": {
                    "mean_roc_auc": 0.81,
                    "mean_top3_hit_rate_all_strains": 0.85,
                    "mean_brier_score": 0.25,
                },
            },
        ]
    )

    assert best["params"]["name"] == "b"


def test_make_lightgbm_estimator_enables_determinism_without_forcing_single_thread(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_lightgbm = types.SimpleNamespace(LGBMClassifier=FakeLGBMClassifier)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lightgbm)

    estimator = make_lightgbm_estimator({"num_leaves": 31}, 2, base_random_state=17)

    assert estimator.__class__.__name__ == "FakeLGBMClassifier"
    assert captured["objective"] == "binary"
    assert captured["class_weight"] == "balanced"
    assert captured["random_state"] == 19
    assert captured["deterministic"] is True
    assert captured["force_col_wise"] is True
    assert "n_jobs" not in captured


def test_partition_track_c_columns_splits_defense_from_remaining_host_genomic() -> None:
    partitioned = partition_track_c_columns(
        [
            "host_defense_subtype_abi_a",
            "host_defense_diversity",
            "host_receptor_variant_btub_01",
            "host_surface_lps_core_type",
            "host_phylogeny_umap_00",
        ]
    )

    assert partitioned["defense_subtypes"] == (
        "host_defense_subtype_abi_a",
        "host_defense_diversity",
    )
    assert partitioned["host_genomic_remainder"] == (
        "host_receptor_variant_btub_01",
        "host_surface_lps_core_type",
        "host_phylogeny_umap_00",
    )


def test_build_ablation_arms_matches_acceptance_sequence() -> None:
    arms = build_ablation_arms(
        FeatureSpace(
            categorical_columns=("host_pathotype", "host_surface_lps_core_type"),
            numeric_columns=(
                "host_mouse_killed_10",
                "host_defense_subtype_abi_a",
                "host_defense_diversity",
                "host_receptor_variant_btub_01",
                "host_phylogeny_umap_00",
                "phage_gc_content",
                "target_receptor_present",
            ),
            track_c_additional_columns=(
                "host_defense_subtype_abi_a",
                "host_defense_diversity",
                "host_receptor_variant_btub_01",
                "host_surface_lps_core_type",
                "host_phylogeny_umap_00",
            ),
            track_d_columns=("phage_gc_content",),
            track_e_columns=("target_receptor_present",),
        )
    )

    assert [arm.arm_id for arm in arms] == [
        "v0_features_only",
        "plus_defense_subtypes",
        "plus_omp_receptors",
        "plus_phage_genomic",
        "plus_pairwise_compatibility",
        "all_features",
    ]
    assert "host_defense_subtype_abi_a" in arms[1].numeric_columns
    assert "phage_gc_content" not in arms[1].numeric_columns
    assert "host_surface_lps_core_type" in arms[2].categorical_columns
    assert "host_receptor_variant_btub_01" in arms[2].numeric_columns
    assert "phage_gc_content" in arms[3].numeric_columns
    assert "target_receptor_present" in arms[4].numeric_columns
    assert arms[-1].categorical_columns == ("host_pathotype", "host_surface_lps_core_type")


def test_run_track_g_dispatches_training_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "train-v1-binary"])
    assert calls == ["train-v1-binary"]

    calls.clear()
    run_track_g.main(["--step", "all"])
    assert calls == [
        "train-v1-binary",
        "calibrate-gbm",
        "feature-block-ablation",
        "compute-shap",
        "feature-subset-sweep",
        "non-leaky-candidate-search",
    ]


def test_run_track_g_dispatches_calibration_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "calibrate-gbm"])
    assert calls == ["calibrate-gbm"]


def test_run_track_g_dispatches_tg03_ablation_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "feature-block-ablation"])
    assert calls == ["feature-block-ablation"]


def test_run_track_g_dispatches_tg04_shap_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "compute-shap"])
    assert calls == ["compute-shap"]


def test_build_subset_sweep_arms_emits_all_required_2_and_3_block_combinations() -> None:
    arms = build_subset_sweep_arms(
        FeatureSpace(
            categorical_columns=("host_pathotype", "host_surface_lps_core_type"),
            numeric_columns=(
                "legacy_host_label_count",
                "host_defense_subtype_abi_a",
                "host_defense_diversity",
                "host_receptor_variant_btub_01",
                "host_phylogeny_umap_00",
                "phage_gc_content",
                "target_receptor_present",
                "legacy_receptor_support_count",
            ),
            track_c_additional_columns=(
                "host_defense_subtype_abi_a",
                "host_defense_diversity",
                "host_receptor_variant_btub_01",
                "host_surface_lps_core_type",
                "host_phylogeny_umap_00",
            ),
            track_d_columns=("phage_gc_content",),
            track_e_columns=("target_receptor_present", "legacy_receptor_support_count"),
        )
    )

    assert len(arms) == 10
    assert arms[0].arm_id == "subset_defense__omp"
    assert arms[-1].arm_id == "subset_omp__phage_genomic__pairwise"
    assert {len(arm.subset_blocks) for arm in arms} == {2, 3}


def test_select_winning_subset_requires_non_degrading_auc_before_top3() -> None:
    winner = select_winning_subset(
        [
            {
                "arm_id": "subset_defense__phage_genomic",
                "holdout_roc_auc": 0.9105,
                "holdout_brier_score": 0.1120,
                "holdout_top3_hit_rate_all_strains": 0.92,
            },
            {
                "arm_id": "subset_defense__pairwise",
                "holdout_roc_auc": 0.9099,
                "holdout_brier_score": 0.1110,
                "holdout_top3_hit_rate_all_strains": 0.95,
            },
            {
                "arm_id": "tg01_all_features_reference",
                "holdout_roc_auc": 0.9100,
                "holdout_brier_score": 0.1131,
                "holdout_top3_hit_rate_all_strains": 0.892308,
            },
        ],
        all_features_auc=0.9100,
    )

    assert winner["arm_id"] == "subset_defense__phage_genomic"
    assert winner["auc_non_degrading_vs_tg01_all_features"] is True


def test_run_track_g_dispatches_tg05_feature_subset_sweep(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "feature-subset-sweep"])
    assert calls == ["feature-subset-sweep"]


def test_run_track_g_dispatches_tg11_non_leaky_candidate_search(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("compute-shap"),
    )
    monkeypatch.setattr(
        run_track_g.run_feature_subset_sweep,
        "main",
        lambda argv: calls.append("feature-subset-sweep"),
    )
    monkeypatch.setattr(
        run_track_g.investigate_non_leaky_candidate_features,
        "main",
        lambda argv: calls.append("non-leaky-candidate-search"),
    )

    run_track_g.main(["--step", "non-leaky-candidate-search"])
    assert calls == ["non-leaky-candidate-search"]


def test_select_recommendation_rows_keeps_top_k_per_bacteria() -> None:
    selected = select_recommendation_rows(
        [
            {"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "pred_lightgbm_isotonic": "0.6"},
            {"pair_id": "B1__P2", "bacteria": "B1", "phage": "P2", "pred_lightgbm_isotonic": "0.9"},
            {"pair_id": "B1__P3", "bacteria": "B1", "phage": "P3", "pred_lightgbm_isotonic": "0.8"},
            {"pair_id": "B2__P1", "bacteria": "B2", "phage": "P1", "pred_lightgbm_isotonic": "0.4"},
            {"pair_id": "B2__P2", "bacteria": "B2", "phage": "P2", "pred_lightgbm_isotonic": "0.5"},
        ],
        recommendation_count=2,
    )

    assert [(row["bacteria"], row["phage"], row["recommendation_rank"]) for row in selected] == [
        ("B1", "P2", "1"),
        ("B1", "P3", "2"),
        ("B2", "P2", "1"),
        ("B2", "P1", "2"),
    ]


def test_top_feature_contributions_splits_positive_and_negative_drivers() -> None:
    contributions = top_feature_contributions(
        np.array([0.7, -0.2, 0.1, -0.5]),
        np.array([1.0, 0.0, 0.3, 1.0]),
        ["feat_a", "feat_b", "feat_c", "feat_d"],
        top_k=2,
    )

    assert [item["feature_name"] for item in contributions["positive"]] == ["feat_a", "feat_c"]
    assert [item["feature_name"] for item in contributions["negative"]] == ["feat_d", "feat_b"]


def test_build_global_feature_importance_rows_orders_by_mean_abs_shap() -> None:
    feature_space = FeatureSpace(
        categorical_columns=("host_pathotype",),
        numeric_columns=("phage_gc_content", "lookup_available", "host_defense_diversity"),
        track_c_additional_columns=("host_defense_diversity",),
        track_d_columns=("phage_gc_content",),
        track_e_columns=("lookup_available",),
    )

    rows = build_global_feature_importance_rows(
        np.array(
            [
                [0.8, 0.2, -0.1],
                [0.4, 0.1, -0.3],
            ]
        ),
        ["phage_gc_content", "lookup_available", "host_defense_diversity"],
        feature_space,
    )

    assert [row["feature_name"] for row in rows] == [
        "phage_gc_content",
        "host_defense_diversity",
        "lookup_available",
    ]
    assert rows[0]["feature_block"] == "track_d_phage_genomic"
    assert rows[1]["feature_block"] == "track_c_host_genomic"
    assert rows[2]["feature_block"] == "track_e_pairwise"


def test_classify_strain_difficulty_uses_confidence_and_hits() -> None:
    assert (
        classify_strain_difficulty(
            top_score=0.84,
            score_gap=0.22,
            mean_margin_from_half=0.24,
            top3_hit=True,
        )
        == "easy"
    )
    assert (
        classify_strain_difficulty(
            top_score=0.41,
            score_gap=0.02,
            mean_margin_from_half=0.05,
            top3_hit=False,
        )
        == "hard"
    )
    assert (
        classify_strain_difficulty(
            top_score=0.62,
            score_gap=0.07,
            mean_margin_from_half=0.12,
            top3_hit=True,
        )
        == "moderate"
    )


def test_tg02_calibration_outputs_expected_files_and_rows(tmp_path) -> None:
    predictions_path = tmp_path / "tg01_pair_predictions.csv"
    st02_path = tmp_path / "st02_pair_table.csv"
    st03_path = tmp_path / "st03_split_assignments.csv"
    protocol_path = tmp_path / "st03_split_protocol.json"
    output_dir = tmp_path / "tg02"

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "bacteria",
                "phage",
                "split_holdout",
                "split_cv5_fold",
                "label_hard_any_lysis",
                "prediction_context",
                "lightgbm_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "pair_id": "B1__P1",
                    "bacteria": "B1",
                    "phage": "P1",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.9",
                },
                {
                    "pair_id": "B1__P2",
                    "bacteria": "B1",
                    "phage": "P2",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.2",
                },
                {
                    "pair_id": "B2__P1",
                    "bacteria": "B2",
                    "phage": "P1",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.8",
                },
                {
                    "pair_id": "B2__P2",
                    "bacteria": "B2",
                    "phage": "P2",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.1",
                },
                {
                    "pair_id": "B3__P1",
                    "bacteria": "B3",
                    "phage": "P1",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.7",
                },
                {
                    "pair_id": "B3__P2",
                    "bacteria": "B3",
                    "phage": "P2",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.4",
                },
                {
                    "pair_id": "B4__P1",
                    "bacteria": "B4",
                    "phage": "P1",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.6",
                },
                {
                    "pair_id": "B4__P2",
                    "bacteria": "B4",
                    "phage": "P2",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.3",
                },
            ]
        )

    with st02_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "phage_family", "label_strict_confidence_tier"],
        )
        writer.writeheader()
        for pair_id, phage_family, tier in [
            ("B1__P1", "fam1", "high"),
            ("B1__P2", "fam2", "high"),
            ("B2__P1", "fam1", "medium"),
            ("B2__P2", "fam2", "high"),
            ("B3__P1", "fam1", "high"),
            ("B3__P2", "fam2", "high"),
            ("B4__P1", "fam1", "medium"),
            ("B4__P2", "fam2", "high"),
        ]:
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "phage_family": phage_family,
                    "label_strict_confidence_tier": tier,
                }
            )

    with st03_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "is_strict_trainable"])
        writer.writeheader()
        for pair_id, flag in [
            ("B1__P1", "1"),
            ("B1__P2", "1"),
            ("B2__P1", "0"),
            ("B2__P2", "1"),
            ("B3__P1", "1"),
            ("B3__P2", "1"),
            ("B4__P1", "0"),
            ("B4__P2", "1"),
        ]:
            writer.writerow({"pair_id": pair_id, "is_strict_trainable": flag})

    protocol_path.write_text(
        json.dumps(
            {
                "split_protocol_id": "steel_thread_v0_st03_split_v1",
                "split_type": "grouped_host_split",
                "split_rules": {
                    "holdout_group_fraction": 0.2,
                    "n_cv_folds": 5,
                    "split_salt": "steel_thread_v0_st03_split_v1",
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = calibrate_gbm_outputs.main(
        [
            "--tg01-predictions-path",
            str(predictions_path),
            "--st02-pair-table-path",
            str(st02_path),
            "--st03-split-assignments-path",
            str(st03_path),
            "--st03-split-protocol-path",
            str(protocol_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "50",
            "--skip-prerequisites",
        ]
    )

    assert exit_code == 0

    with (output_dir / "tg02_calibration_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with (output_dir / "tg02_pair_predictions_calibrated.csv").open("r", newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with (output_dir / "tg02_ranked_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        ranking_rows = list(csv.DictReader(handle))
    with (output_dir / "tg02_benchmark_summary.json").open("r", encoding="utf-8") as handle:
        benchmark_summary = json.load(handle)

    assert len(summary_rows) == 12
    assert {row["variant"] for row in summary_rows} == {"raw", "isotonic", "platt"}
    assert {row["label_slice"] for row in summary_rows} == {"full_label", "strict_confidence"}
    assert len(prediction_rows) == 8
    assert prediction_rows[0]["pred_lightgbm_isotonic"] != ""
    assert prediction_rows[0]["is_strict_trainable"] in {"0", "1"}
    assert len(ranking_rows) == 8
    assert ranking_rows[0]["rank_lightgbm_isotonic"] == "1"
    assert benchmark_summary["split_protocol"]["split_protocol_id"] == "steel_thread_v0_st03_split_v1"
    assert set(benchmark_summary["label_slices"]) == {"full_label", "strict_confidence"}
    assert benchmark_summary["label_slices"]["full_label"]["metrics"]["roc_auc"] is not None
    assert benchmark_summary["label_slices"]["full_label"]["bootstrap_ci"]["roc_auc"]["bootstrap_samples"] == 50
    assert (
        0.0
        <= benchmark_summary["label_slices"]["strict_confidence"]["bootstrap_ci"]["topk_hit_rate_all_strains"][
            "ci_low_topk_hit_rate_all_strains"
        ]
        <= 1.0
    )
