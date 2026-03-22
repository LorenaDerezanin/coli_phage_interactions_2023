from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import FeatureSpace
from lyzortx.pipeline.track_p.steps.build_digital_phagogram import (
    build_arm_display_rows,
    build_locked_arm_feature_spaces,
    build_phagogram_bundle,
    bootstrap_probability_intervals,
    render_digital_phagogram_html,
)


def test_build_locked_arm_feature_spaces_excludes_label_derived_columns_from_deployment() -> None:
    feature_space = FeatureSpace(
        categorical_columns=("host_pathotype", "host_surface_lps_core_type"),
        numeric_columns=(
            "host_n_infections",
            "host_defense_subtype_abi_a",
            "host_defense_diversity",
            "host_receptor_variant_btub_01",
            "host_phylogeny_umap_00",
            "phage_gc_content",
            "target_receptor_present",
            "receptor_variant_training_positive_count",
        ),
        track_c_additional_columns=(
            "host_defense_subtype_abi_a",
            "host_defense_diversity",
            "host_receptor_variant_btub_01",
            "host_surface_lps_core_type",
            "host_phylogeny_umap_00",
        ),
        track_d_columns=("phage_gc_content",),
        track_e_columns=("target_receptor_present", "receptor_variant_training_positive_count"),
    )

    spaces = build_locked_arm_feature_spaces(
        feature_space,
        winner_subset_blocks=("defense", "omp", "phage_genomic"),
        excluded_columns=("host_n_infections", "receptor_variant_training_positive_count"),
    )

    assert "host_n_infections" in spaces["panel"].numeric_columns
    assert "host_n_infections" not in spaces["deployment"].numeric_columns
    assert "receptor_variant_training_positive_count" not in spaces["panel"].numeric_columns
    assert "receptor_variant_training_positive_count" not in spaces["deployment"].numeric_columns
    assert "host_pathotype" in spaces["panel"].categorical_columns


def test_bootstrap_probability_intervals_return_bounded_confidence_band() -> None:
    calibration_rows = [
        {"pair_id": "c1", "predicted_probability": 0.1, "label_hard_any_lysis": 0},
        {"pair_id": "c2", "predicted_probability": 0.2, "label_hard_any_lysis": 0},
        {"pair_id": "c3", "predicted_probability": 0.8, "label_hard_any_lysis": 1},
        {"pair_id": "c4", "predicted_probability": 0.9, "label_hard_any_lysis": 1},
    ]
    candidate_rows = [{"pair_id": "x1", "predicted_probability": 0.75, "label_hard_any_lysis": 1}]

    intervals = bootstrap_probability_intervals(
        calibration_rows,
        candidate_rows,
        bootstrap_samples=32,
        random_state=7,
    )

    assert set(intervals) == {"x1"}
    assert (
        0.0 <= intervals["x1"]["ci_low"] <= intervals["x1"]["calibrated_p_lysis"] <= intervals["x1"]["ci_high"] <= 1.0
    )


def test_build_arm_display_rows_attaches_shap_and_confidence_information() -> None:
    scored_rows = [
        {
            "pair_id": "a1",
            "bacteria": "A",
            "phage": "p2",
            "phage_family": "F1",
            "predicted_probability": 0.6,
            "label_hard_any_lysis": "1",
        },
        {
            "pair_id": "a2",
            "bacteria": "A",
            "phage": "p1",
            "phage_family": "F1",
            "predicted_probability": 0.8,
            "label_hard_any_lysis": "0",
        },
    ]
    confidence_intervals = {
        "a1": {"calibrated_p_lysis": 0.55, "ci_low": 0.5, "ci_high": 0.6, "bootstrap_samples_used": 12.0},
        "a2": {"calibrated_p_lysis": 0.75, "ci_low": 0.7, "ci_high": 0.8, "bootstrap_samples_used": 12.0},
    }
    shap_rows_by_pair = {
        "a1": {
            "top_positive": [{"feature_name": "host_n_infections", "shap_value": 0.9}],
            "top_negative": [{"feature_name": "phage_gc_content", "shap_value": -0.3}],
            "top_shap_summary": "+ host_n_infections=1 ( +0.9000 ); - phage_gc_content=0.4 ( -0.3000 )",
        },
        "a2": {
            "top_positive": [{"feature_name": "defense_evasion_score", "shap_value": 0.7}],
            "top_negative": [{"feature_name": "host_lps_type=R1", "shap_value": -0.2}],
            "top_shap_summary": "+ defense_evasion_score=0.7 ( +0.7000 ); - host_lps_type=R1=1 ( -0.2000 )",
        },
    }

    display_rows = build_arm_display_rows(scored_rows, confidence_intervals, shap_rows_by_pair)

    assert [row["phage"] for row in display_rows["A"]] == ["p1", "p2"]
    assert display_rows["A"][0]["rank"] == 1
    assert display_rows["A"][0]["p_lysis"] == 0.75
    assert display_rows["A"][0]["top_positive"][0]["feature_name"] == "defense_evasion_score"


def test_render_digital_phagogram_html_embeds_both_arm_labels() -> None:
    bundle = build_phagogram_bundle(
        config={
            "locked_v1_feature_configuration": {
                "winner_label": "defense + OMP + phage-genomic",
                "winner_subset_blocks": ["defense", "omp", "phage_genomic"],
                "panel_default": {"holdout_top3_hit_rate_all_strains": 0.88},
                "deployment_realistic": {
                    "holdout_top3_hit_rate_all_strains": 0.92,
                    "excluded_label_derived_columns": ["host_n_infections"],
                },
            }
        },
        tg05_summary={"locked_lightgbm_hyperparameters": {"n_estimators": 300}},
        initial_bacteria="A",
        display_limit=12,
        panel_rows_by_strain={
            "A": [
                {
                    "rank": 1,
                    "phage": "p1",
                    "p_lysis": 0.8,
                    "ci_low": 0.7,
                    "ci_high": 0.9,
                    "top_shap_summary": "x",
                    "top_positive": [],
                    "top_negative": [],
                }
            ]
        },
        deployment_rows_by_strain={
            "A": [
                {
                    "rank": 1,
                    "phage": "p1",
                    "p_lysis": 0.7,
                    "ci_low": 0.6,
                    "ci_high": 0.8,
                    "top_shap_summary": "y",
                    "top_positive": [],
                    "top_negative": [],
                }
            ]
        },
    )

    html = render_digital_phagogram_html(bundle)

    assert "Digital phagogram" in html
    assert "panel-default" in html
    assert "deployment-realistic" in html
    assert "host_n_infections" in html
