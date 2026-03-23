"""Tests for TP01: Digital phagogram visualization.

Fixtures are derived from the checked-in v1_feature_configuration.json so that
key-name drift between the config and the code is caught immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import FeatureSpace
from lyzortx.pipeline.track_p.steps.build_digital_phagogram import (
    build_arm_display_rows,
    build_locked_arm_feature_spaces,
    build_phagogram_bundle,
    bootstrap_probability_intervals,
    render_digital_phagogram_html,
)

V1_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")


@pytest.fixture(scope="module")
def v1_config() -> dict:
    return json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_lock(v1_config: dict) -> dict:
    return v1_config["locked_v1_feature_configuration"]


# ---------------------------------------------------------------------------
# Schema guard: catch key-name drift between config and production code
# ---------------------------------------------------------------------------


def test_v1_config_has_keys_read_by_digital_phagogram(v1_lock: dict) -> None:
    assert "winner_subset_blocks" in v1_lock
    assert isinstance(v1_lock["winner_subset_blocks"], list)
    assert "deployment_realistic_sensitivity" in v1_lock
    deployment = v1_lock["deployment_realistic_sensitivity"]
    assert "excluded_label_derived_columns" in deployment
    assert isinstance(deployment["excluded_label_derived_columns"], list)


def test_v1_config_has_panel_default_metrics(v1_lock: dict) -> None:
    assert "panel_default" in v1_lock
    panel = v1_lock["panel_default"]
    assert "holdout_top3_hit_rate_all_strains" in panel


# ---------------------------------------------------------------------------
# build_locked_arm_feature_spaces: core feature-space splitting logic
# ---------------------------------------------------------------------------


def _make_feature_space() -> FeatureSpace:
    return FeatureSpace(
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


def test_build_locked_arm_feature_spaces_uses_config_keys(v1_lock: dict) -> None:
    """Prove the function can consume the real config keys without KeyError."""
    feature_space = _make_feature_space()
    spaces = build_locked_arm_feature_spaces(
        feature_space,
        winner_subset_blocks=v1_lock["winner_subset_blocks"],
        excluded_columns=v1_lock["deployment_realistic_sensitivity"]["excluded_label_derived_columns"],
    )
    assert set(spaces) == {"panel", "deployment"}


def test_panel_arm_keeps_label_derived_columns() -> None:
    feature_space = _make_feature_space()
    spaces = build_locked_arm_feature_spaces(
        feature_space,
        winner_subset_blocks=("defense", "omp", "phage_genomic"),
        excluded_columns=("host_n_infections",),
    )
    assert "host_n_infections" in spaces["panel"].numeric_columns


def test_deployment_arm_excludes_label_derived_columns() -> None:
    feature_space = _make_feature_space()
    spaces = build_locked_arm_feature_spaces(
        feature_space,
        winner_subset_blocks=("defense", "omp", "phage_genomic"),
        excluded_columns=("host_n_infections", "receptor_variant_training_positive_count"),
    )
    assert "host_n_infections" not in spaces["deployment"].numeric_columns
    assert "receptor_variant_training_positive_count" not in spaces["deployment"].numeric_columns


# ---------------------------------------------------------------------------
# bootstrap_probability_intervals
# ---------------------------------------------------------------------------


def test_bootstrap_intervals_are_bounded_and_ordered() -> None:
    calibration = [
        {"pair_id": "c1", "predicted_probability": 0.1, "label_hard_any_lysis": 0},
        {"pair_id": "c2", "predicted_probability": 0.2, "label_hard_any_lysis": 0},
        {"pair_id": "c3", "predicted_probability": 0.8, "label_hard_any_lysis": 1},
        {"pair_id": "c4", "predicted_probability": 0.9, "label_hard_any_lysis": 1},
    ]
    candidates = [{"pair_id": "x1", "predicted_probability": 0.75, "label_hard_any_lysis": 1}]
    intervals = bootstrap_probability_intervals(calibration, candidates, bootstrap_samples=32, random_state=7)

    assert "x1" in intervals
    iv = intervals["x1"]
    assert 0.0 <= iv["ci_low"] <= iv["calibrated_p_lysis"] <= iv["ci_high"] <= 1.0


# ---------------------------------------------------------------------------
# build_arm_display_rows: ranking + display assembly
# ---------------------------------------------------------------------------


def test_display_rows_ranked_descending_by_probability() -> None:
    scored = [
        {
            "pair_id": "a1",
            "bacteria": "A",
            "phage": "p_low",
            "phage_family": "F1",
            "predicted_probability": 0.3,
            "label_hard_any_lysis": "0",
        },
        {
            "pair_id": "a2",
            "bacteria": "A",
            "phage": "p_high",
            "phage_family": "F1",
            "predicted_probability": 0.9,
            "label_hard_any_lysis": "1",
        },
    ]
    ci = {
        "a1": {"calibrated_p_lysis": 0.25, "ci_low": 0.2, "ci_high": 0.3, "bootstrap_samples_used": 10.0},
        "a2": {"calibrated_p_lysis": 0.85, "ci_low": 0.8, "ci_high": 0.9, "bootstrap_samples_used": 10.0},
    }
    shap = {
        "a1": {"top_positive": [], "top_negative": [], "top_shap_summary": ""},
        "a2": {"top_positive": [], "top_negative": [], "top_shap_summary": ""},
    }

    rows = build_arm_display_rows(scored, ci, shap)

    assert list(rows.keys()) == ["A"]
    assert rows["A"][0]["phage"] == "p_high"
    assert rows["A"][0]["rank"] == 1
    assert rows["A"][1]["phage"] == "p_low"
    assert rows["A"][1]["rank"] == 2


# ---------------------------------------------------------------------------
# build_phagogram_bundle + render: end-to-end with real config keys
# ---------------------------------------------------------------------------


def test_phagogram_bundle_and_html_use_config_keys(v1_config: dict) -> None:
    """Prove bundle + render accept the checked-in config shape without KeyError."""
    panel_row = {
        "rank": 1,
        "phage": "p1",
        "p_lysis": 0.8,
        "ci_low": 0.7,
        "ci_high": 0.9,
        "top_shap_summary": "x",
        "top_positive": [],
        "top_negative": [],
    }
    deployment_row = {
        "rank": 1,
        "phage": "p1",
        "p_lysis": 0.7,
        "ci_low": 0.6,
        "ci_high": 0.8,
        "top_shap_summary": "y",
        "top_positive": [],
        "top_negative": [],
    }
    bundle = build_phagogram_bundle(
        config=v1_config,
        tg05_summary={"locked_lightgbm_hyperparameters": {"n_estimators": 300}},
        initial_bacteria="A",
        display_limit=12,
        panel_rows_by_strain={"A": [panel_row]},
        deployment_rows_by_strain={"A": [deployment_row]},
    )

    html = render_digital_phagogram_html(bundle)
    assert "Digital phagogram" in html
    assert "panel-default" in html
    assert "deployment-realistic" in html
