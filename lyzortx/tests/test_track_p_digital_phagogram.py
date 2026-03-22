from __future__ import annotations

from pathlib import Path

from lyzortx.pipeline.track_p import run_track_p
from lyzortx.pipeline.track_p.steps import build_digital_phagogram as tp01


def test_build_model_arms_respects_locked_configuration() -> None:
    feature_space = tp01.train_v1_binary_classifier.FeatureSpace(
        categorical_columns=("host_pathotype", "host_surface_lps_core_type"),
        numeric_columns=(
            "host_n_infections",
            "host_mouse_killed_10",
            "host_defense_subtype_abi_a",
            "host_receptor_variant_btub_01",
            "phage_gc_content",
            "target_receptor_present",
        ),
        track_c_additional_columns=(
            "host_n_infections",
            "host_defense_subtype_abi_a",
            "host_receptor_variant_btub_01",
            "host_surface_lps_core_type",
        ),
        track_d_columns=("phage_gc_content",),
        track_e_columns=("target_receptor_present",),
    )
    locked_configuration = {
        "winner_label": "defense + OMP + phage-genomic",
        "winner_subset_blocks": ["defense", "omp", "phage_genomic"],
    }

    panel_arm, deployment_arm = tp01.build_model_arms(feature_space, locked_configuration)

    assert panel_arm.subset_blocks == ("defense", "omp", "phage_genomic")
    assert "host_n_infections" in panel_arm.numeric_columns
    assert "host_n_infections" not in deployment_arm.numeric_columns


def test_build_dashboard_payload_includes_both_models_and_selected_strain() -> None:
    panel_artifact = tp01.ModelArtifact(
        arm=tp01.SweepArm(
            arm_id="panel",
            display_name="panel",
            subset_blocks=("defense",),
            evaluation_mode="panel_evaluation",
            categorical_columns=("host_pathotype",),
            numeric_columns=("host_n_infections",),
        ),
        rows=[
            {
                "bacteria": "B1",
                "phage": "P1",
                "predicted_probability": 0.8,
                "confidence_band_low": 0.7,
                "confidence_band_high": 0.9,
            }
        ],
        ranked_rows=[],
        recommendation_rows=[
            {
                "bacteria": "B1",
                "phage": "P1",
                "recommendation_rank": 1,
                "predicted_probability": 0.8,
                "confidence_band_low": 0.7,
                "confidence_band_high": 0.9,
                "shap_summary": "host_n_infections=1 (+0.8)",
            }
        ],
        holdout_binary_metrics={"roc_auc": 0.9, "brier_score": 0.1},
        holdout_top3_metrics={"top3_hit_rate_all_strains": 1.0, "top3_hit_rate_susceptible_only": 1.0},
    )
    deployment_artifact = tp01.ModelArtifact(
        arm=tp01.SweepArm(
            arm_id="deployment",
            display_name="deployment",
            subset_blocks=("defense",),
            evaluation_mode="deployment_realistic",
            categorical_columns=("host_pathotype",),
            numeric_columns=(),
        ),
        rows=[
            {
                "bacteria": "B1",
                "phage": "P2",
                "predicted_probability": 0.6,
                "confidence_band_low": 0.5,
                "confidence_band_high": 0.7,
            }
        ],
        ranked_rows=[],
        recommendation_rows=[
            {
                "bacteria": "B1",
                "phage": "P2",
                "recommendation_rank": 1,
                "predicted_probability": 0.6,
                "confidence_band_low": 0.5,
                "confidence_band_high": 0.7,
                "shap_summary": "defense_subtype=1 (+0.4)",
            }
        ],
        holdout_binary_metrics={"roc_auc": 0.8, "brier_score": 0.2},
        holdout_top3_metrics={"top3_hit_rate_all_strains": 1.0, "top3_hit_rate_susceptible_only": 1.0},
    )
    payload = tp01.build_dashboard_payload(
        panel_artifact,
        deployment_artifact,
        locked_configuration={
            "winner_label": "defense + OMP + phage-genomic",
            "winner_subset_blocks": ["defense", "omp", "phage_genomic"],
            "panel_default": {},
            "deployment_realistic": {"excluded_label_derived_columns": ["host_n_infections"]},
            "selection_policy": "locked",
            "label_derived_columns_reviewed": ["host_n_infections"],
        },
        tg05_summary={"source_path": str(Path("summary.json")), "sha256": "abc"},
        initial_strain="B1",
    )

    assert payload["initial_strain"] == "B1"
    assert payload["models"]["panel"]["rows_by_strain"]["B1"][0]["phage"] == "P1"
    assert payload["models"]["deployment"]["rows_by_strain"]["B1"][0]["phage"] == "P2"


def test_render_dashboard_html_contains_interactive_controls() -> None:
    payload = {
        "generated_at_utc": "2026-03-22T00:00:00+00:00",
        "task_id": "TP01",
        "initial_strain": "B1",
        "strains": ["B1"],
        "locked_configuration": {
            "winner_label": "defense + OMP + phage-genomic",
            "winner_subset_blocks": ["defense", "omp", "phage_genomic"],
            "panel_default": {"holdout_top3_hit_rate_all_strains": 0.87},
            "deployment_realistic": {
                "excluded_label_derived_columns": ["host_n_infections"],
                "holdout_top3_hit_rate_all_strains": 0.92,
            },
            "selection_policy": "locked",
            "label_derived_columns_reviewed": ["host_n_infections"],
        },
        "panel_summary": {
            "model_label": "panel",
            "holdout_roc_auc": 0.9,
            "holdout_brier_score": 0.1,
            "holdout_top3_all_strains": 1.0,
            "holdout_top3_susceptible_only": 1.0,
        },
        "deployment_summary": {
            "model_label": "deployment",
            "holdout_roc_auc": 0.8,
            "holdout_brier_score": 0.2,
            "holdout_top3_all_strains": 1.0,
            "holdout_top3_susceptible_only": 1.0,
        },
        "models": {
            "panel": {
                "arm_id": "panel",
                "label": "panel",
                "rows_by_strain": {"B1": [{"recommendation_rank": 1, "phage": "P1", "predicted_probability": 0.8}]},
            },
            "deployment": {
                "arm_id": "deployment",
                "label": "deployment",
                "rows_by_strain": {"B1": [{"recommendation_rank": 1, "phage": "P2", "predicted_probability": 0.6}]},
            },
        },
        "inputs": {"tg05_summary": {"path": "summary.json", "sha256": "abc"}},
    }

    html_output = tp01.render_dashboard_html(payload)

    assert "digital phagogram" in html_output.lower()
    assert "strain-input" in html_output
    assert "host_n_infections" in html_output
    assert "Plotly.newPlot" in html_output


def test_run_track_p_forwards_skip_flag(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        run_track_p.build_digital_phagogram,
        "main",
        lambda argv: calls.append(list(argv)),
    )

    run_track_p.main(["--step", "digital-phagogram", "--skip-prerequisites"])

    assert calls == [["--skip-prerequisites"]]
