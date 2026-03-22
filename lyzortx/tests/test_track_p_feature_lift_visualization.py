from lyzortx.pipeline.track_p import run_track_p
from lyzortx.pipeline.track_p.steps.build_feature_lift_visualization import (
    build_feature_lift_bundle,
    build_feature_lift_rows,
    render_feature_lift_visualization_html,
)


def _tg03_summary() -> dict[str, object]:
    return {
        "task_id": "TG03",
        "reference_arm": "v0_features_only",
        "arms": {
            "v0_features_only": {
                "display_name": "Metadata-only",
                "holdout_binary_metrics": {"roc_auc": 0.908023, "brier_score": 0.113537},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.861538},
            },
            "plus_defense_subtypes": {
                "display_name": "+defense subtypes",
                "holdout_binary_metrics": {"roc_auc": 0.906666, "brier_score": 0.114083},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.907692},
            },
            "plus_omp_receptors": {
                "display_name": "+OMP receptors",
                "holdout_binary_metrics": {"roc_auc": 0.910112, "brier_score": 0.112338},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
            "plus_phage_genomic": {
                "display_name": "+phage genomic",
                "holdout_binary_metrics": {"roc_auc": 0.908743, "brier_score": 0.112097},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.907692},
            },
            "plus_pairwise_compatibility": {
                "display_name": "+pairwise compatibility",
                "holdout_binary_metrics": {"roc_auc": 0.905398, "brier_score": 0.117343},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
            "all_features": {
                "display_name": "All features combined",
                "holdout_binary_metrics": {"roc_auc": 0.909089, "brier_score": 0.113112},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
        },
    }


def _tg05_summary() -> dict[str, object]:
    return {
        "task_id": "TG05",
        "final_feature_lock": {
            "winner_label": "defense + OMP + phage-genomic",
            "panel_default": {
                "holdout_roc_auc": 0.910766,
                "holdout_top3_hit_rate_all_strains": 0.876923,
                "holdout_brier_score": 0.109543,
            },
            "deployment_realistic": {
                "holdout_roc_auc": 0.835178,
                "holdout_top3_hit_rate_all_strains": 0.923077,
                "holdout_top3_hit_rate_susceptible_only": 0.952381,
                "holdout_brier_score": 0.157767,
                "excluded_label_derived_columns": ["host_n_infections"],
            },
        },
    }


def test_build_feature_lift_rows_orders_sequence_and_computes_lift_pp() -> None:
    rows = build_feature_lift_rows(_tg03_summary())

    assert [row["arm_id"] for row in rows] == [
        "v0_features_only",
        "plus_defense_subtypes",
        "plus_omp_receptors",
        "plus_phage_genomic",
        "plus_pairwise_compatibility",
        "all_features",
    ]
    assert rows[0]["top3_lift_pp_vs_metadata"] == 0.0
    assert rows[1]["top3_lift_pp_vs_metadata"] == 4.6
    assert rows[3]["roc_auc_delta_vs_metadata"] == 0.00072


def test_render_feature_lift_visualization_html_contains_tg05_callout() -> None:
    bundle = build_feature_lift_bundle(
        tg03_summary=_tg03_summary(),
        tg05_summary=_tg05_summary(),
        tg03_source={"path": None, "sha256": None},
        tg05_source={"path": None, "sha256": None},
    )

    html = render_feature_lift_visualization_html(bundle)

    assert "Feature lift from the TG03 ablation suite" in html
    assert "92.3%" in html
    assert "0.835" in html
    assert "host_n_infections" in html


def test_run_track_p_dispatches_feature_lift_visualization_step(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(run_track_p.build_digital_phagogram, "main", lambda argv: calls.append("digital"))
    monkeypatch.setattr(
        run_track_p.build_feature_lift_visualization,
        "main",
        lambda argv: calls.append("feature-lift"),
    )
    monkeypatch.setattr(run_track_p.build_panel_coverage_heatmap, "main", lambda argv: calls.append("heatmap"))

    run_track_p.main(["--step", "feature-lift-visualization"])

    assert calls == ["feature-lift"]
