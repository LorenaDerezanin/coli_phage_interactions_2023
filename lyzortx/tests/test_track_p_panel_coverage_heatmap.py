from lyzortx.pipeline.track_p.steps.build_panel_coverage_heatmap import (
    build_panel_coverage_bundle,
    render_panel_coverage_heatmap_html,
)


def _row(bacteria: str, phage: str, phylogroup: str, family: str, probability: float) -> dict[str, object]:
    return {
        "pair_id": f"{bacteria}:{phage}",
        "bacteria": bacteria,
        "phage": phage,
        "host_phylogroup": phylogroup,
        "phage_family": family,
        "predicted_probability": probability,
    }


def test_build_panel_coverage_bundle_orders_hardest_phylogroup_first() -> None:
    config = {
        "locked_v1_feature_configuration": {
            "winner_label": "defense + OMP + phage-genomic",
            "panel_default": {"holdout_top3_hit_rate_all_strains": 0.88},
            "deployment_realistic": {
                "holdout_top3_hit_rate_all_strains": 0.81,
                "excluded_label_derived_columns": ["host_n_infections"],
            },
        }
    }
    tg05_summary = {"locked_lightgbm_hyperparameters": {"n_estimators": 300}}
    panel_rows_by_strain = {
        "A1": [_row("A1", "P1", "A", "F1", 0.90), _row("A1", "P2", "A", "F2", 0.85)],
        "B1": [_row("B1", "P1", "B", "F1", 0.25), _row("B1", "P2", "B", "F2", 0.15)],
    }
    deployment_rows_by_strain = {
        "A1": [_row("A1", "P1", "A", "F1", 0.80), _row("A1", "P2", "A", "F2", 0.75)],
        "B1": [_row("B1", "P1", "B", "F1", 0.10), _row("B1", "P2", "B", "F2", 0.05)],
    }

    bundle = build_panel_coverage_bundle(
        config=config,
        tg05_summary=tg05_summary,
        initial_bacteria="B1",
        panel_rows_by_strain=panel_rows_by_strain,
        deployment_rows_by_strain=deployment_rows_by_strain,
    )

    assert bundle["row_order"][0] == "B"
    assert set(bundle["column_order"]) == {"F1", "F2"}
    assert bundle["summaries"]["hardest_panel_phylogroups"][0] == "B"
    assert bundle["delta_heatmap"]["rows"][0]["delta_probability"] == 0.1
    assert bundle["delta_heatmap"]["rows"][-1]["delta_probability"] == 0.1


def test_render_panel_coverage_heatmap_html_mentions_deployment_layer_and_host_gap() -> None:
    config = {
        "locked_v1_feature_configuration": {
            "winner_label": "defense + OMP + phage-genomic",
            "panel_default": {"holdout_top3_hit_rate_all_strains": 0.88},
            "deployment_realistic": {
                "holdout_top3_hit_rate_all_strains": 0.81,
                "excluded_label_derived_columns": ["host_n_infections"],
            },
        }
    }
    tg05_summary = {"locked_lightgbm_hyperparameters": {"n_estimators": 300}}
    panel_rows_by_strain = {"A1": [_row("A1", "P1", "A", "F1", 0.90)]}
    deployment_rows_by_strain = {"A1": [_row("A1", "P1", "A", "F1", 0.75)]}

    bundle = build_panel_coverage_bundle(
        config=config,
        tg05_summary=tg05_summary,
        initial_bacteria="A1",
        panel_rows_by_strain=panel_rows_by_strain,
        deployment_rows_by_strain=deployment_rows_by_strain,
    )
    html = render_panel_coverage_heatmap_html(bundle)

    assert "Panel coverage heatmap" in html
    assert "deployment-realistic" in html
    assert "host_n_infections" in html
    assert "Panel minus deployment gap" in html
