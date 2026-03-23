"""Tests for TP02: Panel coverage heatmap.

Fixtures are derived from the checked-in v1_feature_configuration.json so that
key-name drift between the config and the code is caught immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyzortx.pipeline.track_p.steps.build_panel_coverage_heatmap import (
    aggregate_heatmap_layer,
    build_panel_coverage_bundle,
    render_panel_coverage_heatmap_html,
)

V1_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")


@pytest.fixture(scope="module")
def v1_config() -> dict:
    return json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))


def _row(bacteria: str, phage: str, phylogroup: str, family: str, probability: float) -> dict:
    return {
        "pair_id": f"{bacteria}:{phage}",
        "bacteria": bacteria,
        "phage": phage,
        "host_phylogroup": phylogroup,
        "phage_family": family,
        "predicted_probability": probability,
    }


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------


def test_v1_config_has_keys_read_by_panel_coverage_heatmap(v1_config: dict) -> None:
    lock = v1_config["locked_v1_feature_configuration"]
    assert "winner_label" in lock
    assert "deployment_realistic_sensitivity" in lock
    assert "excluded_label_derived_columns" in lock["deployment_realistic_sensitivity"]


# ---------------------------------------------------------------------------
# aggregate_heatmap_layer
# ---------------------------------------------------------------------------


def test_aggregate_heatmap_layer_groups_by_row_and_col() -> None:
    rows = [
        _row("A1", "P1", "A", "Myoviridae", 0.9),
        _row("A2", "P1", "A", "Myoviridae", 0.7),
        _row("B1", "P2", "B", "Siphoviridae", 0.3),
    ]
    result = aggregate_heatmap_layer(rows, row_key="host_phylogroup", col_key="phage_family")
    assert ("A", "Myoviridae") in result["cell_values"]
    assert result["cell_values"][("A", "Myoviridae")] == [0.9, 0.7]
    assert ("B", "Siphoviridae") in result["cell_values"]


def test_aggregate_heatmap_layer_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No scored rows"):
        aggregate_heatmap_layer([], row_key="host_phylogroup", col_key="phage_family")


# ---------------------------------------------------------------------------
# build_panel_coverage_bundle: ordering and delta logic
# ---------------------------------------------------------------------------


def test_hardest_phylogroup_sorted_first(v1_config: dict) -> None:
    panel = {
        "A1": [_row("A1", "P1", "A", "F1", 0.90), _row("A1", "P2", "A", "F2", 0.85)],
        "B1": [_row("B1", "P1", "B", "F1", 0.25), _row("B1", "P2", "B", "F2", 0.15)],
    }
    deployment = {
        "A1": [_row("A1", "P1", "A", "F1", 0.80), _row("A1", "P2", "A", "F2", 0.75)],
        "B1": [_row("B1", "P1", "B", "F1", 0.10), _row("B1", "P2", "B", "F2", 0.05)],
    }
    bundle = build_panel_coverage_bundle(
        config=v1_config,
        tg05_summary={"locked_lightgbm_hyperparameters": {"n_estimators": 300}},
        initial_bacteria="B1",
        panel_rows_by_strain=panel,
        deployment_rows_by_strain=deployment,
    )

    assert bundle["row_order"][0] == "B"
    assert bundle["summaries"]["hardest_panel_phylogroups"][0] == "B"


def test_delta_heatmap_computes_panel_minus_deployment(v1_config: dict) -> None:
    panel = {"A1": [_row("A1", "P1", "A", "F1", 0.90)]}
    deployment = {"A1": [_row("A1", "P1", "A", "F1", 0.70)]}
    bundle = build_panel_coverage_bundle(
        config=v1_config,
        tg05_summary={"locked_lightgbm_hyperparameters": {"n_estimators": 300}},
        initial_bacteria="A1",
        panel_rows_by_strain=panel,
        deployment_rows_by_strain=deployment,
    )

    delta_row = bundle["delta_heatmap"]["rows"][0]
    assert delta_row["delta_probability"] == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# render: HTML contains expected markers
# ---------------------------------------------------------------------------


def test_render_html_mentions_deployment_and_host_gap(v1_config: dict) -> None:
    panel = {"A1": [_row("A1", "P1", "A", "F1", 0.90)]}
    deployment = {"A1": [_row("A1", "P1", "A", "F1", 0.75)]}
    bundle = build_panel_coverage_bundle(
        config=v1_config,
        tg05_summary={"locked_lightgbm_hyperparameters": {"n_estimators": 300}},
        initial_bacteria="A1",
        panel_rows_by_strain=panel,
        deployment_rows_by_strain=deployment,
    )
    html = render_panel_coverage_heatmap_html(bundle)

    assert "Panel coverage heatmap" in html
    assert "deployment-realistic" in html
    assert "Panel minus deployment gap" in html
