"""Tests for TP03: Feature lift visualization.

The TG05 summary is a generated file (not checked in), so the TG05 fixture here
mirrors the schema produced by run_feature_subset_sweep.py.  The schema guard
test validates the fallback's keys match what build_feature_lift_bundle reads.
"""

from __future__ import annotations

from lyzortx.pipeline.track_p import run_track_p
from lyzortx.pipeline.track_p.steps.build_feature_lift_visualization import (
    _fallback_tg05_summary,
    build_feature_lift_bundle,
    build_feature_lift_rows,
    render_feature_lift_visualization_html,
)


def _tg03_summary() -> dict:
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


def _tg05_summary() -> dict:
    """TG05 summary fixture matching the schema produced by run_feature_subset_sweep."""
    return {
        "task_id": "TG05",
        "final_feature_lock": {
            "winner_label": "defense + OMP + phage-genomic",
            "panel_evaluation_metrics": {
                "holdout_roc_auc": 0.910766,
                "holdout_top3_hit_rate_all_strains": 0.876923,
                "holdout_brier_score": 0.109543,
            },
            "deployment_realistic_metrics": {
                "holdout_roc_auc": 0.835178,
                "holdout_top3_hit_rate_all_strains": 0.923077,
                "holdout_top3_hit_rate_susceptible_only": 0.952381,
                "holdout_brier_score": 0.157767,
                "excluded_columns": ["host_n_infections"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Schema guard: fallback must match the keys build_feature_lift_bundle reads
# ---------------------------------------------------------------------------


def test_fallback_tg05_summary_has_required_keys() -> None:
    fallback = _fallback_tg05_summary()
    lock = fallback["final_feature_lock"]
    assert "panel_evaluation_metrics" in lock
    assert "deployment_realistic_metrics" in lock
    deployment = lock["deployment_realistic_metrics"]
    for key in (
        "holdout_roc_auc",
        "holdout_top3_hit_rate_all_strains",
        "holdout_top3_hit_rate_susceptible_only",
        "holdout_brier_score",
    ):
        assert key in deployment, f"missing {key} in fallback deployment metrics"
    assert "excluded_columns" in deployment or "excluded_label_derived_columns" in deployment


# ---------------------------------------------------------------------------
# build_feature_lift_rows: ordering and lift computation
# ---------------------------------------------------------------------------


def test_lift_rows_ordered_by_ablation_sequence() -> None:
    rows = build_feature_lift_rows(_tg03_summary())
    assert [r["arm_id"] for r in rows] == [
        "v0_features_only",
        "plus_defense_subtypes",
        "plus_omp_receptors",
        "plus_phage_genomic",
        "plus_pairwise_compatibility",
        "all_features",
    ]


def test_reference_arm_has_zero_lift() -> None:
    rows = build_feature_lift_rows(_tg03_summary())
    assert rows[0]["top3_lift_pp_vs_metadata"] == 0.0


def test_positive_lift_computed_correctly() -> None:
    rows = build_feature_lift_rows(_tg03_summary())
    defense = rows[1]
    assert defense["arm_id"] == "plus_defense_subtypes"
    assert defense["top3_lift_pp_vs_metadata"] == 4.6


def test_auc_delta_vs_metadata() -> None:
    rows = build_feature_lift_rows(_tg03_summary())
    phage_genomic = rows[3]
    assert phage_genomic["arm_id"] == "plus_phage_genomic"
    assert phage_genomic["roc_auc_delta_vs_metadata"] == 0.00072


# ---------------------------------------------------------------------------
# build_feature_lift_bundle: end-to-end with TG05 summary keys
# ---------------------------------------------------------------------------


def test_bundle_reads_tg05_summary_keys_without_error() -> None:
    bundle = build_feature_lift_bundle(
        tg03_summary=_tg03_summary(),
        tg05_summary=_tg05_summary(),
        tg03_source={"path": None, "sha256": None},
        tg05_source={"path": None, "sha256": None},
    )
    callout = bundle["tg05_callout"]
    assert "deployment_realistic_sensitivity" in callout
    assert callout["deployment_realistic_sensitivity"]["holdout_roc_auc"] == 0.835178


def test_bundle_uses_fallback_keys_without_error() -> None:
    """The fallback path must also work end-to-end."""
    fallback = _fallback_tg05_summary()
    bundle = build_feature_lift_bundle(
        tg03_summary=_tg03_summary(),
        tg05_summary=fallback,
        tg03_source={"path": None, "sha256": None},
        tg05_source={"path": None, "sha256": None},
    )
    assert bundle["tg05_callout"]["deployment_realistic_sensitivity"]["holdout_top3_hit_rate_all_strains"] == 0.923077


# ---------------------------------------------------------------------------
# render: HTML contains expected markers
# ---------------------------------------------------------------------------


def test_html_contains_tg05_deployment_callout() -> None:
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


# ---------------------------------------------------------------------------
# run_track_p dispatcher
# ---------------------------------------------------------------------------


def test_run_track_p_dispatches_feature_lift_step(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(run_track_p.build_digital_phagogram, "main", lambda argv: calls.append("digital"))
    monkeypatch.setattr(run_track_p.build_feature_lift_visualization, "main", lambda argv: calls.append("feature-lift"))
    monkeypatch.setattr(run_track_p.build_panel_coverage_heatmap, "main", lambda argv: calls.append("heatmap"))

    run_track_p.main(["--step", "feature-lift-visualization"])
    assert calls == ["feature-lift"]


def test_run_track_p_dispatches_all_steps(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(run_track_p.build_digital_phagogram, "main", lambda argv: calls.append("digital"))
    monkeypatch.setattr(run_track_p.build_feature_lift_visualization, "main", lambda argv: calls.append("feature-lift"))
    monkeypatch.setattr(run_track_p.build_panel_coverage_heatmap, "main", lambda argv: calls.append("heatmap"))

    run_track_p.main(["--step", "all"])
    assert calls == ["digital", "feature-lift", "heatmap"]
