"""Unit tests for ST0.4b pure helper logic."""

from __future__ import annotations

from lyzortx.pipeline.steel_thread_v0.steps.st04b_ablation_suite import build_split_ap_snapshot
from lyzortx.pipeline.steel_thread_v0.steps.st04b_ablation_suite import infer_feature_columns
from lyzortx.pipeline.steel_thread_v0.steps.st04b_ablation_suite import summarize_signal_sources


def test_infer_feature_columns_partitions_identity_and_non_identity() -> None:
    columns = {
        "pair_id",
        "bacteria",
        "phage",
        "host_origin",
        "host_n_defense_systems",
        "phage_family",
        "phage_genus",
        "pair_host_phylo_equals_phage_host_phylo",
    }

    out = infer_feature_columns(columns)

    assert "bacteria" in out["host_only"]
    assert "phage" in out["phage_only"]
    assert "bacteria" not in out["no_identity"]
    assert "phage" not in out["no_identity"]
    assert "pair_host_phylo_equals_phage_host_phylo" in out["no_identity"]


def test_summarize_signal_sources_reports_dominant_axis() -> None:
    matrix_rows = [
        {"model": "host_only_logreg", "split": "all_eval", "average_precision": 0.71},
        {"model": "phage_only_logreg", "split": "all_eval", "average_precision": 0.54},
        {"model": "no_identity_logreg", "split": "all_eval", "average_precision": 0.49},
        {"model": "full_reference_logreg", "split": "all_eval", "average_precision": 0.75},
        {"model": "dummy_prior", "split": "all_eval", "average_precision": 0.40},
    ]

    lines = summarize_signal_sources(matrix_rows)

    assert any("Dominant single-axis signal is host identity" in line for line in lines)
    assert any("No-identity control AP=0.490" in line for line in lines)


def test_build_split_ap_snapshot_handles_missing_average_precision() -> None:
    rows = [
        {"split": "all_eval", "model": "host_only_logreg", "average_precision": 0.7},
        {"split": "all_eval", "model": "dummy_prior", "average_precision": 0.4},
        {"split": "dual_holdout_test", "model": "host_only_logreg", "average_precision": None},
    ]

    out = build_split_ap_snapshot(rows)

    assert out["all_eval"]["host_only_logreg"] == 0.7
    assert out["all_eval"]["dummy_prior"] == 0.4
    assert out["dual_holdout_test"]["host_only_logreg"] is None
