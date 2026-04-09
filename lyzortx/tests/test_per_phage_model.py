"""Tests for per-phage LightGBM sub-models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lyzortx.autoresearch.per_phage_model import (
    PerPhageResult,
    fit_per_phage_models,
    predict_per_phage,
)


def _make_train_design(
    *,
    n_bacteria: int = 20,
    phage_ids: tuple[str, ...] = ("PA", "PB", "PC"),
    positive_rate: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a synthetic train design matrix with host-only features and labels.

    Returns (design_frame, host_feature_columns).
    """
    rng = np.random.default_rng(seed)
    if positive_rate is None:
        positive_rate = {"PA": 0.5, "PB": 0.3, "PC": 0.05}

    bacteria_ids = [f"B{i:03d}" for i in range(n_bacteria)]
    host_feature_columns = [
        "host_surface__receptor_score",
        "host_surface__o_antigen_type_num",
        "host_typing__phylogroup_num",
    ]

    rows = []
    for phage in phage_ids:
        for bact in bacteria_ids:
            features = {col: rng.normal(0, 1) for col in host_feature_columns}
            label = int(rng.random() < positive_rate.get(phage, 0.2))
            rows.append(
                {
                    "pair_id": f"{bact}__{phage}",
                    "bacteria": bact,
                    "phage": phage,
                    "label_any_lysis": label,
                    "training_weight_v3": 1.0,
                    **features,
                }
            )

    return pd.DataFrame(rows), host_feature_columns


def test_fit_per_phage_models_returns_models_for_eligible_phages():
    """Phages with enough positives get a model; others are omitted."""
    design, host_cols = _make_train_design(
        n_bacteria=30,
        positive_rate={"PA": 0.5, "PB": 0.4, "PC": 0.0},
    )
    models = fit_per_phage_models(design, host_cols)

    # PA and PB should be fitted (have positives); PC has zero positives.
    assert "PA" in models
    assert "PB" in models
    assert "PC" not in models


def test_fit_per_phage_models_skips_all_positive_phage():
    """A phage with all-positive labels (n_pos == len(y)) is skipped."""
    design, host_cols = _make_train_design(
        n_bacteria=30,
        positive_rate={"PA": 1.0, "PB": 0.5, "PC": 0.4},
    )
    models = fit_per_phage_models(design, host_cols)

    assert "PA" not in models  # all positives — skipped
    assert "PB" in models
    assert "PC" in models


def test_fit_per_phage_models_respects_min_positives():
    """Phages with fewer than min_positives positive examples are skipped."""
    # With 5 bacteria and 0.1 positive rate, most seeds give <3 positives.
    design, host_cols = _make_train_design(
        n_bacteria=5,
        positive_rate={"PA": 0.5, "PB": 0.5, "PC": 0.0},
        seed=99,
    )
    # Use a high min_positives threshold to ensure PC and maybe PA/PB are skipped.
    models = fit_per_phage_models(design, host_cols, min_positives=100)
    assert len(models) == 0


def test_predict_per_phage_blends_fitted_and_fallback():
    """Fitted phages are blended; unfitted phages use all-pairs predictions."""
    train_design, host_cols = _make_train_design(n_bacteria=30)
    models = fit_per_phage_models(train_design, host_cols)

    # Build an eval design with the same phages.
    eval_design, _ = _make_train_design(n_bacteria=10, seed=123)
    all_pairs_pred = np.full(len(eval_design), 0.4)

    blended, result = predict_per_phage(
        models,
        eval_design,
        host_cols,
        all_pairs_pred,
        blend_alpha=0.5,
    )

    assert isinstance(result, PerPhageResult)
    assert result.n_phages_total == 3
    assert result.n_phages_fitted + result.n_phages_fallback == result.n_phages_total
    assert len(blended) == len(eval_design)

    # For fallback phages, blended predictions should equal all-pairs predictions.
    for phage in result.fallback_phages:
        mask = (eval_design["phage"] == phage).to_numpy()
        np.testing.assert_array_equal(blended[mask], all_pairs_pred[mask])

    # For fitted phages, blended predictions should differ from all-pairs
    # (unless the per-phage model coincidentally predicts 0.4).
    for phage in result.fitted_phages:
        mask = (eval_design["phage"] == phage).to_numpy()
        # At least some predictions should differ.
        assert not np.allclose(blended[mask], all_pairs_pred[mask], atol=1e-6)


def test_predict_per_phage_alpha_zero_equals_all_pairs():
    """With blend_alpha=0, blended predictions should equal all-pairs exactly."""
    train_design, host_cols = _make_train_design(n_bacteria=30)
    models = fit_per_phage_models(train_design, host_cols)

    eval_design, _ = _make_train_design(n_bacteria=10, seed=123)
    all_pairs_pred = np.random.default_rng(0).random(len(eval_design))

    blended, _ = predict_per_phage(
        models,
        eval_design,
        host_cols,
        all_pairs_pred,
        blend_alpha=0.0,
    )
    np.testing.assert_array_almost_equal(blended, all_pairs_pred)


def test_predict_per_phage_alpha_one_ignores_all_pairs():
    """With blend_alpha=1, blended predictions should ignore all-pairs for fitted phages."""
    train_design, host_cols = _make_train_design(n_bacteria=30)
    models = fit_per_phage_models(train_design, host_cols)

    eval_design, _ = _make_train_design(n_bacteria=10, seed=123)
    all_pairs_pred = np.full(len(eval_design), 0.999)

    blended, result = predict_per_phage(
        models,
        eval_design,
        host_cols,
        all_pairs_pred,
        blend_alpha=1.0,
    )

    # For fitted phages, blended should be pure per-phage predictions (not 0.999).
    for phage in result.fitted_phages:
        mask = (eval_design["phage"] == phage).to_numpy()
        assert not np.allclose(blended[mask], 0.999, atol=0.1)


def test_predict_per_phage_no_models_returns_all_pairs():
    """With an empty model dict, all phages fall back to all-pairs."""
    eval_design, host_cols = _make_train_design(n_bacteria=10, seed=123)
    all_pairs_pred = np.random.default_rng(0).random(len(eval_design))

    blended, result = predict_per_phage(
        {},
        eval_design,
        host_cols,
        all_pairs_pred,
    )

    np.testing.assert_array_equal(blended, all_pairs_pred)
    assert result.n_phages_fitted == 0
    assert result.n_phages_fallback == result.n_phages_total
