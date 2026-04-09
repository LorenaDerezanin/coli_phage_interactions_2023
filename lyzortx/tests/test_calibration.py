"""Tests for post-hoc isotonic calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lyzortx.autoresearch.calibration import (
    calibrate_predictions,
    fit_isotonic_calibrator_cv,
)
from lyzortx.autoresearch.train import PAIR_SCORER_PARAMS


def _make_calibration_data(
    n_pairs: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Build synthetic training data for calibration tests.

    Returns (design_frame, feature_columns, y_train, sample_weight).
    """
    rng = np.random.default_rng(seed)
    feature_columns = ["host_surface__feat_a", "host_surface__feat_b", "phage_stats__feat_c"]

    features = {col: rng.normal(0, 1, n_pairs) for col in feature_columns}
    # Labels correlated with feat_a for a learnable signal.
    logits = features["host_surface__feat_a"] * 1.5 + rng.normal(0, 0.5, n_pairs)
    y = (logits > 0).astype(int)

    design = pd.DataFrame(
        {
            "pair_id": [f"pair_{i}" for i in range(n_pairs)],
            "bacteria": [f"B{i % 10}" for i in range(n_pairs)],
            "phage": [f"P{i % 5}" for i in range(n_pairs)],
            **{col: vals for col, vals in features.items()},
        }
    )
    sample_weight = np.ones(n_pairs, dtype=float)
    return design, feature_columns, y, sample_weight


def test_fit_isotonic_calibrator_cv_returns_calibrator():
    """Calibrator should be fitted and produce valid predictions."""
    design, feature_cols, y, weight = _make_calibration_data()
    calibrator = fit_isotonic_calibrator_cv(
        design,
        feature_cols,
        y,
        weight,
        categorical_features=[],
        device_type="cpu",
        model_params=PAIR_SCORER_PARAMS,
    )

    # Calibrator should predict values in [0, 1].
    test_scores = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    calibrated = calibrate_predictions(calibrator, test_scores)
    assert all(0.0 <= val <= 1.0 for val in calibrated)


def test_calibration_is_monotonic():
    """Isotonic calibration should preserve monotonic ordering."""
    design, feature_cols, y, weight = _make_calibration_data(n_pairs=500)
    calibrator = fit_isotonic_calibrator_cv(
        design,
        feature_cols,
        y,
        weight,
        categorical_features=[],
        device_type="cpu",
        model_params=PAIR_SCORER_PARAMS,
    )

    # Apply to a sorted sequence — output should be non-decreasing.
    raw_scores = np.linspace(0.0, 1.0, 50)
    calibrated = calibrate_predictions(calibrator, raw_scores)
    assert all(calibrated[i] <= calibrated[i + 1] + 1e-10 for i in range(len(calibrated) - 1))


def test_calibrate_predictions_preserves_length():
    """Output array should have the same length as input."""
    design, feature_cols, y, weight = _make_calibration_data()
    calibrator = fit_isotonic_calibrator_cv(
        design,
        feature_cols,
        y,
        weight,
        categorical_features=[],
        device_type="cpu",
        model_params=PAIR_SCORER_PARAMS,
    )

    raw = np.random.default_rng(0).random(100)
    calibrated = calibrate_predictions(calibrator, raw)
    assert len(calibrated) == len(raw)
