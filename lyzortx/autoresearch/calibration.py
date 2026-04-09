"""Post-hoc probability calibration for AUTORESEARCH predictions.

Fits an isotonic calibrator on out-of-fold (OOF) predictions from k-fold
cross-validation on the training set. This avoids circular calibration
(fitting and evaluating on the same data).

The calibrator maps raw model probabilities to calibrated probabilities
that better approximate true event rates.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

LOGGER = logging.getLogger(__name__)

CALIBRATION_CV_FOLDS = 3
CALIBRATION_RANDOM_STATE = 42


def _build_fold_model(
    device_type: str,
    params: dict[str, Any],
    random_state: int,
) -> LGBMClassifier:
    """Build a LightGBM classifier for a single CV fold."""
    estimator_params: dict[str, Any] = {
        **params,
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": -1,
        "device_type": device_type,
    }
    if device_type == "cpu":
        estimator_params["deterministic"] = True
        estimator_params["force_col_wise"] = True
    return LGBMClassifier(**estimator_params)


def fit_isotonic_calibrator_cv(
    train_design: pd.DataFrame,
    feature_columns: Sequence[str],
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    categorical_features: Sequence[str],
    *,
    device_type: str,
    model_params: dict[str, Any],
    n_folds: int = CALIBRATION_CV_FOLDS,
) -> IsotonicRegression:
    """Fit an isotonic calibrator on out-of-fold predictions from k-fold CV.

    1. Split training data into k folds (stratified by label).
    2. For each fold: train a model on k-1 folds, predict on the held-out fold.
    3. Concatenate OOF predictions → unbiased base-model scores.
    4. Fit IsotonicRegression(OOF predictions, true labels).

    Returns the fitted calibrator.
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CALIBRATION_RANDOM_STATE)
    oof_predictions = np.zeros(len(y_train), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_design[feature_columns], y_train)):
        fold_model = _build_fold_model(device_type, model_params, random_state=CALIBRATION_RANDOM_STATE + fold_idx)
        fold_model.fit(
            train_design.iloc[train_idx][list(feature_columns)],
            y_train[train_idx],
            sample_weight=sample_weight[train_idx],
            categorical_feature=list(categorical_features),
        )
        oof_predictions[val_idx] = fold_model.predict_proba(train_design.iloc[val_idx][list(feature_columns)])[:, 1]
        LOGGER.info(
            "Calibration CV fold %d/%d: %d train, %d val",
            fold_idx + 1,
            n_folds,
            len(train_idx),
            len(val_idx),
        )

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(oof_predictions, y_train, sample_weight=sample_weight)
    LOGGER.info("Isotonic calibrator fitted on %d OOF predictions", len(oof_predictions))
    return calibrator


def calibrate_predictions(calibrator: IsotonicRegression, predictions: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to raw model predictions."""
    return calibrator.predict(predictions)
