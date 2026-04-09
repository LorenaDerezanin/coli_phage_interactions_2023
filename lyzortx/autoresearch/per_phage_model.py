"""Per-phage LightGBM sub-models for host-range prediction.

Each phage gets its own classifier trained on bacterial features only,
learning phage-specific host-range patterns. This breaks the Straboviridae
prior collapse where the all-pairs model's top-3 stays all-Straboviridae.

For phages with too few positive examples (< MIN_POSITIVES_FOR_FIT), the
per-phage prediction falls back to the all-pairs model score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

LOGGER = logging.getLogger(__name__)

# Minimum number of positive training examples required to fit a per-phage model.
# Below this, the per-phage model would overfit on noise.
MIN_POSITIVES_FOR_FIT = 3

# Per-phage LightGBM hyperparameters — smaller than the all-pairs model because
# each sub-model has fewer training examples (~294 bacteria per phage).
PER_PHAGE_PARAMS = {
    "n_estimators": 32,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
}

# Fixed blending weight for per-phage vs all-pairs predictions.
# 0.5 = equal weight; higher = more trust in per-phage model.
DEFAULT_BLEND_ALPHA = 0.5


@dataclass(frozen=True)
class PerPhageResult:
    """Result of per-phage model fitting and prediction."""

    n_phages_total: int
    n_phages_fitted: int
    n_phages_fallback: int
    fitted_phages: tuple[str, ...]
    fallback_phages: tuple[str, ...]


def fit_per_phage_models(
    train_design: pd.DataFrame,
    host_feature_columns: list[str],
    *,
    device_type: str = "cpu",
    min_positives: int = MIN_POSITIVES_FOR_FIT,
    random_state: int = 42,
) -> dict[str, LGBMClassifier]:
    """Fit one LightGBM per phage on host-only features.

    Returns a dict mapping phage_id -> fitted model. Phages with fewer than
    min_positives positive examples are omitted (caller should fall back to
    all-pairs predictions for these).
    """
    models: dict[str, LGBMClassifier] = {}
    phages = sorted(train_design["phage"].unique())
    n_fitted = 0
    n_skipped = 0

    for phage in phages:
        phage_rows = train_design.loc[train_design["phage"] == phage]
        y = phage_rows["label_any_lysis"].astype(int).to_numpy(dtype=int)
        n_pos = int(y.sum())

        if n_pos < min_positives or n_pos == len(y):
            n_skipped += 1
            continue

        X = phage_rows[host_feature_columns].copy()
        weight = phage_rows["training_weight_v3"].astype(float).to_numpy(dtype=float)

        estimator_params = {
            **PER_PHAGE_PARAMS,
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

        model = LGBMClassifier(**estimator_params)
        model.fit(X, y, sample_weight=weight)
        models[phage] = model
        n_fitted += 1

    LOGGER.info(
        "Per-phage models: fitted %d/%d phages (%d skipped, min_pos=%d)",
        n_fitted,
        len(phages),
        n_skipped,
        min_positives,
    )
    return models


def predict_per_phage(
    models: dict[str, LGBMClassifier],
    eval_design: pd.DataFrame,
    host_feature_columns: list[str],
    all_pairs_predictions: np.ndarray,
    *,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
) -> tuple[np.ndarray, PerPhageResult]:
    """Generate blended predictions using per-phage models + all-pairs fallback.

    For phages with a fitted per-phage model: blend per-phage and all-pairs
    predictions using blend_alpha. For phages without a model: use all-pairs
    predictions directly.

    Returns (blended_predictions, result_summary).
    """
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"blend_alpha must be between 0.0 and 1.0, got {blend_alpha}")
    blended = all_pairs_predictions.copy()
    phages = sorted(eval_design["phage"].unique())
    fitted_phages: list[str] = []
    fallback_phages: list[str] = []

    for phage in phages:
        mask = (eval_design["phage"] == phage).to_numpy()
        if phage in models:
            X = eval_design.loc[mask, host_feature_columns]
            per_phage_pred = models[phage].predict_proba(X)[:, 1]
            blended[mask] = blend_alpha * per_phage_pred + (1 - blend_alpha) * all_pairs_predictions[mask]
            fitted_phages.append(phage)
        else:
            fallback_phages.append(phage)

    result = PerPhageResult(
        n_phages_total=len(phages),
        n_phages_fitted=len(fitted_phages),
        n_phages_fallback=len(fallback_phages),
        fitted_phages=tuple(fitted_phages),
        fallback_phages=tuple(fallback_phages),
    )
    return blended, result
