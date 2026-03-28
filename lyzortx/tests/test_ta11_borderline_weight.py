"""Tests for TA11: borderline matrix_score=0 noise-positive downweighting."""

from __future__ import annotations

import numpy as np

from lyzortx.pipeline.steel_thread_v0.steps.st02_build_pair_table import BORDERLINE_NOISE_WEIGHT
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    FeatureSpace,
    fit_final_estimator,
    prepare_fold_datasets,
)


def test_borderline_noise_weight_constant() -> None:
    assert BORDERLINE_NOISE_WEIGHT == 0.1


def test_prepare_fold_datasets_carries_sample_weights() -> None:
    """FoldDataset.sample_weights should reflect training_weight_v3 from rows."""
    rows = [
        {
            "pair_id": f"B{i}__P1",
            "bacteria": f"B{i}",
            "phage": "P1",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "split_cv5_fold": str(i % 2),
            "label_hard_any_lysis": "1",
            "training_weight_v3": "0.1",
            "feat": str(float(i)),
        }
        for i in range(6)
    ] + [
        {
            "pair_id": f"B{i}__P2",
            "bacteria": f"B{i}",
            "phage": "P2",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "split_cv5_fold": str(i % 2),
            "label_hard_any_lysis": "0",
            "training_weight_v3": "1.0",
            "feat": str(float(i + 10)),
        }
        for i in range(6)
    ]
    feature_space = FeatureSpace(
        categorical_columns=(),
        numeric_columns=("feat",),
        track_c_additional_columns=(),
        track_d_columns=(),
        track_e_columns=(),
    )
    datasets = prepare_fold_datasets(rows, feature_space)
    assert len(datasets) == 2
    for ds in datasets:
        assert len(ds.sample_weights) == len(ds.y_train)
        for w in ds.sample_weights:
            assert w in (0.1, 1.0)


def test_prepare_fold_datasets_defaults_missing_weight_to_one() -> None:
    """Rows without training_weight_v3 should default to weight 1.0."""
    rows = [
        {
            "pair_id": f"B{i}__P1",
            "bacteria": f"B{i}",
            "phage": "P1",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "split_cv5_fold": str(i % 2),
            "label_hard_any_lysis": str(i % 2),
            "feat": str(float(i)),
        }
        for i in range(6)
    ]
    feature_space = FeatureSpace(
        categorical_columns=(),
        numeric_columns=("feat",),
        track_c_additional_columns=(),
        track_d_columns=(),
        track_e_columns=(),
    )
    datasets = prepare_fold_datasets(rows, feature_space)
    for ds in datasets:
        assert all(w == 1.0 for w in ds.sample_weights)


def test_fit_final_estimator_passes_v3_weights() -> None:
    """fit_final_estimator with sample_weight_key='training_weight_v3' should forward weights."""
    captured: dict[str, object] = {}

    class FakeEstimator:
        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = list(sample_weight) if sample_weight is not None else None
            return self

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * X.shape[0])

    rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "1",
            "training_weight_v3": "0.1",
            "feat": "1.0",
        },
        {
            "pair_id": "B2__P1",
            "bacteria": "B2",
            "phage": "P1",
            "split_holdout": "train_non_holdout",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "0",
            "training_weight_v3": "1.0",
            "feat": "2.0",
        },
        {
            "pair_id": "B3__P1",
            "bacteria": "B3",
            "phage": "P1",
            "split_holdout": "holdout_test",
            "is_hard_trainable": "1",
            "label_hard_any_lysis": "0",
            "training_weight_v3": "1.0",
            "feat": "3.0",
        },
    ]
    feature_space = FeatureSpace(
        categorical_columns=(),
        numeric_columns=("feat",),
        track_c_additional_columns=(),
        track_d_columns=(),
        track_e_columns=(),
    )

    fit_final_estimator(
        rows,
        feature_space,
        estimator_factory=lambda params, seed_offset: FakeEstimator(),
        params={},
        sample_weight_key="training_weight_v3",
    )

    assert captured["sample_weight"] == [0.1, 1.0]
