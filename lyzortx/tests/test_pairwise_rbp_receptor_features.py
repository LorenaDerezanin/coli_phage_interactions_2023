"""Tests for pairwise RBP × receptor cross-term features."""

from __future__ import annotations

import pandas as pd

from lyzortx.pipeline.autoresearch.derive_pairwise_rbp_receptor_features import (
    PAIRWISE_PREFIX,
    RBP_COUNT_COLUMN,
    RBP_PRESENT_COLUMN,
    compute_pairwise_rbp_receptor_features,
    pairwise_feature_names,
)


def _make_design_matrix() -> pd.DataFrame:
    """Minimal design matrix with RBP and receptor columns."""
    return pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B1__P2", "B2__P1", "B2__P2"],
            "bacteria": ["B1", "B1", "B2", "B2"],
            "phage": ["P1", "P1", "P2", "P2"],
            RBP_PRESENT_COLUMN: [1, 1, 0, 0],
            RBP_COUNT_COLUMN: [3, 3, 0, 0],
            "host_surface__host_receptor_btub_score": [100.0, 100.0, 50.0, 50.0],
            "host_surface__host_receptor_ompC_score": [200.0, 200.0, 0.0, 0.0],
        }
    )


def test_compute_pairwise_features_adds_correct_columns():
    """Cross-term columns are added with the pairwise prefix."""
    design = _make_design_matrix()
    added = compute_pairwise_rbp_receptor_features(design)

    assert all(col.startswith(PAIRWISE_PREFIX) for col in added)
    # 2 receptor columns × 2 RBP terms = 4 features.
    assert len(added) == 4
    assert f"{PAIRWISE_PREFIX}has_rbp_x_btub" in added
    assert f"{PAIRWISE_PREFIX}rbp_count_x_btub" in added
    assert f"{PAIRWISE_PREFIX}has_rbp_x_ompC" in added
    assert f"{PAIRWISE_PREFIX}rbp_count_x_ompC" in added


def test_cross_terms_are_correct_products():
    """has_rbp × receptor and rbp_count × receptor are element-wise products."""
    design = _make_design_matrix()
    compute_pairwise_rbp_receptor_features(design)

    # P1 (has_rbp=1, count=3) × B1 (btub=100) → has_rbp_x_btub=100, count_x_btub=300
    row0 = design.iloc[0]
    assert row0[f"{PAIRWISE_PREFIX}has_rbp_x_btub"] == 100.0
    assert row0[f"{PAIRWISE_PREFIX}rbp_count_x_btub"] == 300.0

    # P2 (has_rbp=0, count=0) × B2 (btub=50) → has_rbp_x_btub=0, count_x_btub=0
    row3 = design.iloc[3]
    assert row3[f"{PAIRWISE_PREFIX}has_rbp_x_btub"] == 0.0
    assert row3[f"{PAIRWISE_PREFIX}rbp_count_x_btub"] == 0.0


def test_pairwise_feature_names_returns_all_24():
    """Full feature name list should have 24 entries (12 receptors × 2)."""
    names = pairwise_feature_names()
    assert len(names) == 24
    assert all(name.startswith(PAIRWISE_PREFIX) for name in names)


def test_missing_rbp_column_raises():
    """Raises ValueError when RBP present column is missing."""
    design = pd.DataFrame(
        {
            "pair_id": ["B1__P1"],
            "bacteria": ["B1"],
            "phage": ["P1"],
            "host_surface__host_receptor_btub_score": [100.0],
        }
    )
    try:
        compute_pairwise_rbp_receptor_features(design)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "phage_rbp_struct__has_annotated_rbp" in str(exc)


def test_missing_receptor_columns_raises():
    """Raises ValueError when no receptor score columns are present."""
    design = pd.DataFrame(
        {
            "pair_id": ["B1__P1"],
            "bacteria": ["B1"],
            "phage": ["P1"],
            RBP_PRESENT_COLUMN: [1],
            RBP_COUNT_COLUMN: [2],
        }
    )
    try:
        compute_pairwise_rbp_receptor_features(design)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "receptor" in str(exc).lower()


def test_partial_receptors_only_adds_available():
    """When only some receptors are present, only those get cross-terms."""
    design = pd.DataFrame(
        {
            "pair_id": ["B1__P1"],
            "bacteria": ["B1"],
            "phage": ["P1"],
            RBP_PRESENT_COLUMN: [1],
            RBP_COUNT_COLUMN: [2],
            "host_surface__host_receptor_fhua_score": [150.0],
        }
    )
    added = compute_pairwise_rbp_receptor_features(design)
    assert len(added) == 2  # Only fhua × 2 RBP terms
    assert f"{PAIRWISE_PREFIX}has_rbp_x_fhua" in added
