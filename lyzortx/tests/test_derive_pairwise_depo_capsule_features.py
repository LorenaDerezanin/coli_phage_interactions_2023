"""Tests for derive_pairwise_depo_capsule_features (GT01)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lyzortx.pipeline.autoresearch.derive_pairwise_depo_capsule_features import (
    PAIRWISE_PREFIX,
    build_cluster_index,
    compute_pairwise_depo_capsule_features,
    load_deposcope_clusters,
    load_deposcope_predictions,
)


@pytest.fixture()
def deposcope_dir(tmp_path: Path) -> Path:
    """Create minimal DepoScope output files."""
    d = tmp_path / "deposcope"
    d.mkdir()

    # predictions.csv: 3 phages, 2 with depolymerases
    predictions = d / "predictions.csv"
    predictions.write_text(
        "phage,protein_id,length,deposcope_score,is_depolymerase\n"
        "P1,P1|prot_001,500,0.99,True\n"
        "P1,P1|prot_002,300,0.95,True\n"
        "P1,P1|prot_003,200,0.01,False\n"
        "P2,P2|prot_001,400,0.88,True\n"
        "P2,P2|prot_002,150,0.02,False\n"
        "P3,P3|prot_001,350,0.03,False\n",
        encoding="utf-8",
    )

    # Cluster file: P1|prot_001 and P2|prot_001 in same cluster, P1|prot_002 in its own
    clusters = d / "depo_clusters_cluster.tsv"
    clusters.write_text(
        "P1|P1|prot_001|score=1.0\tP1|P1|prot_001|score=1.0\n"
        "P1|P1|prot_001|score=1.0\tP2|P2|prot_001|score=1.0\n"
        "P1|P1|prot_002|score=1.0\tP1|P1|prot_002|score=1.0\n",
        encoding="utf-8",
    )
    return d


def test_load_deposcope_predictions(deposcope_dir: Path) -> None:
    counts = load_deposcope_predictions(deposcope_dir / "predictions.csv")
    assert counts == {"P1": 2, "P2": 1}
    assert "P3" not in counts


def test_load_deposcope_clusters(deposcope_dir: Path) -> None:
    clusters = load_deposcope_clusters(deposcope_dir / "depo_clusters_cluster.tsv")
    # P1 is in both clusters (it has prot_001 in cluster A and prot_002 in cluster B)
    assert len(clusters["P1"]) == 2
    # P2 is in cluster A only
    assert len(clusters["P2"]) == 1
    # P3 has no depolymerases so no clusters
    assert "P3" not in clusters


def test_build_cluster_index(deposcope_dir: Path) -> None:
    clusters = load_deposcope_clusters(deposcope_dir / "depo_clusters_cluster.tsv")
    index = build_cluster_index(clusters)
    assert len(index) == 2
    # Sorted alphabetically
    assert index == sorted(index)


def test_compute_features_on_design_matrix(deposcope_dir: Path) -> None:
    """Full integration: compute features on a pair-level design matrix."""
    # Build a minimal design matrix: 3 phages × 2 hosts = 6 pairs
    rows = []
    for phage in ["P1", "P2", "P3"]:
        for host in ["H1", "H2"]:
            rows.append(
                {
                    "phage": phage,
                    "bacteria": host,
                    "host_surface__host_capsule_profile_cluster_19_score": 500.0 if host == "H1" else 0.0,
                    "host_surface__host_capsule_profile_kfoa_score": 200.0 if host == "H2" else 100.0,
                }
            )
    design = pd.DataFrame(rows)

    added = compute_pairwise_depo_capsule_features(design, deposcope_dir=deposcope_dir)

    # All added columns start with the prefix
    assert all(col.startswith(PAIRWISE_PREFIX) for col in added)

    # Check phage-level features
    p1_rows = design[design["phage"] == "P1"]
    p3_rows = design[design["phage"] == "P3"]

    assert (p1_rows[f"{PAIRWISE_PREFIX}has_depo"] == 1.0).all()
    assert (p1_rows[f"{PAIRWISE_PREFIX}depo_count"] == 2.0).all()
    assert (p1_rows[f"{PAIRWISE_PREFIX}depo_cluster_count"] == 2.0).all()

    assert (p3_rows[f"{PAIRWISE_PREFIX}has_depo"] == 0.0).all()
    assert (p3_rows[f"{PAIRWISE_PREFIX}depo_count"] == 0.0).all()
    assert (p3_rows[f"{PAIRWISE_PREFIX}depo_cluster_count"] == 0.0).all()

    # Check cross-terms: P1 × H1 should have has_depo(1) × capsule_cluster_19(500) = 500
    p1_h1 = design[(design["phage"] == "P1") & (design["bacteria"] == "H1")]
    assert p1_h1[f"{PAIRWISE_PREFIX}has_depo_x_cluster_19"].iloc[0] == pytest.approx(500.0)
    assert p1_h1[f"{PAIRWISE_PREFIX}depo_count_x_cluster_19"].iloc[0] == pytest.approx(1000.0)

    # P3 × H1 should have all cross-terms = 0 (no depolymerase)
    p3_h1 = design[(design["phage"] == "P3") & (design["bacteria"] == "H1")]
    assert p3_h1[f"{PAIRWISE_PREFIX}has_depo_x_cluster_19"].iloc[0] == pytest.approx(0.0)

    # Check cluster membership features
    cluster_cols = [col for col in added if col.startswith(f"{PAIRWISE_PREFIX}in_cluster_")]
    assert len(cluster_cols) == 2  # 2 clusters in fixture


def test_missing_capsule_columns_raises(deposcope_dir: Path) -> None:
    """Raises when no capsule columns are found."""
    design = pd.DataFrame({"phage": ["P1"], "bacteria": ["H1"], "other_col": [1.0]})
    with pytest.raises(ValueError, match="No host capsule profile columns"):
        compute_pairwise_depo_capsule_features(design, deposcope_dir=deposcope_dir)


def test_missing_phage_column_raises(deposcope_dir: Path) -> None:
    """Raises when phage column is missing."""
    design = pd.DataFrame({"bacteria": ["H1"], "host_surface__host_capsule_profile_x_score": [1.0]})
    with pytest.raises(ValueError, match="phage"):
        compute_pairwise_depo_capsule_features(design, deposcope_dir=deposcope_dir)


def test_missing_deposcope_files_raises(tmp_path: Path) -> None:
    """Raises when DepoScope output files don't exist."""
    design = pd.DataFrame({"phage": ["P1"], "bacteria": ["H1"], "host_surface__host_capsule_profile_x_score": [1.0]})
    with pytest.raises(FileNotFoundError, match="predictions"):
        compute_pairwise_depo_capsule_features(design, deposcope_dir=tmp_path)
