"""Tests for the annotation-interaction enrichment module."""

from __future__ import annotations

import numpy as np
import pytest

from lyzortx.pipeline.track_l.steps.annotation_interaction_enrichment import (
    EnrichmentResult,
    benjamini_hochberg,
    compute_enrichment,
    results_to_rows,
)


class TestBenjaminiHochberg:
    """Tests for the BH correction function."""

    def test_empty_array(self) -> None:
        result = benjamini_hochberg(np.array([], dtype=np.float64))
        assert len(result) == 0

    def test_single_pvalue(self) -> None:
        result = benjamini_hochberg(np.array([0.03]))
        assert result[0] == pytest.approx(0.03)

    def test_all_significant_stay_significant(self) -> None:
        p_values = np.array([0.001, 0.002, 0.003])
        result = benjamini_hochberg(p_values)
        # All should remain < 0.05 since they're very small
        assert all(q < 0.05 for q in result)

    def test_known_correction(self) -> None:
        """Verify BH against a hand-calculated example.

        p-values: [0.01, 0.04, 0.03, 0.20]
        Sorted: [0.01, 0.03, 0.04, 0.20], sorted indices: [0, 2, 1, 3]
        BH raw: [0.01*4/1, 0.03*4/2, 0.04*4/3, 0.20*4/4] = [0.04, 0.06, 0.0533, 0.20]
        Reverse cummin: [0.04, 0.0533, 0.0533, 0.20]
        Map back: idx 0 -> 0.04, idx 2 -> 0.0533, idx 1 -> 0.0533, idx 3 -> 0.20
        """
        p_values = np.array([0.01, 0.04, 0.03, 0.20])
        result = benjamini_hochberg(p_values)
        assert result[0] == pytest.approx(0.04, abs=1e-10)
        assert result[1] == pytest.approx(4 * 0.04 / 3, abs=1e-10)  # 0.05333
        assert result[2] == pytest.approx(4 * 0.04 / 3, abs=1e-10)  # monotonicity -> same as rank 3
        assert result[3] == pytest.approx(0.20, abs=1e-10)

    def test_capped_at_one(self) -> None:
        p_values = np.array([0.5, 0.8, 0.9])
        result = benjamini_hochberg(p_values)
        assert all(q <= 1.0 for q in result)


class TestComputeEnrichment:
    """Tests for the main enrichment computation."""

    def test_perfect_enrichment(self) -> None:
        """When phage feature + host feature perfectly predicts lysis."""
        # 4 phages, 4 hosts
        # Phage 0,1 have feature; phage 2,3 don't
        # Host 0,1 have feature; host 2,3 don't
        # Lysis only when both have the feature
        phage_matrix = np.array([[1], [1], [0], [0]], dtype=np.int8)
        host_matrix = np.array([[1], [1], [0], [0]], dtype=np.int8)
        # interaction_matrix[host, phage]
        interaction = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        results = compute_enrichment(phage_matrix, host_matrix, interaction, ["phage_f"], ["host_f"])
        assert len(results) == 1
        r = results[0]
        assert r.a_lysis_both == 4  # 2 hosts x 2 phages, all lysis
        assert r.n_both == 4
        assert r.p_value < 0.05
        assert r.odds_ratio == float("inf")

    def test_no_enrichment(self) -> None:
        """When features are independent of lysis."""
        rng = np.random.RandomState(42)
        n_phages, n_hosts = 50, 50
        phage_matrix = rng.randint(0, 2, size=(n_phages, 1)).astype(np.int8)
        host_matrix = rng.randint(0, 2, size=(n_hosts, 1)).astype(np.int8)
        # Random lysis at ~30% rate, independent of features
        interaction = (rng.random((n_hosts, n_phages)) < 0.3).astype(np.int8)

        results = compute_enrichment(phage_matrix, host_matrix, interaction, ["phage_f"], ["host_f"])
        assert len(results) == 1
        # Should NOT be significant (at least not consistently)
        # With random data, p-value should be > 0.05 most of the time
        # We can't assert this deterministically, but with seed 42 it should hold
        assert results[0].p_value > 0.01

    def test_multiple_features(self) -> None:
        """Test with multiple phage and host features."""
        phage_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
        host_matrix = np.array([[1, 0], [0, 1]], dtype=np.int8)
        interaction = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

        results = compute_enrichment(
            phage_matrix,
            host_matrix,
            interaction,
            ["pf1", "pf2"],
            ["hf1", "hf2"],
        )
        assert len(results) == 4  # 2 phage x 2 host features
        # Verify all feature pairs are covered
        pairs = {(r.phage_feature, r.host_feature) for r in results}
        assert pairs == {("pf1", "hf1"), ("pf1", "hf2"), ("pf2", "hf1"), ("pf2", "hf2")}

    def test_dimension_mismatch_raises(self) -> None:
        phage_matrix = np.array([[1], [0]], dtype=np.int8)
        host_matrix = np.array([[1], [0], [1]], dtype=np.int8)
        interaction = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)  # 2 hosts x 3 phages

        with pytest.raises(ValueError, match="phage_matrix rows"):
            compute_enrichment(phage_matrix, host_matrix, interaction, ["pf"], ["hf"])

    def test_contingency_table_sums(self) -> None:
        """Verify that contingency table cells sum to total interactions."""
        rng = np.random.RandomState(123)
        n_phages, n_hosts = 20, 30
        phage_matrix = rng.randint(0, 2, size=(n_phages, 2)).astype(np.int8)
        host_matrix = rng.randint(0, 2, size=(n_hosts, 2)).astype(np.int8)
        interaction = rng.randint(0, 2, size=(n_hosts, n_phages)).astype(np.int8)

        results = compute_enrichment(phage_matrix, host_matrix, interaction, ["pf1", "pf2"], ["hf1", "hf2"])

        total_pairs = n_hosts * n_phages
        total_lysis = int(interaction.sum())
        for r in results:
            assert r.n_both + r.n_phage_only + r.n_host_only + r.n_neither == total_pairs
            assert r.a_lysis_both + r.b_lysis_phage_only + r.c_lysis_host_only + r.d_lysis_neither == total_lysis


class TestResultsToRows:
    """Tests for CSV row conversion."""

    def test_basic_conversion(self) -> None:
        result = EnrichmentResult(
            phage_feature="pf1",
            host_feature="hf1",
            a_lysis_both=10,
            b_lysis_phage_only=5,
            c_lysis_host_only=3,
            d_lysis_neither=2,
            n_both=20,
            n_phage_only=30,
            n_host_only=25,
            n_neither=25,
            odds_ratio=2.5,
            p_value=0.01,
            bh_p_value=0.03,
        )
        rows = results_to_rows([result])
        assert len(rows) == 1
        row = rows[0]
        assert row["phage_feature"] == "pf1"
        assert row["host_feature"] == "hf1"
        assert row["lysis_rate_both"] == pytest.approx(0.5)
        assert row["significant"] is True

    def test_infinite_odds_ratio(self) -> None:
        result = EnrichmentResult(
            phage_feature="pf1",
            host_feature="hf1",
            a_lysis_both=5,
            b_lysis_phage_only=0,
            c_lysis_host_only=0,
            d_lysis_neither=0,
            n_both=5,
            n_phage_only=10,
            n_host_only=10,
            n_neither=10,
            odds_ratio=float("inf"),
            p_value=0.001,
            bh_p_value=0.005,
        )
        rows = results_to_rows([result])
        assert rows[0]["odds_ratio"] == "inf"
