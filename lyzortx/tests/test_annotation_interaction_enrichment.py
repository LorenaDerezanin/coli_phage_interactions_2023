"""Tests for the annotation-interaction enrichment module.

Test fixtures use a 20×10 slice of the real 369×96 interaction matrix
(strided sample covering resistant-to-susceptible bacteria and narrow-
to-broad phages, 30% lysis rate).
"""

from __future__ import annotations

import numpy as np
import pytest

from lyzortx.pipeline.track_l.steps.annotation_interaction_enrichment import (
    EnrichmentResult,
    benjamini_hochberg,
    compute_enrichment,
    results_to_rows,
)

# Use fewer permutations in tests for speed
TEST_N_PERMS = 200

# 20 bacteria x 10 phages slice from the real interaction matrix.
# Strided sample across susceptibility spectrum: row sums range 0-9,
# col sums range 0-14. Lysis rate 30%.
REAL_INTERACTION_SLICE = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
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
    """Tests for the main enrichment computation using real data slices."""

    def test_strong_interaction_detected(self) -> None:
        """Construct a clear interaction from the real slice.

        Phages 6-9 (broad range, col sums 8-14) have a feature.
        Hosts 14-19 (susceptible, row sums 3-9) have a feature.
        These pairs are mostly lytic in the real data — permutation
        test should detect the interaction.
        """
        interaction = REAL_INTERACTION_SLICE
        n_hosts, n_phages = interaction.shape
        phage_matrix = np.zeros((n_phages, 1), dtype=np.int8)
        phage_matrix[6:, 0] = 1  # phages 6-9
        host_matrix = np.zeros((n_hosts, 1), dtype=np.int8)
        host_matrix[14:, 0] = 1  # hosts 14-19

        results = compute_enrichment(
            phage_matrix, host_matrix, interaction, ["pf"], ["hf"], n_permutations=TEST_N_PERMS
        )
        assert len(results) == 1
        r = results[0]
        # Verify contingency counts are consistent
        assert r.n_both + r.n_phage_only + r.n_host_only + r.n_neither == n_hosts * n_phages
        total_lysis = int(interaction.sum())
        assert r.a_lysis_both + r.b_lysis_phage_only + r.c_lysis_host_only + r.d_lysis_neither == total_lysis
        # Among phages 6-9: hosts 14-19 should lyse more than hosts 0-13
        assert r.a_lysis_both > 0
        assert r.p_value < 0.10  # real signal should be detectable

    def test_random_features_not_significant(self) -> None:
        """Random binary features should not show enrichment on real data."""
        interaction = REAL_INTERACTION_SLICE
        n_hosts, n_phages = interaction.shape
        rng = np.random.RandomState(42)
        phage_matrix = rng.randint(0, 2, size=(n_phages, 1)).astype(np.int8)
        host_matrix = rng.randint(0, 2, size=(n_hosts, 1)).astype(np.int8)

        results = compute_enrichment(
            phage_matrix, host_matrix, interaction, ["pf"], ["hf"], n_permutations=TEST_N_PERMS
        )
        assert len(results) == 1
        assert results[0].p_value > 0.01

    def test_multiple_features(self) -> None:
        """Test with multiple phage and host features on real data."""
        interaction = REAL_INTERACTION_SLICE
        n_hosts, n_phages = interaction.shape
        phage_matrix = np.zeros((n_phages, 2), dtype=np.int8)
        phage_matrix[:5, 0] = 1
        phage_matrix[5:, 1] = 1
        host_matrix = np.zeros((n_hosts, 2), dtype=np.int8)
        host_matrix[:10, 0] = 1
        host_matrix[10:, 1] = 1

        results = compute_enrichment(
            phage_matrix,
            host_matrix,
            interaction,
            ["pf1", "pf2"],
            ["hf1", "hf2"],
            n_permutations=TEST_N_PERMS,
        )
        assert len(results) == 4
        pairs = {(r.phage_feature, r.host_feature) for r in results}
        assert pairs == {("pf1", "hf1"), ("pf1", "hf2"), ("pf2", "hf1"), ("pf2", "hf2")}

    def test_dimension_mismatch_raises(self) -> None:
        phage_matrix = np.array([[1], [0]], dtype=np.int8)
        host_matrix = np.array([[1], [0], [1]], dtype=np.int8)
        interaction = REAL_INTERACTION_SLICE[:3, :3]  # 3 hosts x 3 phages

        with pytest.raises(ValueError, match="phage_matrix rows"):
            compute_enrichment(phage_matrix, host_matrix, interaction, ["pf"], ["hf"], n_permutations=10)

    def test_contingency_table_sums(self) -> None:
        """Verify that contingency table cells sum to total interactions."""
        interaction = REAL_INTERACTION_SLICE
        n_hosts, n_phages = interaction.shape
        rng = np.random.RandomState(123)
        phage_matrix = rng.randint(0, 2, size=(n_phages, 2)).astype(np.int8)
        host_matrix = rng.randint(0, 2, size=(n_hosts, 2)).astype(np.int8)

        results = compute_enrichment(
            phage_matrix,
            host_matrix,
            interaction,
            ["pf1", "pf2"],
            ["hf1", "hf2"],
            n_permutations=TEST_N_PERMS,
        )

        total_pairs = n_hosts * n_phages
        total_lysis = int(interaction.sum())
        for r in results:
            assert r.n_both + r.n_phage_only + r.n_host_only + r.n_neither == total_pairs
            assert r.a_lysis_both + r.b_lysis_phage_only + r.c_lysis_host_only + r.d_lysis_neither == total_lysis

    def test_generalist_phrog_not_enriched(self) -> None:
        """Phages with a feature that lyse all hosts should not show enrichment.

        Use real interaction slice but set phage feature = 1 for phages 8,9
        (col sums 12,14 — broad range). These lyse most hosts regardless of
        host feature, so no specific host feature should be enriched.
        """
        interaction = REAL_INTERACTION_SLICE
        n_hosts, n_phages = interaction.shape
        phage_matrix = np.zeros((n_phages, 1), dtype=np.int8)
        phage_matrix[8:, 0] = 1  # phages 8,9 — broadest range
        # Host feature: first half
        host_matrix = np.zeros((n_hosts, 1), dtype=np.int8)
        host_matrix[:10, 0] = 1

        results = compute_enrichment(
            phage_matrix, host_matrix, interaction, ["pf"], ["hf"], n_permutations=TEST_N_PERMS
        )
        assert len(results) == 1
        # Broad-range phages lyse most hosts — host feature shouldn't matter much
        # Not asserting p > 0.5 because there could be real structure in this slice,
        # but the odds ratio should be moderate, not extreme
        assert results[0].odds_ratio < 10.0


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
