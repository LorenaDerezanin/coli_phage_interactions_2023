"""Tests for the TL02 enrichment runner."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_l.steps import run_enrichment_analysis


def test_main_excludes_holdout_bacteria_before_enrichment(monkeypatch, tmp_path: Path) -> None:
    label_path = tmp_path / "label_set_v1_pairs.csv"
    split_path = tmp_path / "st03_split_assignments.csv"
    output_dir = tmp_path / "out"

    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bacteria", "phage", "any_lysis"])
        writer.writeheader()
        writer.writerow({"bacteria": "B1", "phage": "P1", "any_lysis": "1"})
        writer.writerow({"bacteria": "B1", "phage": "P2", "any_lysis": "0"})
        writer.writerow({"bacteria": "B2", "phage": "P1", "any_lysis": "0"})
        writer.writerow({"bacteria": "B2", "phage": "P2", "any_lysis": "1"})

    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "bacteria", "split_holdout"])
        writer.writeheader()
        writer.writerow({"pair_id": "B1__P1", "bacteria": "B1", "split_holdout": "train_non_holdout"})
        writer.writerow({"pair_id": "B1__P2", "bacteria": "B1", "split_holdout": "train_non_holdout"})
        writer.writerow({"pair_id": "B2__P1", "bacteria": "B2", "split_holdout": "holdout_test"})
        writer.writerow({"pair_id": "B2__P2", "bacteria": "B2", "split_holdout": "holdout_test"})

    recorded_bacteria: list[list[str]] = []
    captured_calls: list[dict[str, object]] = []

    def fake_load_pharokka_phrog_matrices(cached_dir: Path, phages: list[str]):
        assert phages == ["P1", "P2"]
        return (
            np.array([[1], [0]], dtype=np.int8),
            ["RBP_PHROG_1"],
            np.array([[0], [1]], dtype=np.int8),
            ["ANTIDEF_PHROG_1"],
        )

    def fake_host_loader(*args: object, **kwargs: object):
        bacteria = args[-1]
        assert isinstance(bacteria, list)
        recorded_bacteria.append(bacteria)
        return np.zeros((len(bacteria), 1), dtype=np.int8), ["host_feature"]

    def fake_compute_enrichment(
        phage_matrix: np.ndarray,
        host_matrix: np.ndarray,
        interaction_matrix: np.ndarray,
        phage_feature_names: list[str],
        host_feature_names: list[str],
        n_permutations: int = 0,
        random_seed: int = 0,
        resolved_mask: np.ndarray | None = None,
    ) -> list[object]:
        captured_calls.append(
            {
                "phage_matrix": phage_matrix.copy(),
                "host_matrix": host_matrix.copy(),
                "interaction_matrix": interaction_matrix.copy(),
                "phage_feature_names": list(phage_feature_names),
                "host_feature_names": list(host_feature_names),
                "resolved_mask": None if resolved_mask is None else resolved_mask.copy(),
            }
        )
        return []

    monkeypatch.setattr(run_enrichment_analysis, "load_pharokka_phrog_matrices", fake_load_pharokka_phrog_matrices)
    monkeypatch.setattr(run_enrichment_analysis, "load_omp_receptor_host_matrix", fake_host_loader)
    monkeypatch.setattr(run_enrichment_analysis, "load_lps_host_matrix", fake_host_loader)
    monkeypatch.setattr(run_enrichment_analysis, "load_defense_host_matrix", fake_host_loader)
    monkeypatch.setattr(run_enrichment_analysis, "compute_enrichment", fake_compute_enrichment)

    run_enrichment_analysis.main(
        [
            "--label-path",
            str(label_path),
            "--st03-split-assignments-path",
            str(split_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert recorded_bacteria == [["B1"], ["B1"], ["B1"]]
    assert len(captured_calls) == 3
    for call in captured_calls:
        assert call["interaction_matrix"].shape == (1, 2)
        assert call["host_matrix"].shape == (1, 1)
        assert call["resolved_mask"].shape == (1, 2)
        np.testing.assert_array_equal(call["interaction_matrix"], np.array([[1, 0]], dtype=np.int8))
        np.testing.assert_array_equal(call["resolved_mask"], np.ones((1, 2), dtype=np.int8))
