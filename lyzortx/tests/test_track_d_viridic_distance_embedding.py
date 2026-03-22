import csv
import json
from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_d import run_track_d
from lyzortx.pipeline.track_d.steps.build_phage_distance_embedding import (
    build_phage_distance_embedding_feature_block,
    compute_mds_embedding,
    compute_pairwise_leaf_distances,
    distance_dict_to_matrix,
    parse_newick_tree,
)
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages


def _write_panel_metadata(path: Path, phages: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phage"], delimiter=";")
        writer.writeheader()
        for phage in phages:
            writer.writerow({"phage": phage})


def test_compute_pairwise_leaf_distances_from_newick_tree() -> None:
    tree = parse_newick_tree("((P1:0.1,P2:0.2)inner:0.3,P3:0.4);")

    distances = compute_pairwise_leaf_distances(tree)

    assert distances["P1"]["P1"] == 0.0
    assert np.isclose(distances["P1"]["P2"], 0.3)
    assert np.isclose(distances["P1"]["P3"], 0.8)
    assert np.isclose(distances["P2"]["P3"], 0.9)


def test_compute_mds_embedding_pads_when_requested_dimension_exceeds_rank() -> None:
    distance_matrix = np.array(
        [
            [0.0, 0.3, 0.8],
            [0.3, 0.0, 0.9],
            [0.8, 0.9, 0.0],
        ]
    )

    embedding, metadata = compute_mds_embedding(distance_matrix, embedding_dim=5, random_state=0)

    assert embedding.shape == (3, 5)
    assert metadata["effective_embedding_dim"] == 2
    assert np.allclose(embedding[:, 2:], 0.0)


def test_build_phage_distance_embedding_feature_block_writes_joinable_csvs(tmp_path: Path) -> None:
    metadata_path = tmp_path / "panel.csv"
    tree_path = tmp_path / "tree.nwk"
    output_dir = tmp_path / "out"

    _write_panel_metadata(metadata_path, ["P1", "P2", "P3"])
    tree_path.write_text("((P1:0.1,P2:0.2)inner:0.3,P3:0.4);", encoding="utf-8")

    panel_phages = read_panel_phages(metadata_path, expected_panel_count=3)
    manifest = build_phage_distance_embedding_feature_block(
        panel_phages=panel_phages,
        tree_path=tree_path,
        output_dir=output_dir,
        metadata_path=metadata_path,
        embedding_dim=5,
    )

    features_path = output_dir / "phage_distance_embedding_features.csv"
    pairwise_path = output_dir / "phage_viridic_tree_pairwise_distances.csv"
    metadata_csv_path = output_dir / "phage_distance_embedding_feature_metadata.csv"
    manifest_path = output_dir / "manifest.json"

    rows = list(csv.DictReader(features_path.open(encoding="utf-8")))
    pairwise_rows = list(csv.DictReader(pairwise_path.open(encoding="utf-8")))
    metadata_rows = list(csv.DictReader(metadata_csv_path.open(encoding="utf-8")))
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [row["phage"] for row in rows] == ["P1", "P2", "P3"]
    assert len(rows[0]) == 1 + 5
    assert len(pairwise_rows) == 6
    assert pairwise_rows[0] == {"phage_1": "P1", "phage_2": "P1", "viridic_tree_distance": "0.0"}
    assert len(metadata_rows) == 5
    assert manifest["counts"]["tree_leaf_count"] == 3
    assert manifest["counts"]["embedding_dim_effective"] == 2
    assert manifest_json["counts"]["pairwise_distance_row_count"] == 6


def test_distance_dict_to_matrix_respects_requested_order() -> None:
    distances = {
        "P1": {"P1": 0.0, "P2": 0.1},
        "P2": {"P1": 0.1, "P2": 0.0},
    }

    matrix = distance_dict_to_matrix(distances, phage_order=["P2", "P1"])

    assert np.allclose(matrix, np.array([[0.0, 0.1], [0.1, 0.0]]))


def test_run_track_d_dispatches_viridic_distance_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_d.build_phage_protein_sets,
        "main",
        lambda argv: calls.append("protein-sets"),
    )
    monkeypatch.setattr(
        run_track_d.build_phage_genome_kmer_features,
        "main",
        lambda argv: calls.append("genome-kmers"),
    )
    monkeypatch.setattr(
        run_track_d.build_phage_distance_embedding,
        "main",
        lambda argv: calls.append("viridic-distance"),
    )

    run_track_d.main(["--step", "viridic-distance"])
    assert calls == ["viridic-distance"]

    calls.clear()
    run_track_d.main(["--step", "all"])
    assert calls == ["protein-sets", "genome-kmers", "viridic-distance"]
