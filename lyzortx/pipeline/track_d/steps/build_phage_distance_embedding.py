#!/usr/bin/env python3
"""TD03: Build phage distance embedding features from the VIRIDIC phylogenetic tree."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.manifold import MDS

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages

PAIRWISE_DISTANCE_COLUMNS: Tuple[str, ...] = ("phage_1", "phage_2", "viridic_tree_distance")
METADATA_COLUMNS: Tuple[str, ...] = (
    "column_name",
    "feature_group",
    "feature_type",
    "source_path",
    "source_column",
    "transform",
    "provenance_note",
)


@dataclass
class NewickNode:
    """One parsed node from a Newick tree."""

    name: Optional[str]
    length_to_parent: Optional[float]
    children: List["NewickNode"] = field(default_factory=list)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Semicolon-delimited phage panel metadata containing the canonical phage names.",
    )
    parser.add_argument(
        "--tree-path",
        type=Path,
        default=Path("data/genomics/phages/tree/96_viridic_distance_phylogenetic_tree_algo=upgma.nwk"),
        help="Newick tree produced from VIRIDIC distances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_distance_embedding"),
        help="Directory for generated Track D VIRIDIC distance embedding artifacts.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=8,
        help="Requested metric MDS embedding dimension. Output is padded to this width if needed.",
    )
    parser.add_argument(
        "--expected-panel-count",
        type=int,
        default=96,
        help="Expected number of phages in the panel metadata.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for metric MDS.",
    )
    return parser.parse_args(argv)


def parse_newick_tree(text: str) -> NewickNode:
    """Parse a Newick tree into a recursive node structure."""

    stripped = text.strip()
    if not stripped.endswith(";"):
        raise ValueError("Newick tree must end with ';'")

    index = 0

    def skip_ws() -> None:
        nonlocal index
        while index < len(stripped) and stripped[index].isspace():
            index += 1

    def parse_name() -> Optional[str]:
        nonlocal index
        skip_ws()
        start = index
        while index < len(stripped) and stripped[index] not in ",():;":
            index += 1
        name = stripped[start:index].strip()
        return name or None

    def parse_length() -> Optional[float]:
        nonlocal index
        skip_ws()
        if index >= len(stripped) or stripped[index] != ":":
            return None
        index += 1
        skip_ws()
        start = index
        while index < len(stripped) and stripped[index] not in ",();":
            index += 1
        value = stripped[start:index].strip()
        if not value:
            raise ValueError("Encountered empty branch length in Newick tree")
        return float(value)

    def parse_subtree() -> NewickNode:
        nonlocal index
        skip_ws()
        if index >= len(stripped):
            raise ValueError("Unexpected end of Newick tree")

        if stripped[index] == "(":
            index += 1
            children = [parse_subtree()]
            skip_ws()
            while index < len(stripped) and stripped[index] == ",":
                index += 1
                children.append(parse_subtree())
                skip_ws()
            if index >= len(stripped) or stripped[index] != ")":
                raise ValueError("Expected ')' in Newick tree")
            index += 1
            name = parse_name()
            length = parse_length()
            return NewickNode(name=name, length_to_parent=length, children=children)

        name = parse_name()
        if name is None:
            raise ValueError("Leaf nodes in the Newick tree must have names")
        length = parse_length()
        return NewickNode(name=name, length_to_parent=length, children=[])

    root = parse_subtree()
    skip_ws()
    if index >= len(stripped) or stripped[index] != ";":
        raise ValueError("Expected ';' at the end of the Newick tree")
    skip_ws()
    if index != len(stripped) - 1:
        raise ValueError("Unexpected trailing content after the Newick tree")
    return root


def compute_pairwise_leaf_distances(root: NewickNode) -> Dict[str, Dict[str, float]]:
    """Compute all leaf-to-leaf patristic distances from a rooted tree."""

    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    leaf_names: Dict[str, int] = {}
    next_node_id = 0

    def attach(node: NewickNode, parent_id: Optional[int]) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        adjacency.setdefault(node_id, [])

        if parent_id is not None:
            if node.length_to_parent is None:
                raise ValueError("All non-root nodes must have a branch length")
            adjacency[node_id].append((parent_id, node.length_to_parent))
            adjacency[parent_id].append((node_id, node.length_to_parent))

        if node.children:
            for child in node.children:
                attach(child, node_id)
        else:
            if node.name is None:
                raise ValueError("Leaf nodes must have names")
            if node.name in leaf_names:
                raise ValueError(f"Duplicate leaf name in Newick tree: {node.name}")
            leaf_names[node.name] = node_id

        return node_id

    attach(root, parent_id=None)

    distances: Dict[str, Dict[str, float]] = {}
    for leaf_name, leaf_node_id in sorted(leaf_names.items()):
        pending: List[Tuple[int, int, float]] = [(leaf_node_id, -1, 0.0)]
        leaf_distances: Dict[str, float] = {}
        while pending:
            node_id, parent_id, total_distance = pending.pop()
            for candidate_name, candidate_node_id in leaf_names.items():
                if candidate_node_id == node_id:
                    leaf_distances[candidate_name] = total_distance
                    break
            for neighbor_id, edge_length in adjacency[node_id]:
                if neighbor_id == parent_id:
                    continue
                pending.append((neighbor_id, node_id, total_distance + edge_length))
        distances[leaf_name] = dict(sorted(leaf_distances.items()))

    return dict(sorted(distances.items()))


def distance_dict_to_matrix(
    pairwise_distances: Mapping[str, Mapping[str, float]],
    *,
    phage_order: Sequence[str],
) -> np.ndarray:
    missing = sorted(set(phage_order) - set(pairwise_distances))
    if missing:
        raise ValueError("Missing phages from pairwise distance map: " + ", ".join(missing))

    matrix = np.zeros((len(phage_order), len(phage_order)), dtype=np.float64)
    for row_index, phage_1 in enumerate(phage_order):
        row = pairwise_distances[phage_1]
        for column_index, phage_2 in enumerate(phage_order):
            matrix[row_index, column_index] = float(row[phage_2])
    return matrix


def compute_mds_embedding(
    distance_matrix: np.ndarray,
    *,
    embedding_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] == 0
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        raise ValueError("distance_matrix must be a non-empty square matrix")

    effective_dim = min(embedding_dim, max(1, distance_matrix.shape[0] - 1))
    mds = MDS(
        n_components=effective_dim,
        metric_mds=True,
        metric="precomputed",
        n_init=1,
        init="classical_mds",
        max_iter=300,
        eps=1e-9,
        normalized_stress="auto",
        random_state=random_state,
    )
    reduced = mds.fit_transform(distance_matrix)

    if effective_dim < embedding_dim:
        padded = np.zeros((distance_matrix.shape[0], embedding_dim), dtype=np.float64)
        padded[:, :effective_dim] = reduced
        reduced = padded

    metadata = {
        "requested_embedding_dim": embedding_dim,
        "effective_embedding_dim": effective_dim,
        "stress": float(mds.stress_),
    }
    return reduced, metadata


def _feature_columns(embedding_dim: int) -> Tuple[str, ...]:
    return tuple(f"phage_viridic_mds_{index:02d}" for index in range(embedding_dim))


def build_pairwise_distance_rows(
    *,
    phage_order: Sequence[str],
    distance_matrix: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_index, phage_1 in enumerate(phage_order):
        for column_index in range(row_index, len(phage_order)):
            rows.append(
                {
                    "phage_1": phage_1,
                    "phage_2": phage_order[column_index],
                    "viridic_tree_distance": round(float(distance_matrix[row_index, column_index]), 6),
                }
            )
    return rows


def build_feature_rows(
    *,
    phage_order: Sequence[str],
    embedding: np.ndarray,
    embedding_dim: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_index, phage in enumerate(phage_order):
        row: Dict[str, object] = {"phage": phage}
        for column_index, value in enumerate(embedding[row_index, :embedding_dim]):
            row[f"phage_viridic_mds_{column_index:02d}"] = round(float(value), 6)
        rows.append(row)
    return rows


def build_metadata_rows(*, tree_path: Path, embedding_dim: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for column_name in _feature_columns(embedding_dim):
        rows.append(
            {
                "column_name": column_name,
                "feature_group": "viridic_tree_embedding",
                "feature_type": "continuous",
                "source_path": str(tree_path),
                "source_column": "patristic distance matrix from leaf-to-leaf branch lengths",
                "transform": "Compute pairwise tree distances from the Newick tree and project them with metric MDS.",
                "provenance_note": "Rows are emitted only for canonical panel phages and preserve panel join keys.",
            }
        )
    return rows


def build_phage_distance_embedding_feature_block(
    *,
    panel_phages: Sequence[str],
    tree_path: Path,
    output_dir: Path,
    metadata_path: Path,
    embedding_dim: int,
    random_state: int = 0,
) -> Dict[str, object]:
    tree = parse_newick_tree(tree_path.read_text(encoding="utf-8"))
    pairwise_distances = compute_pairwise_leaf_distances(tree)

    missing = sorted(set(panel_phages) - set(pairwise_distances))
    extra = sorted(set(pairwise_distances) - set(panel_phages))
    if missing:
        raise ValueError("Panel phages missing from VIRIDIC tree: " + ", ".join(missing))
    if extra:
        raise ValueError("VIRIDIC tree contains non-panel leaves: " + ", ".join(extra))

    phage_order = sorted(panel_phages)
    distance_matrix = distance_dict_to_matrix(pairwise_distances, phage_order=phage_order)
    embedding, mds_metadata = compute_mds_embedding(
        distance_matrix,
        embedding_dim=embedding_dim,
        random_state=random_state,
    )

    feature_rows = build_feature_rows(
        phage_order=phage_order,
        embedding=embedding,
        embedding_dim=embedding_dim,
    )
    pairwise_rows = build_pairwise_distance_rows(
        phage_order=phage_order,
        distance_matrix=distance_matrix,
    )
    metadata_rows = build_metadata_rows(tree_path=tree_path, embedding_dim=embedding_dim)

    ensure_directory(output_dir)
    feature_columns = _feature_columns(embedding_dim)
    feature_fieldnames = ("phage", *feature_columns)
    features_path = output_dir / "phage_distance_embedding_features.csv"
    pairwise_path = output_dir / "phage_viridic_tree_pairwise_distances.csv"
    metadata_csv_path = output_dir / "phage_distance_embedding_feature_metadata.csv"
    write_csv(features_path, feature_fieldnames, feature_rows)
    write_csv(pairwise_path, PAIRWISE_DISTANCE_COLUMNS, pairwise_rows)
    write_csv(metadata_csv_path, METADATA_COLUMNS, metadata_rows)

    manifest = {
        "step_name": "build_phage_distance_embedding",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "phage_metadata_path": str(metadata_path),
            "tree_path": str(tree_path),
            "embedding_dim_requested": embedding_dim,
            "random_state": random_state,
        },
        "counts": {
            "panel_phage_count": len(panel_phages),
            "tree_leaf_count": len(pairwise_distances),
            "output_row_count": len(feature_rows),
            "pairwise_distance_row_count": len(pairwise_rows),
            "embedding_dim_effective": int(mds_metadata["effective_embedding_dim"]),
        },
        "output_format": {
            "feature_csv": str(features_path),
            "pairwise_distance_csv": str(pairwise_path),
            "feature_metadata_csv": str(metadata_csv_path),
            "feature_columns": list(feature_fieldnames),
        },
        "reproducibility": {
            "one_command": "python lyzortx/pipeline/track_d/run_track_d.py --step viridic-distance",
            "mds_stress": round(float(mds_metadata["stress"]), 6),
            "distance_definition": (
                "Pairwise distances are the summed branch lengths between phage leaves in the VIRIDIC UPGMA tree."
            ),
            "panel_output_policy": "Require exact leaf-name agreement with the canonical panel before writing outputs.",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    panel_phages = read_panel_phages(args.phage_metadata_path, expected_panel_count=args.expected_panel_count)
    manifest = build_phage_distance_embedding_feature_block(
        panel_phages=panel_phages,
        tree_path=args.tree_path,
        output_dir=args.output_dir,
        metadata_path=args.phage_metadata_path,
        embedding_dim=args.embedding_dim,
        random_state=args.random_state,
    )
    print("Built phage VIRIDIC distance embedding features.")
    print(f"- Panel rows written: {manifest['counts']['output_row_count']}")
    print(f"- Tree leaves used: {manifest['counts']['tree_leaf_count']}")
    print(f"- Pairwise distance rows written: {manifest['counts']['pairwise_distance_row_count']}")


if __name__ == "__main__":
    main()
