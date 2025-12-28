"""
Build GroupKFold cluster assignments ("cv_clusters") from a pairwise core-genome distance matrix.

The README describes the rule used in the paper:
  - strains with a core genome distance < 1e-4 substitutions/site must be placed in the same CV fold

This script implements that rule as connected components in a graph:
  - create an undirected edge between i,j when distance(i,j) <= threshold
  - each connected component is a "group" label for GroupKFold

Output format matches what `dev/predictions/predict_all_phages.py` expects:
  - CSV with columns: bacteria;group

It relies on pandas/numpy/scipy (all already required by the main pipeline) so the implementation
stays small and readable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-matrix", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path (sep=';')")
    args = parser.parse_args()

    dist = pd.read_csv(args.distance_matrix, sep="\t").set_index("bacteria")
    dist = dist.loc[dist.index, dist.index]

    arr = dist.to_numpy(copy=False)
    adj = (arr <= args.threshold).astype(np.uint8)
    np.fill_diagonal(adj, 1)

    _, labels = connected_components(csr_matrix(adj), directed=False, return_labels=True)
    groups = pd.DataFrame({"bacteria": dist.index, "group": labels})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.out, sep=";", index=False)
    print(
        f"Wrote {args.out} with {groups.shape[0]} bacteria across {groups['group'].nunique()} "
        f"groups (threshold={args.threshold:g})."
    )


if __name__ == "__main__":
    main()
