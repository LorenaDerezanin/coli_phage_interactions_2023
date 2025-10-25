#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
IN_TSV = REPO_ROOT / "dev" / "cocktails" / "data" / "picard+test_collection_phylogenetic_distances.tsv"
OUT_DIR = REPO_ROOT / "dev" / "predictions" / "results" / "figures"


def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    # Symmetry guard (average with transpose if shapes match)
    if df.shape == df.T.shape:
        df = (df + df.T) / 2.0
    return df


def plot_heatmap(df: pd.DataFrame, out_png: Path, vmax: Optional[float] = None):
    n = df.shape[0]
    # Scale figure size to matrix size, but cap to keep reasonable
    side = max(8, min(24, n * 0.12))
    plt.figure(figsize=(side, side))
    ax = sns.heatmap(
        df,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "phylogenetic distance"},
        vmax=vmax,
    )
    ax.set_title("Bacterial phylogenetic distances (picard + test collection)")
    ax.set_xlabel("bacteria")
    ax.set_ylabel("bacteria")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_clustered(df: pd.DataFrame, out_png: Path, vmax: Optional[float] = None):
    # Seaborn clustermap for hierarchical ordering
    cg = sns.clustermap(
        df,
        cmap="viridis",
        figsize=(12, 12),
        cbar_kws={"label": "phylogenetic distance"},
        metric="euclidean",
        method="average",
        vmax=vmax,
    )
    cg.figure.suptitle("Clustered phylogenetic distances")
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    df = load_distance_matrix(IN_TSV)
    # Optional cap for color scale to reduce extreme influence
    vmax = np.nanpercentile(df.values, 99)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap(df, OUT_DIR / "phylo_distances_heatmap.png", vmax=vmax)
    plot_clustered(df, OUT_DIR / "phylo_distances_clustermap.png", vmax=vmax)
    print("Saved:")
    print(OUT_DIR / "phylo_distances_heatmap.png")
    print(OUT_DIR / "phylo_distances_clustermap.png")


if __name__ == "__main__":
    main()


