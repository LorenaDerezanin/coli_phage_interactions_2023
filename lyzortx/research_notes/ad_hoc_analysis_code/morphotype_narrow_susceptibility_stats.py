#!/usr/bin/env python3
"""
One-off analysis for:
1) Morphotype-level lytic breadth comparisons.
2) Bacteria with zero lysis.
3) Bacteria with narrow susceptibility (1-3 lytic phages).
4) "Unique rescuer" phages (single-lyser cases).

This script reproduces the numbers summarized in:
`lyzortx/research_notes/GIST Prediction Ecoli nature paper.md`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    matrix_path = root / "data" / "interactions" / "interaction_matrix.csv"
    phage_meta_path = root / "data" / "genomics" / "phages" / "guelin_collection.csv"
    out_dir = root / "lyzortx" / "generated_outputs" / "raw_interactions_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = pd.read_csv(matrix_path, sep=";").set_index("bacteria")
    phage_meta = pd.read_csv(phage_meta_path, sep=";")
    bin_mat = (mat > 0).astype(int)

    phage_to_morph = dict(zip(phage_meta["phage"], phage_meta["Morphotype"]))

    # Morphotype lytic breadth stats at phage level.
    phage_lysis_counts = (bin_mat.sum(axis=0).rename("n_bacteria_lysed").to_frame())
    phage_lysis_counts["morphotype"] = phage_lysis_counts.index.map(phage_to_morph)

    groups = {
        morph: phage_lysis_counts.loc[phage_lysis_counts["morphotype"] == morph, "n_bacteria_lysed"]
        for morph in sorted(phage_lysis_counts["morphotype"].dropna().unique())
    }
    omnibus = kruskal(*groups.values())

    pairwise = []
    for a, b in [("Myoviridae", "Podoviridae"), ("Myoviridae", "Siphoviridae"), ("Podoviridae", "Siphoviridae")]:
        u = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        pairwise.append(
            {
                "comparison": f"{a} vs {b}",
                "U": float(u.statistic),
                "p_value": float(u.pvalue),
                "median_a": float(groups[a].median()),
                "median_b": float(groups[b].median()),
                "mean_a": float(groups[a].mean()),
                "mean_b": float(groups[b].mean()),
            }
        )
    pairwise_df = pd.DataFrame(pairwise)

    # Bacteria-level susceptibility summaries.
    lysis_counts = bin_mat.sum(axis=1).sort_values()
    none_lysed = lysis_counts[lysis_counts == 0]
    low_lysed = lysis_counts[(lysis_counts > 0) & (lysis_counts <= 3)]

    none_df = none_lysed.reset_index()
    none_df.columns = ["bacteria", "n_lytic_phages"]
    none_df.to_csv(out_dir / "bacteria_none_lysed.csv", index=False)

    low_rows = []
    for bact, n_lytic in low_lysed.items():
        lytic_phages = bin_mat.columns[bin_mat.loc[bact] == 1].tolist()
        lytic_morph = [phage_to_morph.get(ph, "NA") for ph in lytic_phages]
        low_rows.append(
            {
                "bacteria": bact,
                "n_lytic_phages": int(n_lytic),
                "lytic_phages": ",".join(lytic_phages),
                "lytic_morphotypes": ",".join(lytic_morph),
            }
        )
    low_df = pd.DataFrame(low_rows).sort_values(["n_lytic_phages", "bacteria"])
    low_df.to_csv(out_dir / "bacteria_low_lysed_1_to_3.csv", index=False)

    # Single-lyser bacteria and "unique rescuer" phages.
    single = lysis_counts[lysis_counts == 1].index.tolist()
    single_df = pd.DataFrame({"bacteria": single})
    single_df["unique_lyser"] = [bin_mat.columns[bin_mat.loc[b] == 1][0] for b in single]
    single_df["morphotype"] = single_df["unique_lyser"].map(phage_to_morph)
    single_df.to_csv(out_dir / "bacteria_lysed_by_exactly_one_phage.csv", index=False)

    unique_lyser_counts = (
        single_df["unique_lyser"]
        .value_counts()
        .rename_axis("phage")
        .reset_index(name="n_bacteria_lysed_exclusively")
    )
    unique_lyser_counts["morphotype"] = unique_lyser_counts["phage"].map(phage_to_morph)
    unique_lyser_counts.to_csv(out_dir / "phage_counts_for_exactly_one_phage_bacteria.csv", index=False)

    # Print summary.
    print("=== Morphotype breadth ===")
    print(f"Kruskal-Wallis H={omnibus.statistic:.6f}, p={omnibus.pvalue:.12g}")
    print(pairwise_df.to_string(index=False))
    print(
        "\nMedians by morphotype:",
        phage_lysis_counts.groupby("morphotype")["n_bacteria_lysed"].median().to_dict(),
    )

    print("\n=== Zero and narrow susceptibility bacteria ===")
    print(f"bacteria_total={bin_mat.shape[0]}")
    print(f"none_lysed_n={len(none_lysed)} ({len(none_lysed) / bin_mat.shape[0] * 100:.2f}%)")
    print(f"low_lysed_1_to3_n={len(low_lysed)} ({len(low_lysed) / bin_mat.shape[0] * 100:.2f}%)")
    print("none_lysed_bacteria=", ", ".join(none_lysed.index.tolist()))

    print("\n=== Unique rescuer phages ===")
    print(f"single_lyser_bacteria_n={len(single_df)}")
    print("single_lyser_morphotype_counts=", single_df["morphotype"].value_counts().to_dict())
    print(unique_lyser_counts.head(20).to_string(index=False))

    # Also save stats table for reuse in notes.
    pairwise_df.to_csv(out_dir / "morphotype_pairwise_stats.csv", index=False)
    phage_lysis_counts.to_csv(out_dir / "phage_lysis_counts_by_morphotype.csv")


if __name__ == "__main__":
    main()
