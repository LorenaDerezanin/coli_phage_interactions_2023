#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPL_DIR = REPO_ROOT / "dev" / "predictions" / "results" / "explanations"


def beautify_axes(ax, title: str):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("SHAP value (positive → increases infection probability)")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=False, bottom=False)


def plot_one(df: pd.DataFrame, out_path: Path, top_n: int = 12):
    # Keep only positive contributions and take top_n
    d = df.copy()
    d = d[d["shap_value"] > 0]
    d = d.sort_values("shap_value", ascending=True).tail(top_n)

    # Nice feature labels
    d["label"] = d["feature"].str.replace("_", " ")

    plt.figure(figsize=(8, max(3, 0.4 * len(d))))
    ax = sns.barplot(
        data=d,
        x="shap_value",
        y="label",
        color="#4C78A8"
    )
    title = f"{d['bacteria'].iloc[0]} — {d['phage'].iloc[0]} (rank {int(d['rank'].iloc[0])})"
    beautify_axes(ax, title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    csvs = sorted(EXPL_DIR.glob("shap_top_features_*.csv"))
    if not csvs:
        print(f"No SHAP CSVs found in {EXPL_DIR}")
        return

    for csv in csvs:
        df = pd.read_csv(csv, sep=";")
        # One plot per phage recommendation within the bacterium
        for (bact, phage, rank), part in df.groupby(["bacteria", "phage", "rank"]):
            safe_bact = str(bact).replace(":", "_")
            out_png = EXPL_DIR / f"plot_shap_{safe_bact}_{phage}_rank{int(rank)}.png"
            plot_one(part, out_png)
            print(f"Saved {out_png}")


if __name__ == "__main__":
    main()





