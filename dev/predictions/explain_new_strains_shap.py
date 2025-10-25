#!/usr/bin/env python3
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import shap

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
BACT_DIR = DATA_DIR / "genomics" / "bacteria"
PHAGE_DIR = DATA_DIR / "genomics" / "phages"
MODELS_DIR = REPO_ROOT / "dev" / "predictions" / "results" / "models"
TYPING_DIR = REPO_ROOT / "dev" / "predictions" / "new_strains" / "typing"
EXTRACTED_DIR = BACT_DIR / "new_strains" / "extracted_features"

from predict_new_strains_cocktails import (
    read_ectyper_outputs,
    read_mlst_st,
    load_phage_metadata,
    load_host_features,
    load_umap_and_embeddings,
    infer_umap_for_new_strain,
    build_feature_row,
)


def choose_model_file(phage_model_dir: Path) -> Path:
    pickles = [p for p in phage_model_dir.glob("*.pickle")]
    if not pickles:
        return None
    rf6 = [p for p in pickles if "RF_6" in p.name]
    return rf6[0] if rf6 else pickles[0]


def main():
    # Load metadata and features
    phage_meta = load_phage_metadata(PHAGE_DIR)
    host_feats = load_host_features(BACT_DIR)
    umap_means, embeddings = load_umap_and_embeddings(REPO_ROOT)

    typings = read_ectyper_outputs(TYPING_DIR)
    if not typings:
        print(f"No ectyper outputs found in {TYPING_DIR}")
        return

    new_strain_to_st = read_mlst_st(TYPING_DIR)

    # Read recommendations to pick targets to explain
    rec_path = EXTRACTED_DIR / "cocktail_recommendations.csv"
    if not rec_path.exists():
        print(f"Missing recommendations file: {rec_path}")
        return
    rec = pd.read_csv(rec_path, sep=";")

    out_dir = REPO_ROOT / "dev" / "predictions" / "results" / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each bacterium and each recommended phage, compute SHAP values
    for bacteria in rec["bacteria"].unique():
        bdf = rec[rec["bacteria"] == bacteria].sort_values("rank")
        new_otype = typings.get(bacteria, {}).get("O-type", "")
        new_htype = typings.get(bacteria, {}).get("H-type", "")
        new_st = new_strain_to_st.get(bacteria, "")
        umap_vec = infer_umap_for_new_strain(embeddings, host_feats, new_st, new_otype)

        rows = []
        for _, r in bdf.iterrows():
            phage = r["phage"]
            ph_dir = MODELS_DIR / phage
            model_file = choose_model_file(ph_dir)
            if model_file is None:
                continue
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"Failed loading model for {phage}: {e}")
                continue
            if not hasattr(model, "feature_names_in_"):
                continue
            feat_names = list(model.feature_names_in_)
            host_name = phage_meta.loc[phage, "Phage_host"] if phage in phage_meta.index else None
            host_row = host_feats.loc[host_name] if (host_name in host_feats.index) else None

            x = build_feature_row(
                feat_names,
                umap_vec,
                new_otype,
                new_htype,
                new_st,
                host_row,
            )
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x.reshape(1, -1))
            # For binary classifiers, index 1 corresponds to positive class
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            contrib = pd.Series(sv[0], index=feat_names).sort_values(ascending=False)
            top_pos = contrib.head(15).reset_index()
            top_pos.columns = ["feature", "shap_value"]
            top_pos.insert(0, "bacteria", bacteria)
            top_pos.insert(1, "phage", phage)
            top_pos.insert(2, "rank", int(r["rank"]))
            rows.append(top_pos)

        if rows:
            df_out = pd.concat(rows, ignore_index=True)
            df_out.to_csv(out_dir / f"shap_top_features_{bacteria}.csv", sep=";", index=False)
            print(f"Wrote {out_dir / f'shap_top_features_{bacteria}.csv'}")


if __name__ == "__main__":
    main()



