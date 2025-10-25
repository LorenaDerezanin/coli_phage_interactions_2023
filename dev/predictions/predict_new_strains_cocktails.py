#!/usr/bin/env python3
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
BACT_DIR = DATA_DIR / "genomics" / "bacteria"
PHAGE_DIR = DATA_DIR / "genomics" / "phages"
MODELS_DIR = REPO_ROOT / "dev" / "predictions" / "results" / "models"
TYPING_DIR = REPO_ROOT / "dev" / "predictions" / "new_strains" / "typing"
EXTRACTED_DIR = BACT_DIR / "new_strains" / "extracted_features"


def read_ectyper_outputs(typing_dir: Path) -> Dict[str, Dict[str, str]]:
    """Parse ectyper output.tsv files to extract O-type and H-type per genome.

    Returns mapping: new_bacteria_id -> {"O-type": str, "H-type": str}
    """
    typings: Dict[str, Dict[str, str]] = {}
    if not typing_dir.exists():
        return typings

    for sub in typing_dir.iterdir():
        if not sub.is_dir():
            continue
        out_tsv = sub / "output.tsv"
        if not out_tsv.exists():
            continue
        try:
            df = pd.read_csv(out_tsv, sep="\t")
        except Exception:
            # ectyper sometimes uses spaces; fallback robust read
            df = pd.read_csv(out_tsv, sep=None, engine="python")

        # Expect columns: Name, O-type, H-type, ...
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            otype = str(row.get("O-type", "")).strip()
            htype = str(row.get("H-type", "")).strip()
            typings[name] = {"O-type": otype, "H-type": htype}
    return typings


def read_mlst_st(typing_dir: Path) -> Dict[str, str]:
    """Parse mlst.txt files to extract ST per genome path.

    Returns mapping: new_bacteria_id -> ST string (e.g., '10')
    """
    id_to_st: Dict[str, str] = {}
    if not typing_dir.exists():
        return id_to_st
    for sub in typing_dir.iterdir():
        if not sub.is_dir():
            continue
        f = sub / "mlst.txt"
        if not f.exists():
            continue
        try:
            line = f.read_text().strip()
        except Exception:
            continue
        if not line:
            continue
        # Format: <path>\tscheme\tST\tadk(...)\t...
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        path, scheme, st = parts[0], parts[1], parts[2]
        # Derive bacteria id from ectyper naming: basename without extension
        name = Path(path).stem  # e.g., GCF_..._genomic
        id_to_st[name] = st
    return id_to_st


def load_umap_and_embeddings(repo_root: Path) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Load cohort UMAP means and full embeddings matrix.

    Returns (means_dict, embeddings_df) where embeddings_df is indexed by bacteria and
    contains columns UMAP0..UMAP7. If file missing, embeddings_df is empty and means are 0.
    """
    umap_file = repo_root / "dev" / "predictions" / "core_genome" / "UMAP_dim_reduction_from_phylogeny" / "data" / "coli_umap_8_dims.tsv"
    umap_cols = [f"UMAP{i}" for i in range(8)]
    if umap_file.exists():
        emb = pd.read_csv(umap_file, sep="\t").set_index("bacteria")
        means = {c: float(emb[c].mean()) if c in emb.columns else 0.0 for c in umap_cols}
        # As training standardized, we will still use 0.0 as neutral imputations during feature build,
        # but keep real means in case needed later.
        means = {c: 0.0 for c in means}
        return means, emb
    else:
        means = {c: 0.0 for c in umap_cols}
        return means, pd.DataFrame(columns=umap_cols)


def infer_umap_for_new_strain(
    embeddings: pd.DataFrame,
    host_feats: pd.DataFrame,
    new_st: str,
    new_otype: str,
) -> np.ndarray:
    """Heuristic UMAP for new strain by borrowing from nearest cohort strains.

    Priority: exact ST match (average UMAP), else exact O-type match (average), else zeros.
    """
    umap_cols = [c for c in embeddings.columns if c.startswith("UMAP")]
    if embeddings.empty or not umap_cols:
        return np.zeros(8, dtype=float)

    # Merge host_feats with embeddings to get ST and O-type per row with UMAP
    hf = host_feats.copy()
    # Ensure we have necessary columns
    if "ST_Warwick" not in hf.columns:
        hf["ST_Warwick"] = ""
    if "O-type" not in hf.columns:
        hf["O-type"] = ""
    merged = hf.join(embeddings, how="inner")

    # Try ST
    if new_st:
        st_matches = merged[merged["ST_Warwick"].astype(str) == str(new_st)]
        if not st_matches.empty:
            return st_matches[umap_cols].mean(axis=0).values.astype(float)

    # Try O-type
    if new_otype:
        o_matches = merged[merged["O-type"].astype(str) == str(new_otype)]
        if not o_matches.empty:
            return o_matches[umap_cols].mean(axis=0).values.astype(float)

    # Fallback zeros
    return np.zeros(len(umap_cols), dtype=float) if umap_cols else np.zeros(8, dtype=float)


def load_phage_metadata(phage_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(phage_dir / "guelin_collection.csv", sep=";")
    meta = meta.set_index("phage")
    return meta


def load_host_features(bact_dir: Path) -> pd.DataFrame:
    """Load training bacteria features to look up host O-type and ST."""
    picard = pd.read_csv(bact_dir / "picard_collection.csv", sep=";")
    picard = picard.set_index("bacteria")
    return picard


def choose_model_file(phage_model_dir: Path) -> Path:
    """Prefer RF_6 model, otherwise first .pickle."""
    pickles = [p for p in phage_model_dir.glob("*.pickle")]
    if not pickles:
        return None
    rf6 = [p for p in pickles if "RF_6" in p.name]
    return rf6[0] if rf6 else pickles[0]


def build_feature_row(
    feature_names: List[str],
    umap_vector: np.ndarray,
    new_otype: str,
    new_htype: str,
    new_st: str,
    host_row: pd.Series,
) -> np.ndarray:
    """Construct a feature vector aligned to feature_names.

    - UMAP* -> 0.0 (centered mean)
    - O-type_* / H-type_* / ST_Warwick_* -> 1 for observed category if present; else all zeros
    - same_* booleans computed when possible, else 0
    - other features default 0
    """
    x = np.zeros(len(feature_names), dtype=float)

    host_otype = str(host_row.get("O-type")) if host_row is not None else ""
    host_st = str(host_row.get("ST_Warwick")) if host_row is not None else ""

    # Precompute boolean features
    same_o = 1.0 if new_otype and host_otype and (new_otype == host_otype) else 0.0
    same_st = 1.0 if new_st and host_st and (str(new_st) == str(host_st)) else 0.0
    same_o_and_st = 1.0 if (same_o == 1.0 and same_st == 1.0) else 0.0

    # Determine available category vocabularies from model feature names
    available_o_types = [fn.replace("O-type_", "") for fn in feature_names if fn.startswith("O-type_")]
    available_h_types = [fn.replace("H-type_", "") for fn in feature_names if fn.startswith("H-type_")]
    available_st_types = [fn.replace("ST_Warwick_", "") for fn in feature_names if fn.startswith("ST_Warwick_")]

    # Choose mapped categories for this new strain
    # O-type: map to 'Other' if unseen and 'Other' column exists
    target_o = None
    if new_otype:
        if new_otype in available_o_types:
            target_o = new_otype
        elif "Other" in available_o_types:
            target_o = "Other"

    # H-type: use exact match only if present
    target_h = new_htype if (new_htype and (new_htype in available_h_types)) else None

    # ST: use exact match only if present
    target_st = str(new_st) if (new_st and (str(new_st) in available_st_types)) else None

    for i, fname in enumerate(feature_names):
        if fname.startswith("UMAP"):
            # Fill from inferred UMAP vector if available (aligned by suffix index)
            try:
                idx = int(fname.replace("UMAP", ""))
                if 0 <= idx < len(umap_vector):
                    x[i] = float(umap_vector[idx])
                else:
                    x[i] = 0.0
            except Exception:
                x[i] = 0.0
        elif fname == "same_O_as_host":
            x[i] = same_o
        elif fname == "same_ST_as_host":
            x[i] = same_st
        elif fname == "same_O_and_ST_as_host":
            x[i] = same_o_and_st
        elif fname.startswith("O-type_"):
            cat = fname.replace("O-type_", "")
            x[i] = 1.0 if (target_o is not None and target_o == cat) else 0.0
        elif fname.startswith("H-type_"):
            cat = fname.replace("H-type_", "")
            x[i] = 1.0 if (target_h is not None and target_h == cat) else 0.0
        elif fname.startswith("ST_Warwick_"):
            cat = fname.replace("ST_Warwick_", "")
            x[i] = 1.0 if (target_st is not None and target_st == cat) else 0.0
        else:
            # Unused/unknown feature -> 0.0
            x[i] = 0.0

    return x


def main():
    # Load inputs
    phage_meta = load_phage_metadata(PHAGE_DIR)
    host_feats = load_host_features(BACT_DIR)
    umap_means, embeddings = load_umap_and_embeddings(REPO_ROOT)

    # Discover new strains from ectyper outputs
    typings = read_ectyper_outputs(TYPING_DIR)
    if not typings:
        print(f"No ectyper outputs found in {TYPING_DIR}")
        sys.exit(1)

    # Read MLST STs if available
    new_strain_to_st: Dict[str, str] = read_mlst_st(TYPING_DIR)

    # Prepare predictions
    all_results: List[Dict[str, object]] = []

    # Iterate models per phage
    phage_dirs = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])
    if not phage_dirs:
        print(f"No phage model directories found in {MODELS_DIR}")
        sys.exit(1)

    for bacterium, th in typings.items():
        new_otype = th.get("O-type", "").strip()
        new_htype = th.get("H-type", "").strip()
        new_st = new_strain_to_st.get(bacterium, "")

        per_phage_scores: List[Tuple[str, float]] = []

        # Infer UMAP vector for this new strain
        umap_vec = infer_umap_for_new_strain(embeddings, host_feats, new_st, new_otype)

        for phage_dir in phage_dirs:
            phage = phage_dir.name
            model_file = choose_model_file(phage_dir)
            if model_file is None:
                continue
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"Failed loading model for {phage}: {e}")
                continue

            # Determine host row for this phage
            host_name = None
            if phage in phage_meta.index:
                host_name = phage_meta.loc[phage, "Phage_host"]
            host_row = host_feats.loc[host_name] if (host_name in host_feats.index) else None

            # Build aligned feature vector
            if not hasattr(model, "feature_names_in_"):
                # Fallback: skip if we can't align features
                continue
            feat_names = list(model.feature_names_in_)
            x = build_feature_row(
                feat_names,
                umap_vec,
                new_otype,
                new_htype,
                new_st,
                host_row,
            )

            try:
                proba = float(model.predict_proba(x.reshape(1, -1))[0, 1])
            except Exception as e:
                print(f"Prediction failed for {phage} on {bacterium}: {e}")
                proba = 0.0

            per_phage_scores.append((phage, proba))

        # Apply diversity constraints
        # Sort by probability
        per_phage_scores.sort(key=lambda t: t[1], reverse=True)

        selected: List[Tuple[str, float]] = []
        used_genera = set()
        used_hosts = set()
        for phage, score in per_phage_scores:
            if len(selected) >= 3:
                break
            genus = phage_meta.loc[phage, "Genus"] if phage in phage_meta.index else None
            host = phage_meta.loc[phage, "Phage_host"] if phage in phage_meta.index else None
            if (genus is None) or (host is None):
                continue
            if (genus not in used_genera) and (host not in used_hosts):
                selected.append((phage, score))
                used_genera.add(genus)
                used_hosts.add(host)

        # Fill if fewer than 3
        if len(selected) < 3:
            for phage, score in per_phage_scores:
                if len(selected) >= 3:
                    break
                if all(ph != phage for ph, _ in selected):
                    selected.append((phage, score))

        # Save results rows
        for rank, (phage, score) in enumerate(selected[:3], start=1):
            genus = phage_meta.loc[phage, "Genus"] if phage in phage_meta.index else "Unknown"
            all_results.append({
                "bacteria": bacterium,
                "phage": phage,
                "rank": rank,
                "probability": score,
                "genus": genus,
            })

    # Write output CSV
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXTRACTED_DIR / "cocktail_recommendations.csv"
    pd.DataFrame(all_results).to_csv(out_csv, sep=";", index=False)
    print(f"Saved cocktail recommendations to {out_csv}")


if __name__ == "__main__":
    main()


