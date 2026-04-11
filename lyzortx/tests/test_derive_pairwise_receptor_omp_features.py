"""Tests for derive_pairwise_receptor_omp_features (GT02)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lyzortx.pipeline.autoresearch.derive_pairwise_receptor_omp_features import (
    PAIRWISE_PREFIX,
    build_genus_receptor_mapping,
    compute_pairwise_receptor_omp_features,
    load_guelin_genus_mapping,
    load_table_s1_genus_receptors,
)


@pytest.fixture()
def table_s1_path(tmp_path: Path) -> Path:
    """Minimal Table S1 with 3 genera."""
    p = tmp_path / "Table_S1_Phages.tsv"
    p.write_text(
        "Phage\tFull name\tProduction host strain\tLifestyle\tGenome accesion\t"
        "Genome size (bp)\tMorphotype\tClass\tFamily\tSubfamily\tGenus\t"
        "BW25113 susceptibility\tBL21 susceptibility\tBW25113 receptor\t"
        "BW25113 LPS sugar\tBL21 receptor\tReceptor-binding protein\t"
        "LPS sugar-binding protein\tAF3 model\tPredictive models set\t"
        "Reference (doi)\tNotes\n"
        # Tequatrovirus: Tsx dominant
        "T6\tPhage T6\tMG1655\tlytic\tX\t168000\tMyo\tC\tStrabo\tTeven\tTequatrovirus\t"
        "Yes\tYes\tTsx\tKdo\tTsx\tCDS1\tCDS2\tsound\ttraining\t\t\n"
        "C16\tPhage C16\tC63\tlytic\tX\t167000\tMyo\tC\tStrabo\tTeven\tTequatrovirus\t"
        "Yes\tYes\tTsx\tHepI\tTsx\tCDS3\tCDS4\tsound\ttraining\t\t\n"
        "RB2\tPhage RB2\tC63\tlytic\tX\t166000\tMyo\tC\tStrabo\tTeven\tTequatrovirus\t"
        "Yes\tYes\tTsx\tKdo\tOmpA\tCDS5\tCDS6\tsound\ttraining\t\t\n"
        # Lambdavirus: LamB clean
        "Lambda\tPhage Lambda\tMG1655\tlytic\tX\t48000\tSipho\tC\tLambdoid\t\tLambdavirus\t"
        "Yes\tYes\tLamB\t\tLamB\tCDS7\t\tsound\ttraining\t\t\n"
        # Felixounavirus: LPS
        "FO1\tPhage FO1\tMG1655\tlytic\tX\t86000\tMyo\tC\tFelixouna\t\tFelixounavirus\t"
        "Yes\tNo\tLPS\t\tNot assayed\tCDS8\t\tsound\ttraining\t\t\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def guelin_path(tmp_path: Path) -> Path:
    """Minimal guelin_collection.csv with 4 phages."""
    p = tmp_path / "guelin_collection.csv"
    p.write_text(
        "phage;Morphotype;Family;Genus;Species;Genome_size;Phage_host;Phage_host_phylo;Old_Family;Subfamily;Old_Genus\n"
        "PHG_TSX;Myoviridae;Straboviridae;Tequatrovirus;sp;168000;H1;A;Strabo;Teven;Tequatro\n"
        "PHG_LAMB;Siphoviridae;Lambdoviridae;Lambdavirus;sp;48000;H1;A;Lambdoid;;Lambda\n"
        "PHG_LPS;Myoviridae;Felixounaviridae;Felixounavirus;sp;86000;H1;A;Felixouna;;Felix\n"
        "PHG_UNK;Myoviridae;Unknown;Unknownvirus;sp;50000;H1;A;Unk;;Unk\n",
        encoding="utf-8",
    )
    return p


def test_load_table_s1_genus_receptors(table_s1_path: Path) -> None:
    counts = load_table_s1_genus_receptors(table_s1_path)
    assert "Tequatrovirus" in counts
    # 3 phages × 2 receptor columns, but one BL21 is OmpA
    assert counts["Tequatrovirus"]["Tsx"] == 5  # 3 BW25113 + 2 BL21
    assert counts["Tequatrovirus"]["OmpA"] == 1
    assert counts["Lambdavirus"]["LamB"] == 2


def test_build_genus_receptor_mapping(table_s1_path: Path) -> None:
    counts = load_table_s1_genus_receptors(table_s1_path)
    mapping = build_genus_receptor_mapping(counts)

    # Tequatrovirus: Tsx dominant (5/6 = 83%)
    assert "Tequatrovirus" in mapping
    receptor, rtype, fraction = mapping["Tequatrovirus"]
    assert receptor == "Tsx"
    assert rtype == "omp"
    assert fraction > 0.6

    # Lambdavirus: LamB clean
    assert mapping["Lambdavirus"][0] == "LamB"
    assert mapping["Lambdavirus"][1] == "omp"

    # Felixounavirus: LPS
    assert mapping["Felixounavirus"][0] == "LPS"
    assert mapping["Felixounavirus"][1] == "lps"


def test_load_guelin_genus_mapping(guelin_path: Path) -> None:
    mapping = load_guelin_genus_mapping(guelin_path)
    assert mapping["PHG_TSX"] == "Tequatrovirus"
    assert mapping["PHG_LAMB"] == "Lambdavirus"
    assert len(mapping) == 4


def test_compute_features_directed_cross_terms(table_s1_path: Path, guelin_path: Path) -> None:
    """Verify directed cross-terms are computed correctly."""
    rows = []
    for phage in ["PHG_TSX", "PHG_LAMB", "PHG_LPS", "PHG_UNK"]:
        for host in ["H1", "H2"]:
            rows.append(
                {
                    "phage": phage,
                    "bacteria": host,
                    "host_surface__host_receptor_tsx_score": 800.0 if host == "H1" else 200.0,
                    "host_surface__host_receptor_lamB_score": 600.0 if host == "H2" else 100.0,
                    "host_surface__host_receptor_ompA_score": 300.0,
                    "host_surface__host_receptor_btub_score": 400.0,
                    "host_surface__host_receptor_fadL_score": 0.0,
                    "host_surface__host_receptor_fhua_score": 500.0,
                    "host_surface__host_receptor_lptD_score": 0.0,
                    "host_surface__host_receptor_nfrA_score": 0.0,
                    "host_surface__host_receptor_ompC_score": 700.0,
                    "host_surface__host_receptor_ompF_score": 0.0,
                    "host_surface__host_receptor_tolC_score": 0.0,
                    "host_surface__host_receptor_yncD_score": 0.0,
                    "host_surface__host_o_antigen_score": 150.0 if host == "H1" else 0.0,
                }
            )
    design = pd.DataFrame(rows)

    added = compute_pairwise_receptor_omp_features(design, table_s1_path=table_s1_path, guelin_path=guelin_path)

    assert all(col.startswith(PAIRWISE_PREFIX) for col in added)

    # PHG_TSX is predicted to target Tsx
    tsx_rows = design[design["phage"] == "PHG_TSX"]
    assert (tsx_rows[f"{PAIRWISE_PREFIX}predicted_is_tsx"] == 1.0).all()
    assert (tsx_rows[f"{PAIRWISE_PREFIX}predicted_is_lamB"] == 0.0).all()

    # Directed cross-term: PHG_TSX × H1 = 1.0 × 800.0 = 800.0
    tsx_h1 = design[(design["phage"] == "PHG_TSX") & (design["bacteria"] == "H1")]
    assert tsx_h1[f"{PAIRWISE_PREFIX}predicted_tsx_x_host_tsx"].iloc[0] == pytest.approx(800.0)

    # PHG_LAMB is predicted to target LamB
    lamb_rows = design[design["phage"] == "PHG_LAMB"]
    assert (lamb_rows[f"{PAIRWISE_PREFIX}predicted_is_lamB"] == 1.0).all()
    assert (lamb_rows[f"{PAIRWISE_PREFIX}predicted_is_tsx"] == 0.0).all()

    # Directed cross-term: PHG_LAMB × H2 = 1.0 × 600.0 = 600.0
    lamb_h2 = design[(design["phage"] == "PHG_LAMB") & (design["bacteria"] == "H2")]
    assert lamb_h2[f"{PAIRWISE_PREFIX}predicted_lamB_x_host_lamB"].iloc[0] == pytest.approx(600.0)

    # PHG_LPS is predicted to target LPS (not OMP)
    lps_rows = design[design["phage"] == "PHG_LPS"]
    assert (lps_rows[f"{PAIRWISE_PREFIX}predicted_is_lps"] == 1.0).all()
    assert (lps_rows[f"{PAIRWISE_PREFIX}predicted_is_tsx"] == 0.0).all()

    # LPS × O-antigen cross-term: PHG_LPS × H1 = 1.0 × 150.0 = 150.0
    lps_h1 = design[(design["phage"] == "PHG_LPS") & (design["bacteria"] == "H1")]
    assert lps_h1[f"{PAIRWISE_PREFIX}predicted_lps_x_host_o_antigen"].iloc[0] == pytest.approx(150.0)

    # PHG_UNK has no assignment
    unk_rows = design[design["phage"] == "PHG_UNK"]
    assert (unk_rows[f"{PAIRWISE_PREFIX}has_receptor_assignment"] == 0.0).all()


def test_unknown_phage_gets_zero_features(table_s1_path: Path, guelin_path: Path) -> None:
    """Phages with unknown genus get all-zero features."""
    design = pd.DataFrame(
        {
            "phage": ["PHG_UNK"],
            "bacteria": ["H1"],
            "host_surface__host_receptor_tsx_score": [800.0],
            "host_surface__host_receptor_lamB_score": [600.0],
            "host_surface__host_receptor_ompA_score": [300.0],
            "host_surface__host_receptor_btub_score": [400.0],
            "host_surface__host_receptor_fadL_score": [0.0],
            "host_surface__host_receptor_fhua_score": [500.0],
            "host_surface__host_receptor_lptD_score": [0.0],
            "host_surface__host_receptor_nfrA_score": [0.0],
            "host_surface__host_receptor_ompC_score": [700.0],
            "host_surface__host_receptor_ompF_score": [0.0],
            "host_surface__host_receptor_tolC_score": [0.0],
            "host_surface__host_receptor_yncD_score": [0.0],
            "host_surface__host_o_antigen_score": [150.0],
        }
    )
    compute_pairwise_receptor_omp_features(design, table_s1_path=table_s1_path, guelin_path=guelin_path)

    # All cross-terms should be 0 for unknown phage
    cross_cols = [col for col in design.columns if col.startswith(PAIRWISE_PREFIX) and "_x_" in col]
    for col in cross_cols:
        assert design[col].iloc[0] == pytest.approx(0.0), f"{col} should be 0 for unknown phage"
