import pandas as pd
import pytest

from lyzortx.research_notes.ad_hoc_analysis_code.dilution_response_patterns import (
    add_bh_q_values,
    build_bacterial_subgroup_dilution_response_summary,
    build_pair_response_features,
    build_phage_dilution_response_summary,
    prepare_host_metadata,
)


def test_build_pair_response_features_counts_positive_dilutions_per_pair() -> None:
    pair_dilution_summary = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B1__P1", "B1__P1", "B2__P2", "B2__P2"],
            "log_dilution": [0, -1, -2, 0, -2],
            "score_1_count": [1, 1, 0, 0, 1],
        }
    )

    out = build_pair_response_features(pair_dilution_summary)

    b1 = out.loc[out["pair_id"] == "B1__P1"].iloc[0]
    b2 = out.loc[out["pair_id"] == "B2__P2"].iloc[0]

    assert b1["n_positive_dilutions"] == 2
    assert b1["best_positive_dilution"] == -1
    assert b1["has_multidilution_support"] == True
    assert b2["n_positive_dilutions"] == 1
    assert b2["positive_dilution_list"] == "-2"


def test_add_bh_q_values_only_updates_tested_rows() -> None:
    summary = pd.DataFrame(
        {
            "group": ["A", "B", "C"],
            "tested": [True, False, True],
            "p_value": [0.01, None, 0.04],
        }
    )

    out = add_bh_q_values(summary, summary["tested"], "p_value", "q_value")

    assert out.loc[out["group"] == "A", "q_value"].item() == pytest.approx(0.02)
    assert pd.isna(out.loc[out["group"] == "B", "q_value"]).item()
    assert out.loc[out["group"] == "C", "q_value"].item() == pytest.approx(0.04)


def test_build_phage_dilution_response_summary_combines_metadata_and_response_metrics() -> None:
    pair_labels = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B2__P1", "B1__P2", "B2__P2"],
            "bacteria": ["B1", "B2", "B1", "B2"],
            "phage": ["P1", "P1", "P2", "P2"],
            "any_lysis": [1.0, 1.0, 1.0, 0.0],
            "best_lysis_dilution": [-2, 0, -1, pd.NA],
            "dilution_potency_rank": [3.0, 1.0, 2.0, pd.NA],
        }
    )
    pair_response_features = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B2__P1", "B1__P2"],
            "n_positive_dilutions": [2, 1, 3],
            "max_positive_dilution": [0, 0, 0],
            "best_positive_dilution": [-2, 0, -1],
            "has_multidilution_support": [True, False, True],
            "positive_dilution_list": ["-2,0", "0", "-1,0,-2"],
        }
    )
    phage_metadata = pd.DataFrame(
        {
            "phage": ["P1", "P2"],
            "Morphotype": ["Myoviridae", "Podoviridae"],
            "Family": ["Straboviridae", "Autographiviridae"],
        }
    )

    out = build_phage_dilution_response_summary(
        pair_labels=pair_labels,
        pair_response_features=pair_response_features,
        phage_metadata=phage_metadata,
        min_lytic_pairs_for_test=2,
    )

    p1 = out.loc[out["phage"] == "P1"].iloc[0]
    p2 = out.loc[out["phage"] == "P2"].iloc[0]

    assert p1["morphotype"] == "Myoviridae"
    assert p1["n_resolved_pairs"] == 2
    assert p1["n_lytic_pairs"] == 2
    assert p1["high_potency_rate"] == 0.5
    assert p1["multi_dilution_support_rate"] == 0.5
    assert p2["n_resolved_pairs"] == 2
    assert p2["n_lytic_pairs"] == 1
    assert p2["tested_for_high_potency_enrichment"] == False


def test_build_bacterial_subgroup_dilution_response_summary_reports_field_names() -> None:
    pair_labels = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B1__P2", "B2__P1", "B2__P2", "B3__P1", "B3__P2"],
            "bacteria": ["B1", "B1", "B2", "B2", "B3", "B3"],
            "phage": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "any_lysis": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "best_lysis_dilution": [-2, pd.NA, 0, -1, pd.NA, -4],
            "dilution_potency_rank": [3.0, pd.NA, 1.0, 2.0, pd.NA, 4.0],
        }
    )
    pair_response_features = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B2__P1", "B2__P2", "B3__P2"],
            "n_positive_dilutions": [2, 1, 2, 3],
            "max_positive_dilution": [0, 0, 0, 0],
            "best_positive_dilution": [-2, 0, -1, -4],
            "has_multidilution_support": [True, False, True, True],
            "positive_dilution_list": ["-2,0", "0", "-1,0", "-4,-2,0"],
        }
    )
    host_metadata = prepare_host_metadata(
        pd.DataFrame(
            {
                "bacteria": ["B1", "B2", "B3"],
                "Clermont_Phylo": ["A", "A", "B2"],
                "ST_Warwick": ["10", "10", "58"],
                "O-type": ["O1", "O1", "O2"],
                "H-type": ["H1", "H1", "H2"],
            }
        )
    )

    out = build_bacterial_subgroup_dilution_response_summary(
        pair_labels=pair_labels,
        pair_response_features=pair_response_features,
        host_metadata=host_metadata,
        min_lytic_pairs_for_test=2,
    )

    phylogroup_a = out[(out["field_name"] == "phylogroup") & (out["subgroup_value"] == "A")].iloc[0]
    st_10 = out[(out["field_name"] == "ST") & (out["subgroup_value"] == "10")].iloc[0]
    serotype_o1h1 = out[(out["field_name"] == "serotype") & (out["subgroup_value"] == "O1:H1")].iloc[0]

    assert phylogroup_a["n_resolved_pairs"] == 4
    assert phylogroup_a["n_lytic_pairs"] == 3
    assert phylogroup_a["high_potency_rate"] == pytest.approx(1 / 3)
    assert st_10["multi_dilution_support_rate"] == pytest.approx(2 / 3)
    assert serotype_o1h1["tested_for_high_potency_enrichment"] == True
