import argparse

import pandas as pd
import pytest

from lyzortx.research_notes.ad_hoc_analysis_code.dilution_response_patterns import (
    DEFAULT_PAIR_DILUTION_SUMMARY_PATH,
    DEFAULT_PAIR_LABELS_PATH,
    TRACK_A_BUILD_COMMAND,
    _aggregate_lytic_pair_metrics,
    add_bh_q_values,
    build_bacterial_subgroup_dilution_response_summary,
    build_pair_response_features,
    build_phage_dilution_response_summary,
    prepare_host_metadata,
    top_rows,
    validate_required_inputs,
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


def test_aggregate_lytic_pair_metrics_matches_both_callers() -> None:
    lytic_pairs = pd.DataFrame(
        {
            "pair_id": ["B1__P1", "B2__P1", "B1__P2"],
            "group": ["G1", "G1", "G2"],
            "is_high_potency": [True, False, True],
            "has_multidilution_support": [True, True, False],
            "dilution_potency_rank": [3.0, 1.0, 2.0],
            "best_lysis_dilution": [-2, 0, -4],
        }
    )

    out = _aggregate_lytic_pair_metrics(lytic_pairs, "group")

    g1 = out.loc[out["group"] == "G1"].iloc[0]
    g2 = out.loc[out["group"] == "G2"].iloc[0]

    assert g1["n_lytic_pairs"] == 2
    assert g1["n_high_potency_pairs"] == 1
    assert g1["n_low_potency_pairs"] == 1
    assert g1["multi_dilution_support_rate"] == 1.0
    assert g1["best_dilution_minus2_count"] == 1
    assert g2["n_lytic_pairs"] == 1
    assert g2["high_potency_rate"] == 1.0
    assert g2["n_single_dilution_pairs"] == 1


def test_top_rows_replaces_nan_with_none() -> None:
    summary = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "score": [1.0, float("nan"), 3.0],
        }
    )

    result = top_rows(summary, limit=2, sort_columns=["score"])
    assert result[0]["name"] == "C"
    assert result[0]["score"] == 3.0
    assert result[1]["name"] == "A"
    assert result[1]["score"] == 1.0
    assert len(result) == 2


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


def test_validate_required_inputs_explains_track_a_prerequisite(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    host_metadata_path = tmp_path / "host.csv"
    phage_metadata_path = tmp_path / "phage.csv"
    host_metadata_path.write_text("bacteria;Clermont_Phylo;ST_Warwick;O-type;H-type\n", encoding="utf-8")
    phage_metadata_path.write_text("phage;Morphotype;Family\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        pair_labels_path=DEFAULT_PAIR_LABELS_PATH,
        pair_dilution_summary_path=DEFAULT_PAIR_DILUTION_SUMMARY_PATH,
        host_metadata_path=host_metadata_path,
        phage_metadata_path=phage_metadata_path,
    )

    with pytest.raises(FileNotFoundError, match="TB05 reuses canonical Track A outputs"):
        validate_required_inputs(args)

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_required_inputs(args)

    message = str(exc_info.value)
    assert str(DEFAULT_PAIR_LABELS_PATH) in message
    assert str(DEFAULT_PAIR_DILUTION_SUMMARY_PATH) in message
    assert TRACK_A_BUILD_COMMAND in message
