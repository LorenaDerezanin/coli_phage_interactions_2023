import pandas as pd
import pytest
from scipy.stats import kruskal

from lyzortx.research_notes.ad_hoc_analysis_code.hard_to_lyse_host_traits import (
    benjamini_hochberg,
    build_per_strain_summary,
    clean_trait_value,
    classify_susceptibility_bucket,
    derive_serotype,
    summarize_trait_field,
)


def test_derive_serotype_prefers_o_h_pair_and_handles_missing() -> None:
    assert derive_serotype("O8", "H9") == "O8:H9"
    assert derive_serotype("O8", "") == "O8"
    assert derive_serotype("", "") == "Missing"


def test_clean_trait_value_treats_pd_na_as_missing() -> None:
    assert clean_trait_value(pd.NA) == "Missing"


def test_classify_susceptibility_bucket_distinguishes_zero_low_and_broad() -> None:
    assert classify_susceptibility_bucket(0, low_threshold=3) == "zero"
    assert classify_susceptibility_bucket(2, low_threshold=3) == "low_1_3"
    assert classify_susceptibility_bucket(5, low_threshold=3) == "broad"


def test_benjamini_hochberg_is_monotone_after_ordering() -> None:
    q_values = benjamini_hochberg([0.01, 0.02, 0.2])

    assert q_values == pytest.approx([0.03, 0.03, 0.2])


def test_build_per_strain_summary_derives_expected_host_fields() -> None:
    interaction_matrix = pd.DataFrame(
        {
            "P1": [1, 0],
            "P2": [0, 0],
            "P3": [0, 1],
        },
        index=["B1", "B2"],
    )
    interaction_matrix.index.name = "bacteria"
    host_metadata = pd.DataFrame(
        {
            "bacteria": ["B1", "B2"],
            "ABC_serotype": ["K1", ""],
            "Clermont_Phylo": ["B2", "A"],
            "ST_Warwick": ["95", "10"],
            "O-type": ["O8", "O89"],
            "H-type": ["H9", "H9"],
        }
    )

    out = build_per_strain_summary(interaction_matrix, host_metadata, low_threshold=1)

    assert out.loc[out["bacteria"] == "B1", "host_serotype"].item() == "O8:H9"
    assert out.loc[out["bacteria"] == "B1", "host_abc_serotype"].item() == "K1"
    assert out.loc[out["bacteria"] == "B2", "host_phylogroup"].item() == "A"
    assert out.loc[out["bacteria"] == "B2", "susceptibility_bucket"].item() == "low_1_1"


def test_build_per_strain_summary_preserves_missing_assays_in_lysis_counts() -> None:
    interaction_matrix = pd.DataFrame(
        {
            "P1": [1, 0],
            "P2": [pd.NA, 0],
            "P3": [0, 1],
        },
        index=["B1", "B2"],
    )
    interaction_matrix.index.name = "bacteria"
    host_metadata = pd.DataFrame(
        {
            "bacteria": ["B1", "B2"],
            "ABC_serotype": ["K1", "K2"],
            "Clermont_Phylo": ["B2", "A"],
            "ST_Warwick": ["95", "10"],
            "O-type": ["O8", "O89"],
            "H-type": ["H9", "H9"],
        }
    )

    out = build_per_strain_summary(interaction_matrix, host_metadata, low_threshold=1)

    b1 = out.loc[out["bacteria"] == "B1"].iloc[0]
    b2 = out.loc[out["bacteria"] == "B2"].iloc[0]

    assert pd.isna(b1["n_lytic_phages"])
    assert b1["is_zero_lysis"] == False
    assert pd.isna(b1["is_low_susceptibility"])
    assert b1["susceptibility_bucket"] == "unknown_missing_assays"
    assert b2["n_lytic_phages"] == 1


def test_build_per_strain_summary_marks_definitely_broad_rows_despite_missing_assays() -> None:
    interaction_matrix = pd.DataFrame(
        {
            "P1": [1, 0],
            "P2": [1, 0],
            "P3": [1, 0],
            "P4": [1, 1],
            "P5": [pd.NA, 0],
        },
        index=["B1", "B2"],
    )
    interaction_matrix.index.name = "bacteria"
    host_metadata = pd.DataFrame(
        {
            "bacteria": ["B1", "B2"],
            "ABC_serotype": ["K1", "K2"],
            "Clermont_Phylo": ["B2", "A"],
            "ST_Warwick": ["95", "10"],
            "O-type": ["O8", "O89"],
            "H-type": ["H9", "H9"],
        }
    )

    out = build_per_strain_summary(interaction_matrix, host_metadata, low_threshold=3)

    b1 = out.loc[out["bacteria"] == "B1"].iloc[0]

    assert pd.isna(b1["n_lytic_phages"])
    assert b1["is_zero_lysis"] == False
    assert b1["is_low_susceptibility"] == False
    assert b1["susceptibility_bucket"] == "broad"


def test_summarize_trait_field_reports_field_and_value_rows() -> None:
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3", "B4", "B5", "B6"],
            "n_lytic_phages": [0, 1, 7, 8, 0, 6],
            "is_zero_lysis": [True, False, False, False, True, False],
            "is_low_susceptibility": [True, True, False, False, True, False],
            "host_phylogroup": ["A", "A", "A", "B", "B", "B"],
        }
    )

    rows = summarize_trait_field(
        strain_summary=strain_summary,
        field_name="host_phylogroup",
        field_label="phylogroup",
        low_threshold=3,
        min_group_size=3,
    )

    field_row = rows[0]
    a_row = next(row for row in rows if row["summary_level"] == "trait_value" and row["trait_value"] == "A")
    b_row = next(row for row in rows if row["summary_level"] == "trait_value" and row["trait_value"] == "B")

    assert field_row["summary_level"] == "field"
    assert field_row["field_name"] == "phylogroup"
    assert a_row["low_susceptibility_count"] == 2
    assert a_row["tested_for_enrichment"] is True
    assert a_row["fisher_q_value"] is not None
    assert b_row["low_susceptibility_rate"] == 1 / 3


def test_summarize_trait_field_excludes_unknown_low_status_from_rates() -> None:
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3", "B4"],
            "n_lytic_phages": [1, pd.NA, 5, 6],
            "is_zero_lysis": [False, pd.NA, False, False],
            "is_low_susceptibility": [True, pd.NA, False, False],
            "host_phylogroup": ["A", "A", "A", "B"],
        }
    )

    rows = summarize_trait_field(
        strain_summary=strain_summary,
        field_name="host_phylogroup",
        field_label="phylogroup",
        low_threshold=3,
        min_group_size=2,
    )

    a_row = next(row for row in rows if row["summary_level"] == "trait_value" and row["trait_value"] == "A")

    assert a_row["low_susceptibility_count"] == 1
    assert a_row["low_susceptibility_rate"] == 0.5
    assert a_row["tested_for_enrichment"] is True


def test_summarize_trait_field_returns_missing_rate_when_no_statuses_are_observed() -> None:
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3"],
            "n_lytic_phages": [pd.NA, pd.NA, 5],
            "is_zero_lysis": [pd.NA, pd.NA, False],
            "is_low_susceptibility": [pd.NA, pd.NA, False],
            "host_serotype": ["O1:H1", "O1:H1", "O2:H2"],
        }
    )

    rows = summarize_trait_field(
        strain_summary=strain_summary,
        field_name="host_serotype",
        field_label="serotype",
        low_threshold=3,
        min_group_size=2,
    )

    unknown_row = next(row for row in rows if row["summary_level"] == "trait_value" and row["trait_value"] == "O1:H1")

    assert unknown_row["low_susceptibility_count"] == 0
    assert unknown_row["low_susceptibility_rate"] is None
    assert unknown_row["mean_lytic_phages"] is None
    assert unknown_row["tested_for_enrichment"] is False


def test_summarize_trait_field_includes_singletons_in_kruskal_test() -> None:
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3", "B4"],
            "n_lytic_phages": [1, 4, 9, 10],
            "is_zero_lysis": [False, False, False, False],
            "is_low_susceptibility": [True, False, False, False],
            "host_st": ["11", "11", "12", "13"],
        }
    )

    rows = summarize_trait_field(
        strain_summary=strain_summary,
        field_name="host_st",
        field_label="ST",
        low_threshold=3,
        min_group_size=2,
    )

    field_row = rows[0]
    expected_p_value = float(kruskal([1, 4], [9], [10]).pvalue)

    assert field_row["summary_level"] == "field"
    assert field_row["kruskal_p_value"] == pytest.approx(expected_p_value)
