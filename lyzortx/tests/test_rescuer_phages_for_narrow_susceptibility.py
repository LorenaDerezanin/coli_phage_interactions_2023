import pandas as pd
import pytest

from lyzortx.research_notes.ad_hoc_analysis_code.rescuer_phages_for_narrow_susceptibility import (
    build_narrow_strain_rescuer_summary,
    build_rescuer_group_summary,
    build_rescuer_phage_summary,
    classify_rescue_mode,
    count_rescue_modes,
    resolved_narrow_susceptibility_strains,
)


def test_classify_rescue_mode_distinguishes_exclusive_and_shared() -> None:
    assert classify_rescue_mode(0) == "non_rescued"
    assert classify_rescue_mode(1) == "exclusive"
    assert classify_rescue_mode(2) == "shared"
    assert classify_rescue_mode(3) == "shared"


def test_count_rescue_modes_excludes_non_rescued_rows_from_shared_totals() -> None:
    narrow_strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B4"],
            "rescue_mode": ["exclusive", "shared", "non_rescued"],
        }
    )

    out = count_rescue_modes(narrow_strain_summary)

    assert out == {"exclusive": 1, "shared": 1, "non_rescued": 1}


def test_resolved_narrow_susceptibility_strains_keeps_only_true_rows() -> None:
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3"],
            "known_lytic_phages": [1, 2, 6],
            "is_low_susceptibility": [True, pd.NA, False],
        }
    )

    out = resolved_narrow_susceptibility_strains(strain_summary)

    assert out["bacteria"].tolist() == ["B1"]


def test_build_narrow_strain_rescuer_summary_emits_expected_rescuer_lists() -> None:
    interaction_matrix = pd.DataFrame(
        {
            "P1": [1, 1, 0, 0],
            "P2": [0, 1, 0, 0],
            "P3": [0, 0, 1, 0],
        },
        index=["B1", "B2", "B3", "B4"],
    )
    interaction_matrix.index.name = "bacteria"
    strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B3", "B4"],
            "known_lytic_phages": [1, 2, 1, 0],
            "is_low_susceptibility": [True, True, False, True],
        }
    )
    phage_metadata = pd.DataFrame(
        {
            "phage": ["P1", "P2", "P3"],
            "Morphotype": ["Myoviridae", "Podoviridae", "Siphoviridae"],
            "Family": ["Straboviridae", "Autographiviridae", "Drexlerviridae"],
        }
    )

    out = build_narrow_strain_rescuer_summary(
        interaction_matrix=interaction_matrix,
        strain_summary=strain_summary,
        phage_metadata=phage_metadata,
    )

    b1 = out.loc[out["bacteria"] == "B1"].iloc[0]
    b2 = out.loc[out["bacteria"] == "B2"].iloc[0]
    b4 = out.loc[out["bacteria"] == "B4"].iloc[0]

    assert b1["rescue_mode"] == "exclusive"
    assert b1["rescuer_phages"] == "P1"
    assert b1["unique_rescuer_phage"] == "P1"
    assert b2["rescue_mode"] == "shared"
    assert b2["rescuer_phages"] == "P1,P2"
    assert b2["rescuer_morphotypes"] == "Myoviridae,Podoviridae"
    assert b2["unique_rescuer_phage"] == ""
    assert b4["rescue_mode"] == "non_rescued"
    assert b4["rescuer_phages"] == ""
    assert b4["rescuer_morphotypes"] == ""
    assert b4["unique_rescuer_phage"] == ""


def test_build_rescuer_phage_summary_counts_exclusive_and_shared_rescues() -> None:
    interaction_matrix = pd.DataFrame(
        {
            "P1": [1, 1, 0, 0],
            "P2": [0, 1, 1, 0],
            "P3": [0, 0, 1, 0],
        },
        index=["B1", "B2", "B3", "B4"],
    )
    interaction_matrix.index.name = "bacteria"
    narrow_strain_summary = pd.DataFrame(
        {
            "bacteria": ["B1", "B2", "B4"],
            "known_lytic_phages": [1, 2, 0],
            "rescue_mode": ["exclusive", "shared", "non_rescued"],
            "rescuer_phages": ["P1", "P1,P2", ""],
            "rescuer_morphotypes": ["Myoviridae", "Myoviridae,Podoviridae", ""],
            "rescuer_families": ["Straboviridae", "Straboviridae,Autographiviridae", ""],
            "unique_rescuer_phage": ["P1", "", ""],
        }
    )
    phage_metadata = pd.DataFrame(
        {
            "phage": ["P1", "P2", "P3"],
            "Morphotype": ["Myoviridae", "Podoviridae", "Siphoviridae"],
            "Family": ["Straboviridae", "Autographiviridae", "Drexlerviridae"],
        }
    )

    out = build_rescuer_phage_summary(
        narrow_strain_summary=narrow_strain_summary,
        interaction_matrix=interaction_matrix,
        phage_metadata=phage_metadata,
    )

    p1 = out.loc[out["phage"] == "P1"].iloc[0]
    p2 = out.loc[out["phage"] == "P2"].iloc[0]
    p3 = out.loc[out["phage"] == "P3"].iloc[0]

    assert p1["narrow_strains_rescued"] == 2
    assert p1["exclusive_rescue_count"] == 1
    assert p1["shared_rescue_count"] == 1
    assert p1["fraction_of_all_lysed_strains_that_are_narrow"] == pytest.approx(1.0)
    assert p2["narrow_strains_rescued"] == 1
    assert p2["exclusive_rescue_count"] == 0
    assert p2["shared_rescue_count"] == 1
    assert p3["is_rescuer_phage"] == False


def test_build_rescuer_group_summary_aggregates_only_rescuer_phages() -> None:
    rescuer_phage_summary = pd.DataFrame(
        {
            "phage": ["P1", "P2", "P3"],
            "morphotype": ["Myoviridae", "Myoviridae", "Podoviridae"],
            "family": ["F1", "F2", "F3"],
            "narrow_strains_rescued": [3, 0, 1],
            "exclusive_rescue_count": [2, 0, 0],
            "shared_rescue_count": [1, 0, 1],
            "fraction_of_all_lysed_strains_that_are_narrow": [0.5, 0.0, 0.25],
            "is_rescuer_phage": [True, False, True],
        }
    )

    rows = build_rescuer_group_summary(rescuer_phage_summary, "morphotype")

    myoviridae = next(row for row in rows if row["group_value"] == "Myoviridae")
    podoviridae = next(row for row in rows if row["group_value"] == "Podoviridae")

    assert myoviridae["n_total_phages"] == 2
    assert myoviridae["n_rescuer_phages"] == 1
    assert myoviridae["exclusive_rescue_count"] == 2
    assert myoviridae["median_narrow_strains_rescued_per_rescuer_phage"] == pytest.approx(3.0)
    assert podoviridae["n_rescuer_phages"] == 1
    assert podoviridae["shared_rescue_count"] == 1
