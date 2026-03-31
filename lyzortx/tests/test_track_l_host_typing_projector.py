from pathlib import Path

import pandas as pd

from lyzortx.pipeline.track_l.steps import build_host_typing_projector as tl16


def test_build_projected_feature_rows_joins_gembase_and_bacteria_keyed_inputs() -> None:
    panel_metadata = pd.DataFrame(
        [
            {
                "bacteria": "B1",
                "Gembase": "G1",
                "Clermont_Phylo": "B2",
                "ST_Warwick": "95",
                "O-type": "O8",
                "H-type": "H9",
                "ABC_serotype": "K1",
                "Klebs_capsule_type": "",
            },
            {
                "bacteria": "B2",
                "Gembase": "G2",
                "Clermont_Phylo": "",
                "ST_Warwick": "",
                "O-type": "",
                "H-type": "",
                "ABC_serotype": "",
                "Klebs_capsule_type": "K54",
            },
        ]
    )
    phylogroup_calls = pd.DataFrame([{"gembase": "G1", "phylogroup": "B2"}])
    sequence_type_calls = pd.DataFrame([{"FILE": "G1", "ST": "95"}])
    serotype_calls = pd.DataFrame(
        [
            {"Name": "B1", "O-type": "O8", "H-type": "H9", "Serotype": "O8:H9"},
            {"Name": "B2", "O-type": "-", "H-type": "-", "Serotype": "-"},
        ]
    )
    capsule_high_hits = pd.DataFrame([{"bacteria": "B2", "Klebs_capsule_type": "K54"}])
    capsule_calls = pd.DataFrame(
        [
            {
                "Assembly": "B1",
                "Best match locus": "KL1",
                "Best match type": "K1",
                "Match confidence": "Low",
                "Problems": "?-*",
                "Coverage": "80.0%",
                "Identity": "99.1%",
            }
        ]
    )

    rows = tl16.build_projected_feature_rows(
        panel_metadata=panel_metadata,
        phylogroup_calls=phylogroup_calls,
        sequence_type_calls=sequence_type_calls,
        serotype_calls=serotype_calls,
        capsule_high_hits=capsule_high_hits,
        capsule_calls=capsule_calls,
    )

    assert rows == [
        {
            "bacteria": "B1",
            "source_gembase": "G1",
            "host_clermont_phylo": "B2",
            "host_clermont_phylo_status": "called",
            "host_st_warwick": "95",
            "host_st_warwick_status": "called",
            "host_o_type": "O8",
            "host_h_type": "H9",
            "host_serotype": "O8:H9",
            "host_serotype_status": "called",
            "host_abc_serotype": "K1",
            "host_abc_serotype_status": "typed_proxy",
            "host_surface_klebsiella_capsule_type": "",
            "host_surface_klebsiella_capsule_type_status": "not_callable",
            "host_capsule_call_status": "typed_proxy",
            "host_capsule_best_match_locus": "KL1",
            "host_capsule_best_match_type": "K1",
            "host_capsule_match_confidence": "Low",
            "host_capsule_problem_flags": "?-*",
            "host_capsule_coverage_pct": "80.000",
            "host_capsule_identity_pct": "99.100",
        },
        {
            "bacteria": "B2",
            "source_gembase": "G2",
            "host_clermont_phylo": "",
            "host_clermont_phylo_status": "not_callable",
            "host_st_warwick": "",
            "host_st_warwick_status": "not_callable",
            "host_o_type": "",
            "host_h_type": "",
            "host_serotype": "",
            "host_serotype_status": "not_callable",
            "host_abc_serotype": "K54",
            "host_abc_serotype_status": "typed_proxy",
            "host_surface_klebsiella_capsule_type": "K54",
            "host_surface_klebsiella_capsule_type_status": "high_hit_typed",
            "host_capsule_call_status": "high_hit_typed",
            "host_capsule_best_match_locus": "",
            "host_capsule_best_match_type": "",
            "host_capsule_match_confidence": "",
            "host_capsule_problem_flags": "",
            "host_capsule_coverage_pct": "",
            "host_capsule_identity_pct": "",
        },
    ]


def test_build_validation_rows_marks_exact_and_proxy_families_separately() -> None:
    panel_metadata = pd.DataFrame(
        [
            {
                "bacteria": "B1",
                "Gembase": "G1",
                "Clermont_Phylo": "B2",
                "ST_Warwick": "95",
                "O-type": "O8",
                "H-type": "H9",
                "ABC_serotype": "K1",
                "Klebs_capsule_type": "",
                "Capsule_ABC": "1.0",
                "Capsule_GroupIV_e": "0.0",
                "Capsule_GroupIV_e_stricte": "0.0",
                "Capsule_GroupIV_s": "1.0",
                "Capsule_Wzy_stricte": "1.0",
            },
            {
                "bacteria": "B2",
                "Gembase": "G2",
                "Clermont_Phylo": "",
                "ST_Warwick": "",
                "O-type": "",
                "H-type": "",
                "ABC_serotype": "CatB",
                "Klebs_capsule_type": "K54",
                "Capsule_ABC": "",
                "Capsule_GroupIV_e": "",
                "Capsule_GroupIV_e_stricte": "",
                "Capsule_GroupIV_s": "",
                "Capsule_Wzy_stricte": "",
            },
        ]
    )
    projected_rows = [
        {
            "bacteria": "B1",
            "source_gembase": "G1",
            "host_clermont_phylo": "B2",
            "host_clermont_phylo_status": "called",
            "host_st_warwick": "95",
            "host_st_warwick_status": "called",
            "host_o_type": "O8",
            "host_h_type": "H9",
            "host_serotype": "O8:H9",
            "host_serotype_status": "called",
            "host_abc_serotype": "K1",
            "host_abc_serotype_status": "typed_proxy",
            "host_surface_klebsiella_capsule_type": "",
            "host_surface_klebsiella_capsule_type_status": "not_callable",
            "host_capsule_call_status": "typed_proxy",
            "host_capsule_best_match_locus": "KL1",
            "host_capsule_best_match_type": "K1",
            "host_capsule_match_confidence": "Low",
            "host_capsule_problem_flags": "?-*",
            "host_capsule_coverage_pct": "80.000",
            "host_capsule_identity_pct": "99.100",
        },
        {
            "bacteria": "B2",
            "source_gembase": "G2",
            "host_clermont_phylo": "",
            "host_clermont_phylo_status": "not_callable",
            "host_st_warwick": "",
            "host_st_warwick_status": "not_callable",
            "host_o_type": "",
            "host_h_type": "",
            "host_serotype": "",
            "host_serotype_status": "not_callable",
            "host_abc_serotype": "K54",
            "host_abc_serotype_status": "typed_proxy",
            "host_surface_klebsiella_capsule_type": "K54",
            "host_surface_klebsiella_capsule_type_status": "high_hit_typed",
            "host_capsule_call_status": "high_hit_typed",
            "host_capsule_best_match_locus": "KL54",
            "host_capsule_best_match_type": "K54",
            "host_capsule_match_confidence": "Good",
            "host_capsule_problem_flags": "",
            "host_capsule_coverage_pct": "99.000",
            "host_capsule_identity_pct": "98.000",
        },
    ]

    summary_rows, detail_rows = tl16.build_validation_rows(
        panel_metadata=panel_metadata,
        projected_feature_rows=projected_rows,
    )

    summary_by_family = {row["feature_family"]: row for row in summary_rows}

    assert summary_by_family["phylogroup"]["validation_status"] == "clean"
    assert summary_by_family["phylogroup"]["exact_match_count"] == 1
    assert summary_by_family["abc_serotype_proxy"]["validation_status"] == "noisy_or_partial"
    assert summary_by_family["abc_serotype_proxy"]["comparable_host_count"] == 2
    assert summary_by_family["legacy_capsule_binary_flags"]["validation_status"] == "unsupported"
    assert any(
        row["feature_family"] == "serotype" and row["bacteria"] == "B1" and row["exact_match"] == 1
        for row in detail_rows
    )


def test_main_writes_expected_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    panel_path.write_text(
        (
            "bacteria;Gembase;Clermont_Phylo;ST_Warwick;O-type;H-type;ABC_serotype;Klebs_capsule_type;"
            "Capsule_ABC;Capsule_GroupIV_e;Capsule_GroupIV_e_stricte;Capsule_GroupIV_s;Capsule_Wzy_stricte\n"
            "B1;G1;B2;95;O8;H9;K1;;1.0;0.0;0.0;1.0;1.0\n"
        ),
        encoding="utf-8",
    )
    phylogroup_path = tmp_path / "phy.tsv"
    phylogroup_path.write_text("G1\tgenes\tpresence\tallele\tB2\tmash\n", encoding="utf-8")
    sequence_type_path = tmp_path / "st.tsv"
    sequence_type_path.write_text("FILE\tSCHEME\tST\nG1\tecoli_achtman_4\t95\n", encoding="utf-8")
    serotype_path = tmp_path / "serotype.tsv"
    serotype_path.write_text(
        "Name\tO-type\tH-type\tSerotype\nB1\tO8\tH9\tO8:H9\n",
        encoding="utf-8",
    )
    capsule_high_hit_path = tmp_path / "capsule_high.tsv"
    capsule_high_hit_path.write_text("bacteria\tKlebs_capsule_type\tlocus\nB9\tK54\tGood\n", encoding="utf-8")
    capsule_all_path = tmp_path / "capsule_all.tsv"
    capsule_all_path.write_text(
        (
            "Assembly\tBest match locus\tBest match type\tMatch confidence\tProblems\tCoverage\tIdentity\n"
            "B1\tKL1\tK1\tLow\t?-*\t80.0%\t99.1%\n"
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = tl16.main(
        [
            "--panel-metadata-path",
            str(panel_path),
            "--phylogroup-path",
            str(phylogroup_path),
            "--sequence-type-path",
            str(sequence_type_path),
            "--serotype-path",
            str(serotype_path),
            "--capsule-high-hit-path",
            str(capsule_high_hit_path),
            "--capsule-all-results-path",
            str(capsule_all_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / tl16.OUTPUT_FEATURES_FILENAME).exists()
    assert (output_dir / tl16.OUTPUT_SCHEMA_FILENAME).exists()
    assert (output_dir / tl16.OUTPUT_STATUS_FILENAME).exists()
    assert (output_dir / tl16.OUTPUT_VALIDATION_SUMMARY_FILENAME).exists()
    assert (output_dir / tl16.OUTPUT_MANIFEST_FILENAME).exists()
