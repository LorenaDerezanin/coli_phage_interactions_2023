from pathlib import Path

from lyzortx.pipeline.track_l.steps.build_host_typing_projector import (
    build_family_validation_rows,
    build_legacy_validation_rows,
    build_projected_feature_row,
    derive_serotype,
    normalize_capsule_model_to_serotype_proxy,
    normalize_legacy_abc_serotype,
    parse_ectyper_output,
    parse_mlst_legacy_output,
    parse_phylogroup_report,
)


def test_parse_phylogroup_report_extracts_predicted_and_mash_groups(tmp_path: Path) -> None:
    report_path = tmp_path / "typing_phylogroups.txt"
    report_path.write_text(
        "LF82.fasta\t['trpA']\t['+']\t['trpAgpC']\tB2\tB2\n",
        encoding="utf-8",
    )

    observed = parse_phylogroup_report(report_path)

    assert observed == {
        "assembly_name": "LF82.fasta",
        "phylogroup": "B2",
        "mash_phylogroup": "B2",
    }


def test_parse_ectyper_output_reads_single_row_tsv(tmp_path: Path) -> None:
    output_path = tmp_path / "output.tsv"
    output_path.write_text(
        ("Name\tSpecies\tO-type\tH-type\tSerotype\tQC\tWarnings\nLF82\tEscherichia coli\tO83\tH1\tO83:H1\t-\t-\n"),
        encoding="utf-8",
    )

    observed = parse_ectyper_output(output_path)

    assert observed == {
        "species": "Escherichia coli",
        "o_type": "O83",
        "h_type": "H1",
        "serotype": "O83:H1",
        "qc": "",
        "warnings": "",
    }


def test_parse_mlst_legacy_output_ignores_banner_lines(tmp_path: Path) -> None:
    output_path = tmp_path / "mlst_legacy.tsv"
    output_path.write_text(
        (
            "This is mlst 2.32.2 running on linux\n"
            "Done.\n"
            "FILE\tSCHEME\tST\tadk\n"
            "data/genomics/bacteria/validation_subset/fastas/LF82.fasta\tecoli_achtman_4\t135\t13\n"
        ),
        encoding="utf-8",
    )

    observed = parse_mlst_legacy_output(output_path)

    assert observed == {
        "assembly_path": "data/genomics/bacteria/validation_subset/fastas/LF82.fasta",
        "scheme": "ecoli_achtman_4",
        "st_warwick": "135",
    }


def test_normalize_legacy_abc_serotype_adds_k_prefix_for_numeric_calls() -> None:
    assert normalize_legacy_abc_serotype("5") == "K5"
    assert normalize_legacy_abc_serotype("K1") == "K1"
    assert normalize_legacy_abc_serotype("Unknown") == ""


def test_normalize_capsule_model_to_serotype_proxy_only_keeps_simple_k_models() -> None:
    assert normalize_capsule_model_to_serotype_proxy("K4_KfoFGCA_2_unknown") == "K4"
    assert normalize_capsule_model_to_serotype_proxy("K10_like") == "K10"
    assert normalize_capsule_model_to_serotype_proxy("class_1_3") == ""


def test_derive_serotype_prefers_combined_oh_call_and_handles_missing() -> None:
    assert derive_serotype("O83", "H1") == "O83:H1"
    assert derive_serotype("O83", "") == "O83"
    assert derive_serotype("", "") == ""


def test_build_projected_feature_row_keeps_direct_and_proxy_fields_separate() -> None:
    observed = build_projected_feature_row(
        bacteria="LF82",
        phylogroup_call={"phylogroup": "B2"},
        mlst_call={"st_warwick": "135"},
        serotype_call={"o_type": "O83", "h_type": "H1"},
        capsule_proxy={
            "host_capsule_abc_proxy_present": 1,
            "host_abc_serotype_proxy": "K5",
            "host_capsule_proxy_top_model": "K5",
            "host_capsule_proxy_model_count": 2,
            "host_capsule_proxy_candidate_models": "K5|class_1_5",
        },
    )

    assert observed == {
        "bacteria": "LF82",
        "host_clermont_phylo": "B2",
        "host_st_warwick": "135",
        "host_o_type": "O83",
        "host_h_type": "H1",
        "host_serotype": "O83:H1",
        "host_capsule_abc_proxy_present": 1,
        "host_abc_serotype_proxy": "K5",
        "host_capsule_proxy_top_model": "K5",
        "host_capsule_proxy_model_count": 2,
        "host_capsule_proxy_candidate_models": "K5|class_1_5",
    }


def test_build_legacy_validation_rows_marks_direct_proxy_and_non_derivable_statuses() -> None:
    projected_rows = [
        {
            "bacteria": "LF82",
            "host_clermont_phylo": "B2",
            "host_st_warwick": "135",
            "host_o_type": "O83",
            "host_h_type": "H1",
            "host_serotype": "O83:H1",
            "host_capsule_abc_proxy_present": 1,
            "host_abc_serotype_proxy": "K4",
            "host_capsule_proxy_top_model": "K4",
            "host_capsule_proxy_model_count": 3,
            "host_capsule_proxy_candidate_models": "K4|class_1_5|class_1_3",
        }
    ]
    panel_metadata = {
        "LF82": {
            "Clermont_Phylo": "B2",
            "ST_Warwick": "135",
            "O-type": "O83",
            "H-type": "H1",
            "Capsule_ABC": "1.0",
            "ABC_serotype": "5",
            "Capsule_GroupIV_e": "",
            "Capsule_GroupIV_e_stricte": "",
            "Capsule_GroupIV_s": "",
            "Capsule_Wzy_stricte": "",
            "Origin": "",
            "Pathotype": "",
            "Collection": "Host",
            "Mouse_killed_10": "",
        }
    }

    observed = build_legacy_validation_rows(projected_rows=projected_rows, panel_metadata=panel_metadata)

    direct_row = next(row for row in observed if row["legacy_field_name"] == "host_clermont_phylo")
    proxy_row = next(row for row in observed if row["legacy_field_name"] == "host_abc_serotype")
    unsupported_row = next(row for row in observed if row["legacy_field_name"] == "host_capsule_groupiv_e")
    non_derivable_row = next(row for row in observed if row["legacy_field_name"] == "host_collection")

    assert direct_row["validation_outcome"] == "reproduced_directly"
    assert direct_row["exact_match_count"] == 1
    assert proxy_row["validation_outcome"] == "noisy_proxy"
    assert proxy_row["exact_match_count"] == 0
    assert unsupported_row["validation_outcome"] == "unsupported"
    assert non_derivable_row["validation_outcome"] == "non_derivable"


def test_build_family_validation_rows_rolls_up_family_outcomes() -> None:
    field_rows = [
        {
            "feature_family": "serotype",
            "legacy_field_name": "host_o_type",
            "projected_feature_name": "host_o_type",
            "projection_status": "direct",
            "validation_outcome": "reproduced_directly",
            "legacy_resolved_host_count": 3,
            "exact_match_count": 3,
            "rationale": "direct",
        },
        {
            "feature_family": "capsule_typed_serotype",
            "legacy_field_name": "host_abc_serotype",
            "projected_feature_name": "host_abc_serotype_proxy",
            "projection_status": "deployable_proxy",
            "validation_outcome": "noisy_proxy",
            "legacy_resolved_host_count": 1,
            "exact_match_count": 0,
            "rationale": "proxy",
        },
        {
            "feature_family": "non_derivable_metadata",
            "legacy_field_name": "host_collection",
            "projected_feature_name": "",
            "projection_status": "non_derivable",
            "validation_outcome": "non_derivable",
            "legacy_resolved_host_count": "",
            "exact_match_count": "",
            "rationale": "metadata",
        },
    ]

    observed = build_family_validation_rows(field_rows)

    serotype_row = next(row for row in observed if row["feature_family"] == "serotype")
    capsule_row = next(row for row in observed if row["feature_family"] == "capsule_typed_serotype")
    metadata_row = next(row for row in observed if row["feature_family"] == "non_derivable_metadata")

    assert serotype_row["family_outcome"] == "reproduced_directly"
    assert capsule_row["family_outcome"] == "noisy_proxy"
    assert metadata_row["family_outcome"] == "non_derivable"
