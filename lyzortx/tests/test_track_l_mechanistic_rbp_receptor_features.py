import csv
from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_l.steps.build_mechanistic_rbp_receptor_features import (
    build_feature_rows,
    build_sanity_check_rows,
    collapse_duplicate_profiles,
    collapse_significant_associations,
    main,
)


MERGED_TSV_HEADER = (
    "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
    "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\t"
    "pyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\tcustom_hmm_bitscore\t"
    "custom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\t"
    "annot\tcategory\tvfdb_hit\tvfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\t"
    "vfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\tCARD_alnScore\t"
    "CARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
    "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\t"
    "Resistance_Mechanism\ttransl_table"
)

NONE_FIELDS = "\t".join(["None"] * 20)


def _make_merged_row(gene: str, phrog: str, annot: str, category: str = "tail") -> str:
    return (
        f"{gene}\t1\t100\t+\tcontig_1\t-10.0\t{phrog}\t50\t0.6\t1e-5\t"
        f"No_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\tNo_custom_hmm\tNo_custom_hmm\t"
        f"{phrog}\tPHANOTATE\tCDS\t#838383\t"
        f"{annot}\t{category}\t{NONE_FIELDS}"
    )


def test_collapse_duplicate_profiles_groups_identical_carrier_patterns() -> None:
    feature_names = ["RBP_PHROG_136", "RBP_PHROG_15437", "RBP_PHROG_1002", "RBP_PHROG_4277"]
    matrix = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=np.int8,
    )

    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(feature_names, matrix)

    assert collapsed_matrix.shape == (3, 3)
    assert len(profiles) == 3
    assert feature_to_profile_id["RBP_PHROG_136"] == feature_to_profile_id["RBP_PHROG_15437"]
    assert profiles[0].member_features == ("RBP_PHROG_1002",)
    duplicate_profile = next(profile for profile in profiles if len(profile.member_features) == 2)
    assert duplicate_profile.member_features == ("RBP_PHROG_136", "RBP_PHROG_15437")


def test_collapse_significant_associations_merges_duplicate_phrogs() -> None:
    _, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        ["RBP_PHROG_136", "RBP_PHROG_15437", "RBP_PHROG_1002"],
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.int8),
    )
    profile_by_id = {profile.profile_id: profile for profile in profiles}

    enrichment_rows = [
        {
            "phage_feature": "RBP_PHROG_136",
            "host_feature": "OMPC_99_1",
            "lysis_rate_diff": "0.6",
            "significant": "True",
        },
        {
            "phage_feature": "RBP_PHROG_15437",
            "host_feature": "OMPC_99_1",
            "lysis_rate_diff": "0.6",
            "significant": "True",
        },
        {"phage_feature": "RBP_PHROG_1002", "host_feature": "LPS_R1", "lysis_rate_diff": "0.2", "significant": "True"},
        {"phage_feature": "RBP_PHROG_1002", "host_feature": "LPS_R2", "lysis_rate_diff": "0.1", "significant": "False"},
    ]

    associations = collapse_significant_associations(enrichment_rows, feature_to_profile_id, profile_by_id)
    by_host_feature = {association.host_feature: association for association in associations}

    assert len(associations) == 2
    assert by_host_feature["OMPC_99_1"].weight == 0.6
    assert by_host_feature["OMPC_99_1"].pairwise_column.endswith("ompc_99_1_weight")


def test_build_feature_rows_emits_direct_and_pairwise_blocks() -> None:
    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        ["RBP_PHROG_136", "RBP_PHROG_15437", "RBP_PHROG_1002"],
        np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8),
    )
    phage_to_profile_presence = {
        "P1": {profile.profile_id: int(collapsed_matrix[0, index]) for index, profile in enumerate(profiles)},
        "P2": {profile.profile_id: int(collapsed_matrix[1, index]) for index, profile in enumerate(profiles)},
    }
    associations = [
        collapse_significant_associations(
            [
                {
                    "phage_feature": "RBP_PHROG_136",
                    "host_feature": "OMPC_99_1",
                    "lysis_rate_diff": "0.6",
                    "significant": "True",
                }
            ],
            feature_to_profile_id,
            {profile.profile_id: profile for profile in profiles},
        )[0]
    ]
    pair_rows = [
        {"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1"},
        {"pair_id": "B2__P1", "bacteria": "B2", "phage": "P1"},
        {"pair_id": "B1__P2", "bacteria": "B1", "phage": "P2"},
    ]
    bacteria_to_host_features = {"B1": {"OMPC_99_1"}, "B2": set()}

    feature_rows, feature_columns = build_feature_rows(
        pair_rows,
        phage_to_profile_presence,
        bacteria_to_host_features,
        profiles,
        associations,
    )

    pairwise_column = associations[0].pairwise_column
    by_pair_id = {row["pair_id"]: row for row in feature_rows}
    direct_column = next(
        profile.direct_column for profile in profiles if profile.profile_id == feature_to_profile_id["RBP_PHROG_136"]
    )
    assert pairwise_column in feature_columns
    assert by_pair_id["B1__P1"][pairwise_column] == 0.6
    assert by_pair_id["B2__P1"][pairwise_column] == 0.0
    assert by_pair_id["B1__P2"][pairwise_column] == 0.0
    assert by_pair_id["B1__P1"][direct_column] == 1


def test_build_sanity_check_rows_tracks_curated_vs_pharokka_presence() -> None:
    rows = build_sanity_check_rows(
        panel_phages=["P1", "P2"],
        curated_summary={
            "P1": {"curated_rbp_count": 2, "curated_types": {"fiber", "spike"}, "curated_has_rbp": 1},
            "P2": {"curated_rbp_count": 0, "curated_types": set(), "curated_has_rbp": 0},
        },
        pharokka_summary={
            "P1": {"pharokka_rbp_gene_count": 3, "pharokka_has_rbp": 1},
            "P2": {"pharokka_rbp_gene_count": 1, "pharokka_has_rbp": 1},
        },
        phage_to_profile_presence={
            "P1": {"profile_001": 1, "profile_002": 1},
            "P2": {"profile_001": 0, "profile_002": 1},
        },
    )

    by_phage = {row["phage"]: row for row in rows}
    assert by_phage["P1"]["agreement_has_rbp"] == 1
    assert by_phage["P1"]["collapsed_profile_count"] == 2
    assert by_phage["P2"]["agreement_has_rbp"] == 0


def test_main_writes_mechanistic_feature_outputs(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.csv"
    cached_dir = tmp_path / "cached"
    omp_path = tmp_path / "omp.tsv"
    lps_primary_path = tmp_path / "lps_primary.tsv"
    lps_supplemental_path = tmp_path / "lps_supplemental.tsv"
    omp_enrichment_path = tmp_path / "omp_enrichment.csv"
    lps_enrichment_path = tmp_path / "lps_enrichment.csv"
    rbp_list_path = tmp_path / "RBP_list.csv"
    output_dir = tmp_path / "out"

    cached_dir.mkdir()

    label_path.write_text("bacteria,phage\nB1,P1\nB2,P1\nB1,P2\n", encoding="utf-8")
    (cached_dir / "P1_cds_final_merged_output.tsv").write_text(
        MERGED_TSV_HEADER
        + "\n"
        + _make_merged_row("P1_G1", "136", "tail fiber protein")
        + "\n"
        + _make_merged_row("P1_G2", "15437", "tail fiber protein")
        + "\n",
        encoding="utf-8",
    )
    (cached_dir / "P2_cds_final_merged_output.tsv").write_text(
        MERGED_TSV_HEADER
        + "\n"
        + _make_merged_row("P2_G1", "136", "tail fiber protein")
        + "\n"
        + _make_merged_row("P2_G2", "15437", "tail fiber protein")
        + "\n",
        encoding="utf-8",
    )
    omp_path.write_text(
        (
            "bacteria\tBTUB\tFADL\tFHUA\tLAMB\tLPTD\tNFRA\tOMPA\tOMPC\tOMPF\tTOLC\tTSX\tYNCD\n"
            "B1\t\t\t\t\t\t\t\t99_1\t\t\t\t\n"
            "B2\t\t\t\t\t\t\t\t99_2\t\t\t\t\n"
        ),
        encoding="utf-8",
    )
    lps_primary_path.write_text("bacteria\tLPS_type\nB1\tR1\nB2\tR2\n", encoding="utf-8")
    lps_supplemental_path.write_text("Strain\tLPS_type\n", encoding="utf-8")
    omp_enrichment_path.write_text(
        "phage_feature,host_feature,lysis_rate_diff,significant\nRBP_PHROG_136,OMPC_99_1,0.1,False\n",
        encoding="utf-8",
    )
    lps_enrichment_path.write_text(
        "phage_feature,host_feature,lysis_rate_diff,significant\n"
        "RBP_PHROG_136,LPS_R1,0.6,True\n"
        "RBP_PHROG_15437,LPS_R1,0.6,True\n",
        encoding="utf-8",
    )
    rbp_list_path.write_text(
        "phage;Morphotype;Family;Subfamily;Genus;RBP;type\n"
        "P1;Myoviridae;Other;Other;X;P1_gene;fiber\n"
        "P2;Myoviridae;Other;Other;X;P2_gene;spike\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--label-path",
            str(label_path),
            "--cached-annotations-dir",
            str(cached_dir),
            "--omp-path",
            str(omp_path),
            "--lps-primary-path",
            str(lps_primary_path),
            "--lps-supplemental-path",
            str(lps_supplemental_path),
            "--omp-enrichment-path",
            str(omp_enrichment_path),
            "--lps-enrichment-path",
            str(lps_enrichment_path),
            "--rbp-list-path",
            str(rbp_list_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    feature_rows = list(
        csv.DictReader((output_dir / "mechanistic_rbp_receptor_features_v1.csv").open(encoding="utf-8"))
    )
    profile_rows = list(csv.DictReader((output_dir / "mechanistic_rbp_profile_metadata_v1.csv").open(encoding="utf-8")))
    sanity_rows = list(csv.DictReader((output_dir / "mechanistic_rbp_sanity_check_v1.csv").open(encoding="utf-8")))

    assert len(feature_rows) == 3
    assert len(profile_rows) == 1
    duplicate_profile = next(row for row in profile_rows if row["member_features"] == "RBP_PHROG_136|RBP_PHROG_15437")
    pairwise_column = f"tl03_pair_{duplicate_profile['profile_id']}_x_lps_r1_weight"
    assert any(float(row[pairwise_column]) == 0.6 for row in feature_rows if row["pair_id"] == "B1__P1")
    assert any(row["agreement_has_rbp"] == "1" for row in sanity_rows)
