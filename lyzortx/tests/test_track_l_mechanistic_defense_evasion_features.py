import csv
from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_l.steps.build_mechanistic_defense_evasion_features import (
    EXPERIMENTAL_STATUS,
    build_feature_rows,
    collapse_duplicate_profiles,
    collapse_significant_associations,
    main,
)


def test_collapse_duplicate_profiles_groups_identical_antidef_patterns() -> None:
    feature_names = ["ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22", "ANTIDEF_PHROG_33"]
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(feature_names, matrix)

    assert collapsed_matrix.shape == (3, 2)
    assert len(profiles) == 2
    assert feature_to_profile_id["ANTIDEF_PHROG_11"] == feature_to_profile_id["ANTIDEF_PHROG_22"]
    duplicate_profile = next(profile for profile in profiles if len(profile.member_features) == 2)
    assert duplicate_profile.direct_column.startswith("tl04_phage_antidef_")


def test_collapse_significant_associations_merges_duplicate_profiles() -> None:
    _, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        ["ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22", "ANTIDEF_PHROG_33"],
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.int8),
    )
    profile_by_id = {profile.profile_id: profile for profile in profiles}

    enrichment_rows = [
        {
            "phage_feature": "ANTIDEF_PHROG_11",
            "host_feature": "defense_BREX_I",
            "lysis_rate_diff": "0.4",
            "significant": "True",
        },
        {
            "phage_feature": "ANTIDEF_PHROG_22",
            "host_feature": "defense_BREX_I",
            "lysis_rate_diff": "0.4",
            "significant": "True",
        },
        {
            "phage_feature": "ANTIDEF_PHROG_33",
            "host_feature": "defense_RM_Type_I",
            "lysis_rate_diff": "0.2",
            "significant": "True",
        },
    ]

    associations = collapse_significant_associations(enrichment_rows, feature_to_profile_id, profile_by_id)
    by_host_feature = {association.host_feature: association for association in associations}

    assert len(associations) == 2
    assert by_host_feature["defense_BREX_I"].weight == 0.4
    assert by_host_feature["defense_BREX_I"].pairwise_column.endswith("defense_brex_i_weight")


def test_build_feature_rows_emits_defense_evasion_blocks() -> None:
    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        ["ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22", "ANTIDEF_PHROG_33"],
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
                    "phage_feature": "ANTIDEF_PHROG_11",
                    "host_feature": "defense_BREX_I",
                    "lysis_rate_diff": "0.4",
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
    bacteria_to_defense_features = {"B1": {"defense_BREX_I"}, "B2": set()}

    feature_rows, feature_columns = build_feature_rows(
        pair_rows,
        phage_to_profile_presence,
        bacteria_to_defense_features,
        profiles,
        associations,
    )

    pairwise_column = associations[0].pairwise_column
    by_pair_id = {row["pair_id"]: row for row in feature_rows}
    direct_column = next(
        profile.direct_column for profile in profiles if profile.profile_id == feature_to_profile_id["ANTIDEF_PHROG_11"]
    )
    assert pairwise_column in feature_columns
    assert by_pair_id["B1__P1"][pairwise_column] == 0.4
    assert by_pair_id["B2__P1"][pairwise_column] == 0.0
    assert by_pair_id["B1__P2"][pairwise_column] == 0.0
    assert by_pair_id["B1__P1"][direct_column] == 1


def test_main_writes_mechanistic_defense_outputs(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.csv"
    cached_dir = tmp_path / "cached"
    defense_path = tmp_path / "defense.csv"
    antidef_enrichment_path = tmp_path / "antidef_enrichment.csv"
    output_dir = tmp_path / "out"

    cached_dir.mkdir()

    label_path.write_text(
        "bacteria,phage\nB1,P1\nB2,P1\nB3,P1\nB4,P1\nB5,P1\nB1,P2\nB2,P2\nB3,P2\nB4,P2\nB5,P2\n",
        encoding="utf-8",
    )
    (cached_dir / "P1_cds_final_merged_output.tsv").write_text(
        "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
        "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\t"
        "custom_hmm_bitscore\tcustom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\t"
        "vfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\t"
        "CARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
        "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table\n"
        "P1_G1\t1\t100\t+\tcontig_1\t-10.0\t11\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
        "No_custom_hmm\tNo_custom_hmm\t11\tPHANOTATE\tCDS\t#838383\tanti-CRISPR protein\tother\t"
        "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
        "None\tNone\tNone\tNone\n"
        "P1_G2\t1\t100\t+\tcontig_1\t-10.0\t22\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
        "No_custom_hmm\tNo_custom_hmm\t22\tPHANOTATE\tCDS\t#838383\tDNA methyltransferase\tother\t"
        "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
        "None\tNone\tNone\tNone\n",
        encoding="utf-8",
    )
    (cached_dir / "P2_cds_final_merged_output.tsv").write_text(
        "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
        "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\t"
        "custom_hmm_bitscore\tcustom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\t"
        "vfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\t"
        "CARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
        "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table\n"
        "P2_G1\t1\t100\t+\tcontig_1\t-10.0\t11\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
        "No_custom_hmm\tNo_custom_hmm\t11\tPHANOTATE\tCDS\t#838383\tanti-CRISPR protein\tother\t"
        "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
        "None\tNone\tNone\tNone\n"
        "P2_G2\t1\t100\t+\tcontig_1\t-10.0\t22\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
        "No_custom_hmm\tNo_custom_hmm\t22\tPHANOTATE\tCDS\t#838383\tDNA methyltransferase\tother\t"
        "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
        "None\tNone\tNone\tNone\n",
        encoding="utf-8",
    )
    defense_path.write_text(
        "bacteria;BREX_I;RM_Type_I\nB1;1;0\nB2;1;0\nB3;1;0\nB4;1;0\nB5;1;0\n",
        encoding="utf-8",
    )
    antidef_enrichment_path.write_text(
        "phage_feature,host_feature,lysis_rate_diff,significant\n"
        "ANTIDEF_PHROG_11,defense_BREX_I,0.4,True\n"
        "ANTIDEF_PHROG_22,defense_BREX_I,0.4,True\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--label-path",
            str(label_path),
            "--cached-annotations-dir",
            str(cached_dir),
            "--defense-path",
            str(defense_path),
            "--antidef-enrichment-path",
            str(antidef_enrichment_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    feature_rows = list(
        csv.DictReader((output_dir / "mechanistic_defense_evasion_features_v1.csv").open(encoding="utf-8"))
    )
    profile_rows = list(
        csv.DictReader((output_dir / "mechanistic_defense_evasion_profile_metadata_v1.csv").open(encoding="utf-8"))
    )
    metadata_rows = list(
        csv.DictReader((output_dir / "mechanistic_defense_evasion_feature_metadata_v1.csv").open(encoding="utf-8"))
    )

    assert len(feature_rows) == 10
    assert len(profile_rows) == 1
    duplicate_profile = next(
        row for row in profile_rows if row["member_features"] == "ANTIDEF_PHROG_11|ANTIDEF_PHROG_22"
    )
    pairwise_column = f"tl04_pair_{duplicate_profile['profile_id']}_x_defense_brex_i_weight"
    assert any(float(row[pairwise_column]) == 0.4 for row in feature_rows if row["pair_id"] == "B1__P1")
    assert {row["experimental_status"] for row in profile_rows} == {EXPERIMENTAL_STATUS}
    assert {row["experimental_status"] for row in metadata_rows} == {EXPERIMENTAL_STATUS}
