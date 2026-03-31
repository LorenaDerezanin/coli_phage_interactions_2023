from pathlib import Path

from lyzortx.pipeline.track_l.steps.deployable_tl04_runtime import (
    build_defense_host_feature_lookup,
    build_direct_feature_values,
    build_pairwise_feature_values,
    build_profile_presence,
    build_tl04_runtime_payload,
    extract_antidef_feature_names,
    parse_tl04_runtime_payload,
)


def test_build_defense_host_feature_lookup_maps_enrichment_names_to_projected_columns() -> None:
    defense_mask = {
        "retained_subtype_columns": ["BREX_I", "RM_Type_I"],
        "retained_feature_columns": ["host_defense_subtype_brex_i", "host_defense_subtype_rm_type_i"],
    }

    observed = build_defense_host_feature_lookup(defense_mask)

    assert observed == {
        "defense_BREX_I": "host_defense_subtype_brex_i",
        "defense_RM_Type_I": "host_defense_subtype_rm_type_i",
    }


def test_extract_antidef_feature_names_ignores_no_phrog_and_non_antidef_hits(tmp_path: Path) -> None:
    annotation_path = tmp_path / "P1_cds_final_merged_output.tsv"
    annotation_path.write_text(
        (
            "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\t"
            "mmseqs_seqIdentity\tmmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\t"
            "custom_hmm_bitscore\tcustom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\t"
            "vfdb_alnScore\tvfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\t"
            "CARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
            "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table\n"
            "G1\t1\t100\t+\tcontig_1\t-1\t11\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
            "No_custom_hmm\tNo_custom_hmm\t11\tPHANOTATE\tCDS\t#838383\tanti-CRISPR protein\tother\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\n"
            "G2\t1\t100\t+\tcontig_1\t-1\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_PHROGs_HMM\tNo_PHROGs_HMM\t"
            "No_PHROGs_HMM\tNo_custom_hmm\tNo_custom_hmm\tNo_custom_hmm\tNo_PHROG\tPHANOTATE\tCDS\t#838383\t"
            "DNA methyltransferase\tother\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n"
            "G3\t1\t100\t+\tcontig_1\t-1\t42\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
            "No_custom_hmm\tNo_custom_hmm\t42\tPHANOTATE\tCDS\t#838383\ttail fiber protein\ttail\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\n"
        ),
        encoding="utf-8",
    )

    observed = extract_antidef_feature_names(annotation_path)

    assert observed == {"ANTIDEF_PHROG_11"}


def test_profile_presence_uses_any_member_of_collapsed_profile() -> None:
    payload = {
        "profiles": [
            {
                "profile_id": "profile_001",
                "direct_column": "tl04_phage_antidef_profile_001",
                "member_features": ["ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22"],
            }
        ],
        "associations": [],
    }
    profiles, _ = parse_tl04_runtime_payload(payload)

    presence = build_profile_presence({"ANTIDEF_PHROG_22"}, profiles)
    direct_values = build_direct_feature_values(presence, profiles)

    assert presence == {"profile_001": 1}
    assert direct_values == {"tl04_phage_antidef_profile_001": 1}


def test_build_pairwise_feature_values_uses_projected_host_column_lookup() -> None:
    payload = {
        "profiles": [
            {
                "profile_id": "profile_001",
                "direct_column": "tl04_phage_antidef_profile_001",
                "member_features": ["ANTIDEF_PHROG_11"],
            }
        ],
        "associations": [
            {
                "profile_id": "profile_001",
                "host_feature": "defense_BREX_I",
                "host_feature_column": "host_defense_subtype_brex_i",
                "pairwise_column": "tl04_pair_profile_001_x_defense_brex_i_weight",
                "weight": 0.4,
            }
        ],
    }
    profiles, associations = parse_tl04_runtime_payload(payload)
    presence = build_profile_presence({"ANTIDEF_PHROG_11"}, profiles)

    observed = build_pairwise_feature_values(
        host_row={"bacteria": "B1", "host_defense_subtype_brex_i": 1},
        profile_presence=presence,
        associations=associations,
    )

    assert observed == {"tl04_pair_profile_001_x_defense_brex_i_weight": 0.4}


def test_build_tl04_runtime_payload_maps_metadata_to_runtime_contract() -> None:
    payload = build_tl04_runtime_payload(
        profile_rows=[
            {
                "profile_id": "profile_001",
                "direct_column": "tl04_phage_antidef_profile_001",
                "member_features": "ANTIDEF_PHROG_11|ANTIDEF_PHROG_22",
            }
        ],
        metadata_rows=[
            {
                "column_name": "tl04_pair_profile_001_x_defense_brex_i_weight",
                "block_type": "pairwise_defense_evasion",
                "profile_id": "profile_001",
                "host_feature": "defense_BREX_I",
                "member_features": "ANTIDEF_PHROG_11|ANTIDEF_PHROG_22",
                "weight": "0.4",
            }
        ],
        defense_mask={
            "retained_subtype_columns": ["BREX_I"],
            "retained_feature_columns": ["host_defense_subtype_brex_i"],
        },
    )

    profiles, associations = parse_tl04_runtime_payload(payload)

    assert profiles[0].member_features == ("ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22")
    assert associations[0].host_feature_column == "host_defense_subtype_brex_i"
