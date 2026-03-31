from pathlib import Path

from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import (
    build_direct_feature_values,
    build_profile_presence,
    build_tl17_runtime_payload,
    extract_rbp_runtime_inputs,
    parse_tl17_runtime_payload,
)


def test_extract_rbp_runtime_inputs_ignores_no_phrog_and_non_rbp_hits(tmp_path: Path) -> None:
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
            "No_custom_hmm\tNo_custom_hmm\t11\tPHANOTATE\tCDS\t#838383\ttail fiber protein\ttail\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\n"
            "G2\t1\t100\t+\tcontig_1\t-1\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_PHROGs_HMM\tNo_PHROGs_HMM\t"
            "No_PHROGs_HMM\tNo_custom_hmm\tNo_custom_hmm\tNo_custom_hmm\tNo_PHROG\tPHANOTATE\tCDS\t#838383\t"
            "host specificity protein\ttail\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n"
            "G3\t1\t100\t+\tcontig_1\t-1\t42\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
            "No_custom_hmm\tNo_custom_hmm\t42\tPHANOTATE\tCDS\t#838383\tDNA methyltransferase\tother\t"
            "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
            "None\tNone\tNone\tNone\n"
        ),
        encoding="utf-8",
    )

    features, rbp_gene_count = extract_rbp_runtime_inputs(annotation_path)

    assert features == {"RBP_PHROG_11"}
    assert rbp_gene_count == 2


def test_build_direct_feature_values_uses_any_member_of_collapsed_profile() -> None:
    payload = {
        "profiles": [
            {
                "profile_id": "profile_001",
                "direct_column": "tl17_phage_rbp_profile_001_present",
                "member_features": ["RBP_PHROG_11", "RBP_PHROG_22"],
            }
        ],
        "summary_columns": {
            "profile_count_column": "tl17_phage_rbp_profile_count",
            "gene_count_column": "tl17_phage_rbp_gene_count",
            "unique_phrog_count_column": "tl17_phage_rbp_unique_phrog_count",
        },
    }
    profiles, summary = parse_tl17_runtime_payload(payload)

    profile_presence = build_profile_presence({"RBP_PHROG_22"}, profiles)
    values = build_direct_feature_values(
        present_features={"RBP_PHROG_22"},
        rbp_gene_count=3,
        profile_presence=profile_presence,
        profiles=profiles,
        summary=summary,
    )

    assert values == {
        "tl17_phage_rbp_profile_001_present": 1,
        "tl17_phage_rbp_profile_count": 1,
        "tl17_phage_rbp_gene_count": 3,
        "tl17_phage_rbp_unique_phrog_count": 1,
    }


def test_build_tl17_runtime_payload_round_trips_profiles_and_summary_columns() -> None:
    payload = build_tl17_runtime_payload(
        profile_rows=[
            {
                "profile_id": "profile_001",
                "direct_column": "tl17_phage_rbp_profile_001_present",
                "member_features": "RBP_PHROG_11|RBP_PHROG_22",
            }
        ],
        summary_columns={
            "profile_count_column": "tl17_phage_rbp_profile_count",
            "gene_count_column": "tl17_phage_rbp_gene_count",
            "unique_phrog_count_column": "tl17_phage_rbp_unique_phrog_count",
        },
    )

    profiles, summary = parse_tl17_runtime_payload(payload)

    assert profiles[0].member_features == ("RBP_PHROG_11", "RBP_PHROG_22")
    assert summary.profile_count_column == "tl17_phage_rbp_profile_count"
    assert summary.gene_count_column == "tl17_phage_rbp_gene_count"
    assert summary.unique_phrog_count_column == "tl17_phage_rbp_unique_phrog_count"
