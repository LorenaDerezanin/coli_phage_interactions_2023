from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_l.steps.build_tl17_phage_preprocessor import (
    SUMMARY_COLUMNS,
    build_candidate_audit_rows,
    build_panel_feature_rows,
    build_profile_metadata_rows,
    build_projection_validation_rows,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import CollapsedProfile


def test_build_candidate_audit_rows_marks_rbp_profiles_as_selected() -> None:
    rows = build_candidate_audit_rows(
        kmer_feature_columns=("phage_gc_content", "phage_genome_tetra_svd_00"),
        rbp_profiles=[
            CollapsedProfile(
                profile_id="profile_001",
                member_features=("RBP_PHROG_11",),
                representative_feature="RBP_PHROG_11",
                carrier_count=2,
                direct_column="tl17_phage_rbp_profile_001_present",
            )
        ],
        anti_def_profile_count=3,
        rbp_feature_rows=[
            {
                "phage": "P1",
                "tl17_phage_rbp_profile_001_present": 1,
                SUMMARY_COLUMNS["profile_count_column"]: 1,
                SUMMARY_COLUMNS["gene_count_column"]: 2,
                SUMMARY_COLUMNS["unique_phrog_count_column"]: 1,
            },
            {
                "phage": "P2",
                "tl17_phage_rbp_profile_001_present": 0,
                SUMMARY_COLUMNS["profile_count_column"]: 0,
                SUMMARY_COLUMNS["gene_count_column"]: 0,
                SUMMARY_COLUMNS["unique_phrog_count_column"]: 0,
            },
        ],
    )

    chosen = next(row for row in rows if row["candidate_block_id"] == "tl17_rbp_phrog_profiles")
    assert chosen["chosen_for_tl17"] == 1
    assert chosen["panel_phage_coverage"] == 1


def test_build_panel_feature_rows_and_projection_validation_round_trip(tmp_path: Path) -> None:
    for phage_name, annot, phrog in (("P1", "tail fiber protein", "11"), ("P2", "tail spike protein", "22")):
        (tmp_path / f"{phage_name}_cds_final_merged_output.tsv").write_text(
            (
                "gene\tstart\tstop\tstrand\tcontig\tscore\tmmseqs_phrog\tmmseqs_alnScore\tmmseqs_seqIdentity\t"
                "mmseqs_eVal\tpyhmmer_phrog\tpyhmmer_bitscore\tpyhmmer_evalue\tcustom_hmm_id\tcustom_hmm_bitscore\t"
                "custom_hmm_evalue\tphrog\tMethod\tRegion\tcolor\tannot\tcategory\tvfdb_hit\tvfdb_alnScore\t"
                "vfdb_seqIdentity\tvfdb_eVal\tvfdb_short_name\tvfdb_description\tvfdb_species\tCARD_hit\t"
                "CARD_alnScore\tCARD_seqIdentity\tCARD_eVal\tCARD_species\tARO_Accession\tCARD_short_name\t"
                "Protein_Accession\tDNA_Accession\tAMR_Gene_Family\tDrug_Class\tResistance_Mechanism\ttransl_table\n"
                f"G1\t1\t100\t+\tcontig_1\t-1\t{phrog}\t50\t0.6\t1e-5\tNo_PHROG\tNo_PHROG\tNo_PHROG\tNo_custom_hmm\t"
                f"No_custom_hmm\tNo_custom_hmm\t{phrog}\tPHANOTATE\tCDS\t#838383\t{annot}\ttail\t"
                "None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\t"
                "None\tNone\tNone\tNone\n"
            ),
            encoding="utf-8",
        )

    profiles = [
        CollapsedProfile(
            profile_id="profile_001",
            member_features=("RBP_PHROG_11",),
            representative_feature="RBP_PHROG_11",
            carrier_count=1,
            direct_column="tl17_phage_rbp_profile_001_present",
        ),
        CollapsedProfile(
            profile_id="profile_002",
            member_features=("RBP_PHROG_22",),
            representative_feature="RBP_PHROG_22",
            carrier_count=1,
            direct_column="tl17_phage_rbp_profile_002_present",
        ),
    ]
    collapsed_matrix = np.asarray([[1, 0], [0, 1]], dtype=np.int8)

    panel_rows = build_panel_feature_rows(
        phages=("P1", "P2"),
        profiles=profiles,
        collapsed_matrix=collapsed_matrix,
        cached_annotations_dir=tmp_path,
    )
    profile_rows = build_profile_metadata_rows(profiles)
    runtime_payload = {
        "profiles": [
            {
                "profile_id": row["profile_id"],
                "direct_column": row["direct_column"],
                "member_features": row["member_features"].split("|"),
            }
            for row in profile_rows
        ],
        "summary_columns": dict(SUMMARY_COLUMNS),
    }

    validation_rows, validation_summary_rows = build_projection_validation_rows(
        phage_feature_rows=panel_rows,
        runtime_payload=runtime_payload,
        cached_annotations_dir=tmp_path,
    )

    assert all(row["exact_match"] == 1 for row in validation_rows)
    assert validation_summary_rows == [
        {
            "panel_phage_count": 2,
            "exact_match_count": 2,
            "nonzero_profile_count_phages": 2,
        }
    ]
