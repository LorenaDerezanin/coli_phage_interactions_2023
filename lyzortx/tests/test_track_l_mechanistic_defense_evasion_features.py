import json
import csv
from pathlib import Path

import numpy as np

from lyzortx.pipeline.track_l.steps import build_mechanistic_defense_evasion_features as defense_module
from lyzortx.pipeline.track_l.steps.build_mechanistic_defense_evasion_features import (
    EXPERIMENTAL_STATUS,
    build_feature_rows,
    collapse_duplicate_profiles,
    collapse_significant_associations,
    main,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import _sha256


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


def test_build_feature_rows_zeroes_pairwise_weight_when_host_lacks_defense_feature() -> None:
    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        ["ANTIDEF_PHROG_11", "ANTIDEF_PHROG_22"],
        np.array([[1, 1], [0, 0]], dtype=np.int8),
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
    ]
    bacteria_to_defense_features = {"B1": {"defense_BREX_I"}, "B2": set()}

    feature_rows, _ = build_feature_rows(
        pair_rows,
        phage_to_profile_presence,
        bacteria_to_defense_features,
        profiles,
        associations,
    )

    pairwise_column = associations[0].pairwise_column
    by_pair_id = {row["pair_id"]: row for row in feature_rows}
    assert by_pair_id["B1__P1"][pairwise_column] == 0.4
    assert by_pair_id["B2__P1"][pairwise_column] == 0.0


def test_main_writes_mechanistic_defense_outputs(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.csv"
    cached_dir = tmp_path / "cached"
    defense_path = tmp_path / "defense.csv"
    antidef_enrichment_path = tmp_path / "antidef_enrichment.csv"
    split_path = tmp_path / "st03_split_assignments.csv"
    output_dir = tmp_path / "out"

    cached_dir.mkdir()

    label_path.write_text(
        (
            "bacteria,phage\n"
            "B1,P1\nB2,P1\nB3,P1\nB4,P1\nB5,P1\nB6,P1\nB7,P1\nB8,P1\n"
            "B1,P2\nB2,P2\nB3,P2\nB4,P2\nB5,P2\nB6,P2\nB7,P2\nB8,P2\n"
        ),
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
        "bacteria;BREX_I;RM_Type_I\nB1;1;0\nB2;1;0\nB3;1;0\nB4;1;0\nB5;1;0\nB6;0;1\nB7;1;0\nB8;0;1\n",
        encoding="utf-8",
    )
    antidef_enrichment_path.write_text(
        "phage_feature,host_feature,lysis_rate_diff,significant\n"
        "ANTIDEF_PHROG_11,defense_BREX_I,0.4,True\n"
        "ANTIDEF_PHROG_22,defense_BREX_I,0.4,True\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "pair_id,bacteria,split_holdout,split_cv5_fold\n"
        "B1__P1,B1,train_non_holdout,0\n"
        "B2__P1,B2,train_non_holdout,1\n"
        "B3__P1,B3,train_non_holdout,2\n"
        "B4__P1,B4,train_non_holdout,3\n"
        "B5__P1,B5,train_non_holdout,2\n"
        "B6__P1,B6,train_non_holdout,4\n"
        "B7__P1,B7,holdout_test,-1\n"
        "B8__P1,B8,holdout_test,-1\n"
        "B1__P2,B1,train_non_holdout,0\n"
        "B2__P2,B2,train_non_holdout,1\n"
        "B3__P2,B3,train_non_holdout,2\n"
        "B4__P2,B4,train_non_holdout,3\n"
        "B5__P2,B5,train_non_holdout,2\n"
        "B6__P2,B6,train_non_holdout,4\n"
        "B7__P2,B7,holdout_test,-1\n"
        "B8__P2,B8,holdout_test,-1\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "step": "TL02_enrichment_analysis",
                "holdout_exclusion": {
                    "split_assignments": {"path": str(split_path), "sha256": _sha256(split_path)},
                    "excluded_holdout_bacteria_ids": ["B7", "B8"],
                    "excluded_holdout_bacteria_count": 2,
                },
                "outputs": {
                    "antidef_phrog_x_defense_subtype": {
                        "path": str(antidef_enrichment_path),
                        "sha256": _sha256(antidef_enrichment_path),
                    }
                },
            }
        ),
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
            "--st03-split-assignments-path",
            str(split_path),
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

    assert len(feature_rows) == 12
    assert len(profile_rows) == 1
    duplicate_profile = next(
        row for row in profile_rows if row["member_features"] == "ANTIDEF_PHROG_11|ANTIDEF_PHROG_22"
    )
    pairwise_column = f"tl04_pair_{duplicate_profile['profile_id']}_x_defense_brex_i_weight"
    feature_rows_by_pair = {row["pair_id"]: row for row in feature_rows}
    assert "B7__P1" not in feature_rows_by_pair
    assert "B8__P1" not in feature_rows_by_pair
    assert "B7__P2" not in feature_rows_by_pair
    assert "B8__P2" not in feature_rows_by_pair
    assert float(feature_rows_by_pair["B1__P1"][pairwise_column]) == 0.4
    assert float(feature_rows_by_pair["B6__P1"][pairwise_column]) == 0.0
    assert {row["experimental_status"] for row in profile_rows} == {EXPERIMENTAL_STATUS}
    assert {row["experimental_status"] for row in metadata_rows} == {EXPERIMENTAL_STATUS}
    manifest = json.loads((output_dir / "mechanistic_defense_evasion_manifest_v1.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["excluded_holdout_bacteria_ids"] == ["B7", "B8"]
    assert manifest["provenance"]["split_assignments"]["path"] == str(split_path)
    assert manifest["holdout_exclusion"]["excluded_pair_rows"] == 4
    assert manifest["outputs"]["feature_csv_sha256"] == _sha256(
        output_dir / "mechanistic_defense_evasion_features_v1.csv"
    )


def test_main_reports_holdout_exclusion_independently_of_feature_row_drops(monkeypatch, tmp_path: Path) -> None:
    label_path = tmp_path / "labels.csv"
    cached_dir = tmp_path / "cached"
    defense_path = tmp_path / "defense.csv"
    antidef_enrichment_path = tmp_path / "antidef_enrichment.csv"
    split_path = tmp_path / "st03_split_assignments.csv"
    output_dir = tmp_path / "out"

    cached_dir.mkdir()
    label_path.write_text("bacteria,phage\nB1,P1\nB2,P1\nB1,P2\n", encoding="utf-8")
    defense_path.write_text("bacteria;BREX_I\nB1;1\n", encoding="utf-8")
    antidef_enrichment_path.write_text(
        "phage_feature,host_feature,lysis_rate_diff,significant\nANTIDEF_PHROG_11,defense_BREX_I,0.4,True\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "pair_id,bacteria,split_holdout,split_cv5_fold\n"
        "B1__P1,B1,train_non_holdout,0\n"
        "B2__P1,B2,holdout_test,-1\n"
        "B1__P2,B1,train_non_holdout,1\n",
        encoding="utf-8",
    )

    fake_provenance = {
        "manifest_path": tmp_path / "manifest.json",
        "split_assignments_path": split_path,
        "split_assignments_sha256": "split-sha",
        "holdout_bacteria_ids": ["B2"],
        "enrichment_inputs": {},
        "enrichment_manifest_sha256": "manifest-sha",
    }

    def fake_read_delimited_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
        if path == label_path:
            return [
                {"bacteria": "B1", "phage": "P1"},
                {"bacteria": "B2", "phage": "P1"},
                {"bacteria": "B1", "phage": "P2"},
            ]
        if path == antidef_enrichment_path:
            return [
                {
                    "phage_feature": "ANTIDEF_PHROG_11",
                    "host_feature": "defense_BREX_I",
                    "lysis_rate_diff": "0.4",
                    "significant": "True",
                }
            ]
        raise AssertionError(f"unexpected path: {path}")

    real_build_feature_rows = defense_module.build_feature_rows

    def fake_build_feature_rows(*args, **kwargs):
        feature_rows, feature_columns = real_build_feature_rows(*args, **kwargs)
        return feature_rows[:-1], feature_columns

    monkeypatch.setattr(defense_module, "ensure_default_label_path", lambda *_: None)
    monkeypatch.setattr(defense_module, "ensure_default_tl02_output", lambda *_: None)
    monkeypatch.setattr(defense_module, "load_tl02_holdout_clean_provenance", lambda *_: fake_provenance)
    monkeypatch.setattr(defense_module, "read_delimited_rows", fake_read_delimited_rows)
    monkeypatch.setattr(
        defense_module,
        "load_pharokka_phrog_matrices",
        lambda *_: (None, None, np.array([[1], [0]], dtype=np.int8), ["11"]),
    )
    monkeypatch.setattr(
        defense_module, "load_defense_host_matrix", lambda *_: (np.array([[1]], dtype=np.int8), ["BREX_I"])
    )
    monkeypatch.setattr(defense_module, "build_feature_rows", fake_build_feature_rows)

    exit_code = defense_module.main(
        [
            "--label-path",
            str(label_path),
            "--cached-annotations-dir",
            str(cached_dir),
            "--defense-path",
            str(defense_path),
            "--antidef-enrichment-path",
            str(antidef_enrichment_path),
            "--st03-split-assignments-path",
            str(split_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "mechanistic_defense_evasion_manifest_v1.json").read_text(encoding="utf-8"))
    assert manifest["holdout_exclusion"]["excluded_pair_rows"] == 1
