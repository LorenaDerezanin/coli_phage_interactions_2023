import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_e.steps.build_rbp_receptor_compatibility_feature_block import (
    LookupEntry,
    build_feature_rows,
    build_host_target_states,
    build_training_variant_indices,
    load_lookup,
    main,
    merge_pair_rows,
)


def test_load_lookup_prefers_genus_and_validates_targets(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.csv"
    lookup_path.write_text(
        (
            "match_level,taxon,target_receptors,target_family,evidence_note\n"
            "genus,Lambdavirus,LAMB,protein,exact override\n"
            "subfamily,Tevenvirinae,OMPC|LPS_CORE,mixed,fallback\n"
        ),
        encoding="utf-8",
    )

    lookup_by_genus, lookup_by_subfamily = load_lookup(lookup_path)

    assert lookup_by_genus["Lambdavirus"] == LookupEntry(
        match_level="genus",
        taxon="Lambdavirus",
        target_receptors=("LAMB",),
        target_family="protein",
        evidence_note="exact override",
    )
    assert lookup_by_subfamily["Tevenvirinae"].target_receptors == ("OMPC", "LPS_CORE")


def test_build_feature_rows_uses_leakage_safe_training_views() -> None:
    st02_rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "phage_genus": "Tequatrovirus",
            "phage_subfamily": "Tevenvirinae",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_lps_type": "R1",
            "host_o_type": "O1",
            "host_abc_serotype": "",
            "host_capsule_abc": "0",
            "host_capsule_groupiv_e": "0",
            "host_capsule_groupiv_e_stricte": "0",
            "host_capsule_groupiv_s": "0",
            "host_capsule_wzy_stricte": "0",
        },
        {
            "pair_id": "B2__P1",
            "bacteria": "B2",
            "phage": "P1",
            "phage_genus": "Tequatrovirus",
            "phage_subfamily": "Tevenvirinae",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_lps_type": "R2",
            "host_o_type": "O2",
            "host_abc_serotype": "",
            "host_capsule_abc": "0",
            "host_capsule_groupiv_e": "0",
            "host_capsule_groupiv_e_stricte": "0",
            "host_capsule_groupiv_s": "0",
            "host_capsule_wzy_stricte": "0",
        },
        {
            "pair_id": "B3__P1",
            "bacteria": "B3",
            "phage": "P1",
            "phage_genus": "Tequatrovirus",
            "phage_subfamily": "Tevenvirinae",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_lps_type": "R3",
            "host_o_type": "O3",
            "host_abc_serotype": "",
            "host_capsule_abc": "0",
            "host_capsule_groupiv_e": "0",
            "host_capsule_groupiv_e_stricte": "0",
            "host_capsule_groupiv_s": "0",
            "host_capsule_wzy_stricte": "0",
        },
        {
            "pair_id": "B4__P2",
            "bacteria": "B4",
            "phage": "P2",
            "phage_genus": "Vectrevirus",
            "phage_subfamily": "Molineuxvirinae",
            "label_hard_any_lysis": "0",
            "include_in_training": "1",
            "host_lps_type": "",
            "host_o_type": "",
            "host_abc_serotype": "K1",
            "host_capsule_abc": "1",
            "host_capsule_groupiv_e": "0",
            "host_capsule_groupiv_e_stricte": "0",
            "host_capsule_groupiv_s": "0",
            "host_capsule_wzy_stricte": "0",
        },
    ]
    split_rows = [
        {"pair_id": "B1__P1", "split_holdout": "train_non_holdout", "split_cv5_fold": "0"},
        {"pair_id": "B2__P1", "split_holdout": "train_non_holdout", "split_cv5_fold": "1"},
        {"pair_id": "B3__P1", "split_holdout": "holdout_test", "split_cv5_fold": "-1"},
        {"pair_id": "B4__P2", "split_holdout": "train_non_holdout", "split_cv5_fold": "0"},
    ]
    receptor_rows = [
        {
            "bacteria": "B1",
            "BTUB": "",
            "FADL": "",
            "FHUA": "",
            "LAMB": "",
            "LPTD": "",
            "NFRA": "",
            "OMPA": "",
            "OMPC": "99_1",
            "OMPF": "",
            "TOLC": "",
            "TSX": "",
            "YNCD": "",
        },
        {
            "bacteria": "B2",
            "BTUB": "",
            "FADL": "",
            "FHUA": "",
            "LAMB": "",
            "LPTD": "",
            "NFRA": "",
            "OMPA": "",
            "OMPC": "99_1",
            "OMPF": "",
            "TOLC": "",
            "TSX": "",
            "YNCD": "",
        },
        {
            "bacteria": "B3",
            "BTUB": "",
            "FADL": "",
            "FHUA": "",
            "LAMB": "",
            "LPTD": "",
            "NFRA": "",
            "OMPA": "",
            "OMPC": "99_1",
            "OMPF": "",
            "TOLC": "",
            "TSX": "",
            "YNCD": "",
        },
        {
            "bacteria": "B4",
            "BTUB": "",
            "FADL": "",
            "FHUA": "",
            "LAMB": "",
            "LPTD": "",
            "NFRA": "",
            "OMPA": "",
            "OMPC": "",
            "OMPF": "",
            "TOLC": "",
            "TSX": "",
            "YNCD": "",
        },
    ]

    lookup_by_genus = {
        "Vectrevirus": LookupEntry(
            match_level="genus",
            taxon="Vectrevirus",
            target_receptors=("CAPSULE",),
            target_family="surface_glycan",
            evidence_note="capsule-targeting override",
        )
    }
    lookup_by_subfamily = {
        "Tevenvirinae": LookupEntry(
            match_level="subfamily",
            taxon="Tevenvirinae",
            target_receptors=("OMPC", "LPS_CORE"),
            target_family="mixed",
            evidence_note="T-even fallback",
        )
    }

    receptor_index = {row["bacteria"]: row for row in receptor_rows}
    host_target_states = build_host_target_states(st02_rows, receptor_index)
    merged_rows = merge_pair_rows(
        st02_rows,
        split_rows,
        host_target_states,
        lookup_by_genus,
        lookup_by_subfamily,
        "include_in_training",
    )
    scenario_indices = build_training_variant_indices(merged_rows)
    feature_rows = build_feature_rows(merged_rows, scenario_indices)

    by_pair_id = {row["pair_id"]: row for row in feature_rows}

    assert by_pair_id["B1__P1"]["target_receptor_present"] == 1
    assert by_pair_id["B1__P1"]["protein_target_present"] == 1
    assert by_pair_id["B1__P1"]["receptor_cluster_matches"] == 1
    assert by_pair_id["B1__P1"]["receptor_variant_seen_in_training_positives"] == 1

    assert by_pair_id["B3__P1"]["receptor_cluster_matches"] == 1
    assert by_pair_id["B4__P2"]["surface_target_present"] == 1
    assert by_pair_id["B4__P2"]["receptor_variant_seen_in_training_positives"] == 0


def test_main_writes_feature_matrix_metadata_lookup_summary_and_manifest(tmp_path: Path) -> None:
    st02_path = tmp_path / "st02_pair_table.csv"
    st03_path = tmp_path / "st03_split_assignments.csv"
    receptor_path = tmp_path / "receptors.tsv"
    lookup_path = tmp_path / "lookup.csv"
    output_dir = tmp_path / "out"

    st02_path.write_text(
        (
            "pair_id,bacteria,phage,phage_genus,phage_subfamily,label_hard_any_lysis,include_in_training,"
            "host_lps_type,host_o_type,host_abc_serotype,host_capsule_abc,host_capsule_groupiv_e,"
            "host_capsule_groupiv_e_stricte,host_capsule_groupiv_s,host_capsule_wzy_stricte\n"
            "B1__P1,B1,P1,Tequatrovirus,Tevenvirinae,1,1,R1,O1,,0,0,0,0,0\n"
            "B2__P1,B2,P1,Tequatrovirus,Tevenvirinae,1,1,R2,O2,,0,0,0,0,0\n"
            "B3__P2,B3,P2,Vectrevirus,Molineuxvirinae,0,1,,,K1,1,0,0,0,0\n"
        ),
        encoding="utf-8",
    )
    st03_path.write_text(
        (
            "pair_id,split_holdout,split_cv5_fold\n"
            "B1__P1,train_non_holdout,0\n"
            "B2__P1,train_non_holdout,1\n"
            "B3__P2,holdout_test,-1\n"
        ),
        encoding="utf-8",
    )
    receptor_path.write_text(
        (
            "bacteria\tBTUB\tFADL\tFHUA\tLAMB\tLPTD\tNFRA\tOMPA\tOMPC\tOMPF\tTOLC\tTSX\tYNCD\n"
            "B1\t\t\t\t\t\t\t\t99_1\t\t\t\t\n"
            "B2\t\t\t\t\t\t\t\t99_1\t\t\t\t\n"
            "B3\t\t\t\t\t\t\t\t\t\t\t\t\n"
        ),
        encoding="utf-8",
    )
    lookup_path.write_text(
        (
            "match_level,taxon,target_receptors,target_family,evidence_note\n"
            "subfamily,Tevenvirinae,OMPC|LPS_CORE,mixed,T-even fallback\n"
            "genus,Vectrevirus,CAPSULE,surface_glycan,capsule override\n"
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--st02-pair-table-path",
            str(st02_path),
            "--st03-split-assignments-path",
            str(st03_path),
            "--receptor-clusters-path",
            str(receptor_path),
            "--lookup-path",
            str(lookup_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
        ]
    )

    assert exit_code == 0

    feature_path = output_dir / "rbp_receptor_compatibility_features_test.csv"
    metadata_path = output_dir / "rbp_receptor_compatibility_feature_metadata_test.csv"
    lookup_summary_path = output_dir / "rbp_receptor_lookup_summary_test.csv"
    manifest_path = output_dir / "rbp_receptor_compatibility_manifest_test.json"

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with lookup_summary_path.open("r", encoding="utf-8", newline="") as handle:
        lookup_summary_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(feature_rows) == 3
    assert feature_rows[0]["pair_id"] == "B1__P1"
    assert feature_rows[0]["receptor_cluster_matches"] == "1"
    assert any(row["column_name"] == "target_receptor_present" for row in metadata_rows)
    assert lookup_summary_rows[0]["phage"] == "P1"
    assert manifest["pair_count"] == 3
    assert manifest["feature_count"] == 6
    assert manifest["lookup_coverage"]["covered_phage_count"] == 2
