import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_e.steps.build_defense_evasion_proxy_feature_block import (
    build_feature_rows,
    build_training_family_defense_profiles,
    main,
    merge_pair_rows,
)


def test_build_feature_rows_uses_leakage_safe_family_subtype_rates() -> None:
    pair_rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "phage_family": "FamA",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_defense_subtype_abi_d": "1",
            "host_defense_subtype_cas_type_i": "0",
        },
        {
            "pair_id": "B2__P2",
            "bacteria": "B2",
            "phage": "P2",
            "phage_family": "FamA",
            "label_hard_any_lysis": "0",
            "include_in_training": "1",
            "host_defense_subtype_abi_d": "1",
            "host_defense_subtype_cas_type_i": "0",
        },
        {
            "pair_id": "B3__P3",
            "bacteria": "B3",
            "phage": "P3",
            "phage_family": "FamA",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_defense_subtype_abi_d": "0",
            "host_defense_subtype_cas_type_i": "1",
        },
        {
            "pair_id": "B4__P4",
            "bacteria": "B4",
            "phage": "P4",
            "phage_family": "FamA",
            "label_hard_any_lysis": "1",
            "include_in_training": "1",
            "host_defense_subtype_abi_d": "1",
            "host_defense_subtype_cas_type_i": "0",
        },
        {
            "pair_id": "B5__P5",
            "bacteria": "B5",
            "phage": "P5",
            "phage_family": "FamA",
            "label_hard_any_lysis": "1",
            "include_in_training": "0",
            "host_defense_subtype_abi_d": "1",
            "host_defense_subtype_cas_type_i": "0",
        },
    ]
    split_rows = [
        {"pair_id": "B1__P1", "split_holdout": "train_non_holdout", "split_cv5_fold": "0"},
        {"pair_id": "B2__P2", "split_holdout": "train_non_holdout", "split_cv5_fold": "1"},
        {"pair_id": "B3__P3", "split_holdout": "train_non_holdout", "split_cv5_fold": "1"},
        {"pair_id": "B4__P4", "split_holdout": "holdout_test", "split_cv5_fold": "-1"},
        {"pair_id": "B5__P5", "split_holdout": "train_non_holdout", "split_cv5_fold": "1"},
    ]
    defense_subtype_columns = ["host_defense_subtype_abi_d", "host_defense_subtype_cas_type_i"]

    merged_rows = merge_pair_rows(pair_rows, split_rows, "include_in_training", defense_subtype_columns)
    scenario_profiles = build_training_family_defense_profiles(merged_rows, defense_subtype_columns)
    feature_rows = build_feature_rows(merged_rows, scenario_profiles, defense_subtype_columns)

    by_pair_id = {row["pair_id"]: row for row in feature_rows}

    assert by_pair_id["B1__P1"]["defense_evasion_expected_score"] == 0.0
    assert by_pair_id["B1__P1"]["defense_evasion_supported_subtype_count"] == 1
    assert by_pair_id["B1__P1"]["defense_evasion_family_training_pair_count"] == 2

    assert by_pair_id["B4__P4"]["defense_evasion_expected_score"] == 0.5
    assert by_pair_id["B4__P4"]["defense_evasion_mean_score"] == 0.5
    assert by_pair_id["B4__P4"]["defense_evasion_family_training_pair_count"] == 3

    holdout_rates = scenario_profiles["holdout"]["subtype_rates"]
    assert holdout_rates[("FamA", "host_defense_subtype_abi_d")] == 0.5
    assert holdout_rates[("FamA", "host_defense_subtype_cas_type_i")] == 1.0


def test_main_writes_feature_matrix_metadata_rate_table_and_manifest(tmp_path: Path) -> None:
    pair_table_path = tmp_path / "pair_table.csv"
    st03_path = tmp_path / "st03_split_assignments.csv"
    output_dir = tmp_path / "out"

    pair_table_path.write_text(
        (
            "pair_id,bacteria,phage,phage_family,label_hard_any_lysis,include_in_training,"
            "host_defense_subtype_abi_d,host_defense_subtype_cas_type_i\n"
            "B1__P1,B1,P1,FamA,1,1,1,0\n"
            "B2__P2,B2,P2,FamA,0,1,1,0\n"
            "B3__P3,B3,P3,FamA,1,1,0,1\n"
        ),
        encoding="utf-8",
    )
    st03_path.write_text(
        (
            "pair_id,split_holdout,split_cv5_fold\n"
            "B1__P1,train_non_holdout,0\n"
            "B2__P2,train_non_holdout,1\n"
            "B3__P3,holdout_test,-1\n"
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--pair-table-path",
            str(pair_table_path),
            "--st03-split-assignments-path",
            str(st03_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
        ]
    )

    assert exit_code == 0

    feature_path = output_dir / "defense_evasion_proxy_features_test.csv"
    metadata_path = output_dir / "defense_evasion_proxy_feature_metadata_test.csv"
    family_rates_path = output_dir / "family_defense_lysis_rates_test.csv"
    manifest_path = output_dir / "defense_evasion_proxy_manifest_test.json"

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with family_rates_path.open("r", encoding="utf-8", newline="") as handle:
        family_rate_rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(feature_rows) == 3
    by_pair_id = {row["pair_id"]: row for row in feature_rows}
    assert feature_rows[0]["pair_id"] == "B1__P1"
    assert by_pair_id["B2__P2"]["defense_evasion_expected_score"] == "1.0"
    assert by_pair_id["B3__P3"]["defense_evasion_expected_score"] == "0.0"
    assert any(row["column_name"] == "defense_evasion_expected_score" for row in metadata_rows)
    assert any(row["scenario"] == "holdout" for row in family_rate_rows)
    assert manifest["pair_count"] == 3
    assert manifest["feature_count"] == 4
    assert manifest["training_profiles"]["holdout_training_pair_count"] == 2
