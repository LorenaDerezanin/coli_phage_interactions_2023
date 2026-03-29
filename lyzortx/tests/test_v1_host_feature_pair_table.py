import csv
import json
from pathlib import Path

import joblib

from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import (
    DEFENSE_SUBTYPE_MASK_NAME,
    build_defense_column_mask,
    build_defense_feature_rows,
    main,
    merge_host_feature_blocks,
    run_lightgbm_sanity_check,
)


def test_build_defense_feature_rows_filters_support_and_adds_derived_features() -> None:
    rows = [
        {"bacteria": "B1", "AbiD": "1", "AbiE": "0", "CAS_Class1-Type-I": "1", "RareThing": "1"},
        {"bacteria": "B2", "AbiD": "1", "AbiE": "1", "CAS_Class1-Type-I": "0", "RareThing": "0"},
        {"bacteria": "B3", "AbiD": "0", "AbiE": "1", "CAS_Class1-Type-I": "0", "RareThing": "0"},
    ]

    feature_rows, numeric_columns, manifest = build_defense_feature_rows(
        rows,
        min_present_count=1,
        max_present_count=2,
    )
    mask = build_defense_column_mask(rows, min_present_count=1, max_present_count=2)

    assert numeric_columns == [
        "host_defense_subtype_abi_d",
        "host_defense_subtype_abi_e",
        "host_defense_subtype_cas_class1_type_i",
        "host_defense_subtype_rare_thing",
        "host_defense_diversity",
        "host_defense_has_crispr",
        "host_defense_abi_burden",
    ]
    assert feature_rows[0]["host_defense_diversity"] == 3
    assert feature_rows[0]["host_defense_has_crispr"] == 1
    assert feature_rows[1]["host_defense_abi_burden"] == 2
    assert manifest["retained_subtype_count"] == 4
    assert mask["ordered_feature_columns"] == numeric_columns


def test_merge_host_feature_blocks_preserves_expected_missingness() -> None:
    merged_rows, join_audit, columns = merge_host_feature_blocks(
        ["B1", "B2"],
        blocks={
            "block_a": [
                {"bacteria": "B1", "feature_a": 1},
                {"bacteria": "B2", "feature_a": 0},
            ],
            "block_b": [
                {"bacteria": "B1", "feature_b": ""},
                {"bacteria": "B2", "feature_b": "typed"},
            ],
        },
    )

    assert columns == ["feature_a", "feature_b"]
    assert merged_rows[0]["feature_b"] == ""
    assert (
        join_audit["block_summaries"]["block_b"]["column_missingness"]["feature_b"]["unexpected_missing_increase"] == 0
    )


def test_run_lightgbm_sanity_check_confirms_lift_with_strong_v1_signal() -> None:
    pair_rows = []
    split_rows = []
    host_signal = {
        "B1": 1,
        "B2": 0,
        "B3": 1,
        "B4": 0,
    }
    fold_by_bacteria = {"B1": 0, "B2": 0, "B3": 1, "B4": 1}

    for bacteria in ["B1", "B2", "B3", "B4"]:
        for phage in ["P1", "P2"]:
            pair_id = f"{bacteria}__{phage}"
            label = str(host_signal[bacteria])
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "bacteria": bacteria,
                    "phage": phage,
                    "label_hard_any_lysis": label,
                    "host_pathotype": "shared",
                    "host_clermont_phylo": "A",
                    "host_origin": "gut",
                    "host_lps_type": "R1",
                    "host_o_type": "O1",
                    "host_h_type": "H1",
                    "host_collection": "set1",
                    "host_abc_serotype": "",
                    "host_mouse_killed_10": "0",
                    "host_capsule_abc": "0",
                    "host_capsule_groupiv_e": "0",
                    "host_capsule_groupiv_e_stricte": "0",
                    "host_capsule_groupiv_s": "0",
                    "host_capsule_wzy_stricte": "0",
                    "host_n_defense_systems": "0",
                    "phage_morphotype": "morph",
                    "phage_family": "fam",
                    "phage_genus": "gen",
                    "phage_species": "spec",
                    "phage_subfamily": "sub",
                    "phage_old_family": "oldfam",
                    "phage_old_genus": "oldgen",
                    "phage_host": "H",
                    "phage_host_phylo": "A",
                    "phage_genome_size": "100",
                    "pair_host_phylo_equals_phage_host_phylo": "1",
                    "host_surface_klebsiella_capsule_type": "",
                    "host_surface_lps_core_type": "R1",
                    "host_surface_klebsiella_capsule_type_missing": "1",
                    "host_phylogeny_umap_00": str(host_signal[bacteria]),
                    "host_phylogeny_umap_01": "0",
                    "host_phylogeny_umap_02": "0",
                    "host_phylogeny_umap_03": "0",
                    "host_phylogeny_umap_04": "0",
                    "host_phylogeny_umap_05": "0",
                    "host_phylogeny_umap_06": "0",
                    "host_phylogeny_umap_07": "0",
                    "host_defense_subtype_signal": str(host_signal[bacteria]),
                    "host_defense_diversity": str(host_signal[bacteria]),
                    "host_defense_has_crispr": str(host_signal[bacteria]),
                    "host_defense_abi_burden": "0",
                    "host_omp_receptor_btub_cluster_99_1": str(host_signal[bacteria]),
                }
            )
            split_rows.append(
                {
                    "pair_id": pair_id,
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": str(fold_by_bacteria[bacteria]),
                    "is_hard_trainable": "1",
                }
            )

    sanity = run_lightgbm_sanity_check(
        pair_rows,
        split_rows,
        v1_categorical_columns=[
            "host_pathotype",
            "host_clermont_phylo",
            "host_origin",
            "host_lps_type",
            "host_o_type",
            "host_h_type",
            "host_collection",
            "host_abc_serotype",
            "phage",
            "phage_morphotype",
            "phage_family",
            "phage_genus",
            "phage_species",
            "phage_subfamily",
            "phage_old_family",
            "phage_old_genus",
            "phage_host",
            "phage_host_phylo",
            "pair_host_phylo_equals_phage_host_phylo",
            "host_surface_klebsiella_capsule_type",
            "host_surface_lps_core_type",
        ],
        v1_numeric_columns=[
            "host_mouse_killed_10",
            "host_capsule_abc",
            "host_capsule_groupiv_e",
            "host_capsule_groupiv_e_stricte",
            "host_capsule_groupiv_s",
            "host_capsule_wzy_stricte",
            "host_n_defense_systems",
            "phage_genome_size",
            "host_surface_klebsiella_capsule_type_missing",
            "host_phylogeny_umap_00",
            "host_phylogeny_umap_01",
            "host_phylogeny_umap_02",
            "host_phylogeny_umap_03",
            "host_phylogeny_umap_04",
            "host_phylogeny_umap_05",
            "host_phylogeny_umap_06",
            "host_phylogeny_umap_07",
            "host_defense_subtype_signal",
            "host_defense_diversity",
            "host_defense_has_crispr",
            "host_defense_abi_burden",
            "host_omp_receptor_btub_cluster_99_1",
        ],
        random_state=7,
    )

    assert sanity["summary"]["lift_confirmed"] is True
    assert sanity["summary"]["average_precision_lift"] > 0


def test_main_writes_host_matrix_pair_table_and_manifests(tmp_path: Path, monkeypatch) -> None:
    st02_path = tmp_path / "st02.csv"
    st03_path = tmp_path / "st03.csv"
    defense_path = tmp_path / "defense.csv"
    omp_path = tmp_path / "omp.tsv"
    umap_path = tmp_path / "umap.tsv"
    capsule_path = tmp_path / "capsule.tsv"
    lps_primary_path = tmp_path / "lps_primary.tsv"
    lps_supplemental_path = tmp_path / "lps_supplemental.tsv"
    output_dir = tmp_path / "out"

    st02_path.write_text(
        (
            "pair_id,bacteria,phage,label_hard_any_lysis,host_pathotype,host_clermont_phylo,host_origin,host_lps_type,"
            "host_o_type,host_h_type,host_collection,host_abc_serotype,host_mouse_killed_10,host_capsule_abc,"
            "host_capsule_groupiv_e,host_capsule_groupiv_e_stricte,host_capsule_groupiv_s,host_capsule_wzy_stricte,"
            "host_n_defense_systems,phage_morphotype,phage_family,phage_genus,phage_species,"
            "phage_subfamily,phage_old_family,phage_old_genus,phage_host,phage_host_phylo,phage_genome_size,"
            "pair_host_phylo_equals_phage_host_phylo\n"
            "B1__P1,B1,P1,1,pt,A,gut,R1,O1,H1,set1,,0,0,0,0,0,0,1,0,morph,fam,gen,spec,sub,oldfam,oldgen,H,A,100,1\n"
            "B2__P1,B2,P1,0,pt,A,gut,R1,O1,H1,set1,,0,0,0,0,0,0,1,0,morph,fam,gen,spec,sub,oldfam,oldgen,H,A,100,1\n"
        ),
        encoding="utf-8",
    )
    st03_path.write_text(
        (
            "pair_id,split_holdout,split_cv5_fold,is_hard_trainable\n"
            "B1__P1,train_non_holdout,0,1\n"
            "B2__P1,train_non_holdout,1,1\n"
        ),
        encoding="utf-8",
    )
    defense_path.write_text(
        "bacteria;AbiD;CAS_Class1-Type-I\nB1;1;1\nB2;0;0\n",
        encoding="utf-8",
    )
    omp_path.write_text(
        (
            "bacteria\tBTUB\tFADL\tFHUA\tLAMB\tLPTD\tNFRA\tOMPA\tOMPC\tOMPF\tTOLC\tTSX\tYNCD\n"
            "B1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\n"
            "B2\t99_2\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\t99_1\n"
        ),
        encoding="utf-8",
    )
    umap_path.write_text(
        (
            "bacteria\tUMAP0\tUMAP1\tUMAP2\tUMAP3\tUMAP4\tUMAP5\tUMAP6\tUMAP7\n"
            "B1\t0\t0\t0\t0\t0\t0\t0\t0\n"
            "B2\t1\t1\t1\t1\t1\t1\t1\t1\n"
        ),
        encoding="utf-8",
    )
    capsule_path.write_text("bacteria\tKlebs_capsule_type\tlocus\nB1\tK2\tgood\n", encoding="utf-8")
    lps_primary_path.write_text("bacteria\tgembase\tLPS_type\nB1\tGB1\tR1\nB2\tGB2\tR3\n", encoding="utf-8")
    lps_supplemental_path.write_text("Strain\tLPS_type\n", encoding="utf-8")

    monkeypatch.setattr(
        "lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table.run_lightgbm_sanity_check",
        lambda *args, **kwargs: {
            "summary": {
                "v0_mean_average_precision": 0.4,
                "v1_mean_average_precision": 0.5,
                "average_precision_lift": 0.1,
                "lift_confirmed": True,
            }
        },
    )

    exit_code = main(
        [
            "--st02-pair-table-path",
            str(st02_path),
            "--st03-split-assignments-path",
            str(st03_path),
            "--defense-subtypes-path",
            str(defense_path),
            "--receptor-clusters-path",
            str(omp_path),
            "--umap-path",
            str(umap_path),
            "--capsule-path",
            str(capsule_path),
            "--lps-primary-path",
            str(lps_primary_path),
            "--lps-supplemental-path",
            str(lps_supplemental_path),
            "--output-dir",
            str(output_dir),
            "--version",
            "test",
            "--defense-min-present-count",
            "1",
            "--defense-max-present-count",
            "2",
            "--omp-min-cluster-count",
            "1",
            "--omp-max-feature-count",
            "12",
        ]
    )

    assert exit_code == 0

    host_matrix_path = output_dir / "host_feature_matrix_test.csv"
    pair_table_path = output_dir / "pair_table_test.csv"
    join_audit_path = output_dir / "host_feature_join_audit_test.json"
    manifest_path = output_dir / "pair_table_manifest_test.json"

    with host_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        host_rows = list(csv.DictReader(handle))
    with pair_table_path.open("r", encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    join_audit = json.loads(join_audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mask = joblib.load(output_dir / DEFENSE_SUBTYPE_MASK_NAME)

    assert len(host_rows) == 2
    assert len(pair_rows) == 2
    assert pair_rows[0]["host_defense_diversity"] == "2"
    assert (
        join_audit["block_summaries"]["extended_surface"]["column_missingness"]["host_surface_klebsiella_capsule_type"][
            "unexpected_missing_increase"
        ]
        == 0
    )
    assert manifest["host_feature_count"] > 0
    assert mask["retained_subtype_columns"] == ["AbiD", "CAS_Class1-Type-I"]
    assert mask["ordered_feature_columns"] == [
        "host_defense_subtype_abi_d",
        "host_defense_subtype_cas_class1_type_i",
        "host_defense_diversity",
        "host_defense_has_crispr",
        "host_defense_abi_burden",
    ]
