from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lyzortx.autoresearch import train as autoresearch_train
from lyzortx.pipeline.autoresearch import build_contract, prepare_cache


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_pair_rows(*, bacteria_ids: list[str], phage_ids: list[str], split_name: str) -> list[dict[str, object]]:
    positive_pairs = {
        ("B1", "P1"),
        ("B1", "P2"),
        ("B2", "P1"),
        ("B2", "P2"),
        ("B3", "P3"),
        ("B4", "P1"),
        ("B4", "P2"),
        ("B5", "P3"),
    }
    rows: list[dict[str, object]] = []
    for bacteria in bacteria_ids:
        for phage in phage_ids:
            rows.append(
                {
                    "pair_id": f"{bacteria}__{phage}",
                    "bacteria": bacteria,
                    "phage": phage,
                    "split": split_name,
                    "label_any_lysis": "1" if (bacteria, phage) in positive_pairs else "0",
                    "label_reason": "fixture",
                    "training_weight_v3": "1.0",
                    "label_read_only": "1",
                    "retained_for_autoresearch": "1",
                    "exclusion_reasons": "",
                    "host_fasta_path": f"hosts/{bacteria}.fna",
                    "phage_fasta_path": f"phages/{phage}.fna",
                }
            )
    return rows


def slot_fixture_rows() -> dict[str, list[dict[str, object]]]:
    host_surface_rows = [
        {"bacteria": "B1", "host_surface__host_o_antigen_type": "O1", "host_surface__host_receptor_btub_score": "2.0"},
        {"bacteria": "B2", "host_surface__host_o_antigen_type": "O1", "host_surface__host_receptor_btub_score": "1.8"},
        {"bacteria": "B3", "host_surface__host_o_antigen_type": "O2", "host_surface__host_receptor_btub_score": "0.4"},
        {"bacteria": "B4", "host_surface__host_o_antigen_type": "O1", "host_surface__host_receptor_btub_score": "1.9"},
        {"bacteria": "B5", "host_surface__host_o_antigen_type": "O2", "host_surface__host_receptor_btub_score": "0.3"},
    ]
    host_typing_rows = [
        {"bacteria": "B1", "host_typing__host_clermont_phylo": "A", "host_typing__host_serotype": "O1:H1"},
        {"bacteria": "B2", "host_typing__host_clermont_phylo": "A", "host_typing__host_serotype": "O1:H1"},
        {"bacteria": "B3", "host_typing__host_clermont_phylo": "B1", "host_typing__host_serotype": "O2:H2"},
        {"bacteria": "B4", "host_typing__host_clermont_phylo": "A", "host_typing__host_serotype": "O1:H1"},
        {"bacteria": "B5", "host_typing__host_clermont_phylo": "B1", "host_typing__host_serotype": "O2:H2"},
    ]
    host_stats_rows = [
        {
            "bacteria": "B1",
            "host_stats__host_sequence_record_count": "1",
            "host_stats__host_genome_length_nt": "5100000",
            "host_stats__host_gc_content": "0.51",
        },
        {
            "bacteria": "B2",
            "host_stats__host_sequence_record_count": "1",
            "host_stats__host_genome_length_nt": "5050000",
            "host_stats__host_gc_content": "0.50",
        },
        {
            "bacteria": "B3",
            "host_stats__host_sequence_record_count": "1",
            "host_stats__host_genome_length_nt": "4700000",
            "host_stats__host_gc_content": "0.47",
        },
        {
            "bacteria": "B4",
            "host_stats__host_sequence_record_count": "1",
            "host_stats__host_genome_length_nt": "5080000",
            "host_stats__host_gc_content": "0.50",
        },
        {
            "bacteria": "B5",
            "host_stats__host_sequence_record_count": "1",
            "host_stats__host_genome_length_nt": "4680000",
            "host_stats__host_gc_content": "0.47",
        },
    ]
    phage_projection_rows = [
        {
            "phage": "P1",
            "phage_projection__tl17_phage_rbp_family_11_percent_identity": "0.95",
            "phage_projection__tl17_rbp_reference_hit_count": "3",
        },
        {
            "phage": "P2",
            "phage_projection__tl17_phage_rbp_family_11_percent_identity": "0.90",
            "phage_projection__tl17_rbp_reference_hit_count": "3",
        },
        {
            "phage": "P3",
            "phage_projection__tl17_phage_rbp_family_11_percent_identity": "0.25",
            "phage_projection__tl17_rbp_reference_hit_count": "1",
        },
        {
            "phage": "P4",
            "phage_projection__tl17_phage_rbp_family_11_percent_identity": "0.05",
            "phage_projection__tl17_rbp_reference_hit_count": "0",
        },
    ]
    phage_stats_rows = [
        {
            "phage": "P1",
            "phage_stats__phage_sequence_record_count": "1",
            "phage_stats__phage_genome_length_nt": "42000",
            "phage_stats__phage_gc_content": "0.45",
        },
        {
            "phage": "P2",
            "phage_stats__phage_sequence_record_count": "1",
            "phage_stats__phage_genome_length_nt": "41500",
            "phage_stats__phage_gc_content": "0.45",
        },
        {
            "phage": "P3",
            "phage_stats__phage_sequence_record_count": "1",
            "phage_stats__phage_genome_length_nt": "39000",
            "phage_stats__phage_gc_content": "0.41",
        },
        {
            "phage": "P4",
            "phage_stats__phage_sequence_record_count": "1",
            "phage_stats__phage_genome_length_nt": "36000",
            "phage_stats__phage_gc_content": "0.39",
        },
    ]
    return {
        "host_defense": [],
        "host_surface": host_surface_rows,
        "host_typing": host_typing_rows,
        "host_stats": host_stats_rows,
        "phage_projection": phage_projection_rows,
        "phage_stats": phage_stats_rows,
    }


def write_slot(cache_dir: Path, slot_name: str, rows: list[dict[str, object]]) -> tuple[list[str], str]:
    slot_spec = prepare_cache.SLOT_SPEC_BY_NAME[slot_name]
    slot_dir = cache_dir / "feature_slots" / slot_name
    slot_dir.mkdir(parents=True, exist_ok=True)

    entity_values = (
        sorted({str(row[slot_spec.entity_key]) for row in rows})
        if rows
        else ["B1", "B2", "B3", "B4", "B5"]
        if slot_spec.entity_key == "bacteria"
        else ["P1", "P2", "P3", "P4"]
    )
    write_csv_rows(
        slot_dir / prepare_cache.ENTITY_INDEX_FILENAME, [{slot_spec.entity_key: value} for value in entity_values]
    )

    feature_columns = [column for column in rows[0].keys() if column != slot_spec.entity_key] if rows else []
    if rows:
        write_csv_rows(slot_dir / prepare_cache.SLOT_FEATURES_FILENAME, rows)

    slot_schema = prepare_cache.build_slot_schema_manifest(
        slot_spec,
        row_count=len(entity_values),
        reserved_feature_columns=feature_columns,
    )
    schema_path = slot_dir / prepare_cache.SLOT_SCHEMA_FILENAME
    write_json(schema_path, slot_schema)
    return feature_columns, str(schema_path)


def create_minimal_autoresearch_cache(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    cache_dir = output_root / "search_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_pair_rows(
        bacteria_ids=["B1", "B2", "B3"],
        phage_ids=["P1", "P2", "P3", "P4"],
        split_name=build_contract.TRAIN_SPLIT,
    )
    inner_rows = build_pair_rows(
        bacteria_ids=["B4", "B5"],
        phage_ids=["P1", "P2", "P3", "P4"],
        split_name=build_contract.INNER_VAL_SPLIT,
    )
    train_path = cache_dir / "search_pairs" / prepare_cache.TRAIN_PAIR_TABLE_FILENAME
    inner_path = cache_dir / "search_pairs" / prepare_cache.INNER_VAL_PAIR_TABLE_FILENAME
    write_csv_rows(train_path, train_rows)
    write_csv_rows(inner_path, inner_rows)

    slot_rows = slot_fixture_rows()
    slot_feature_columns: dict[str, list[str]] = {}
    slot_schema_paths: dict[str, str] = {}
    for slot_name in [spec.slot_name for spec in prepare_cache.SLOT_SPECS]:
        feature_columns, schema_path = write_slot(cache_dir, slot_name, slot_rows[slot_name])
        slot_feature_columns[slot_name] = feature_columns
        slot_schema_paths[slot_name] = schema_path

    schema_manifest = prepare_cache.build_top_level_schema_manifest(slot_feature_columns=slot_feature_columns)
    schema_manifest_path = cache_dir / prepare_cache.SCHEMA_MANIFEST_FILENAME
    write_json(schema_manifest_path, schema_manifest)

    contract_manifest = {
        "task_id": "AR01",
        "split_contract": {
            "split_hashes": {
                build_contract.TRAIN_SPLIT: {
                    "retained_pair_ids_sha256": build_contract.sha256_strings([row["pair_id"] for row in train_rows])
                },
                build_contract.INNER_VAL_SPLIT: {
                    "retained_pair_ids_sha256": build_contract.sha256_strings([row["pair_id"] for row in inner_rows])
                },
                build_contract.HOLDOUT_SPLIT: {"retained_pair_ids_sha256": build_contract.sha256_strings(["H0__P0"])},
            }
        },
    }
    contract_manifest_path = output_root / build_contract.CONTRACT_MANIFEST_FILENAME
    write_json(contract_manifest_path, contract_manifest)

    input_checksums_path = output_root / build_contract.INPUT_CHECKSUMS_FILENAME
    write_json(input_checksums_path, {"task_id": "AR01"})
    provenance_manifest = {
        "task_id": "AR02",
        "source_contract": {
            "pair_contract_manifest_path": str(contract_manifest_path),
            "input_checksums_manifest_path": str(input_checksums_path),
            "output_root": str(output_root),
        },
    }
    provenance_manifest_path = cache_dir / prepare_cache.PROVENANCE_MANIFEST_FILENAME
    write_json(provenance_manifest_path, provenance_manifest)

    cache_manifest = {
        "task_id": "AR02",
        "cache_contract_id": prepare_cache.CACHE_CONTRACT_ID,
        "schema_manifest_path": str(schema_manifest_path),
        "provenance_manifest_path": str(provenance_manifest_path),
        "pair_tables": {
            build_contract.TRAIN_SPLIT: {"path": str(train_path)},
            build_contract.INNER_VAL_SPLIT: {"path": str(inner_path)},
        },
        "feature_slots": {
            slot_name: {
                "schema_manifest_path": slot_schema_paths[slot_name],
            }
            for slot_name in slot_schema_paths
        },
    }
    write_json(cache_dir / prepare_cache.CACHE_MANIFEST_FILENAME, cache_manifest)
    return cache_dir


def test_train_runs_adsorption_first_baseline_without_host_defense(tmp_path: Path) -> None:
    cache_dir = create_minimal_autoresearch_cache(tmp_path)
    output_dir = tmp_path / "train_outputs"

    exit_code = autoresearch_train.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--device-type",
            "cpu",
        ]
    )
    assert exit_code == 0

    summary = json.loads((output_dir / "ar07_baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["search_metric"]["name"] == "roc_auc"
    assert 0.0 <= float(summary["search_metric"]["value"]) <= 1.0
    assert summary["baseline_contract"]["minimum_slots"] == [
        "host_surface",
        "host_typing",
        "host_stats",
        "phage_projection",
        "phage_stats",
    ]
    assert summary["baseline_contract"]["host_defense_active"] is False
    assert summary["host_encoder"]["slots"] == ["host_surface", "host_typing", "host_stats"]
    assert "host_defense" not in summary["host_encoder"]["slots"]
    assert set(summary["inner_val_metrics"]) == {"roc_auc", "top3_hit_rate", "brier_score"}


def test_train_rejects_holdout_labels_in_search_cache(tmp_path: Path) -> None:
    cache_dir = create_minimal_autoresearch_cache(tmp_path)
    holdout_path = cache_dir / "search_pairs" / "holdout_pairs.csv"
    write_csv_rows(
        holdout_path,
        [
            {
                "pair_id": "H1__P1",
                "bacteria": "H1",
                "phage": "P1",
                "split": "holdout",
                "label_any_lysis": "1",
                "training_weight_v3": "1.0",
                "retained_for_autoresearch": "1",
                "host_fasta_path": "hosts/H1.fna",
                "phage_fasta_path": "phages/P1.fna",
            }
        ],
    )

    with pytest.raises(ValueError, match="holdout"):
        autoresearch_train.main(
            ["--cache-dir", str(cache_dir), "--output-dir", str(tmp_path / "out"), "--device-type", "cpu"]
        )


def test_train_rejects_split_membership_drift(tmp_path: Path) -> None:
    cache_dir = create_minimal_autoresearch_cache(tmp_path)
    train_path = cache_dir / "search_pairs" / prepare_cache.TRAIN_PAIR_TABLE_FILENAME
    train_rows = list(csv.DictReader(train_path.open("r", newline="", encoding="utf-8")))
    train_rows.pop()
    write_csv_rows(train_path, train_rows)

    with pytest.raises(ValueError, match="split membership drift"):
        autoresearch_train.main(
            ["--cache-dir", str(cache_dir), "--output-dir", str(tmp_path / "out"), "--device-type", "cpu"]
        )


def test_train_rejects_schema_bypass(tmp_path: Path) -> None:
    cache_dir = create_minimal_autoresearch_cache(tmp_path)
    feature_path = cache_dir / "feature_slots" / "host_surface" / prepare_cache.SLOT_FEATURES_FILENAME
    rows = list(csv.DictReader(feature_path.open("r", newline="", encoding="utf-8")))
    for row in rows:
        row["unexpected_column"] = "1"
    write_csv_rows(feature_path, rows)

    with pytest.raises(ValueError, match="frozen cache schema"):
        autoresearch_train.main(
            ["--cache-dir", str(cache_dir), "--output-dir", str(tmp_path / "out"), "--device-type", "cpu"]
        )
