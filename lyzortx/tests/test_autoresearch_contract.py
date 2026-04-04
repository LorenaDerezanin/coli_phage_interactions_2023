from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lyzortx.pipeline.autoresearch import build_contract


def write_raw_interactions(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "bacteria",
        "bacteria_index",
        "phage",
        "image",
        "replicate",
        "plate",
        "log_dilution",
        "X",
        "Y",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(path: Path, stem: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{stem}\nATGC\n", encoding="utf-8")


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_locked_comparator_artifacts(root: Path) -> dict[str, object]:
    benchmark_summary_path = root / "track_g" / "tg02_gbm_calibration" / "tg02_benchmark_summary.json"
    feature_lock_path = root / "track_g" / "tg05_feature_subset_sweep" / "tg05_locked_v1_feature_config.json"
    model_summary_path = root / "track_g" / "tg01_v1_binary_classifier" / "tg01_model_summary.json"

    write_json_file(benchmark_summary_path, {"metric": "auprc", "value": 0.5})
    write_json_file(feature_lock_path, {"feature_blocks": ["defense", "phage_genomic"]})
    write_json_file(model_summary_path, {"model_type": "gbm"})

    return {
        "artifact_id": "track_g_clean_v1_locked_benchmark",
        "benchmark_summary_path": str(benchmark_summary_path),
        "feature_lock_path": str(feature_lock_path),
        "model_summary_path": str(model_summary_path),
        "evaluation_protocol_id": "steel_thread_v0_st03_split_v1",
        "locked_feature_blocks": ["defense", "phage_genomic"],
        "selection_source": "test comparator fixture",
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_pair_rows_reuses_track_a_label_v1_and_raw_weight() -> None:
    pair_rows = {
        ("B1", "P1"): [
            {
                "bacteria": "B1",
                "phage": "P1",
                "log_dilution": dilution,
                "score": score,
            }
            for dilution, score in (
                ("0", "1"),
                ("0", "0"),
                ("0", "0"),
                ("-1", "0"),
                ("-1", "0"),
                ("-2", "0"),
                ("-2", "0"),
                ("-2", "0"),
                ("-4", "0"),
            )
        ],
        ("B2", "P1"): [
            {
                "bacteria": "B2",
                "phage": "P1",
                "log_dilution": dilution,
                "score": score,
            }
            for dilution, score in (
                ("0", "0"),
                ("0", "0"),
                ("0", "0"),
                ("-1", "0"),
                ("-1", "0"),
            )
        ],
    }
    rows = build_contract.build_pair_rows(
        pair_rows=pair_rows,
        host_fasta_map={"B1": Path("hosts/B1.fna"), "B2": Path("hosts/B2.fna")},
        phage_fasta_map={"P1": Path("phages/P1.fna")},
        bacteria_split_map={"B1": "train", "B2": "holdout"},
    )

    by_pair = {(row["bacteria"], row["phage"]): row for row in rows}
    assert by_pair[("B1", "P1")]["label_any_lysis"] == "1"
    assert by_pair[("B1", "P1")]["training_weight_v3"] == "0.1"
    assert by_pair[("B1", "P1")]["retained_for_autoresearch"] == 1
    assert by_pair[("B2", "P1")]["label_any_lysis"] == "0"
    assert by_pair[("B2", "P1")]["training_weight_v3"] == "1"


def test_build_pair_rows_marks_unresolved_and_missing_fastas() -> None:
    pair_rows = {
        ("B1", "P1"): [
            {"bacteria": "B1", "phage": "P1", "log_dilution": "0", "score": "0"},
            {"bacteria": "B1", "phage": "P1", "log_dilution": "0", "score": "0"},
            {"bacteria": "B1", "phage": "P1", "log_dilution": "0", "score": "n"},
        ],
    }
    rows = build_contract.build_pair_rows(
        pair_rows=pair_rows,
        host_fasta_map={},
        phage_fasta_map={"P1": Path("phages/P1.fna")},
        bacteria_split_map={"B1": "inner_val"},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["retained_for_autoresearch"] == 0
    assert row["exclusion_reasons"] == "unresolved_label|missing_host_fasta"


def test_assign_bacteria_splits_is_disjoint_and_deterministic() -> None:
    assignments = build_contract.assign_bacteria_splits(
        [f"B{i}" for i in range(10)],
        holdout_fraction=0.2,
        inner_val_fraction=0.2,
        split_salt="seed",
    )
    repeated_assignments = build_contract.assign_bacteria_splits(
        [f"B{i}" for i in range(10)],
        holdout_fraction=0.2,
        inner_val_fraction=0.2,
        split_salt="seed",
    )
    assert set(assignments.values()) == {"train", "inner_val", "holdout"}
    assert assignments == repeated_assignments
    buckets = {
        split: {b for b, assigned in assignments.items() if assigned == split} for split in build_contract.SPLIT_ORDER
    }
    assert buckets["train"].isdisjoint(buckets["inner_val"])
    assert buckets["train"].isdisjoint(buckets["holdout"])
    assert buckets["inner_val"].isdisjoint(buckets["holdout"])


def test_build_locked_comparator_benchmark_requires_existing_files(tmp_path: Path) -> None:
    comparator_benchmark = {
        "artifact_id": "track_g_clean_v1_locked_benchmark",
        "benchmark_summary_path": str(tmp_path / "missing_benchmark_summary.json"),
        "feature_lock_path": str(tmp_path / "missing_feature_lock.json"),
        "model_summary_path": str(tmp_path / "missing_model_summary.json"),
        "evaluation_protocol_id": "steel_thread_v0_st03_split_v1",
        "locked_feature_blocks": ["defense", "phage_genomic"],
        "selection_source": "test comparator fixture",
    }

    with pytest.raises(FileNotFoundError, match="Locked comparator artifact not found"):
        build_contract.build_locked_comparator_benchmark(comparator_benchmark)


def test_validate_split_contract_requires_retained_rows_per_split() -> None:
    split_summary = {
        "leakage_checks": {"max_overlap_count": 0},
        "retained_row_counts": {"train": 2, "inner_val": 0, "holdout": 1},
    }

    with pytest.raises(ValueError, match="inner_val"):
        build_contract.validate_split_contract(split_summary)


def test_main_writes_manifest_and_pair_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path = tmp_path / "raw_interactions.csv"
    host_dir = tmp_path / "hosts"
    phage_dir = tmp_path / "phages"
    output_dir = tmp_path / "outputs"
    comparator_root = tmp_path / "comparator"

    rows: list[dict[str, str]] = []
    for bacteria in ("B1", "B2", "B3"):
        for phage in ("P1", "P2"):
            for replicate, dilution, score in (
                ("1", "0", "1" if bacteria == "B1" and phage == "P1" else "0"),
                ("2", "0", "0"),
                ("3", "0", "0"),
                ("1", "-1", "0"),
                ("2", "-1", "0"),
                ("1", "-2", "0"),
                ("2", "-2", "0"),
                ("3", "-2", "0"),
                ("1", "-4", "0"),
            ):
                rows.append(
                    {
                        "bacteria": bacteria,
                        "bacteria_index": bacteria,
                        "phage": phage,
                        "image": "",
                        "replicate": replicate,
                        "plate": "plate1",
                        "log_dilution": dilution,
                        "X": "0",
                        "Y": "0",
                        "score": score,
                    }
                )
    write_raw_interactions(raw_path, rows)

    for bacteria in ("B1", "B2", "B3"):
        write_fasta(host_dir / f"{bacteria}.fna", bacteria)
    for phage in ("P1", "P2"):
        write_fasta(phage_dir / f"{phage}.fna", phage)

    monkeypatch.setattr(
        build_contract,
        "COMPARATOR_BENCHMARK",
        write_locked_comparator_artifacts(comparator_root),
    )

    exit_code = build_contract.main(
        [
            "--raw-interactions-path",
            str(raw_path),
            "--host-assembly-dir",
            str(host_dir),
            "--phage-fasta-dir",
            str(phage_dir),
            "--output-dir",
            str(output_dir),
            "--skip-host-assembly-resolution",
        ]
    )
    assert exit_code == 0

    pair_rows = read_csv_rows(output_dir / build_contract.PAIR_TABLE_FILENAME)
    assert len(pair_rows) == 6

    manifest = json.loads((output_dir / build_contract.CONTRACT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["split_contract"]["leakage_checks"]["max_overlap_count"] == 0
    assert manifest["current_locked_comparator_benchmark"]["artifact_id"] == "track_g_clean_v1_locked_benchmark"
    assert set(manifest["current_locked_comparator_benchmark"]["artifact_checksums"]) == {
        "benchmark_summary_path",
        "feature_lock_path",
        "model_summary_path",
    }
    assert set(manifest["split_contract"]["row_counts"]) == {"train", "inner_val", "holdout"}
