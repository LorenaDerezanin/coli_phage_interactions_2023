from __future__ import annotations

import csv
import json
from pathlib import Path

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


def test_assign_bacteria_splits_is_disjoint() -> None:
    assignments = build_contract.assign_bacteria_splits(
        [f"B{i}" for i in range(10)],
        holdout_fraction=0.2,
        inner_val_fraction=0.2,
        split_salt="seed",
    )
    assert set(assignments.values()) == {"train", "inner_val", "holdout"}
    buckets = {
        split: {b for b, assigned in assignments.items() if assigned == split} for split in build_contract.SPLIT_ORDER
    }
    assert buckets["train"].isdisjoint(buckets["inner_val"])
    assert buckets["train"].isdisjoint(buckets["holdout"])
    assert buckets["inner_val"].isdisjoint(buckets["holdout"])


def test_main_writes_manifest_and_pair_table(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_interactions.csv"
    host_dir = tmp_path / "hosts"
    phage_dir = tmp_path / "phages"
    output_dir = tmp_path / "outputs"

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
    assert set(manifest["split_contract"]["row_counts"]) == {"train", "inner_val", "holdout"}
