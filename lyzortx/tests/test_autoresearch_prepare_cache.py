from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lyzortx.pipeline.autoresearch import build_contract, prepare_cache


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
    write_json_file(feature_lock_path, {"feature_blocks": ["adsorption"]})
    write_json_file(model_summary_path, {"model_type": "gbm"})

    return {
        "artifact_id": "track_g_clean_v1_locked_benchmark",
        "benchmark_summary_path": str(benchmark_summary_path),
        "feature_lock_path": str(feature_lock_path),
        "model_summary_path": str(model_summary_path),
        "evaluation_protocol_id": "steel_thread_v0_st03_split_v1",
        "locked_feature_blocks": ["adsorption"],
        "selection_source": "test comparator fixture",
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_fixture_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bacteria in ("B1", "B2", "B3", "B4", "B5"):
        for phage in ("P1", "P2"):
            for replicate, dilution, score in (
                ("1", "0", "1" if (bacteria, phage) in {("B1", "P1"), ("B2", "P2")} else "0"),
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
    return rows


def test_build_top_level_schema_manifest_freezes_reserved_slots() -> None:
    schema = prepare_cache.build_top_level_schema_manifest()

    assert schema["supported_search_splits"] == ["train", "inner_val"]
    assert schema["disallowed_search_splits"] == ["holdout"]
    assert schema["slot_order"] == [
        "host_defense",
        "host_surface",
        "host_typing",
        "host_stats",
        "phage_projection",
        "phage_stats",
    ]
    assert schema["feature_slots"]["host_surface"] == {
        "entity_key": "bacteria",
        "join_keys": ["bacteria"],
        "column_family_prefix": "host_surface__",
        "block_role": "host",
        "reserved_feature_columns": [],
        "reserved_feature_column_count": 0,
    }
    assert schema["feature_slots"]["phage_projection"]["join_keys"] == ["phage"]


def test_main_writes_search_cache_without_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path = tmp_path / "raw_interactions.csv"
    host_dir = tmp_path / "hosts"
    phage_dir = tmp_path / "phages"
    output_root = tmp_path / "outputs"
    cache_dir = output_root / "search_cache_v1"
    comparator_root = tmp_path / "comparator"

    write_raw_interactions(raw_path, build_fixture_rows())
    for bacteria in ("B1", "B2", "B3", "B4", "B5"):
        write_fasta(host_dir / f"{bacteria}.fna", bacteria)
    for phage in ("P1", "P2"):
        write_fasta(phage_dir / f"{phage}.fna", phage)

    monkeypatch.setattr(
        build_contract,
        "COMPARATOR_BENCHMARK",
        write_locked_comparator_artifacts(comparator_root),
    )

    def fake_build_host_defense_slot_artifact(
        *,
        args: object,
        cache_dir: Path,
        split_rows: dict[str, list[dict[str, str]]],
    ) -> dict[str, object]:
        slot_dir = cache_dir / "feature_slots" / "host_defense"
        slot_dir.mkdir(parents=True, exist_ok=True)
        retained_bacteria = sorted(
            {
                str(row["bacteria"])
                for rows in split_rows.values()
                for row in rows
                if str(row["retained_for_autoresearch"]) == "1"
            }
        )
        artifact_path = slot_dir / prepare_cache.SLOT_FEATURE_TABLE_FILENAME
        with artifact_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bacteria", "host_defense__AbiD"])
            writer.writeheader()
            for bacteria in retained_bacteria:
                writer.writerow({"bacteria": bacteria, "host_defense__AbiD": "1"})
        return {
            "artifact_path": str(artifact_path),
            "artifact_sha256": build_contract.sha256_file(artifact_path),
            "columns": ["host_defense__AbiD"],
            "column_count": 1,
            "entity_count": len(retained_bacteria),
            "build_manifest_path": str(slot_dir / prepare_cache.HOST_DEFENSE_BUILD_MANIFEST_FILENAME),
        }

    monkeypatch.setattr(
        prepare_cache,
        "_build_host_defense_slot_artifact",
        fake_build_host_defense_slot_artifact,
    )

    exit_code = prepare_cache.main(
        [
            "--raw-interactions-path",
            str(raw_path),
            "--host-assembly-dir",
            str(host_dir),
            "--phage-fasta-dir",
            str(phage_dir),
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            "--skip-host-assembly-resolution",
            "--holdout-fraction",
            "0.2",
            "--inner-val-fraction",
            "0.2",
        ]
    )
    assert exit_code == 0

    train_rows = read_csv_rows(cache_dir / "search_pairs" / prepare_cache.TRAIN_PAIR_TABLE_FILENAME)
    inner_val_rows = read_csv_rows(cache_dir / "search_pairs" / prepare_cache.INNER_VAL_PAIR_TABLE_FILENAME)
    assert train_rows
    assert inner_val_rows
    assert {row["split"] for row in train_rows} == {"train"}
    assert {row["split"] for row in inner_val_rows} == {"inner_val"}

    cache_manifest = json.loads((cache_dir / prepare_cache.CACHE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert set(cache_manifest["pair_tables"]) == {"train", "inner_val"}
    assert cache_manifest["feature_slots"]["host_defense"]["column_count"] == 1

    provenance = json.loads((cache_dir / prepare_cache.PROVENANCE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert provenance["build_mode"] == "raw_inputs_only"
    assert provenance["sealed_holdout"]["exported_to_search_cache"] is False
    assert provenance["sealed_holdout"]["retained_row_count"] > 0

    host_index_rows = read_csv_rows(cache_dir / "feature_slots" / "host_surface" / prepare_cache.ENTITY_INDEX_FILENAME)
    exported_bacteria = {row["bacteria"] for row in host_index_rows}
    source_pair_rows = read_csv_rows(output_root / build_contract.PAIR_TABLE_FILENAME)
    holdout_bacteria = {row["bacteria"] for row in source_pair_rows if row["split"] == "holdout"}
    assert exported_bacteria
    assert exported_bacteria.isdisjoint(holdout_bacteria)

    schema = json.loads((cache_dir / prepare_cache.SCHEMA_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert schema["feature_slots"]["host_defense"]["reserved_feature_columns"] == ["host_defense__AbiD"]
    host_defense_rows = read_csv_rows(
        cache_dir / "feature_slots" / "host_defense" / prepare_cache.SLOT_FEATURE_TABLE_FILENAME
    )
    assert {row["bacteria"] for row in host_defense_rows} == exported_bacteria


def test_validate_warm_cache_manifest_rejects_schema_mismatch(tmp_path: Path) -> None:
    schema = prepare_cache.build_top_level_schema_manifest()
    warm_cache_dir = tmp_path / "warm"
    warm_cache_dir.mkdir(parents=True)
    artifact_path = warm_cache_dir / "host_surface.csv"
    artifact_path.write_text("bacteria,host_surface__capsule_hits\nB1,1\n", encoding="utf-8")

    manifest_path = warm_cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "warm_cache_manifest_id": "fixture",
                "schema_manifest_id": "wrong_schema",
                "slot_artifacts": {
                    "host_surface": {
                        "path": "host_surface.csv",
                        "join_keys": ["bacteria"],
                        "column_family_prefix": "host_surface__",
                        "columns": ["host_surface__capsule_hits"],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        prepare_cache.validate_warm_cache_manifest(manifest_path, schema_manifest=schema)


def test_validate_warm_cache_manifest_accepts_matching_descriptor(tmp_path: Path) -> None:
    schema = prepare_cache.build_top_level_schema_manifest()
    warm_cache_dir = tmp_path / "warm"
    warm_cache_dir.mkdir(parents=True)
    artifact_path = warm_cache_dir / "phage_stats.csv"
    artifact_path.write_text("phage,phage_stats__gc_content\nP1,0.5\n", encoding="utf-8")

    manifest_path = warm_cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "warm_cache_manifest_id": "fixture",
                "schema_manifest_id": prepare_cache.SCHEMA_MANIFEST_ID,
                "source_kind": "deploy_feature_csv_optional",
                "slot_artifacts": {
                    "phage_stats": {
                        "path": "phage_stats.csv",
                        "join_keys": ["phage"],
                        "column_family_prefix": "phage_stats__",
                        "columns": ["phage_stats__gc_content"],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    validation = prepare_cache.validate_warm_cache_manifest(manifest_path, schema_manifest=schema)
    assert validation["schema_manifest_id"] == prepare_cache.SCHEMA_MANIFEST_ID
    assert validation["validated_slots"]["phage_stats"]["column_count"] == 1
    assert validation["validated_slots"]["phage_stats"]["columns"] == ["phage_stats__gc_content"]


def test_process_one_host_defense_cache_entry_forbids_worker_model_installs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assembly_path = tmp_path / "B1.fna"
    assembly_path.write_text(">B1\nATGC\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    from lyzortx.pipeline.deployment_paired_features import derive_host_defense_features as host_defense_module

    def fake_derive_host_defense_features(
        assembly_path: Path,
        *,
        bacteria_id: str | None = None,
        output_dir: Path,
        panel_defense_subtypes_path: Path,
        models_dir: Path,
        workers: int,
        force_model_update: bool,
        model_install_mode: str,
        force_run: bool,
        preserve_raw: bool,
    ) -> dict[str, object]:
        calls.append(
            {
                "bacteria_id": bacteria_id,
                "output_dir": output_dir,
                "workers": workers,
                "force_model_update": force_model_update,
                "model_install_mode": model_install_mode,
            }
        )
        return {"feature_row": {"bacteria": bacteria_id or assembly_path.stem}}

    monkeypatch.setattr(host_defense_module, "derive_host_defense_features", fake_derive_host_defense_features)

    bacteria_id, ok, message = prepare_cache._process_one_host_defense_cache_entry(
        assembly_path,
        "B1",
        tmp_path / "per_host",
        Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"),
        tmp_path / "models",
        False,
        False,
    )

    assert (bacteria_id, ok, message) == ("B1", True, "ok")
    assert calls == [
        {
            "bacteria_id": "B1",
            "output_dir": tmp_path / "per_host" / "B1",
            "workers": 1,
            "force_model_update": False,
            "model_install_mode": "forbid",
        }
    ]


def test_build_host_defense_slot_artifact_supports_reaggregation_without_worker_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    output_root = tmp_path / "outputs"
    per_host_output_dir = output_root / "host_defense"
    for bacteria in ("B1", "B2"):
        host_dir = per_host_output_dir / bacteria
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / prepare_cache.PER_HOST_COUNTS_FILENAME).write_text(
            f"bacteria,AbiD,RM_Type_I\n{bacteria},1,0\n",
            encoding="utf-8",
        )

    def fail_if_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("ensure_defense_finder_models should not run in aggregate-only mode")

    monkeypatch.setattr(prepare_cache, "ensure_defense_finder_models", fail_if_called)

    args = prepare_cache.parse_args(
        [
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            "--skip-host-assembly-resolution",
            "--host-defense-output-dir",
            str(per_host_output_dir),
            "--host-defense-aggregate-only",
        ]
    )
    split_rows = {
        "train": [
            {
                "bacteria": "B1",
                "host_fasta_path": str(tmp_path / "hosts" / "B1.fna"),
                "retained_for_autoresearch": "1",
            }
        ],
        "inner_val": [
            {
                "bacteria": "B2",
                "host_fasta_path": str(tmp_path / "hosts" / "B2.fna"),
                "retained_for_autoresearch": "1",
            }
        ],
    }

    summary = prepare_cache._build_host_defense_slot_artifact(
        args=args,
        cache_dir=cache_dir,
        split_rows=split_rows,
    )

    assert summary["column_count"] > 0
    artifact_rows = read_csv_rows(
        cache_dir / "feature_slots" / "host_defense" / prepare_cache.SLOT_FEATURE_TABLE_FILENAME
    )
    assert [row["bacteria"] for row in artifact_rows] == ["B1", "B2"]
    assert all(row["host_defense__AbiD"] == "1" for row in artifact_rows)
    assert all(row["host_defense__RM_Type_I"] == "0" for row in artifact_rows)
    assert all(column == "bacteria" or column.startswith("host_defense__") for column in artifact_rows[0])


def test_build_host_defense_slot_artifact_wraps_process_pool_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    output_root = tmp_path / "outputs"
    process_error = OSError("spawn failed")
    logged_errors: list[str] = []

    class FakeFuture:
        def result(self) -> tuple[str, bool, str]:
            raise process_error

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeProcessPoolExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, *args: object, **kwargs: object) -> FakeFuture:
            return FakeFuture()

    def fake_as_completed(futures: dict[FakeFuture, str]) -> list[FakeFuture]:
        return list(futures)

    def fake_ensure_defense_finder_models(*args: object, **kwargs: object) -> str:
        return "already_present"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("aggregate_host_defense_csvs should not run when a worker process fails")

    monkeypatch.setattr(prepare_cache, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(prepare_cache, "as_completed", fake_as_completed)
    monkeypatch.setattr(prepare_cache, "ensure_defense_finder_models", fake_ensure_defense_finder_models)
    monkeypatch.setattr(prepare_cache, "aggregate_host_defense_csvs", fail_if_called)
    monkeypatch.setattr(
        prepare_cache.LOGGER,
        "error",
        lambda message, *args: logged_errors.append(message % args),
    )

    args = prepare_cache.parse_args(
        [
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            "--skip-host-assembly-resolution",
        ]
    )
    split_rows = {
        "train": [
            {
                "bacteria": "B1",
                "host_fasta_path": str(tmp_path / "hosts" / "B1.fna"),
                "retained_for_autoresearch": "1",
            }
        ],
        "inner_val": [],
    }

    with pytest.raises(RuntimeError, match="AR03 host-defense cache build failed for B1"):
        prepare_cache._build_host_defense_slot_artifact(
            args=args,
            cache_dir=cache_dir,
            split_rows=split_rows,
        )

    assert logged_errors == ["AR03 host defense failed for B1: worker process error: spawn failed"]
