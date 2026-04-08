from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lyzortx.pipeline.autoresearch import candidate_replay


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_candidate_bundle(root: Path) -> Path:
    bundle_root = root / "candidate_bundle"
    bundle_root.mkdir(parents=True)
    (bundle_root / "train.py").write_text("print('candidate')\n", encoding="utf-8")
    write_json(
        bundle_root / "local_run_metadata.json",
        {
            "github_run_id": "12345",
            "github_run_attempt": "2",
        },
    )
    write_json(bundle_root / "autoresearch_runpod_bundle_manifest.json", {"bundle_contract_id": "fixture"})
    (bundle_root / "runpod_experiment.log").write_text("ok\n", encoding="utf-8")
    write_json(bundle_root / "runpod_execution_metadata.json", {"experiment_exit_code": "0"})
    write_json(bundle_root / "ar07_baseline_summary.json", {"search_metric": {"name": "roc_auc", "value": 0.71}})
    return bundle_root


def test_import_runpod_candidate_copies_bundle_and_writes_manifest(tmp_path: Path) -> None:
    bundle_root = make_candidate_bundle(tmp_path)

    destination = candidate_replay.import_runpod_candidate(
        candidate_bundle=bundle_root,
        candidates_dir=tmp_path / "imports",
        candidate_id=None,
    )

    assert destination.name == "runpod_run_12345_attempt_2"
    assert (destination / "train.py").exists()
    manifest = json.loads((destination / candidate_replay.IMPORT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "runpod_run_12345_attempt_2"
    assert manifest["runpod_inner_val_summary"]["search_metric"]["value"] == 0.71


def test_candidate_source_root_rejects_path_traversal_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious_candidate.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"unsafe"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.OutsideDestinationError):
        with candidate_replay.candidate_source_root(archive_path):
            pass


def test_resolve_feature_lock_path_uses_only_narrow_default_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback_path = tmp_path / "v1_feature_configuration.json"
    fallback_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(candidate_replay, "FALLBACK_FEATURE_LOCK_PATH", fallback_path)
    monkeypatch.setattr(candidate_replay, "DEFAULT_FEATURE_LOCK_PATH", Path("missing/default.json"))

    assert candidate_replay.resolve_feature_lock_path(Path("missing/default.json")) == fallback_path
    with pytest.raises(FileNotFoundError):
        candidate_replay.resolve_feature_lock_path(Path("missing/other.json"))


def test_load_comparator_params_uses_only_narrow_default_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate_replay, "DEFAULT_TG01_MODEL_SUMMARY_PATH", Path("missing/default_tg01.json"))

    assert candidate_replay.load_comparator_params(Path("missing/default_tg01.json")) == dict(
        candidate_replay.FALLBACK_TG01_BEST_PARAMS
    )
    with pytest.raises(FileNotFoundError):
        candidate_replay.load_comparator_params(Path("missing/other_tg01.json"))


def test_validate_comparator_feature_lock_requires_contract_blocks_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        candidate_replay,
        "load_v1_lock",
        lambda path: {"winner_subset_blocks": ["defense", "phage_genomic"]},
    )

    assert candidate_replay.validate_comparator_feature_lock(
        Path("fixture.json"),
        ("defense", "phage_genomic"),
    ) == {"winner_subset_blocks": ["defense", "phage_genomic"]}

    with pytest.raises(ValueError, match="does not match the AR01 contract"):
        candidate_replay.validate_comparator_feature_lock(
            Path("fixture.json"),
            ("defense",),
        )


def build_fake_candidate_module() -> SimpleNamespace:
    slot_prefixes = (
        "host_surface__",
        "host_typing__",
        "host_stats__",
        "phage_projection__",
        "phage_stats__",
        "host_defense__",
    )

    class FakeEstimator:
        def fit(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            categorical_feature: list[str] | None = None,
        ) -> None:
            host_cols = [c for c in X.columns if c.startswith(("host_surface__", "host_typing__", "host_stats__"))]
            phage_cols = [c for c in X.columns if c.startswith(("phage_projection__", "phage_stats__"))]
            self.host_col = host_cols[0]
            self.phage_col = phage_cols[0]
            product = X[self.host_col].astype(float) * X[self.phage_col].astype(float)
            self.scale = float(np.average(product, weights=sample_weight))

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            product = X[self.host_col].astype(float) * X[self.phage_col].astype(float)
            raw = product.to_numpy() - self.scale
            probs = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - probs, probs])

    def build_entity_feature_table(slot_artifacts, *, slot_names, entity_key):
        merged = None
        for slot_name in slot_names:
            frame = slot_artifacts[slot_name].frame.copy()
            merged = frame if merged is None else merged.merge(frame, on=entity_key, how="inner")
        return merged

    def type_entity_features(entity_table, entity_key):
        feature_cols = [c for c in entity_table.columns if c != entity_key]
        typed = entity_table.copy()
        numeric_cols, categorical_cols = [], []
        for col in feature_cols:
            numeric_candidate = pd.to_numeric(typed[col], errors="coerce")
            non_empty = typed[col].astype(str) != ""
            if bool(non_empty.any()) and bool(numeric_candidate[non_empty].notna().all()):
                typed[col] = numeric_candidate.astype(float)
                numeric_cols.append(col)
            else:
                typed[col] = typed[col].astype("category")
                categorical_cols.append(col)
        return typed, numeric_cols, categorical_cols

    def build_raw_pair_design_matrix(pair_frame, *, host_features, phage_features):
        merged = pair_frame.merge(host_features, on="bacteria", how="left", validate="many_to_one")
        merged = merged.merge(phage_features, on="phage", how="left", validate="many_to_one")
        return merged

    return SimpleNamespace(
        SLOT_PREFIXES=slot_prefixes,
        PAIR_SCORER_RANDOM_STATE=7,
        build_entity_feature_table=build_entity_feature_table,
        type_entity_features=type_entity_features,
        build_raw_pair_design_matrix=build_raw_pair_design_matrix,
        build_pair_scorer=lambda device_type: FakeEstimator(),
    )


def test_build_candidate_holdout_rows_replays_on_combined_non_holdout_training() -> None:
    candidate_module = build_fake_candidate_module()
    context = SimpleNamespace(
        split_frames={
            "train": pd.DataFrame(
                [
                    {
                        "pair_id": "B1__P1",
                        "bacteria": "B1",
                        "phage": "P1",
                        "label_any_lysis": "1",
                        "training_weight_v3": "1.0",
                        "retained_for_autoresearch": "1",
                    }
                ]
            ),
            "inner_val": pd.DataFrame(
                [
                    {
                        "pair_id": "B2__P1",
                        "bacteria": "B2",
                        "phage": "P1",
                        "label_any_lysis": "0",
                        "training_weight_v3": "1.0",
                        "retained_for_autoresearch": "1",
                    }
                ]
            ),
        },
        slot_artifacts={
            "host_surface": SimpleNamespace(
                entity_key="bacteria",
                frame=pd.DataFrame({"bacteria": ["B1", "B2", "B3"], "host_surface__score": ["2.0", "1.0", "0.5"]}),
            ),
            "host_typing": SimpleNamespace(
                entity_key="bacteria",
                frame=pd.DataFrame({"bacteria": ["B1", "B2", "B3"], "host_typing__score": ["2.0", "1.0", "0.5"]}),
            ),
            "host_stats": SimpleNamespace(
                entity_key="bacteria",
                frame=pd.DataFrame({"bacteria": ["B1", "B2", "B3"], "host_stats__score": ["2.0", "1.0", "0.5"]}),
            ),
            "phage_projection": SimpleNamespace(
                entity_key="phage",
                frame=pd.DataFrame({"phage": ["P1", "P2"], "phage_projection__score": ["2.0", "0.1"]}),
            ),
            "phage_stats": SimpleNamespace(
                entity_key="phage", frame=pd.DataFrame({"phage": ["P1", "P2"], "phage_stats__score": ["2.0", "0.1"]})
            ),
        },
    )
    holdout_frame = pd.DataFrame(
        [
            {
                "pair_id": "B3__P1",
                "bacteria": "B3",
                "phage": "P1",
                "label_any_lysis": "1",
                "training_weight_v3": "1.0",
                "retained_for_autoresearch": "1",
            },
            {
                "pair_id": "B3__P2",
                "bacteria": "B3",
                "phage": "P2",
                "label_any_lysis": "0",
                "training_weight_v3": "1.0",
                "retained_for_autoresearch": "1",
            },
        ]
    )

    rows = candidate_replay.build_candidate_holdout_rows(
        candidate_module=candidate_module,
        context=context,
        holdout_frame=holdout_frame,
        seed=7,
        device_type="cpu",
        include_host_defense=False,
    )

    assert [row["pair_id"] for row in rows] == ["B3__P1", "B3__P2"]
    assert rows[0]["predicted_probability"] > rows[1]["predicted_probability"]
    assert {row["label_hard_any_lysis"] for row in rows} == {0, 1}


def test_build_decision_summary_returns_no_honest_lift_without_auc_clear() -> None:
    comparator_rows = [
        {
            "arm_id": candidate_replay.BASELINE_ARM_ID,
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "label_hard_any_lysis": 1,
            "predicted_probability": 0.55,
        },
        {
            "arm_id": candidate_replay.BASELINE_ARM_ID,
            "pair_id": "B1__P2",
            "bacteria": "B1",
            "phage": "P2",
            "label_hard_any_lysis": 0,
            "predicted_probability": 0.45,
        },
        {
            "arm_id": candidate_replay.BASELINE_ARM_ID,
            "pair_id": "B2__P1",
            "bacteria": "B2",
            "phage": "P1",
            "label_hard_any_lysis": 0,
            "predicted_probability": 0.45,
        },
        {
            "arm_id": candidate_replay.BASELINE_ARM_ID,
            "pair_id": "B2__P2",
            "bacteria": "B2",
            "phage": "P2",
            "label_hard_any_lysis": 1,
            "predicted_probability": 0.55,
        },
    ]
    candidate_rows = [
        {
            "arm_id": candidate_replay.CANDIDATE_ARM_ID,
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "label_hard_any_lysis": 1,
            "predicted_probability": 0.56,
        },
        {
            "arm_id": candidate_replay.CANDIDATE_ARM_ID,
            "pair_id": "B1__P2",
            "bacteria": "B1",
            "phage": "P2",
            "label_hard_any_lysis": 0,
            "predicted_probability": 0.44,
        },
        {
            "arm_id": candidate_replay.CANDIDATE_ARM_ID,
            "pair_id": "B2__P1",
            "bacteria": "B2",
            "phage": "P1",
            "label_hard_any_lysis": 0,
            "predicted_probability": 0.44,
        },
        {
            "arm_id": candidate_replay.CANDIDATE_ARM_ID,
            "pair_id": "B2__P2",
            "bacteria": "B2",
            "phage": "P2",
            "label_hard_any_lysis": 1,
            "predicted_probability": 0.56,
        },
    ]
    bootstrap_summary = candidate_replay.bootstrap_holdout_metric_cis(
        {
            candidate_replay.BASELINE_ARM_ID: comparator_rows,
            candidate_replay.CANDIDATE_ARM_ID: candidate_rows,
        },
        bootstrap_samples=32,
        bootstrap_random_state=7,
        baseline_arm_id=candidate_replay.BASELINE_ARM_ID,
    )

    decision = candidate_replay.build_decision_summary(
        candidate_rows=candidate_rows,
        comparator_rows=comparator_rows,
        bootstrap_summary=bootstrap_summary,
    )

    assert decision["decision"] == "no_honest_lift"
    assert "AUC delta stays within bootstrap noise" in decision["decision_rationale"]


def test_replicate_candidate_writes_decision_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "train.py").write_text("# fixture\n", encoding="utf-8")
    write_json(candidate_dir / candidate_replay.IMPORT_MANIFEST_FILENAME, {"candidate_id": "fixture_candidate"})

    pair_table_path = tmp_path / "ar01_pair_table.csv"
    write_csv_rows(
        pair_table_path,
        [
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "split": "holdout",
                "label_any_lysis": "1",
                "retained_for_autoresearch": "1",
                "training_weight_v3": "1.0",
            },
            {
                "pair_id": "B1__P2",
                "bacteria": "B1",
                "phage": "P2",
                "split": "holdout",
                "label_any_lysis": "0",
                "retained_for_autoresearch": "1",
                "training_weight_v3": "1.0",
            },
        ],
    )
    contract_manifest = {
        "pair_table": {"path": str(pair_table_path)},
        "current_locked_comparator_benchmark": {
            "feature_lock_path": str(tmp_path / "missing_lock.json"),
            "model_summary_path": str(tmp_path / "missing_tg01.json"),
        },
    }
    contract_manifest_path = tmp_path / "ar01_manifest.json"
    write_json(contract_manifest_path, contract_manifest)
    fake_context = SimpleNamespace(
        provenance_manifest={"source_contract": {"pair_contract_manifest_path": str(contract_manifest_path)}},
        contract_manifest=contract_manifest,
    )

    monkeypatch.setattr(candidate_replay, "ensure_autoresearch_cache", lambda cache_dir: cache_dir)
    monkeypatch.setattr(
        candidate_replay,
        "load_module_from_path",
        lambda module_name, path: SimpleNamespace(load_and_validate_cache=lambda **kwargs: fake_context),
    )
    monkeypatch.setattr(
        candidate_replay,
        "build_candidate_holdout_rows",
        lambda **kwargs: [
            {
                "arm_id": candidate_replay.CANDIDATE_ARM_ID,
                "seed": kwargs["seed"],
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": 1,
                "predicted_probability": 0.65,
            },
            {
                "arm_id": candidate_replay.CANDIDATE_ARM_ID,
                "seed": kwargs["seed"],
                "pair_id": "B1__P2",
                "bacteria": "B1",
                "phage": "P2",
                "label_hard_any_lysis": 0,
                "predicted_probability": 0.35,
            },
        ],
    )
    monkeypatch.setattr(
        candidate_replay,
        "build_comparator_holdout_rows",
        lambda **kwargs: [
            {
                "arm_id": candidate_replay.BASELINE_ARM_ID,
                "seed": kwargs["seed"],
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": 1,
                "predicted_probability": 0.55,
            },
            {
                "arm_id": candidate_replay.BASELINE_ARM_ID,
                "seed": kwargs["seed"],
                "pair_id": "B1__P2",
                "bacteria": "B1",
                "phage": "P2",
                "label_hard_any_lysis": 0,
                "predicted_probability": 0.45,
            },
        ],
    )

    args = SimpleNamespace(
        candidate_dir=candidate_dir,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "decision",
        device_type="cpu",
        replication_seeds=[7, 17],
        bootstrap_samples=32,
        bootstrap_random_state=7,
        include_host_defense=False,
        skip_track_g_prerequisites=True,
        use_st03_split=False,
    )
    output_dir = candidate_replay.replicate_candidate(args)

    bundle = json.loads((output_dir / candidate_replay.DECISION_BUNDLE_FILENAME).read_text(encoding="utf-8"))
    assert bundle["candidate_id"] == "fixture_candidate"
    assert bundle["decision"] in {"promote", "no_honest_lift"}
    aggregated_rows = list(csv.DictReader((output_dir / candidate_replay.AGGREGATED_PREDICTIONS_FILENAME).open()))
    assert {row["arm_id"] for row in aggregated_rows} == {
        candidate_replay.CANDIDATE_ARM_ID,
        candidate_replay.BASELINE_ARM_ID,
    }
