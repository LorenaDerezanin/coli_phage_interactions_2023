"""Unit tests for TK06 external-data synthesis and lock decisions."""

from __future__ import annotations

import csv
import json

import pytest

from lyzortx.pipeline.track_k import run_track_k
from lyzortx.pipeline.track_k.steps.build_external_data_decision import main


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest(
    *,
    source_system_added: str,
    augmented_source_systems,
    augmented_metrics,
    lock_hash: str = "lock",
    cohort_hash: str = "cohort",
):
    return {
        "locked_feature_config_sha256": lock_hash,
        "tg01_best_params": {"n_estimators": 150, "learning_rate": 0.05},
        "input_hashes_sha256": {
            "st02_pair_table": "st02",
            "st03_split_assignments": "st03",
            "track_c_pair_table": "tc",
            "track_d_genome_kmers": "tdg",
            "track_d_distance": "tdd",
            "track_e_rbp_receptor_compatibility": "ter",
            "track_e_isolation_host_distance": "tei",
            "ti08_training_cohort_rows": cohort_hash,
        },
        "source_system_added": source_system_added,
        "baseline_metrics": {
            "roc_auc": 0.8,
            "top3_hit_rate_all_strains": 0.9,
            "brier_score": 0.12,
        },
        "augmented_metrics": augmented_metrics,
        "augmented_source_systems": augmented_source_systems,
    }


def test_main_keeps_internal_only_when_all_external_arms_are_neutral(tmp_path) -> None:
    tk01 = tmp_path / "tk01.json"
    tk02 = tmp_path / "tk02.json"
    tk03 = tmp_path / "tk03.json"
    tk04 = tmp_path / "tk04.json"
    tk05 = tmp_path / "tk05.json"
    v1_config = tmp_path / "v1_feature_configuration.json"
    output_dir = tmp_path / "out"

    v1_config.write_text(
        json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"], "source_lock_task_id": "TG05"}),
        encoding="utf-8",
    )
    _write_json(
        tk01,
        _manifest(
            source_system_added="vhrdb",
            augmented_source_systems=["internal", "vhrdb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk02,
        _manifest(
            source_system_added="basel",
            augmented_source_systems=["internal", "basel"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk03,
        _manifest(
            source_system_added="klebphacol",
            augmented_source_systems=["internal", "klebphacol"],
            augmented_metrics={"roc_auc": 0.799, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.121},
        ),
    )
    _write_json(
        tk04,
        _manifest(
            source_system_added="gpb",
            augmented_source_systems=["internal", "gpb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.899, "brier_score": 0.121},
        ),
    )
    _write_json(
        tk05,
        _manifest(
            source_system_added="tier_b",
            augmented_source_systems=["internal", "virus_host_db", "ncbi_virus_biosample"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )

    original_config = v1_config.read_text(encoding="utf-8")
    main(
        [
            "--v1-feature-config-path",
            str(v1_config),
            "--tk01-manifest-path",
            str(tk01),
            "--tk02-manifest-path",
            str(tk02),
            "--tk03-manifest-path",
            str(tk03),
            "--tk04-manifest-path",
            str(tk04),
            "--tk05-manifest-path",
            str(tk05),
            "--output-dir",
            str(output_dir),
        ]
    )

    summary_rows = _read_csv_rows(output_dir / "tk06_external_data_decision_summary.csv")
    manifest = json.loads((output_dir / "tk06_external_data_decision_manifest.json").read_text(encoding="utf-8"))

    assert len(summary_rows) == 6
    assert manifest["decision"] == "keep_internal_only_baseline"
    assert manifest["selected_source_systems"] == []
    assert manifest["selected_arm"] == "internal_only"
    assert manifest["best_candidate"] is None
    assert manifest["final_model_outputs"] is None
    assert v1_config.read_text(encoding="utf-8") == original_config


def test_main_updates_config_and_retrains_when_external_arm_adds_lift(tmp_path, monkeypatch) -> None:
    tk01 = tmp_path / "tk01.json"
    tk02 = tmp_path / "tk02.json"
    tk03 = tmp_path / "tk03.json"
    tk04 = tmp_path / "tk04.json"
    tk05 = tmp_path / "tk05.json"
    v1_config = tmp_path / "v1_feature_configuration.json"
    output_dir = tmp_path / "out"

    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    _write_json(
        tk01,
        _manifest(
            source_system_added="vhrdb",
            augmented_source_systems=["internal", "vhrdb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk02,
        _manifest(
            source_system_added="basel",
            augmented_source_systems=["internal", "vhrdb", "basel"],
            augmented_metrics={"roc_auc": 0.82, "top3_hit_rate_all_strains": 0.91, "brier_score": 0.11},
        ),
    )
    _write_json(
        tk03,
        _manifest(
            source_system_added="klebphacol",
            augmented_source_systems=["internal", "vhrdb", "klebphacol"],
            augmented_metrics={"roc_auc": 0.81, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk04,
        _manifest(
            source_system_added="gpb",
            augmented_source_systems=["internal", "gpb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk05,
        _manifest(
            source_system_added="tier_b",
            augmented_source_systems=["internal", "virus_host_db", "ncbi_virus_biosample"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )

    calls = {}

    def _fake_retrain(args, *, selected_source_systems, output_dir):
        calls["selected_source_systems"] = list(selected_source_systems)
        summary_path = output_dir / "fake_summary.json"
        summary_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        rankings_path = output_dir / "fake_rankings.csv"
        rankings_path.write_text("model_label\n", encoding="utf-8")
        return {
            "summary_path": str(summary_path),
            "summary_sha256": "summary",
            "holdout_rankings_path": str(rankings_path),
            "holdout_rankings_sha256": "rankings",
            "metrics": {"roc_auc": 0.82, "top3_hit_rate_all_strains": 0.91, "brier_score": 0.11},
        }

    monkeypatch.setattr(
        "lyzortx.pipeline.track_k.steps.build_external_data_decision._build_final_model_outputs",
        _fake_retrain,
    )

    main(
        [
            "--v1-feature-config-path",
            str(v1_config),
            "--tk01-manifest-path",
            str(tk01),
            "--tk02-manifest-path",
            str(tk02),
            "--tk03-manifest-path",
            str(tk03),
            "--tk04-manifest-path",
            str(tk04),
            "--tk05-manifest-path",
            str(tk05),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "tk06_external_data_decision_manifest.json").read_text(encoding="utf-8"))
    config = json.loads(v1_config.read_text(encoding="utf-8"))

    assert calls["selected_source_systems"] == ["vhrdb", "basel"]
    assert manifest["decision"] == "lock_external_sources"
    assert manifest["selected_source_systems"] == ["vhrdb", "basel"]
    assert config["source_lock_task_id"] == "TK06"
    assert config["locked_external_source_systems"] == ["vhrdb", "basel"]
    assert config["locked_training_arm"] == "internal_plus_vhrdb_plus_basel"


def test_main_rejects_non_comparable_manifests(tmp_path) -> None:
    tk01 = tmp_path / "tk01.json"
    tk02 = tmp_path / "tk02.json"
    tk03 = tmp_path / "tk03.json"
    tk04 = tmp_path / "tk04.json"
    tk05 = tmp_path / "tk05.json"
    v1_config = tmp_path / "v1_feature_configuration.json"

    v1_config.write_text(json.dumps({"winner_subset_blocks": ["defense", "phage_genomic"]}), encoding="utf-8")
    _write_json(
        tk01,
        _manifest(
            source_system_added="vhrdb",
            augmented_source_systems=["internal", "vhrdb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk02,
        _manifest(
            source_system_added="basel",
            augmented_source_systems=["internal", "basel"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
            lock_hash="different-lock",
        ),
    )
    _write_json(
        tk03,
        _manifest(
            source_system_added="klebphacol",
            augmented_source_systems=["internal", "klebphacol"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk04,
        _manifest(
            source_system_added="gpb",
            augmented_source_systems=["internal", "gpb"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )
    _write_json(
        tk05,
        _manifest(
            source_system_added="tier_b",
            augmented_source_systems=["internal", "virus_host_db", "ncbi_virus_biosample"],
            augmented_metrics={"roc_auc": 0.8, "top3_hit_rate_all_strains": 0.9, "brier_score": 0.12},
        ),
    )

    with pytest.raises(ValueError, match="locked feature config hash"):
        main(
            [
                "--v1-feature-config-path",
                str(v1_config),
                "--tk01-manifest-path",
                str(tk01),
                "--tk02-manifest-path",
                str(tk02),
                "--tk03-manifest-path",
                str(tk03),
                "--tk04-manifest-path",
                str(tk04),
                "--tk05-manifest-path",
                str(tk05),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_run_track_k_dispatches_external_data_decision(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(run_track_k.build_vhrdb_lift_report, "main", lambda argv: calls.append("vhrdb-lift"))
    monkeypatch.setattr(run_track_k.build_basel_lift_report, "main", lambda argv: calls.append("basel-lift"))
    monkeypatch.setattr(run_track_k.build_klebphacol_lift_report, "main", lambda argv: calls.append("klebphacol-lift"))
    monkeypatch.setattr(run_track_k.build_gpb_lift_report, "main", lambda argv: calls.append("gpb-lift"))
    monkeypatch.setattr(run_track_k.build_tier_b_lift_report, "main", lambda argv: calls.append("tier-b-lift"))
    monkeypatch.setattr(
        run_track_k.build_external_data_decision,
        "main",
        lambda argv: calls.append("external-data-decision"),
    )

    run_track_k.main(["--step", "external-data-decision"])
    assert calls == ["external-data-decision"]

    calls.clear()
    run_track_k.main(["--step", "all"])
    assert calls == [
        "vhrdb-lift",
        "basel-lift",
        "klebphacol-lift",
        "gpb-lift",
        "tier-b-lift",
        "external-data-decision",
    ]
