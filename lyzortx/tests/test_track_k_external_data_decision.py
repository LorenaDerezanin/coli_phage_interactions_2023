"""Unit tests for TK06 external-data decision synthesis."""

from __future__ import annotations

import csv
import json

from lyzortx.pipeline.track_k.steps.build_external_data_decision_report import main


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_normalizes_all_lift_arms_back_to_internal_only_baseline(tmp_path) -> None:
    v1_config = tmp_path / "v1_feature_configuration.json"
    tk01_manifest = tmp_path / "tk01.json"
    tk02_manifest = tmp_path / "tk02.json"
    tk03_manifest = tmp_path / "tk03.json"
    tk04_manifest = tmp_path / "tk04.json"
    tk05_manifest = tmp_path / "tk05.json"
    output_dir = tmp_path / "out"

    _write_json(v1_config, {"winner_subset_blocks": ["defense", "phage_genomic"]})
    _write_json(
        tk01_manifest,
        {
            "baseline_metrics": {
                "roc_auc": 0.8,
                "top3_hit_rate_all_strains": 0.7,
                "brier_score": 0.2,
            },
            "augmented_metrics": {
                "roc_auc": 0.81,
                "top3_hit_rate_all_strains": 0.72,
                "brier_score": 0.19,
            },
            "lift_decision": "keep_vhrdb_for_followup_arms",
        },
    )
    _write_json(
        tk02_manifest,
        {
            "augmented_source_systems": ["internal", "vhrdb", "basel"],
            "augmented_metrics": {
                "roc_auc": 0.805,
                "top3_hit_rate_all_strains": 0.71,
                "brier_score": 0.198,
            },
            "lift_assessment": "neutral",
        },
    )
    _write_json(
        tk03_manifest,
        {
            "augmented_source_systems": ["internal", "vhrdb", "klebphacol"],
            "augmented_metrics": {
                "roc_auc": 0.82,
                "top3_hit_rate_all_strains": 0.73,
                "brier_score": 0.19,
            },
            "lift_assessment": "adds",
        },
    )
    _write_json(
        tk04_manifest,
        {
            "augmented_source_systems": ["internal", "vhrdb", "klebphacol", "gpb"],
            "augmented_metrics": {
                "roc_auc": 0.809,
                "top3_hit_rate_all_strains": 0.72,
                "brier_score": 0.191,
            },
            "lift_assessment": "neutral",
        },
    )
    _write_json(
        tk05_manifest,
        {
            "augmented_source_systems": [
                "internal",
                "vhrdb",
                "klebphacol",
                "virus_host_db",
                "ncbi_virus_biosample",
            ],
            "augmented_metrics": {
                "roc_auc": 0.815,
                "top3_hit_rate_all_strains": 0.721,
                "brier_score": 0.192,
            },
            "lift_assessment": "neutral",
        },
    )

    main(
        [
            "--v1-feature-config-path",
            str(v1_config),
            "--tk01-manifest-path",
            str(tk01_manifest),
            "--tk02-manifest-path",
            str(tk02_manifest),
            "--tk03-manifest-path",
            str(tk03_manifest),
            "--tk04-manifest-path",
            str(tk04_manifest),
            "--tk05-manifest-path",
            str(tk05_manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = list(csv.DictReader((output_dir / "tk06_external_data_comparison.csv").open("r", encoding="utf-8")))
    assert [row["task_id"] for row in summary] == ["TK01", "TK02", "TK03", "TK04", "TK05"]
    assert summary[1]["source_combination"] == "VHRdb + BASEL"
    assert summary[1]["delta_roc_auc_vs_internal_only"] == "0.005"
    assert summary[2]["source_combination"] == "VHRdb + KlebPhaCol"
    assert summary[2]["delta_top3_vs_internal_only"] == "0.03"

    manifest = json.loads((output_dir / "tk06_external_data_decision_manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_data_decision"] == "include_external_data"
    assert manifest["locked_training_arm"] == "internal_plus_vhrdb_plus_klebphacol"
    assert manifest["locked_external_source_systems"] == ["vhrdb", "klebphacol"]


def test_main_keeps_internal_only_when_no_external_arm_improves_without_harm(tmp_path) -> None:
    v1_config = tmp_path / "v1_feature_configuration.json"
    output_dir = tmp_path / "out"
    _write_json(v1_config, {"winner_subset_blocks": ["defense", "phage_genomic"]})

    for task_name in ("tk01", "tk02", "tk03", "tk04", "tk05"):
        manifest_path = tmp_path / f"{task_name}.json"
        payload = {
            "augmented_source_systems": ["internal", task_name],
            "augmented_metrics": {
                "roc_auc": 0.8,
                "top3_hit_rate_all_strains": 0.7,
                "brier_score": 0.2,
            },
            "lift_assessment": "neutral",
        }
        if task_name == "tk01":
            payload = {
                "baseline_metrics": {
                    "roc_auc": 0.8,
                    "top3_hit_rate_all_strains": 0.7,
                    "brier_score": 0.2,
                },
                "augmented_metrics": {
                    "roc_auc": 0.8,
                    "top3_hit_rate_all_strains": 0.7,
                    "brier_score": 0.2,
                },
                "lift_decision": "pending_external_artifact",
            }
        _write_json(manifest_path, payload)

    main(
        [
            "--v1-feature-config-path",
            str(v1_config),
            "--tk01-manifest-path",
            str(tmp_path / "tk01.json"),
            "--tk02-manifest-path",
            str(tmp_path / "tk02.json"),
            "--tk03-manifest-path",
            str(tmp_path / "tk03.json"),
            "--tk04-manifest-path",
            str(tmp_path / "tk04.json"),
            "--tk05-manifest-path",
            str(tmp_path / "tk05.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "tk06_external_data_decision_manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_data_decision"] == "keep_internal_only_baseline"
    assert manifest["locked_training_arm"] == "internal_only"
    assert manifest["locked_external_source_systems"] == []
