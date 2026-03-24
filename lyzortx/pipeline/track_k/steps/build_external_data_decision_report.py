#!/usr/bin/env python3
"""TK06: Synthesize Track K lift results and lock the external-data decision."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import safe_round
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import arm_name_for_source_systems
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import canonical_source_systems
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import classify_lift
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import sha256

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_k/tk06_external_data_decision")
DEFAULT_V1_FEATURE_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
DEFAULT_TK01_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement/tk01_vhrdb_lift_manifest.json"
)
DEFAULT_TK02_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk02_basel_lift_measurement/tk02_basel_lift_manifest.json"
)
DEFAULT_TK03_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk03_klebphacol_lift_measurement/tk03_klebphacol_lift_manifest.json"
)
DEFAULT_TK04_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk04_gpb_lift_measurement/tk04_gpb_lift_manifest.json"
)
DEFAULT_TK05_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk05_tier_b_lift_measurement/tk05_tier_b_lift_manifest.json"
)
NEGLIGIBLE_DELTA_TOLERANCE = 0.001
SOURCE_DISPLAY_NAMES = {
    "vhrdb": "VHRdb",
    "basel": "BASEL",
    "klebphacol": "KlebPhaCol",
    "gpb": "GPB",
    "virus_host_db": "Virus-Host DB",
    "ncbi_virus_biosample": "NCBI Virus/BioSample",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tk01-manifest-path", type=Path, default=DEFAULT_TK01_MANIFEST_PATH)
    parser.add_argument("--tk02-manifest-path", type=Path, default=DEFAULT_TK02_MANIFEST_PATH)
    parser.add_argument("--tk03-manifest-path", type=Path, default=DEFAULT_TK03_MANIFEST_PATH)
    parser.add_argument("--tk04-manifest-path", type=Path, default=DEFAULT_TK04_MANIFEST_PATH)
    parser.add_argument("--tk05-manifest-path", type=Path, default=DEFAULT_TK05_MANIFEST_PATH)
    parser.add_argument("--v1-feature-config-path", type=Path, default=DEFAULT_V1_FEATURE_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Track K manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Track K manifest is not a JSON object: {path}")
    return payload


def _metric(metrics: Mapping[str, object], key: str, *, manifest_path: Path) -> float:
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"Missing metric {key!r} in {manifest_path}")
    return float(value)


def _source_systems_for_manifest(task_id: str, default_source: str, manifest: Mapping[str, object]) -> Tuple[str, ...]:
    source_systems = manifest.get("augmented_source_systems")
    if isinstance(source_systems, list) and source_systems:
        return canonical_source_systems([str(source_system) for source_system in source_systems])
    if task_id == "TK01":
        return ("internal", default_source)
    raise KeyError(f"Missing augmented_source_systems in manifest for {task_id}")


def _display_label_for_source_systems(source_systems: Sequence[str]) -> str:
    external_sources = [
        source_system for source_system in canonical_source_systems(source_systems) if source_system != "internal"
    ]
    if not external_sources:
        return "internal-only"
    return " + ".join(SOURCE_DISPLAY_NAMES.get(source_system, source_system) for source_system in external_sources)


def _delta(value: float, baseline: float) -> float:
    return float(safe_round(value - baseline))


def _candidate_sort_key(candidate: Mapping[str, object]) -> Tuple[float, float, float, int]:
    delta_top3 = float(candidate["delta_top3_vs_internal_only"])
    delta_roc_auc = float(candidate["delta_roc_auc_vs_internal_only"])
    delta_brier = float(candidate["delta_brier_vs_internal_only"])
    source_count = len([source for source in candidate["source_systems"] if source != "internal"])
    return (delta_top3, delta_roc_auc, -delta_brier, -source_count)


def build_comparison_rows(
    manifest_specs: Sequence[Tuple[str, str, Path]],
) -> Tuple[List[Dict[str, object]], Dict[str, float], Dict[str, str]]:
    loaded_manifests = {task_id: _load_manifest(path) for task_id, _default_source, path in manifest_specs}

    tk01_path = next(path for task_id, _default_source, path in manifest_specs if task_id == "TK01")
    tk01_manifest = loaded_manifests["TK01"]
    baseline_metrics = tk01_manifest.get("baseline_metrics")
    if not isinstance(baseline_metrics, dict):
        raise KeyError(f"Missing baseline_metrics in {tk01_path}")

    internal_metrics = {
        "roc_auc": _metric(baseline_metrics, "roc_auc", manifest_path=tk01_path),
        "top3_hit_rate_all_strains": _metric(
            baseline_metrics,
            "top3_hit_rate_all_strains",
            manifest_path=tk01_path,
        ),
        "brier_score": _metric(baseline_metrics, "brier_score", manifest_path=tk01_path),
    }

    comparison_rows: List[Dict[str, object]] = []
    manifest_decisions: Dict[str, str] = {}

    for task_id, default_source, manifest_path in manifest_specs:
        manifest = loaded_manifests[task_id]
        augmented_metrics = manifest.get("augmented_metrics")
        if not isinstance(augmented_metrics, dict):
            raise KeyError(f"Missing augmented_metrics in {manifest_path}")

        source_systems = _source_systems_for_manifest(task_id, default_source, manifest)
        holdout_roc_auc = _metric(augmented_metrics, "roc_auc", manifest_path=manifest_path)
        holdout_top3 = _metric(augmented_metrics, "top3_hit_rate_all_strains", manifest_path=manifest_path)
        holdout_brier = _metric(augmented_metrics, "brier_score", manifest_path=manifest_path)
        delta_roc_auc = _delta(holdout_roc_auc, internal_metrics["roc_auc"])
        delta_top3 = _delta(holdout_top3, internal_metrics["top3_hit_rate_all_strains"])
        delta_brier = _delta(holdout_brier, internal_metrics["brier_score"])
        step_decision = str(manifest.get("lift_assessment") or manifest.get("lift_decision") or "")
        improves_without_harm = (
            classify_lift(
                delta_roc_auc=delta_roc_auc,
                delta_top3=delta_top3,
                delta_brier=delta_brier,
                tolerance=NEGLIGIBLE_DELTA_TOLERANCE,
            )
            == "adds"
        )
        comparison_rows.append(
            {
                "task_id": task_id,
                "source_combination": _display_label_for_source_systems(source_systems),
                "arm": arm_name_for_source_systems(source_systems),
                "source_systems": list(source_systems),
                "holdout_roc_auc": holdout_roc_auc,
                "holdout_top3_hit_rate_all_strains": holdout_top3,
                "holdout_brier_score": holdout_brier,
                "delta_roc_auc_vs_internal_only": delta_roc_auc,
                "delta_top3_vs_internal_only": delta_top3,
                "delta_brier_vs_internal_only": delta_brier,
                "step_decision": step_decision,
                "improves_without_harm": improves_without_harm,
            }
        )
        manifest_decisions[task_id] = step_decision

    return comparison_rows, internal_metrics, manifest_decisions


def select_locked_candidate(comparison_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    eligible_candidates = [row for row in comparison_rows if bool(row["improves_without_harm"])]
    if not eligible_candidates:
        return {
            "locked_training_arm": "internal_only",
            "locked_external_source_systems": [],
            "decision": "keep_internal_only_baseline",
            "decision_rationale": (
                "No evaluated external-data combination improved at least one metric without harming another metric "
                "on the locked internal-only baseline."
            ),
        }

    best_candidate = max(eligible_candidates, key=_candidate_sort_key)
    return {
        "locked_training_arm": str(best_candidate["arm"]),
        "locked_external_source_systems": [
            source_system for source_system in best_candidate["source_systems"] if source_system != "internal"
        ],
        "decision": "include_external_data",
        "decision_rationale": (
            "At least one external-data combination improved the locked baseline without harming another tracked "
            "metric; the selected arm maximizes top-3 lift, then ROC-AUC lift, then Brier improvement."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("TK06 starting: synthesize Track K lift results and lock the external-data decision")
    ensure_directory(args.output_dir)

    manifest_specs = (
        ("TK01", "vhrdb", args.tk01_manifest_path),
        ("TK02", "basel", args.tk02_manifest_path),
        ("TK03", "klebphacol", args.tk03_manifest_path),
        ("TK04", "gpb", args.tk04_manifest_path),
        ("TK05", "tier_b", args.tk05_manifest_path),
    )
    comparison_rows, internal_metrics, manifest_decisions = build_comparison_rows(manifest_specs)
    locked_candidate = select_locked_candidate(comparison_rows)

    summary_rows = [
        {
            "task_id": str(row["task_id"]),
            "source_combination": str(row["source_combination"]),
            "arm": str(row["arm"]),
            "holdout_roc_auc": row["holdout_roc_auc"],
            "holdout_top3_hit_rate_all_strains": row["holdout_top3_hit_rate_all_strains"],
            "holdout_brier_score": row["holdout_brier_score"],
            "delta_roc_auc_vs_internal_only": row["delta_roc_auc_vs_internal_only"],
            "delta_top3_vs_internal_only": row["delta_top3_vs_internal_only"],
            "delta_brier_vs_internal_only": row["delta_brier_vs_internal_only"],
            "step_decision": str(row["step_decision"]),
            "improves_without_harm": int(bool(row["improves_without_harm"])),
        }
        for row in comparison_rows
    ]
    output_summary_path = args.output_dir / "tk06_external_data_comparison.csv"
    output_manifest_path = args.output_dir / "tk06_external_data_decision_manifest.json"
    write_csv(
        output_summary_path,
        fieldnames=[
            "task_id",
            "source_combination",
            "arm",
            "holdout_roc_auc",
            "holdout_top3_hit_rate_all_strains",
            "holdout_brier_score",
            "delta_roc_auc_vs_internal_only",
            "delta_top3_vs_internal_only",
            "delta_brier_vs_internal_only",
            "step_decision",
            "improves_without_harm",
        ],
        rows=summary_rows,
    )
    write_json(
        output_manifest_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_external_data_decision_report",
            "v1_feature_config_path": str(args.v1_feature_config_path),
            "v1_feature_config_sha256": sha256(args.v1_feature_config_path),
            "input_paths": {task_id.lower() + "_manifest": str(path) for task_id, _source, path in manifest_specs},
            "input_hashes_sha256": {
                task_id.lower() + "_manifest": sha256(path) for task_id, _source, path in manifest_specs
            },
            "internal_only_baseline_metrics": internal_metrics,
            "comparison_rows": summary_rows,
            "manifest_decisions": manifest_decisions,
            "locked_training_arm": locked_candidate["locked_training_arm"],
            "locked_external_source_systems": locked_candidate["locked_external_source_systems"],
            "external_data_decision": locked_candidate["decision"],
            "decision_rationale": locked_candidate["decision_rationale"],
            "output_paths": {
                "comparison_summary": str(output_summary_path),
            },
        },
    )

    logger.info("TK06 completed.")
    logger.info("- External data decision: %s", locked_candidate["decision"])
    logger.info("- Locked training arm: %s", locked_candidate["locked_training_arm"])
    logger.info("- Locked external source systems: %s", locked_candidate["locked_external_source_systems"])
