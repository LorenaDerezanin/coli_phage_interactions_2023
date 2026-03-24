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
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    IDENTIFIER_COLUMNS,
    build_feature_space,
    build_top3_ranking_rows,
    compute_binary_metrics,
    compute_top3_hit_rate,
    fit_final_estimator,
    make_lightgbm_estimator,
    merge_expanded_feature_rows,
)
from lyzortx.pipeline.track_k.steps.build_source_lift_helpers import (
    arm_name_for_source_systems,
    build_locked_feature_space,
    build_training_rows,
    canonical_source_systems,
    classify_lift,
    load_locked_v1_feature_config,
    load_source_training_rows,
    load_source_training_rows_for_systems,
    load_tg01_best_params,
    sha256,
    source_systems_label,
)

logger = logging.getLogger(__name__)

LOCKED_V1_FEATURE_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
TG01_SUMMARY_PATH = Path("lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json")
TI08_TRAINING_COHORT_PATH = Path(
    "lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"
)
TK01_MANIFEST_PATH = Path("lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement/tk01_vhrdb_lift_manifest.json")
TK02_MANIFEST_PATH = Path("lyzortx/generated_outputs/track_k/tk02_basel_lift_measurement/tk02_basel_lift_manifest.json")
TK03_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk03_klebphacol_lift_measurement/tk03_klebphacol_lift_manifest.json"
)
TK04_MANIFEST_PATH = Path("lyzortx/generated_outputs/track_k/tk04_gpb_lift_measurement/tk04_gpb_lift_manifest.json")
TK05_MANIFEST_PATH = Path(
    "lyzortx/generated_outputs/track_k/tk05_tier_b_lift_measurement/tk05_tier_b_lift_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_k/tk06_external_data_decision")
NEGLIGIBLE_DELTA_TOLERANCE = 0.001
TIER_B_SOURCE_SYSTEMS = ("virus_host_db", "ncbi_virus_biosample")
COMMON_INPUT_HASH_KEYS = (
    "st02_pair_table",
    "st03_split_assignments",
    "track_c_pair_table",
    "track_d_genome_kmers",
    "track_d_distance",
    "track_e_rbp_receptor_compatibility",
    "track_e_isolation_host_distance",
)
DEFAULT_SOURCE_LABELS = {
    "TK01": "vhrdb",
    "TK02": "basel",
    "TK03": "klebphacol",
    "TK04": "gpb",
    "TK05": "tier_b",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
    )
    parser.add_argument(
        "--track-c-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
    )
    parser.add_argument(
        "--track-d-genome-kmer-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv"),
    )
    parser.add_argument(
        "--track-d-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_d/phage_distance_embedding/phage_distance_embedding_features.csv"
        ),
    )
    parser.add_argument(
        "--track-e-rbp-compatibility-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/"
            "rbp_receptor_compatibility_features_v1.csv"
        ),
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/isolation_host_distance_feature_block/"
            "isolation_host_distance_features_v1.csv"
        ),
    )
    parser.add_argument("--v1-feature-config-path", type=Path, default=LOCKED_V1_FEATURE_CONFIG_PATH)
    parser.add_argument("--tg01-summary-path", type=Path, default=TG01_SUMMARY_PATH)
    parser.add_argument("--ti08-training-cohort-path", type=Path, default=TI08_TRAINING_COHORT_PATH)
    parser.add_argument("--tk01-manifest-path", type=Path, default=TK01_MANIFEST_PATH)
    parser.add_argument("--tk02-manifest-path", type=Path, default=TK02_MANIFEST_PATH)
    parser.add_argument("--tk03-manifest-path", type=Path, default=TK03_MANIFEST_PATH)
    parser.add_argument("--tk04-manifest-path", type=Path, default=TK04_MANIFEST_PATH)
    parser.add_argument("--tk05-manifest-path", type=Path, default=TK05_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return safe_round(value - baseline)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cohort_hashes_match(reference: Mapping[str, object], candidate: Mapping[str, object]) -> bool:
    reference_hashes = reference.get("input_hashes_sha256", {})
    candidate_hashes = candidate.get("input_hashes_sha256", {})
    if not isinstance(reference_hashes, dict) or not isinstance(candidate_hashes, dict):
        return False
    for key in COMMON_INPUT_HASH_KEYS:
        if reference_hashes.get(key) != candidate_hashes.get(key):
            return False
    return reference_hashes.get("ti08_training_cohort_rows") == candidate_hashes.get("ti08_training_cohort_rows")


def _validate_manifest_chain(manifests: Sequence[Tuple[str, Mapping[str, object]]]) -> None:
    reference_label, reference_manifest = manifests[0]
    reference_lock_hash = reference_manifest.get("locked_feature_config_sha256")
    reference_params = json.dumps(reference_manifest.get("tg01_best_params", {}), sort_keys=True)
    for label, manifest in manifests[1:]:
        if manifest.get("locked_feature_config_sha256") != reference_lock_hash:
            raise ValueError(
                f"{label} is not comparable to {reference_label}: locked feature config hash does not match."
            )
        candidate_params = json.dumps(manifest.get("tg01_best_params", {}), sort_keys=True)
        if candidate_params != reference_params:
            raise ValueError(f"{label} is not comparable to {reference_label}: TG01 best params do not match.")
        if not _cohort_hashes_match(reference_manifest, manifest):
            raise ValueError(f"{label} is not comparable to {reference_label}: core input hashes do not match.")


def _candidate_row(
    *,
    step: str,
    source: str,
    source_systems: Sequence[str],
    metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
) -> Dict[str, object]:
    delta_roc_auc = _delta(
        float(metrics["roc_auc"]) if metrics.get("roc_auc") is not None else None,
        float(baseline_metrics["roc_auc"]) if baseline_metrics.get("roc_auc") is not None else None,
    )
    delta_top3 = _delta(
        float(metrics["top3_hit_rate_all_strains"]) if metrics.get("top3_hit_rate_all_strains") is not None else None,
        float(baseline_metrics["top3_hit_rate_all_strains"])
        if baseline_metrics.get("top3_hit_rate_all_strains") is not None
        else None,
    )
    delta_brier = _delta(
        float(metrics["brier_score"]) if metrics.get("brier_score") is not None else None,
        float(baseline_metrics["brier_score"]) if baseline_metrics.get("brier_score") is not None else None,
    )
    lift_assessment = classify_lift(
        delta_roc_auc=delta_roc_auc,
        delta_top3=delta_top3,
        delta_brier=delta_brier,
        tolerance=NEGLIGIBLE_DELTA_TOLERANCE,
    )
    canonical_sources = canonical_source_systems(source_systems)
    return {
        "step": step,
        "source": source,
        "arm": arm_name_for_source_systems(canonical_sources),
        "evaluated_source_combination": source_systems_label(canonical_sources),
        "holdout_roc_auc": metrics.get("roc_auc"),
        "holdout_top3_hit_rate_all_strains": metrics.get("top3_hit_rate_all_strains"),
        "holdout_brier_score": metrics.get("brier_score"),
        "delta_roc_auc_vs_internal_only": delta_roc_auc,
        "delta_top3_vs_internal_only": delta_top3,
        "delta_brier_vs_internal_only": delta_brier,
        "lift_assessment_vs_internal_only": lift_assessment,
        "eligible_for_lock": "1" if lift_assessment == "adds" else "0",
    }


def _best_candidate(comparison_rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    eligible_rows = [dict(row) for row in comparison_rows if row["lift_assessment_vs_internal_only"] == "adds"]
    if not eligible_rows:
        return None

    def _sort_key(row: Mapping[str, object]) -> Tuple[float, float, float, str]:
        return (
            float(row["holdout_roc_auc"]),
            float(row["holdout_top3_hit_rate_all_strains"]),
            -float(row["holdout_brier_score"]),
            str(row["evaluated_source_combination"]),
        )

    return dict(max(eligible_rows, key=_sort_key))


def _load_manifest_sequence(args: argparse.Namespace) -> List[Tuple[str, Dict[str, object]]]:
    manifest_paths = [
        ("TK01", args.tk01_manifest_path),
        ("TK02", args.tk02_manifest_path),
        ("TK03", args.tk03_manifest_path),
        ("TK04", args.tk04_manifest_path),
        ("TK05", args.tk05_manifest_path),
    ]
    missing = [f"{label} ({path})" for label, path in manifest_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "TK06 requires completed TK01-TK05 manifests before it can lock the external-data decision. Missing: "
            + ", ".join(missing)
        )
    manifests = [(label, _load_json(path)) for label, path in manifest_paths]
    _validate_manifest_chain(manifests)
    return manifests


def _build_final_model_outputs(
    args: argparse.Namespace,
    *,
    selected_source_systems: Sequence[str],
    output_dir: Path,
) -> Dict[str, object]:
    if not args.skip_prerequisites:
        from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import ensure_prerequisite_outputs

        ensure_prerequisite_outputs(args)

    locked_config = load_locked_v1_feature_config(args.v1_feature_config_path)
    locked_subset_blocks = tuple(str(block) for block in locked_config["winner_subset_blocks"])
    tg01_best_params = load_tg01_best_params(args.tg01_summary_path)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)
    cohort_rows = read_csv_rows(args.ti08_training_cohort_path) if args.ti08_training_cohort_path.exists() else []

    track_d_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_d_genome_rows[0].keys() if column != "phage"]
            + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
        )
    )
    track_e_feature_columns = tuple(
        dict.fromkeys(
            [column for column in track_e_rbp_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
            + [column for column in track_e_isolation_rows[0].keys() if column not in IDENTIFIER_COLUMNS]
        )
    )
    full_feature_space = build_feature_space(
        st02_rows, track_c_pair_rows, track_d_feature_columns, track_e_feature_columns
    )
    locked_feature_space = build_locked_feature_space(full_feature_space, locked_subset_blocks)
    merged_rows = merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_isolation_rows),
    )

    source_rows_by_system: Dict[str, List[Dict[str, object]]] = {}
    for source_system in canonical_source_systems(selected_source_systems):
        if source_system in TIER_B_SOURCE_SYSTEMS:
            continue
        source_rows_by_system[source_system], _ = load_source_training_rows(merged_rows, cohort_rows, source_system)

    tier_b_rows: List[Dict[str, object]] = []
    if any(source_system in TIER_B_SOURCE_SYSTEMS for source_system in selected_source_systems):
        tier_b_rows, _ = load_source_training_rows_for_systems(merged_rows, cohort_rows, TIER_B_SOURCE_SYSTEMS)

    training_rows = build_training_rows(
        list(merged_rows), source_rows_by_system, ["internal", *selected_source_systems]
    )
    if tier_b_rows:
        training_rows = [*training_rows, *tier_b_rows]

    estimator_factory = lambda params, seed_offset: make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )
    sample_weight_key = "effective_training_weight" if tier_b_rows else None
    _, _, train_rows, eval_rows, probabilities = fit_final_estimator(
        training_rows,
        locked_feature_space,
        estimator_factory=estimator_factory,
        params=tg01_best_params,
        sample_weight_key=sample_weight_key,
    )
    scored_rows: List[Dict[str, object]] = []
    for row, probability in zip(eval_rows, probabilities):
        scored = dict(row)
        scored["probability"] = probability
        scored_rows.append(scored)

    holdout_metrics = compute_binary_metrics(
        [int(str(row["label_hard_any_lysis"])) for row in scored_rows],
        [float(row["probability"]) for row in scored_rows],
    )
    top3 = compute_top3_hit_rate(scored_rows, probability_key="probability")
    rankings_path = output_dir / "tk06_locked_external_holdout_top3_rankings.csv"
    summary_path = output_dir / "tk06_locked_external_model_summary.json"
    write_csv(
        rankings_path,
        fieldnames=[
            "model_label",
            "bacteria",
            "phage",
            "pair_id",
            "rank",
            "predicted_probability",
            "label_hard_any_lysis",
        ],
        rows=build_top3_ranking_rows(
            scored_rows,
            probability_key="probability",
            model_label=arm_name_for_source_systems(["internal", *selected_source_systems]),
        ),
    )
    summary_payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "selected_source_systems": list(canonical_source_systems(selected_source_systems)),
        "selected_arm": arm_name_for_source_systems(["internal", *selected_source_systems]),
        "train_row_count": len(train_rows),
        "holdout_metrics": {
            "roc_auc": holdout_metrics["roc_auc"],
            "brier_score": holdout_metrics["brier_score"],
            "top3_hit_rate_all_strains": top3["top3_hit_rate_all_strains"],
            "top3_hit_rate_susceptible_only": top3["top3_hit_rate_susceptible_only"],
        },
        "tg01_best_params": tg01_best_params,
        "locked_feature_config_path": str(args.v1_feature_config_path),
        "locked_feature_config_sha256": sha256(args.v1_feature_config_path),
        "output_paths": {
            "holdout_rankings": str(rankings_path),
        },
    }
    write_json(summary_path, summary_payload)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "holdout_rankings_path": str(rankings_path),
        "holdout_rankings_sha256": sha256(rankings_path),
        "metrics": summary_payload["holdout_metrics"],
    }


def _update_locked_config(
    *,
    config_path: Path,
    selected_source_systems: Sequence[str],
    decision_row: Mapping[str, object],
) -> Dict[str, object]:
    payload = _load_json(config_path)
    payload["source_lock_task_id"] = "TK06"
    payload["external_data_selection_policy"] = (
        "Promote only source combinations that improve at least one primary holdout metric "
        "(ROC-AUC, top-3 hit rate, or lower Brier) without worsening any other primary metric beyond tolerance."
    )
    payload["locked_external_source_systems"] = list(canonical_source_systems(selected_source_systems))
    payload["locked_training_sources"] = list(canonical_source_systems(["internal", *selected_source_systems]))
    payload["locked_training_arm"] = str(decision_row["arm"])
    payload["locked_external_holdout_roc_auc"] = decision_row["holdout_roc_auc"]
    payload["locked_external_holdout_top3_hit_rate_all_strains"] = decision_row["holdout_top3_hit_rate_all_strains"]
    payload["locked_external_holdout_brier_score"] = decision_row["holdout_brier_score"]
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("TK06 starting: synthesize Track K lift results and lock the external-data decision")
    ensure_directory(args.output_dir)

    manifests = _load_manifest_sequence(args)
    _, tk01_manifest = manifests[0]
    internal_baseline = dict(tk01_manifest["baseline_metrics"])

    comparison_rows = [
        {
            "step": "baseline",
            "source": "internal",
            "arm": "internal_only",
            "evaluated_source_combination": "internal",
            "holdout_roc_auc": internal_baseline["roc_auc"],
            "holdout_top3_hit_rate_all_strains": internal_baseline["top3_hit_rate_all_strains"],
            "holdout_brier_score": internal_baseline["brier_score"],
            "delta_roc_auc_vs_internal_only": 0.0,
            "delta_top3_vs_internal_only": 0.0,
            "delta_brier_vs_internal_only": 0.0,
            "lift_assessment_vs_internal_only": "baseline",
            "eligible_for_lock": "0",
        }
    ]
    for step_label, manifest in manifests:
        source = str(manifest.get("source_system_added", DEFAULT_SOURCE_LABELS[step_label]))
        source_systems = manifest.get("augmented_source_systems")
        if not isinstance(source_systems, list):
            source_systems = ["internal", source]
        comparison_rows.append(
            _candidate_row(
                step=step_label,
                source=source,
                source_systems=[str(value) for value in source_systems],
                metrics=manifest["augmented_metrics"],
                baseline_metrics=internal_baseline,
            )
        )

    best_candidate = _best_candidate(comparison_rows[1:])
    external_inclusion_earned = best_candidate is not None
    selected_source_systems: Tuple[str, ...] = ()
    locked_config_update: Optional[Dict[str, object]] = None
    final_model_outputs: Optional[Dict[str, object]] = None

    if external_inclusion_earned and best_candidate is not None:
        selected_source_systems = tuple(
            source
            for source in str(best_candidate["evaluated_source_combination"]).split("|")
            if source and source != "internal"
        )
        locked_config_update = _update_locked_config(
            config_path=args.v1_feature_config_path,
            selected_source_systems=selected_source_systems,
            decision_row=best_candidate,
        )
        final_model_outputs = _build_final_model_outputs(
            args,
            selected_source_systems=selected_source_systems,
            output_dir=args.output_dir,
        )

    summary_path = args.output_dir / "tk06_external_data_decision_summary.csv"
    manifest_path = args.output_dir / "tk06_external_data_decision_manifest.json"
    write_csv(
        summary_path,
        fieldnames=[
            "step",
            "source",
            "arm",
            "evaluated_source_combination",
            "holdout_roc_auc",
            "holdout_top3_hit_rate_all_strains",
            "holdout_brier_score",
            "delta_roc_auc_vs_internal_only",
            "delta_top3_vs_internal_only",
            "delta_brier_vs_internal_only",
            "lift_assessment_vs_internal_only",
            "eligible_for_lock",
        ],
        rows=comparison_rows,
    )
    write_json(
        manifest_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_external_data_decision",
            "summary_table_path": str(summary_path),
            "summary_table_sha256": sha256(summary_path),
            "internal_baseline_metrics": internal_baseline,
            "comparison_rows": comparison_rows,
            "best_candidate": best_candidate,
            "external_inclusion_earned": external_inclusion_earned,
            "selected_source_systems": list(selected_source_systems),
            "selected_arm": (
                arm_name_for_source_systems(["internal", *selected_source_systems])
                if selected_source_systems
                else "internal_only"
            ),
            "decision": ("lock_external_sources" if external_inclusion_earned else "keep_internal_only_baseline"),
            "decision_rationale": (
                "At least one external source combination improved the locked v1 holdout metrics without causing a "
                "countervailing regression."
                if external_inclusion_earned
                else "No external source combination improved the locked v1 holdout metrics vs the internal-only "
                "baseline, so v1 remains internal-only."
            ),
            "input_manifests": {
                label: {
                    "path": str(path),
                    "sha256": sha256(path),
                }
                for label, path in (
                    ("tk01", args.tk01_manifest_path),
                    ("tk02", args.tk02_manifest_path),
                    ("tk03", args.tk03_manifest_path),
                    ("tk04", args.tk04_manifest_path),
                    ("tk05", args.tk05_manifest_path),
                )
            },
            "locked_v1_feature_config_path": str(args.v1_feature_config_path),
            "locked_v1_feature_config_sha256": sha256(args.v1_feature_config_path),
            "locked_config_update": locked_config_update,
            "final_model_outputs": final_model_outputs,
        },
    )

    logger.info("TK06 completed.")
    logger.info("- Decision: %s", "lock external sources" if external_inclusion_earned else "keep internal-only")
    logger.info(
        "- Selected arm: %s",
        "internal_only"
        if not selected_source_systems
        else arm_name_for_source_systems(["internal", *selected_source_systems]),
    )
    logger.info("- Summary table: %s", summary_path)
    logger.info("- Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
