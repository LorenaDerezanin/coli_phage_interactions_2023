#!/usr/bin/env python3
"""TL17: Build a deployable phage compatibility preprocessor beyond k-mer SVD."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import (
    CollapsedProfile,
    build_binary_value_index,
    collapse_duplicate_profiles,
    sha256_file,
)
from lyzortx.pipeline.track_l.steps.build_mechanistic_rbp_receptor_features import load_pharokka_rbp_gene_summary
from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import (
    TL17_DIRECT_BLOCK_ID,
    build_direct_feature_values,
    build_profile_presence,
    build_tl17_runtime_payload,
    extract_rbp_runtime_inputs,
)
from lyzortx.pipeline.track_l.steps.run_enrichment_analysis import CACHED_ANNOTATIONS_DIR, load_pharokka_phrog_matrices

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/tl17_phage_preprocessor")
DEFAULT_BASELINE_OUTPUT_DIR = Path(".scratch/tl17_baseline_generalized_inference_bundle")
PANEL_FEATURE_FILENAME = "tl17_panel_phage_features.csv"
PROFILE_METADATA_FILENAME = "tl17_rbp_profile_metadata.csv"
CANDIDATE_AUDIT_FILENAME = "tl17_phage_candidate_audit.csv"
PROJECTION_VALIDATION_FILENAME = "tl17_panel_projection_validation.csv"
PROJECTION_VALIDATION_SUMMARY_FILENAME = "tl17_panel_projection_validation_summary.csv"
HOLDOUT_METRIC_COMPARISON_FILENAME = "tl17_holdout_metric_comparison.csv"
PANEL_SURFACE_DELTA_FILENAME = "tl17_panel_surface_deltas.csv"
PANEL_SURFACE_SUMMARY_FILENAME = "tl17_panel_surface_summary.csv"
PANEL_SURFACE_OVERALL_FILENAME = "tl17_panel_surface_overall.csv"
MANIFEST_FILENAME = "tl17_phage_preprocessor_manifest.json"
PANEL_ANNOTATION_CACHE_DIRNAME = "panel_pharokka_annotations"
SUMMARY_COLUMNS = {
    "profile_count_column": "tl17_phage_rbp_profile_count",
    "gene_count_column": "tl17_phage_rbp_gene_count",
    "unique_phrog_count_column": "tl17_phage_rbp_unique_phrog_count",
}
BLOCK_ID_KMERS = "track_d_phage_genomic_kmers"
BLOCK_ID_RBP = TL17_DIRECT_BLOCK_ID
BLOCK_ID_ANTIDEF = "tl04_antidef_defense_evasion"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_ST02_PAIR_TABLE_PATH,
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_ST03_SPLIT_ASSIGNMENTS_PATH,
    )
    parser.add_argument(
        "--defense-subtypes-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_DEFENSE_SUBTYPES_PATH,
    )
    parser.add_argument(
        "--phage-kmer-feature-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_PHAGE_KMER_FEATURE_PATH,
    )
    parser.add_argument(
        "--phage-kmer-svd-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_PHAGE_KMER_SVD_PATH,
    )
    parser.add_argument(
        "--tg01-summary-path",
        type=Path,
        default=build_generalized_inference_bundle.DEFAULT_TG01_SUMMARY_PATH,
    )
    parser.add_argument("--cached-annotations-dir", type=Path, default=CACHED_ANNOTATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=build_generalized_inference_bundle.DEFAULT_CALIBRATION_FOLD,
    )
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bundle_update(
    *,
    bundle_path: Path,
    manifest_path: Path,
    bundle_updates: Mapping[str, object],
    artifact_updates: Mapping[str, object],
) -> None:
    bundle = joblib.load(bundle_path)
    bundle.update(dict(bundle_updates))
    bundle["artifacts"].update(dict(artifact_updates))
    joblib.dump(bundle, bundle_path)

    manifest = _load_json(manifest_path)
    manifest.update({key: value for key, value in bundle_updates.items() if key != "deployable_runtime"})
    manifest["artifact_hashes"]["bundle"] = sha256_file(bundle_path)
    write_json(manifest_path, manifest)


def build_profile_metadata_rows(profiles: Sequence[CollapsedProfile]) -> list[dict[str, object]]:
    return [
        {
            "profile_id": profile.profile_id,
            "representative_feature": profile.representative_feature,
            "member_features": "|".join(profile.member_features),
            "member_count": len(profile.member_features),
            "carrier_count": profile.carrier_count,
            "direct_column": profile.direct_column,
        }
        for profile in profiles
    ]


def build_panel_feature_rows(
    *,
    phages: Sequence[str],
    profiles: Sequence[CollapsedProfile],
    collapsed_matrix: Any,
    cached_annotations_dir: Path,
) -> list[dict[str, object]]:
    tl17_profiles = [tl17_profile_from_collapsed(profile) for profile in profiles]
    tl17_summary = tl17_summary_runtime()
    profile_presence_by_phage = build_binary_value_index(
        phages,
        [profile.profile_id for profile in profiles],
        collapsed_matrix,
    )
    rbp_summary = load_pharokka_rbp_gene_summary(cached_annotations_dir, phages)
    feature_rows: list[dict[str, object]] = []
    for phage in phages:
        summary = rbp_summary.get(phage)
        if summary is None:
            raise ValueError(f"Missing Pharokka RBP summary for phage {phage}")
        present_features, _ = extract_rbp_runtime_inputs(
            cached_annotations_dir / f"{phage}_cds_final_merged_output.tsv"
        )
        values = build_direct_feature_values(
            present_features=present_features,
            rbp_gene_count=int(summary["pharokka_rbp_gene_count"]),
            profile_presence=profile_presence_by_phage[phage],
            profiles=tl17_profiles,
            summary=tl17_summary,
        )
        feature_rows.append({"phage": phage, **values})
    return feature_rows


def tl17_profile_from_collapsed(profile: CollapsedProfile):
    from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import Tl17ProfileRuntime

    return Tl17ProfileRuntime(
        profile_id=profile.profile_id,
        direct_column=profile.direct_column,
        member_features=profile.member_features,
    )


def tl17_summary_runtime():
    from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import Tl17SummaryRuntime

    return Tl17SummaryRuntime(**SUMMARY_COLUMNS)


def build_candidate_audit_rows(
    *,
    kmer_feature_columns: Sequence[str],
    rbp_profiles: Sequence[CollapsedProfile],
    anti_def_profile_count: int,
    rbp_feature_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rbp_coverage = sum(int(row[SUMMARY_COLUMNS["profile_count_column"]]) > 0 for row in rbp_feature_rows)
    return [
        {
            "candidate_block_id": BLOCK_ID_KMERS,
            "mechanistic_link": "generic_sequence_composition",
            "panel_feature_width": len(kmer_feature_columns),
            "panel_phage_coverage": len(rbp_feature_rows),
            "chosen_for_tl17": 0,
            "rationale": (
                "Already deployable and useful, but tetranucleotide SVD is only an indirect host-range proxy and does "
                "not encode adsorption or defense-specific compatibility on its own."
            ),
        },
        {
            "candidate_block_id": BLOCK_ID_RBP,
            "mechanistic_link": "adsorption_receptor_recognition",
            "panel_feature_width": len(rbp_profiles) + len(SUMMARY_COLUMNS),
            "panel_phage_coverage": rbp_coverage,
            "chosen_for_tl17": 1,
            "rationale": (
                "Chosen strongest next candidate: RBP PHROG profiles are genome-derived, directly linked to "
                "adsorption biology, and provide a richer deployable compatibility surface than k-mer SVD alone."
            ),
        },
        {
            "candidate_block_id": BLOCK_ID_ANTIDEF,
            "mechanistic_link": "host_defense_evasion",
            "panel_feature_width": anti_def_profile_count,
            "panel_phage_coverage": 66,
            "chosen_for_tl17": 0,
            "rationale": (
                "Deployable but weaker as the next phage-side candidate: Track L already showed the anti-defense path "
                "is sparser and dominated by generic methyltransferase-like annotations."
            ),
        },
    ]


def build_projection_validation_rows(
    *,
    phage_feature_rows: Sequence[Mapping[str, object]],
    runtime_payload: Mapping[str, object],
    cached_annotations_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import parse_tl17_runtime_payload

    feature_by_phage = {str(row["phage"]): dict(row) for row in phage_feature_rows}
    profiles, summary = parse_tl17_runtime_payload(runtime_payload)
    validation_rows: list[dict[str, object]] = []
    for phage, expected_row in sorted(feature_by_phage.items()):
        annotation_tsv_path = cached_annotations_dir / f"{phage}_cds_final_merged_output.tsv"
        present_features, rbp_gene_count = extract_rbp_runtime_inputs(annotation_tsv_path)
        profile_presence = build_profile_presence(present_features, profiles)
        projected_values = build_direct_feature_values(
            present_features=present_features,
            rbp_gene_count=rbp_gene_count,
            profile_presence=profile_presence,
            profiles=profiles,
            summary=summary,
        )
        exact_match = int(all(projected_values[column] == int(expected_row[column]) for column in projected_values))
        validation_rows.append(
            {
                "phage": phage,
                "exact_match": exact_match,
                "expected_profile_count": int(expected_row[summary.profile_count_column]),
                "projected_profile_count": projected_values[summary.profile_count_column],
                "expected_rbp_gene_count": int(expected_row[summary.gene_count_column]),
                "projected_rbp_gene_count": projected_values[summary.gene_count_column],
                "expected_unique_phrog_count": int(expected_row[summary.unique_phrog_count_column]),
                "projected_unique_phrog_count": projected_values[summary.unique_phrog_count_column],
            }
        )
    summary_rows = [
        {
            "panel_phage_count": len(validation_rows),
            "exact_match_count": sum(int(row["exact_match"]) for row in validation_rows),
            "nonzero_profile_count_phages": sum(int(row["expected_profile_count"]) > 0 for row in validation_rows),
        }
    ]
    return validation_rows, summary_rows


def persist_candidate_runtime_contract(bundle_path: Path, runtime_payload: Mapping[str, object]) -> None:
    bundle = joblib.load(bundle_path)
    bundle["deployable_runtime"] = {
        TL17_DIRECT_BLOCK_ID: {
            **dict(runtime_payload),
            "panel_annotation_cache_dirname": PANEL_ANNOTATION_CACHE_DIRNAME,
        }
    }
    joblib.dump(bundle, bundle_path)


def copy_panel_annotation_cache(cached_annotations_dir: Path, output_dir: Path) -> Path:
    destination_dir = output_dir / PANEL_ANNOTATION_CACHE_DIRNAME
    ensure_directory(destination_dir)
    for source_path in sorted(cached_annotations_dir.glob("*_cds_final_merged_output.tsv")):
        destination_path = destination_dir / source_path.name
        destination_path.write_bytes(source_path.read_bytes())
    copied_paths = sorted(destination_dir.glob("*_cds_final_merged_output.tsv"))
    if not copied_paths:
        raise FileNotFoundError(f"No cached Pharokka merged TSVs found in {cached_annotations_dir}")
    return destination_dir


def build_holdout_metric_rows(
    *,
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
) -> list[dict[str, object]]:
    metric_directions = {
        "roc_auc": "higher",
        "average_precision": "higher",
        "brier_score": "lower",
        "log_loss": "lower",
    }
    rows = []
    for metric, direction in metric_directions.items():
        baseline_value = float(baseline_metrics[metric])
        candidate_value = float(candidate_metrics[metric])
        if direction == "higher":
            delta = candidate_value - baseline_value
        else:
            delta = baseline_value - candidate_value
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta_in_favor_of_candidate": delta,
            }
        )
    return rows


def build_panel_surface_delta_rows(
    *,
    baseline_prediction_rows: Sequence[Mapping[str, str]],
    candidate_prediction_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    baseline_by_pair = {
        (str(row["bacteria"]), str(row["phage"])): (
            float(row["pred_lightgbm_isotonic"]),
            int(row["rank_lightgbm_isotonic"]),
        )
        for row in baseline_prediction_rows
    }
    rows = []
    for row in candidate_prediction_rows:
        key = (str(row["bacteria"]), str(row["phage"]))
        baseline_probability, baseline_rank = baseline_by_pair[key]
        candidate_probability = float(row["pred_lightgbm_isotonic"])
        candidate_rank = int(row["rank_lightgbm_isotonic"])
        rows.append(
            {
                "bacteria": key[0],
                "phage": key[1],
                "baseline_p_lysis": baseline_probability,
                "candidate_p_lysis": candidate_probability,
                "abs_probability_delta": abs(candidate_probability - baseline_probability),
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "abs_rank_delta": abs(candidate_rank - baseline_rank),
            }
        )
    if not rows:
        raise ValueError("Panel prediction surface comparison produced zero rows.")
    return rows


def summarize_panel_surface_deltas(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_bacteria: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_bacteria.setdefault(str(row["bacteria"]), []).append(row)

    summary_rows = []
    all_probability_deltas = [float(row["abs_probability_delta"]) for row in rows]
    all_rank_deltas = [int(row["abs_rank_delta"]) for row in rows]
    changed_prediction_count = sum(
        float(row["abs_probability_delta"]) > 0.0 or int(row["abs_rank_delta"]) > 0 for row in rows
    )
    for bacteria, bacteria_rows in sorted(by_bacteria.items()):
        summary_rows.append(
            {
                "bacteria": bacteria,
                "changed_prediction_count": sum(
                    float(row["abs_probability_delta"]) > 0.0 or int(row["abs_rank_delta"]) > 0 for row in bacteria_rows
                ),
                "median_abs_probability_delta": float(
                    pd.Series([row["abs_probability_delta"] for row in bacteria_rows]).median()
                ),
                "max_abs_probability_delta": max(float(row["abs_probability_delta"]) for row in bacteria_rows),
                "identical_rank_count": sum(int(row["abs_rank_delta"]) == 0 for row in bacteria_rows),
            }
        )

    overall_rows = [
        {
            "pair_count": len(rows),
            "changed_prediction_count": changed_prediction_count,
            "median_abs_probability_delta": float(pd.Series(all_probability_deltas).median()),
            "max_abs_probability_delta": max(all_probability_deltas),
            "median_abs_rank_delta": float(pd.Series(all_rank_deltas).median()),
            "max_abs_rank_delta": max(all_rank_deltas),
        }
    ]
    return summary_rows, overall_rows


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_directory(args.baseline_output_dir)

    LOGGER.info("Starting TL17 deployable phage compatibility preprocessor build")
    if not args.skip_prerequisites:
        prerequisite_args = build_generalized_inference_bundle.parse_args(
            [
                "--st02-pair-table-path",
                str(args.st02_pair_table_path),
                "--st03-split-assignments-path",
                str(args.st03_split_assignments_path),
                "--defense-subtypes-path",
                str(args.defense_subtypes_path),
                "--phage-kmer-feature-path",
                str(args.phage_kmer_feature_path),
                "--phage-kmer-svd-path",
                str(args.phage_kmer_svd_path),
                "--tg01-summary-path",
                str(args.tg01_summary_path),
            ]
        )
        build_generalized_inference_bundle.ensure_prerequisite_outputs(prerequisite_args)
    if not args.cached_annotations_dir.exists():
        raise FileNotFoundError(f"Missing cached Pharokka annotations: {args.cached_annotations_dir}")

    lightgbm_params = build_generalized_inference_bundle.load_locked_lightgbm_params(args.tg01_summary_path)
    kmer_rows = read_csv_rows(args.phage_kmer_feature_path)
    if not kmer_rows:
        raise ValueError(f"No phage k-mer rows found in {args.phage_kmer_feature_path}")
    phages = sorted(str(row["phage"]) for row in kmer_rows)
    kmer_feature_columns = [column for column in kmer_rows[0].keys() if column != "phage"]

    rbp_matrix, rbp_phrog_names, anti_def_matrix, anti_def_phrog_names = load_pharokka_phrog_matrices(
        args.cached_annotations_dir,
        phages,
    )
    rbp_feature_names = [f"RBP_PHROG_{phrog}" for phrog in rbp_phrog_names]
    anti_def_feature_names = [f"ANTIDEF_PHROG_{phrog}" for phrog in anti_def_phrog_names]
    collapsed_rbp_matrix, rbp_profiles, _ = collapse_duplicate_profiles(
        rbp_feature_names,
        rbp_matrix,
        direct_column_prefix="tl17_phage_rbp",
    )
    _, anti_def_profiles, _ = collapse_duplicate_profiles(
        anti_def_feature_names,
        anti_def_matrix,
        direct_column_prefix="tl17_antidef_reference",
    )

    phage_feature_rows = build_panel_feature_rows(
        phages=phages,
        profiles=rbp_profiles,
        collapsed_matrix=collapsed_rbp_matrix,
        cached_annotations_dir=args.cached_annotations_dir,
    )
    profile_metadata_rows = build_profile_metadata_rows(rbp_profiles)
    runtime_payload = build_tl17_runtime_payload(
        profile_rows=profile_metadata_rows,
        summary_columns=SUMMARY_COLUMNS,
    )
    candidate_audit_rows = build_candidate_audit_rows(
        kmer_feature_columns=kmer_feature_columns,
        rbp_profiles=rbp_profiles,
        anti_def_profile_count=len(anti_def_profiles),
        rbp_feature_rows=phage_feature_rows,
    )
    projection_validation_rows, projection_validation_summary_rows = build_projection_validation_rows(
        phage_feature_rows=phage_feature_rows,
        runtime_payload=runtime_payload,
        cached_annotations_dir=args.cached_annotations_dir,
    )

    panel_feature_path = args.output_dir / PANEL_FEATURE_FILENAME
    profile_metadata_path = args.output_dir / PROFILE_METADATA_FILENAME
    candidate_audit_path = args.output_dir / CANDIDATE_AUDIT_FILENAME
    projection_validation_path = args.output_dir / PROJECTION_VALIDATION_FILENAME
    projection_validation_summary_path = args.output_dir / PROJECTION_VALIDATION_SUMMARY_FILENAME
    write_csv(panel_feature_path, list(phage_feature_rows[0].keys()), phage_feature_rows)
    write_csv(profile_metadata_path, list(profile_metadata_rows[0].keys()), profile_metadata_rows)
    write_csv(candidate_audit_path, list(candidate_audit_rows[0].keys()), candidate_audit_rows)
    write_csv(projection_validation_path, list(projection_validation_rows[0].keys()), projection_validation_rows)
    write_csv(
        projection_validation_summary_path,
        list(projection_validation_summary_rows[0].keys()),
        projection_validation_summary_rows,
    )

    baseline_result = build_generalized_inference_bundle.build_model_bundle(
        st02_pair_table_path=args.st02_pair_table_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
        defense_subtypes_path=args.defense_subtypes_path,
        phage_kmer_feature_path=args.phage_kmer_feature_path,
        phage_kmer_svd_path=args.phage_kmer_svd_path,
        output_dir=args.baseline_output_dir,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
    )
    candidate_result = build_generalized_inference_bundle.build_model_bundle(
        st02_pair_table_path=args.st02_pair_table_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
        defense_subtypes_path=args.defense_subtypes_path,
        phage_kmer_feature_path=args.phage_kmer_feature_path,
        phage_kmer_svd_path=args.phage_kmer_svd_path,
        output_dir=args.output_dir,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
        extra_phage_feature_rows=phage_feature_rows,
        extra_phage_feature_columns=[
            *[profile.direct_column for profile in rbp_profiles],
            *SUMMARY_COLUMNS.values(),
        ],
        bundle_task_id="TL17",
        bundle_format_version="tl17_deployable_phage_preprocessor_v1",
    )
    persist_candidate_runtime_contract(candidate_result["bundle_path"], runtime_payload)
    copy_panel_annotation_cache(args.cached_annotations_dir, args.output_dir)

    holdout_metric_rows = build_holdout_metric_rows(
        baseline_metrics=baseline_result["holdout_metrics"]["isotonic"],
        candidate_metrics=candidate_result["holdout_metrics"]["isotonic"],
    )
    baseline_prediction_rows = read_csv_rows(baseline_result["panel_predictions_path"])
    candidate_prediction_rows = read_csv_rows(candidate_result["panel_predictions_path"])
    surface_delta_rows = build_panel_surface_delta_rows(
        baseline_prediction_rows=baseline_prediction_rows,
        candidate_prediction_rows=candidate_prediction_rows,
    )
    surface_summary_rows, surface_overall_rows = summarize_panel_surface_deltas(surface_delta_rows)
    changed_prediction_count = int(surface_overall_rows[0]["changed_prediction_count"])
    if changed_prediction_count <= 0:
        raise ValueError("TL17 RBP deployable block changed zero panel predictions.")

    write_csv(
        args.output_dir / HOLDOUT_METRIC_COMPARISON_FILENAME,
        list(holdout_metric_rows[0].keys()),
        holdout_metric_rows,
    )
    write_csv(
        args.output_dir / PANEL_SURFACE_DELTA_FILENAME,
        list(surface_delta_rows[0].keys()),
        surface_delta_rows,
    )
    write_csv(
        args.output_dir / PANEL_SURFACE_SUMMARY_FILENAME,
        list(surface_summary_rows[0].keys()),
        surface_summary_rows,
    )
    write_csv(
        args.output_dir / PANEL_SURFACE_OVERALL_FILENAME,
        list(surface_overall_rows[0].keys()),
        surface_overall_rows,
    )

    _bundle_update(
        bundle_path=candidate_result["bundle_path"],
        manifest_path=candidate_result["manifest_path"],
        artifact_updates={
            "tl17_candidate_audit_filename": CANDIDATE_AUDIT_FILENAME,
            "tl17_panel_features_filename": PANEL_FEATURE_FILENAME,
            "tl17_projection_validation_filename": PROJECTION_VALIDATION_FILENAME,
        },
        bundle_updates={
            "deployable_runtime": {
                TL17_DIRECT_BLOCK_ID: {
                    **runtime_payload,
                    "panel_annotation_cache_dirname": PANEL_ANNOTATION_CACHE_DIRNAME,
                }
            },
            "deployable_feature_blocks": [
                {
                    "block_id": "track_c_defense",
                    "status": "deployable_now",
                    "source": "raw_host_assembly",
                },
                {
                    "block_id": "track_d_phage_genomic_kmers",
                    "status": "deployable_now",
                    "source": "raw_phage_genome",
                },
                {
                    "block_id": TL17_DIRECT_BLOCK_ID,
                    "status": "deployable_in_this_task",
                    "source": "raw_phage_annotation",
                },
            ],
            "tl17_panel_surface": {
                "surface_changed_prediction_count": changed_prediction_count,
                "holdout_metric_comparison_path": str(args.output_dir / HOLDOUT_METRIC_COMPARISON_FILENAME),
                "panel_surface_overall_path": str(args.output_dir / PANEL_SURFACE_OVERALL_FILENAME),
            },
        },
    )

    manifest_path = args.output_dir / MANIFEST_FILENAME
    write_json(
        manifest_path,
        {
            "task_id": "TL17",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_selection": {
                "chosen_block_id": TL17_DIRECT_BLOCK_ID,
                "audit_path": str(candidate_audit_path),
                "audit_sha256": sha256_file(candidate_audit_path),
            },
            "runtime_projection": {
                "profile_count": len(rbp_profiles),
                "summary_columns": dict(SUMMARY_COLUMNS),
                "panel_projection_validation_path": str(projection_validation_path),
                "panel_projection_validation_sha256": sha256_file(projection_validation_path),
            },
            "roundtrip_surface": {
                "changed_prediction_count": changed_prediction_count,
                "holdout_metric_comparison_path": str(args.output_dir / HOLDOUT_METRIC_COMPARISON_FILENAME),
                "panel_surface_overall_path": str(args.output_dir / PANEL_SURFACE_OVERALL_FILENAME),
            },
            "bundle": {
                "path": str(candidate_result["bundle_path"]),
                "sha256": sha256_file(candidate_result["bundle_path"]),
            },
        },
    )
    LOGGER.info("Completed TL17 deployable phage compatibility preprocessor build")
    LOGGER.info("Chosen phage-side candidate: %s", TL17_DIRECT_BLOCK_ID)
    LOGGER.info("Panel-surface changed predictions: %d", changed_prediction_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
