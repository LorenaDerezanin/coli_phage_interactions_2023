#!/usr/bin/env python3
"""TL17: Build a deployable phage compatibility preprocessor beyond k-mer SVD."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle
from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import (
    DEFAULT_MIN_FAMILY_PHAGE_SUPPORT,
    DEFAULT_MMSEQS_COMMAND,
    DEFAULT_MMSEQS_MIN_IDENTITY,
    DEFAULT_MMSEQS_MIN_QUERY_COVERAGE,
    SUMMARY_FAMILY_COUNT_COLUMN,
    SUMMARY_HIT_COUNT_COLUMN,
    TL17_BLOCK_ID,
    build_fasta_inventory_rows,
    build_reference_proteins,
    build_runtime_payload,
    project_panel_feature_rows,
    write_family_metadata_csv,
    write_reference_fasta,
    write_reference_metadata_csv,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/tl17_phage_compatibility_preprocessor")
DEFAULT_BASELINE_OUTPUT_DIR = Path(".scratch/tl17_baseline_generalized_inference_bundle")
DEFAULT_SURFACE_HOST_FASTA_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_PHAGE_METADATA_PATH = Path("data/genomics/phages/guelin_collection.csv")
DEFAULT_PHAGE_FASTA_DIR = Path("data/genomics/phages/FNA")
DEFAULT_CACHED_ANNOTATIONS_DIR = Path("data/annotations/pharokka")
DEFAULT_EXPECTED_PANEL_COUNT = 96
REFERENCE_FASTA_FILENAME = "tl17_rbp_reference_bank.faa"
REFERENCE_METADATA_FILENAME = "tl17_rbp_reference_metadata.csv"
FAMILY_METADATA_FILENAME = "tl17_rbp_family_metadata.csv"
FASTA_INVENTORY_FILENAME = "tl17_panel_fasta_inventory.csv"
FEATURE_AUDIT_FILENAME = "tl17_candidate_audit.csv"
PROJECTED_FEATURE_FILENAME = "tl17_panel_projected_phage_features.csv"
VALIDATION_SUMMARY_FILENAME = "tl17_validation_summary.csv"
SURFACE_DELTA_FILENAME = "tl17_surface_delta.csv"
SURFACE_SUMMARY_FILENAME = "tl17_surface_summary.csv"
MANIFEST_FILENAME = "tl17_phage_compatibility_manifest.json"
RUNTIME_FILENAME = "tl17_rbp_runtime.joblib"
SURFACE_HOSTS: tuple[str, ...] = ("EDL933", "55989", "LF82")
DEPLOYABLE_BUNDLE_FORMAT_VERSION = "tl17_surface_probe_bundle_v1"
LOCKED_LIGHTGBM_KEYS = ("learning_rate", "min_child_samples", "n_estimators", "num_leaves")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument(
        "--surface-host-fasta-dir",
        type=Path,
        default=DEFAULT_SURFACE_HOST_FASTA_DIR,
        help="Directory used only to define the real-example host names for the surface probe.",
    )
    parser.add_argument("--phage-metadata-path", type=Path, default=DEFAULT_PHAGE_METADATA_PATH)
    parser.add_argument("--phage-fasta-dir", type=Path, default=DEFAULT_PHAGE_FASTA_DIR)
    parser.add_argument("--cached-annotations-dir", type=Path, default=DEFAULT_CACHED_ANNOTATIONS_DIR)
    parser.add_argument("--expected-panel-count", type=int, default=DEFAULT_EXPECTED_PANEL_COUNT)
    parser.add_argument(
        "--min-family-phage-support",
        type=int,
        default=DEFAULT_MIN_FAMILY_PHAGE_SUPPORT,
        help="Retain only RBP families present in at least this many panel phages.",
    )
    parser.add_argument(
        "--mmseqs-command",
        nargs="+",
        default=list(DEFAULT_MMSEQS_COMMAND),
        help="Command prefix used to invoke mmseqs easy-search.",
    )
    parser.add_argument("--min-percent-identity", type=float, default=DEFAULT_MMSEQS_MIN_IDENTITY)
    parser.add_argument("--min-query-coverage", type=float, default=DEFAULT_MMSEQS_MIN_QUERY_COVERAGE)
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
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=build_generalized_inference_bundle.DEFAULT_CALIBRATION_FOLD,
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_locked_lightgbm_params(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    params = dict(payload["lightgbm"]["best_params"])
    missing = [key for key in LOCKED_LIGHTGBM_KEYS if key not in params]
    if missing:
        raise ValueError(f"TG01 summary at {path} is missing locked LightGBM keys: {', '.join(missing)}")
    return params


def build_candidate_audit_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_block_id": "track_d_phage_genomic_kmers",
            "status": "baseline_only",
            "chosen_for_tl17": 0,
            "rationale": (
                "Tetranucleotide SVD is deployable and already in TL08/TL13, but it mostly captures broad genome "
                "composition rather than a defensible adsorption or defense-evasion mechanism."
            ),
        },
        {
            "candidate_block_id": "track_d_viridic_distance_embedding",
            "status": "not_chosen",
            "chosen_for_tl17": 0,
            "rationale": (
                "A VIRIDIC/tree embedding is generic phage relatedness, not a direct compatibility block, and the repo "
                "does not yet ship a raw-genome projector that would place arbitrary new phages into that space."
            ),
        },
        {
            "candidate_block_id": "tl04_antidef_direct_phage_block",
            "status": "not_chosen",
            "chosen_for_tl17": 0,
            "rationale": (
                "Anti-defense genes are plausibly relevant to host compatibility and TL13 already carries that path, "
                "but defense evasion is downstream of adsorption and therefore a weaker next phage-side candidate than "
                "an explicit adsorption-protein block."
            ),
        },
        {
            "candidate_block_id": TL17_BLOCK_ID,
            "status": "chosen",
            "chosen_for_tl17": 1,
            "rationale": (
                "RBP-family projection is the strongest next deployable candidate because receptor-binding proteins are "
                "the phage-side molecules most directly tied to adsorption and host-range gating. TL17 freezes a panel "
                "RBP reference bank and projects raw phage FASTAs into that family space without using panel-only host "
                "metadata or label-derived weights."
            ),
        },
    ]


def write_candidate_audit(output_dir: Path) -> Path:
    rows = build_candidate_audit_rows()
    output_path = output_dir / FEATURE_AUDIT_FILENAME
    write_csv(output_path, list(rows[0].keys()), rows)
    return output_path


def select_surface_hosts(surface_host_fasta_dir: Path, available_bacteria: set[str]) -> tuple[str, ...]:
    host_names = [path.stem for path in sorted(surface_host_fasta_dir.glob("*.fa*"))]
    selected = tuple(host for host in host_names if host in available_bacteria and host in SURFACE_HOSTS)
    if not selected:
        raise ValueError(
            "TL17 surface probe found no overlap between the committed validation-subset host names and panel predictions."
        )
    return selected


def build_surface_delta_rows(
    *,
    baseline_predictions_path: Path,
    candidate_predictions_path: Path,
    surface_hosts: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline = pd.read_csv(baseline_predictions_path)
    candidate = pd.read_csv(candidate_predictions_path)
    baseline = baseline[baseline["bacteria"].isin(surface_hosts)].copy()
    candidate = candidate[candidate["bacteria"].isin(surface_hosts)].copy()
    if baseline.empty or candidate.empty:
        raise ValueError("TL17 surface probe selected zero predictions for the chosen surface hosts.")
    merged = baseline.merge(
        candidate,
        on=["pair_id", "bacteria", "phage", "split_holdout", "split_cv5_fold", "label_hard_any_lysis"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("TL17 surface probe produced zero overlapping prediction rows.")
    delta_rows = []
    for row in merged.to_dict("records"):
        delta_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "baseline_probability": float(row["pred_lightgbm_isotonic_baseline"]),
                "candidate_probability": float(row["pred_lightgbm_isotonic_candidate"]),
                "abs_probability_delta": abs(
                    float(row["pred_lightgbm_isotonic_candidate"]) - float(row["pred_lightgbm_isotonic_baseline"])
                ),
                "baseline_rank": int(row["rank_lightgbm_isotonic_baseline"]),
                "candidate_rank": int(row["rank_lightgbm_isotonic_candidate"]),
                "rank_delta": int(row["rank_lightgbm_isotonic_candidate"])
                - int(row["rank_lightgbm_isotonic_baseline"]),
            }
        )
    summary_rows = []
    delta_frame = pd.DataFrame(delta_rows)
    for bacteria, group in delta_frame.groupby("bacteria", sort=True):
        summary_rows.append(
            {
                "bacteria": bacteria,
                "changed_prediction_count": int((group["abs_probability_delta"] > 0).sum()),
                "median_abs_probability_delta": float(group["abs_probability_delta"].median()),
                "max_abs_probability_delta": float(group["abs_probability_delta"].max()),
                "identical_rank_count": int((group["rank_delta"] == 0).sum()),
            }
        )
    return delta_rows, summary_rows


def build_validation_summary_rows(
    *,
    projected_feature_rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str],
    surface_summary_rows: Sequence[Mapping[str, object]],
    selected_surface_hosts: Sequence[str],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(projected_feature_rows)
    nonzero_phages = int((frame[SUMMARY_FAMILY_COUNT_COLUMN] > 0).sum())
    nonzero_features = int((frame[list(feature_columns)].sum(axis=0) > 0).sum())
    changed_predictions = sum(int(row["changed_prediction_count"]) for row in surface_summary_rows)
    return [
        {
            "metric": "projected_panel_phage_count",
            "value": len(projected_feature_rows),
        },
        {
            "metric": "tl17_family_feature_count",
            "value": len(feature_columns),
        },
        {
            "metric": "nonzero_family_feature_count",
            "value": nonzero_features,
        },
        {
            "metric": "panel_phages_with_any_tl17_feature",
            "value": nonzero_phages,
        },
        {
            "metric": "surface_probe_host_count",
            "value": len(selected_surface_hosts),
        },
        {
            "metric": "surface_probe_changed_prediction_count",
            "value": changed_predictions,
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_directory(args.baseline_output_dir)

    LOGGER.info("Starting TL17 phage compatibility preprocessor build")

    base_args = build_generalized_inference_bundle.parse_args(
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
            "--output-dir",
            str(args.output_dir / ".tl17_prereq_probe"),
            "--calibration-fold",
            str(args.calibration_fold),
            "--random-state",
            str(args.random_state),
        ]
        + (["--skip-prerequisites"] if args.skip_prerequisites else [])
    )
    if not args.skip_prerequisites:
        build_generalized_inference_bundle.ensure_prerequisite_outputs(base_args)
    lightgbm_params = load_locked_lightgbm_params(args.tg01_summary_path)

    candidate_audit_path = write_candidate_audit(args.output_dir)
    fasta_inventory_rows = build_fasta_inventory_rows(
        phage_metadata_path=args.phage_metadata_path,
        fna_dir=args.phage_fasta_dir,
        expected_panel_count=args.expected_panel_count,
    )
    fasta_inventory_path = args.output_dir / FASTA_INVENTORY_FILENAME
    write_csv(fasta_inventory_path, list(fasta_inventory_rows[0].keys()), fasta_inventory_rows)

    reference_rows, family_rows = build_reference_proteins(
        phage_metadata_path=args.phage_metadata_path,
        fna_dir=args.phage_fasta_dir,
        cached_annotations_dir=args.cached_annotations_dir,
        expected_panel_count=args.expected_panel_count,
        min_family_phage_support=args.min_family_phage_support,
    )
    reference_fasta_path = write_reference_fasta(reference_rows, args.output_dir / REFERENCE_FASTA_FILENAME)
    reference_metadata_path = write_reference_metadata_csv(
        reference_rows, args.output_dir / REFERENCE_METADATA_FILENAME
    )
    family_metadata_path = write_family_metadata_csv(family_rows, args.output_dir / FAMILY_METADATA_FILENAME)

    runtime_payload = build_runtime_payload(
        family_rows=family_rows,
        reference_rows=reference_rows,
        min_percent_identity=args.min_percent_identity,
        min_query_coverage=args.min_query_coverage,
        mmseqs_command=tuple(args.mmseqs_command),
    )
    runtime_path = args.output_dir / RUNTIME_FILENAME
    joblib.dump(runtime_payload, runtime_path)

    panel_feature_rows = project_panel_feature_rows(
        phage_metadata_path=args.phage_metadata_path,
        fna_dir=args.phage_fasta_dir,
        expected_panel_count=args.expected_panel_count,
        runtime_payload=runtime_payload,
        reference_fasta_path=reference_fasta_path,
        scratch_root=args.output_dir / ".scratch_panel_projection",
    )
    projected_feature_path = args.output_dir / PROJECTED_FEATURE_FILENAME
    write_csv(projected_feature_path, list(panel_feature_rows[0].keys()), panel_feature_rows)
    feature_columns = [
        column
        for column in panel_feature_rows[0].keys()
        if column not in {"phage", SUMMARY_HIT_COUNT_COLUMN, SUMMARY_FAMILY_COUNT_COLUMN}
    ]

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
        output_dir=args.output_dir / "surface_probe_bundle",
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
        extra_phage_feature_rows=panel_feature_rows,
        extra_phage_feature_columns=feature_columns,
        bundle_task_id="TL17",
        bundle_format_version=DEPLOYABLE_BUNDLE_FORMAT_VERSION,
    )

    available_bacteria = set(pd.read_csv(candidate_result["panel_predictions_path"])["bacteria"].unique())
    selected_surface_hosts = select_surface_hosts(args.surface_host_fasta_dir, available_bacteria)
    surface_delta_rows, surface_summary_rows = build_surface_delta_rows(
        baseline_predictions_path=baseline_result["panel_predictions_path"],
        candidate_predictions_path=candidate_result["panel_predictions_path"],
        surface_hosts=selected_surface_hosts,
    )
    changed_prediction_count = sum(int(row["changed_prediction_count"]) for row in surface_summary_rows)
    if changed_prediction_count <= 0:
        raise ValueError("TL17 candidate block changed zero panel predictions on the real-example surface probe.")
    surface_delta_path = args.output_dir / SURFACE_DELTA_FILENAME
    surface_summary_path = args.output_dir / SURFACE_SUMMARY_FILENAME
    write_csv(surface_delta_path, list(surface_delta_rows[0].keys()), surface_delta_rows)
    write_csv(surface_summary_path, list(surface_summary_rows[0].keys()), surface_summary_rows)

    validation_summary_rows = build_validation_summary_rows(
        projected_feature_rows=panel_feature_rows,
        feature_columns=feature_columns,
        surface_summary_rows=surface_summary_rows,
        selected_surface_hosts=selected_surface_hosts,
    )
    validation_summary_path = args.output_dir / VALIDATION_SUMMARY_FILENAME
    write_csv(validation_summary_path, list(validation_summary_rows[0].keys()), validation_summary_rows)

    manifest = {
        "step_name": "build_tl17_phage_compatibility_preprocessor",
        "task_id": "TL17",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_block_id": TL17_BLOCK_ID,
        "chosen_block_rationale": next(
            row["rationale"] for row in build_candidate_audit_rows() if row["candidate_block_id"] == TL17_BLOCK_ID
        ),
        "inputs": {
            "phage_metadata_path": str(args.phage_metadata_path),
            "phage_fasta_dir": str(args.phage_fasta_dir),
            "cached_annotations_dir": str(args.cached_annotations_dir),
            "st02_pair_table_path": str(args.st02_pair_table_path),
            "st03_split_assignments_path": str(args.st03_split_assignments_path),
            "defense_subtypes_path": str(args.defense_subtypes_path),
            "phage_kmer_feature_path": str(args.phage_kmer_feature_path),
            "phage_kmer_svd_path": str(args.phage_kmer_svd_path),
            "tg01_summary_path": str(args.tg01_summary_path),
        },
        "frozen_runtime_assets": {
            "runtime_payload_path": str(runtime_path),
            "reference_fasta_path": str(reference_fasta_path),
            "reference_metadata_path": str(reference_metadata_path),
            "family_metadata_path": str(family_metadata_path),
        },
        "matching_policy": {
            "mmseqs_command": list(args.mmseqs_command),
            "min_percent_identity": args.min_percent_identity,
            "min_query_coverage": args.min_query_coverage,
        },
        "outputs": {
            "candidate_audit_csv": str(candidate_audit_path),
            "fasta_inventory_csv": str(fasta_inventory_path),
            "projected_feature_csv": str(projected_feature_path),
            "validation_summary_csv": str(validation_summary_path),
            "surface_delta_csv": str(surface_delta_path),
            "surface_summary_csv": str(surface_summary_path),
            "baseline_bundle_path": str(baseline_result["bundle_path"]),
            "candidate_bundle_path": str(candidate_result["bundle_path"]),
        },
        "counts": {
            "retained_family_count": len(family_rows),
            "retained_reference_protein_count": len(reference_rows),
            "projected_panel_phage_count": len(panel_feature_rows),
            "surface_probe_host_count": len(selected_surface_hosts),
            "surface_probe_changed_prediction_count": changed_prediction_count,
        },
        "surface_probe": {
            "hosts": list(selected_surface_hosts),
            "host_source_dir": str(args.surface_host_fasta_dir),
            "candidate_bundle_path": str(candidate_result["bundle_path"]),
        },
        "deployability_scope": (
            "The runtime projects arbitrary raw phage FASTAs into a frozen panel RBP-family reference space. "
            "Novel phages carrying unseen adsorption families project to zeros rather than pretending to know the "
            "missing family, so the current scope is deployable but intentionally bounded."
        ),
    }
    write_json(args.output_dir / MANIFEST_FILENAME, manifest)

    LOGGER.info("Completed TL17 phage compatibility preprocessor build")
    LOGGER.info("Retained TL17 RBP families: %d", len(family_rows))
    LOGGER.info(
        "Panel phages with any TL17 family hit: %d",
        sum(int(row[SUMMARY_FAMILY_COUNT_COLUMN] > 0) for row in panel_feature_rows),
    )
    LOGGER.info("Real-example surface hosts: %s", ", ".join(selected_surface_hosts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
