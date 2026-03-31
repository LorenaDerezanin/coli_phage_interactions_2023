#!/usr/bin/env python3
"""TL13: Audit and rebuild the deployable generalized inference bundle."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import build_defense_column_mask
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle
from lyzortx.pipeline.track_l.steps import build_mechanistic_defense_evasion_features
from lyzortx.pipeline.track_l.steps import generalized_inference
from lyzortx.pipeline.track_l.steps import validate_vhdb_generalized_inference as tl09
from lyzortx.pipeline.track_l.steps.deployable_tl04_runtime import (
    TL04_DIRECT_BLOCK_ID,
    build_tl04_runtime_payload,
)
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import sha256_file
from lyzortx.pipeline.track_l.steps.run_enrichment_analysis import CACHED_ANNOTATIONS_DIR
from lyzortx.pipeline.track_l.steps.retrain_mechanistic_v1_model import load_tl11_feature_provenance

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/generalized_inference_bundle_tl13")
DEFAULT_BASELINE_OUTPUT_DIR = Path(".scratch/tl13_baseline_generalized_inference_bundle")
DEFAULT_TL04_FEATURE_PATH = build_mechanistic_defense_evasion_features.DEFAULT_OUTPUT_DIR / (
    "mechanistic_defense_evasion_features_v1.csv"
)
DEFAULT_TL04_MANIFEST_PATH = build_mechanistic_defense_evasion_features.DEFAULT_OUTPUT_DIR / (
    "mechanistic_defense_evasion_manifest_v1.json"
)
PARITY_AUDIT_FILENAME = "tl13_feature_parity_audit.csv"
ROUNDTRIP_REFERENCE_FILENAME = "tl13_roundtrip_panel_reference_predictions.csv"
ROUNDTRIP_HOST_COHORT_FILENAME = "tl13_roundtrip_panel_host_cohort.csv"
ROUNDTRIP_METRIC_COMPARISON_FILENAME = "tl13_roundtrip_metric_comparison.csv"
ROUNDTRIP_BASELINE_COMPARISON_FILENAME = "tl13_roundtrip_baseline_comparison.csv"
ROUNDTRIP_CANDIDATE_COMPARISON_FILENAME = "tl13_roundtrip_candidate_comparison.csv"
ROUNDTRIP_SURFACE_DELTA_FILENAME = "tl13_roundtrip_surface_deltas.csv"
ROUNDTRIP_SURFACE_SUMMARY_FILENAME = "tl13_roundtrip_surface_summary.csv"
PANEL_ANNOTATION_CACHE_DIRNAME = "panel_pharokka_annotations"
DEPLOYABLE_BLOCK_ID = "tl04_antidef_defense_evasion"
DEPLOYABLE_BUNDLE_FORMAT_VERSION = "tl13_deployable_inference_bundle_v1"
PREDECLARED_ROUNDTRIP_METRICS = {
    "median_abs_probability_delta_median": "lower",
    "max_abs_probability_delta_max": "lower",
    "identical_rank_count_total": "higher",
}


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--cached-annotations-dir", type=Path, default=CACHED_ANNOTATIONS_DIR)
    parser.add_argument("--tl04-feature-path", type=Path, default=DEFAULT_TL04_FEATURE_PATH)
    parser.add_argument("--tl04-manifest-path", type=Path, default=DEFAULT_TL04_MANIFEST_PATH)
    parser.add_argument("--panel-hosts-path", type=Path, default=tl09.DEFAULT_PANEL_HOSTS_PATH)
    parser.add_argument("--panel-phage-dir", type=Path, default=tl09.DEFAULT_PANEL_PHAGE_DIR)
    parser.add_argument("--panel-phage-metadata-path", type=Path, default=tl09.DEFAULT_PANEL_PHAGE_METADATA_PATH)
    parser.add_argument("--vhdb-url", default=tl09.DEFAULT_VHDB_URL)
    parser.add_argument("--assembly-summary-url", default=tl09.DEFAULT_ASSEMBLY_SUMMARY_URL)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--calibration-fold", type=int, default=build_generalized_inference_bundle.DEFAULT_CALIBRATION_FOLD
    )
    parser.add_argument("--skip-prerequisites", action="store_true")
    return parser.parse_args(argv)


def ensure_tl04_artifacts(
    *,
    tl04_feature_path: Path,
    tl04_manifest_path: Path,
    st03_split_assignments_path: Path,
) -> None:
    should_rebuild = False
    if tl04_feature_path.exists() and tl04_manifest_path.exists():
        try:
            load_tl11_feature_provenance(
                tl04_feature_path,
                tl04_manifest_path,
                "TL04",
                st03_split_assignments_path,
            )
            return
        except (FileNotFoundError, ValueError):
            should_rebuild = True
    else:
        should_rebuild = True
    if tl04_feature_path != DEFAULT_TL04_FEATURE_PATH or tl04_manifest_path != DEFAULT_TL04_MANIFEST_PATH:
        missing = tl04_feature_path if not tl04_feature_path.exists() else tl04_manifest_path
        raise FileNotFoundError(f"Missing TL04 deployable artifact: {missing}")
    if should_rebuild:
        LOGGER.info("TL04 deployable artifacts missing or stale; rebuilding TL04")
        build_mechanistic_defense_evasion_features.main(
            ["--st03-split-assignments-path", str(st03_split_assignments_path)]
        )
    load_tl11_feature_provenance(
        tl04_feature_path,
        tl04_manifest_path,
        "TL04",
        st03_split_assignments_path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tl04_contract(
    *,
    tl04_feature_path: Path,
    tl04_manifest_path: Path,
    defense_subtypes_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(tl04_manifest_path)
    outputs = dict(manifest["outputs"])
    feature_rows = read_csv_rows(tl04_feature_path)
    if not feature_rows:
        raise ValueError(f"No TL04 feature rows found in {tl04_feature_path}")
    metadata_rows = read_csv_rows(Path(outputs["feature_metadata_csv"]))
    profile_rows = read_csv_rows(Path(outputs["profile_metadata_csv"]))
    direct_columns = tuple(column for column in feature_rows[0].keys() if column.startswith("tl04_phage_antidef_"))
    pairwise_columns = tuple(column for column in feature_rows[0].keys() if column.startswith("tl04_pair_"))
    phage_rows_by_name: dict[str, dict[str, object]] = {}
    for row in feature_rows:
        phage = str(row["phage"])
        values = {column: row[column] for column in direct_columns}
        existing = phage_rows_by_name.get(phage)
        if existing is not None and existing != {"phage": phage, **values}:
            raise ValueError(f"TL04 direct phage features were inconsistent across pair rows for {phage}")
        phage_rows_by_name[phage] = {"phage": phage, **values}
    defense_rows = build_generalized_inference_bundle.read_defense_rows(defense_subtypes_path)
    defense_mask = build_defense_column_mask(defense_rows)
    runtime_payload = build_tl04_runtime_payload(
        profile_rows=profile_rows,
        metadata_rows=metadata_rows,
        defense_mask=defense_mask,
    )
    return {
        "feature_rows": feature_rows,
        "direct_phage_rows": list(phage_rows_by_name.values()),
        "direct_columns": direct_columns,
        "pairwise_columns": pairwise_columns,
        "runtime_payload": runtime_payload,
        "manifest": manifest,
        "metadata_path": Path(outputs["feature_metadata_csv"]),
        "profile_path": Path(outputs["profile_metadata_csv"]),
    }


def build_feature_parity_audit_rows() -> list[dict[str, object]]:
    return [
        {
            "training_block_id": "st04_v0_baseline_metadata",
            "deployable_status": "not_deployable",
            "included_in_tl13_bundle": 0,
            "rationale": "Panel-only assay and metadata baseline features are not reconstructable for arbitrary novel pairs.",
        },
        {
            "training_block_id": "track_c_defense",
            "deployable_status": "deployable_now",
            "included_in_tl13_bundle": 1,
            "rationale": "TL07 already projects DefenseFinder-derived host defense features directly from raw host assemblies.",
        },
        {
            "training_block_id": "track_c_omp_surface",
            "deployable_status": "not_deployable",
            "included_in_tl13_bundle": 0,
            "rationale": "The repo has no raw-host OMP/LPS/capsule projector for inference-time genomes, so this host block is not deployable now.",
        },
        {
            "training_block_id": "track_d_phage_genomic_kmers",
            "deployable_status": "deployable_now",
            "included_in_tl13_bundle": 1,
            "rationale": "TL06 already projects tetranucleotide SVD features from raw phage genomes using the saved bundle-local SVD.",
        },
        {
            "training_block_id": "track_e_curated_rbp_receptor_compatibility",
            "deployable_status": "not_deployable",
            "included_in_tl13_bundle": 0,
            "rationale": "This curated lookup depends on panel taxon metadata rather than a raw-genome projection contract for novel phages.",
        },
        {
            "training_block_id": "track_e_isolation_host_distance",
            "deployable_status": "not_deployable",
            "included_in_tl13_bundle": 0,
            "rationale": "Isolation-host distance requires isolation metadata and therefore cannot be derived from raw genomes at inference time.",
        },
        {
            "training_block_id": "tl03_rbp_receptor_pairwise",
            "deployable_status": "not_deployable",
            "included_in_tl13_bundle": 0,
            "rationale": "TL11/TL12 showed the mechanistic pairwise path is dead-ended for the current v1 lock, and the repo still lacks the raw-host receptor projector needed to deploy any TL03 subset honestly.",
        },
        {
            "training_block_id": "tl04_antidef_defense_pairwise",
            "deployable_status": "deployable_in_this_task",
            "included_in_tl13_bundle": 1,
            "rationale": "TL11/TL12 dead-ended TL04 for panel locking, but its anti-defense x defense subset is still deployable for generalized inference because both sides come from raw genomes via Pharokka and DefenseFinder.",
        },
    ]


def write_parity_audit(output_dir: Path) -> Path:
    audit_rows = build_feature_parity_audit_rows()
    audit_path = output_dir / PARITY_AUDIT_FILENAME
    write_csv(
        audit_path,
        ["training_block_id", "deployable_status", "included_in_tl13_bundle", "rationale"],
        audit_rows,
    )
    return audit_path


def copy_panel_annotation_cache(cached_annotations_dir: Path, output_dir: Path) -> Path:
    destination_dir = output_dir / PANEL_ANNOTATION_CACHE_DIRNAME
    ensure_directory(destination_dir)
    for source_path in sorted(cached_annotations_dir.glob("*_cds_final_merged_output.tsv")):
        shutil.copy2(source_path, destination_dir / source_path.name)
    copied_paths = sorted(destination_dir.glob("*_cds_final_merged_output.tsv"))
    if not copied_paths:
        raise FileNotFoundError(f"No cached Pharokka merged TSVs found in {cached_annotations_dir}")
    return destination_dir


def select_roundtrip_panel_hosts(
    *,
    panel_hosts_path: Path,
    vhdb_path: Path,
    assembly_summary_path: Path,
) -> list[tl09.HostCandidate]:
    _, panel_lookup = tl09.load_panel_hosts(panel_hosts_path)
    positive_pairs = tl09.parse_vhdb_positive_pairs(vhdb_path.read_text(encoding="utf-8"))
    assemblies_by_taxid = tl09.parse_assembly_summary(assembly_summary_path.read_text(encoding="utf-8"))
    host_candidates = tl09.build_host_candidates(positive_pairs, assemblies_by_taxid, panel_lookup)
    roundtrip_by_panel_name: dict[str, tl09.HostCandidate] = {}
    for candidate in host_candidates:
        if candidate.panel_match and candidate.panel_match in tl09.ROUNDTRIP_PANEL_HOSTS:
            existing = roundtrip_by_panel_name.get(candidate.panel_match)
            if existing is None or (
                candidate.unique_phage_count,
                candidate.positive_pair_count,
                candidate.host_name,
            ) > (
                existing.unique_phage_count,
                existing.positive_pair_count,
                existing.host_name,
            ):
                roundtrip_by_panel_name[candidate.panel_match] = candidate
    roundtrip_hosts = [
        roundtrip_by_panel_name[name] for name in tl09.ROUNDTRIP_PANEL_HOSTS if name in roundtrip_by_panel_name
    ]
    if not roundtrip_hosts:
        raise ValueError("No predeclared round-trip panel hosts had assembly-backed Virus-Host DB matches.")
    return roundtrip_hosts


def ensure_roundtrip_source_files(
    *,
    output_dir: Path,
    vhdb_url: str,
    assembly_summary_url: str,
) -> tuple[Path, Path]:
    raw_dir = output_dir / "raw_roundtrip_sources"
    ensure_directory(raw_dir)
    vhdb_path = raw_dir / "virushostdb.tsv"
    if not vhdb_path.exists():
        vhdb_path.write_text(tl09._download_text(vhdb_url), encoding="utf-8")
    assembly_summary_path = raw_dir / "assembly_summary_refseq.txt"
    if not assembly_summary_path.exists():
        assembly_summary_path.write_text(tl09._download_text(assembly_summary_url), encoding="utf-8")
    return vhdb_path, assembly_summary_path


def write_roundtrip_reference_predictions(
    *,
    panel_predictions_path: Path,
    roundtrip_hosts: Sequence[tl09.HostCandidate],
    output_dir: Path,
) -> Path:
    panel_matches = {host.panel_match for host in roundtrip_hosts}
    reference_rows = [row for row in read_csv_rows(panel_predictions_path) if str(row["bacteria"]) in panel_matches]
    if not reference_rows:
        raise ValueError("No round-trip reference predictions matched the predeclared panel-host cohort.")
    output_path = output_dir / ROUNDTRIP_REFERENCE_FILENAME
    write_csv(output_path, list(reference_rows[0].keys()), reference_rows)
    return output_path


def write_roundtrip_host_cohort(roundtrip_hosts: Sequence[tl09.HostCandidate], output_dir: Path) -> Path:
    output_path = output_dir / ROUNDTRIP_HOST_COHORT_FILENAME
    write_csv(output_path, list(roundtrip_hosts[0].__dict__.keys()), [host.__dict__ for host in roundtrip_hosts])
    return output_path


def run_roundtrip_bundle_comparison(
    *,
    bundle_path: Path,
    roundtrip_hosts: Sequence[tl09.HostCandidate],
    panel_phage_paths: Sequence[Path],
    host_assembly_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = generalized_inference.load_runtime(bundle_path)
    host_metadata = {host.host_tax_id: host for host in roundtrip_hosts}
    panel_phage_rows = generalized_inference.project_phage_features(panel_phage_paths, runtime=runtime)
    prediction_frames: list[pd.DataFrame] = []
    for host in roundtrip_hosts:
        host_fasta_path = tl09.download_host_assembly(host, host_assembly_dir)
        host_row = generalized_inference.project_host_features(
            host_fasta_path,
            bacteria_id=host.panel_match,
            runtime=runtime,
        )
        predictions = generalized_inference.score_projected_features(host_row, panel_phage_rows, runtime=runtime)
        predictions["host_tax_id"] = host.host_tax_id
        predictions["host_name"] = host.host_name
        predictions["panel_match"] = host.panel_match
        prediction_frames.append(predictions)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison = tl09.build_roundtrip_comparison(
        prediction_frames=prediction_frames,
        host_metadata=host_metadata,
        panel_predictions_path=runtime.bundle_path.parent
        / str(runtime.bundle["artifacts"]["panel_predictions_filename"]),
    )
    return all_predictions, comparison


def summarize_roundtrip_metrics(comparison: pd.DataFrame, *, bundle_label: str) -> list[dict[str, object]]:
    return [
        {
            "bundle_label": bundle_label,
            "metric": "median_abs_probability_delta_median",
            "value": float(comparison["median_abs_probability_delta"].median()),
        },
        {
            "bundle_label": bundle_label,
            "metric": "max_abs_probability_delta_max",
            "value": float(comparison["max_abs_probability_delta"].max()),
        },
        {
            "bundle_label": bundle_label,
            "metric": "identical_rank_count_total",
            "value": float(comparison["identical_rank_count"].sum()),
        },
    ]


def compare_metric_rows(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    baseline_by_metric = {str(row["metric"]): float(row["value"]) for row in baseline_rows}
    candidate_by_metric = {str(row["metric"]): float(row["value"]) for row in candidate_rows}
    output_rows: list[dict[str, object]] = []
    improved_metrics: list[str] = []
    for metric, direction in PREDECLARED_ROUNDTRIP_METRICS.items():
        baseline_value = baseline_by_metric[metric]
        candidate_value = candidate_by_metric[metric]
        if direction == "lower":
            improved = candidate_value < baseline_value
        else:
            improved = candidate_value > baseline_value
        if improved:
            improved_metrics.append(metric)
        output_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "candidate_improved": int(improved),
            }
        )
    return output_rows, improved_metrics


def build_surface_delta_rows(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> list[dict[str, object]]:
    merged = baseline_predictions.merge(
        candidate_predictions,
        on=["host_tax_id", "host_name", "panel_match", "phage"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("Round-trip ablation merge produced zero overlapping prediction rows.")
    rows = []
    for row in merged.to_dict("records"):
        baseline_value = float(row["p_lysis_baseline"])
        candidate_value = float(row["p_lysis_candidate"])
        rows.append(
            {
                "host_tax_id": row["host_tax_id"],
                "host_name": row["host_name"],
                "panel_match": row["panel_match"],
                "phage": row["phage"],
                "baseline_p_lysis": baseline_value,
                "candidate_p_lysis": candidate_value,
                "abs_probability_delta": abs(candidate_value - baseline_value),
                "baseline_rank": int(row["rank_baseline"]),
                "candidate_rank": int(row["rank_candidate"]),
                "rank_delta": int(row["rank_candidate"]) - int(row["rank_baseline"]),
            }
        )
    return rows


def summarize_surface_deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_host: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_host.setdefault(str(row["panel_match"]), []).append(row)
    summary_rows = []
    for panel_match, host_rows in sorted(by_host.items()):
        changed_count = sum(float(row["abs_probability_delta"]) > 0.0 for row in host_rows)
        summary_rows.append(
            {
                "panel_match": panel_match,
                "changed_prediction_count": changed_count,
                "median_abs_probability_delta": float(
                    pd.Series([row["abs_probability_delta"] for row in host_rows]).median()
                ),
                "max_abs_probability_delta": max(float(row["abs_probability_delta"]) for row in host_rows),
            }
        )
    return summary_rows


def update_bundle_and_manifest(
    *,
    bundle_path: Path,
    manifest_path: Path,
    artifact_updates: Mapping[str, object],
    bundle_updates: Mapping[str, object],
) -> None:
    bundle = joblib.load(bundle_path)
    bundle["artifacts"].update(dict(artifact_updates))
    bundle.update(dict(bundle_updates))
    joblib.dump(bundle, bundle_path)

    manifest = _load_json(manifest_path)
    manifest.update({key: value for key, value in bundle_updates.items() if key != "deployable_runtime"})
    manifest["artifact_hashes"]["bundle"] = sha256_file(bundle_path)
    write_json(manifest_path, manifest)


def persist_candidate_runtime_contract(
    *,
    bundle_path: Path,
    tl04_runtime_payload: Mapping[str, object],
) -> None:
    bundle = joblib.load(bundle_path)
    bundle["deployable_runtime"] = {
        TL04_DIRECT_BLOCK_ID: {
            **dict(tl04_runtime_payload),
            "panel_annotation_cache_dirname": PANEL_ANNOTATION_CACHE_DIRNAME,
        }
    }
    joblib.dump(bundle, bundle_path)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    ensure_directory(args.output_dir)
    ensure_directory(args.baseline_output_dir)

    LOGGER.info("Starting TL13 deployable generalized inference bundle rebuild")

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
            str(args.output_dir),
            "--calibration-fold",
            str(args.calibration_fold),
            "--random-state",
            str(args.random_state),
        ]
        + (["--skip-prerequisites"] if args.skip_prerequisites else [])
    )
    if not args.skip_prerequisites:
        build_generalized_inference_bundle.ensure_prerequisite_outputs(base_args)
    lightgbm_params = build_generalized_inference_bundle.load_locked_lightgbm_params(args.tg01_summary_path)

    ensure_tl04_artifacts(
        tl04_feature_path=args.tl04_feature_path,
        tl04_manifest_path=args.tl04_manifest_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
    )
    tl04_contract = load_tl04_contract(
        tl04_feature_path=args.tl04_feature_path,
        tl04_manifest_path=args.tl04_manifest_path,
        defense_subtypes_path=args.defense_subtypes_path,
    )

    parity_audit_path = write_parity_audit(args.output_dir)
    copied_annotation_dir = copy_panel_annotation_cache(args.cached_annotations_dir, args.output_dir)

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
        extra_phage_feature_rows=tl04_contract["direct_phage_rows"],
        extra_phage_feature_columns=tl04_contract["direct_columns"],
        pair_feature_rows=tl04_contract["feature_rows"],
        pair_feature_columns=tl04_contract["pairwise_columns"],
        bundle_task_id="TL13",
        bundle_format_version=DEPLOYABLE_BUNDLE_FORMAT_VERSION,
    )
    persist_candidate_runtime_contract(
        bundle_path=candidate_result["bundle_path"],
        tl04_runtime_payload=tl04_contract["runtime_payload"],
    )

    vhdb_path, assembly_summary_path = ensure_roundtrip_source_files(
        output_dir=args.output_dir,
        vhdb_url=args.vhdb_url,
        assembly_summary_url=args.assembly_summary_url,
    )
    roundtrip_hosts = select_roundtrip_panel_hosts(
        panel_hosts_path=args.panel_hosts_path,
        vhdb_path=vhdb_path,
        assembly_summary_path=assembly_summary_path,
    )
    roundtrip_hosts = tl09.filter_roundtrip_hosts_for_reference(
        roundtrip_hosts,
        candidate_result["panel_predictions_path"],
    )
    roundtrip_reference_path = write_roundtrip_reference_predictions(
        panel_predictions_path=candidate_result["panel_predictions_path"],
        roundtrip_hosts=roundtrip_hosts,
        output_dir=args.output_dir,
    )
    roundtrip_host_cohort_path = write_roundtrip_host_cohort(roundtrip_hosts, args.output_dir)

    panel_phages = tl09.read_panel_phages(args.panel_phage_metadata_path, expected_panel_count=96)
    panel_phage_paths = [args.panel_phage_dir / f"{phage}.fna" for phage in panel_phages]
    missing_panel_fastas = [str(path) for path in panel_phage_paths if not path.exists()]
    if missing_panel_fastas:
        raise FileNotFoundError(f"Missing panel phage FASTA: {missing_panel_fastas[0]}")

    host_assembly_dir = args.output_dir / "roundtrip_host_assemblies"
    baseline_predictions, baseline_comparison = run_roundtrip_bundle_comparison(
        bundle_path=baseline_result["bundle_path"],
        roundtrip_hosts=roundtrip_hosts,
        panel_phage_paths=panel_phage_paths,
        host_assembly_dir=host_assembly_dir,
    )
    candidate_predictions, candidate_comparison = run_roundtrip_bundle_comparison(
        bundle_path=candidate_result["bundle_path"],
        roundtrip_hosts=roundtrip_hosts,
        panel_phage_paths=panel_phage_paths,
        host_assembly_dir=host_assembly_dir,
    )

    baseline_metric_rows = summarize_roundtrip_metrics(baseline_comparison, bundle_label="baseline_tl08")
    candidate_metric_rows = summarize_roundtrip_metrics(candidate_comparison, bundle_label="candidate_tl13")
    metric_comparison_rows, improved_metrics = compare_metric_rows(baseline_metric_rows, candidate_metric_rows)

    surface_delta_rows = build_surface_delta_rows(baseline_predictions, candidate_predictions)
    surface_summary_rows = summarize_surface_deltas(surface_delta_rows)
    changed_prediction_count = sum(int(row["changed_prediction_count"]) for row in surface_summary_rows)
    if changed_prediction_count <= 0:
        raise ValueError("TL13 deployable block changed zero round-trip predictions; the richer bundle has no effect.")
    if not improved_metrics:
        raise ValueError(
            "TL13 deployable block did not improve any predeclared round-trip metric over the TL08 baseline."
        )

    write_csv(
        args.output_dir / ROUNDTRIP_METRIC_COMPARISON_FILENAME,
        list(metric_comparison_rows[0].keys()),
        metric_comparison_rows,
    )
    write_csv(
        args.output_dir / ROUNDTRIP_BASELINE_COMPARISON_FILENAME,
        list(baseline_comparison.columns),
        baseline_comparison.to_dict("records"),
    )
    write_csv(
        args.output_dir / ROUNDTRIP_CANDIDATE_COMPARISON_FILENAME,
        list(candidate_comparison.columns),
        candidate_comparison.to_dict("records"),
    )
    write_csv(
        args.output_dir / ROUNDTRIP_SURFACE_DELTA_FILENAME,
        list(surface_delta_rows[0].keys()),
        surface_delta_rows,
    )
    write_csv(
        args.output_dir / ROUNDTRIP_SURFACE_SUMMARY_FILENAME,
        list(surface_summary_rows[0].keys()),
        surface_summary_rows,
    )

    update_bundle_and_manifest(
        bundle_path=candidate_result["bundle_path"],
        manifest_path=candidate_result["manifest_path"],
        artifact_updates={
            "feature_parity_audit_filename": PARITY_AUDIT_FILENAME,
            "roundtrip_reference_predictions_filename": ROUNDTRIP_REFERENCE_FILENAME,
            "roundtrip_host_cohort_filename": ROUNDTRIP_HOST_COHORT_FILENAME,
        },
        bundle_updates={
            "deployable_runtime": {
                TL04_DIRECT_BLOCK_ID: {
                    **tl04_contract["runtime_payload"],
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
                    "block_id": DEPLOYABLE_BLOCK_ID,
                    "status": "deployable_in_this_task",
                    "source": "raw_host_assembly_plus_raw_phage_annotation",
                },
            ],
            "parity_audit": {
                "path": str(parity_audit_path),
                "sha256": sha256_file(parity_audit_path),
            },
            "roundtrip_gate": {
                "predeclared_metrics": PREDECLARED_ROUNDTRIP_METRICS,
                "improved_metrics": improved_metrics,
                "surface_changed_prediction_count": changed_prediction_count,
                "metric_comparison_path": str(args.output_dir / ROUNDTRIP_METRIC_COMPARISON_FILENAME),
            },
        },
    )

    LOGGER.info("Completed TL13 deployable generalized inference bundle rebuild")
    LOGGER.info("Improved round-trip metrics: %s", ", ".join(improved_metrics))
    LOGGER.info("Bundle-local panel annotation cache: %s", copied_annotation_dir)
    LOGGER.info("Round-trip reference predictions: %s", roundtrip_reference_path)
    LOGGER.info("Round-trip cohort manifest: %s", roundtrip_host_cohort_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
