#!/usr/bin/env python3
"""TL18: rebuild the deployable generalized inference bundle with richer preprocessors."""

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
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle
from lyzortx.pipeline.track_l.steps import build_host_typing_projector as tl16
from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15
from lyzortx.pipeline.track_l.steps import build_tl13_generalized_inference_bundle as tl13
from lyzortx.pipeline.track_l.steps import build_tl17_phage_compatibility_preprocessor as tl17
from lyzortx.pipeline.track_l.steps import generalized_inference
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import sha256_file
from lyzortx.pipeline.track_l.steps.deployable_tl04_runtime import TL04_DIRECT_BLOCK_ID
from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import TL17_BLOCK_ID
from lyzortx.pipeline.track_l.steps.deployable_tl18_host_runtime import (
    TL15_BLOCK_ID,
    TL15_CATEGORICAL_COLUMNS,
    TL15_NUMERIC_COLUMNS,
    TL16_BLOCK_ID,
    TL16_CATEGORICAL_COLUMNS,
    TL16_NUMERIC_COLUMNS,
    build_tl15_panel_training_rows,
    build_tl15_runtime_payload,
    build_tl16_panel_training_rows,
    build_tl16_runtime_payload,
    write_roundtrip_host_inventory,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/generalized_inference_bundle_tl18")
DEFAULT_BASELINE_OUTPUT_DIR = Path(".scratch/tl18_current_deployable_baseline")
DEFAULT_RELOCATION_PROBE_DIR = Path(".scratch/tl18_relocation_probe")
PARITY_AUDIT_FILENAME = "tl18_feature_parity_audit.csv"
ROUNDTRIP_REFERENCE_FILENAME = "tl18_roundtrip_panel_reference_predictions.csv"
ROUNDTRIP_HOST_COHORT_FILENAME = "tl18_roundtrip_panel_host_cohort.csv"
ROUNDTRIP_METRIC_COMPARISON_FILENAME = "tl18_roundtrip_metric_comparison.csv"
ROUNDTRIP_BASELINE_COMPARISON_FILENAME = "tl18_roundtrip_baseline_comparison.csv"
ROUNDTRIP_CANDIDATE_COMPARISON_FILENAME = "tl18_roundtrip_candidate_comparison.csv"
ROUNDTRIP_SURFACE_DELTA_FILENAME = "tl18_roundtrip_surface_deltas.csv"
ROUNDTRIP_SURFACE_SUMMARY_FILENAME = "tl18_roundtrip_surface_summary.csv"
RELOCATION_PROBE_FILENAME = "tl18_relocated_bundle_probe_predictions.csv"
DEPLOYABLE_BUNDLE_FORMAT_VERSION = "tl18_deployable_inference_bundle_v1"
PREDECLARED_ROUNDTRIP_METRICS = {
    "median_abs_probability_delta_median": "lower",
    "max_abs_probability_delta_max": "lower",
    "identical_rank_count_total": "higher",
}
MATERIAL_DEGRADATION_TOLERANCES = {
    "median_abs_probability_delta_median": 0.01,
    "max_abs_probability_delta_max": 0.01,
    "identical_rank_count_total": 3.0,
}
RELOCATION_PROBE_HOST = "EDL933"
RELOCATION_PROBE_PHAGE_COUNT = 5


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
    parser.add_argument("--relocation-probe-dir", type=Path, default=DEFAULT_RELOCATION_PROBE_DIR)
    parser.add_argument("--picard-metadata-path", type=Path, default=tl15.DEFAULT_PICARD_METADATA_PATH)
    parser.add_argument("--o-type-output-path", type=Path, default=tl15.DEFAULT_O_TYPE_OUTPUT_PATH)
    parser.add_argument("--o-type-allele-path", type=Path, default=tl15.DEFAULT_O_TYPE_ALLELE_PATH)
    parser.add_argument("--o-antigen-override-path", type=Path, default=tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH)
    parser.add_argument("--lps-primary-path", type=Path, default=tl15.DEFAULT_LPS_PRIMARY_PATH)
    parser.add_argument("--lps-supplemental-path", type=Path, default=tl15.DEFAULT_LPS_SUPPLEMENTAL_PATH)
    parser.add_argument("--receptor-cluster-path", type=Path, default=tl15.DEFAULT_RECEPTOR_CLUSTER_PATH)
    parser.add_argument("--abc-capsule-profile-dir", type=Path, default=tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR)
    parser.add_argument("--abc-capsule-definition-dir", type=Path, default=tl15.DEFAULT_ABC_CAPSULE_DEFINITION_DIR)
    parser.add_argument("--omp-reference-path", type=Path, default=tl15.DEFAULT_OMP_REFERENCE_PATH)
    parser.add_argument("--validation-manifest-path", type=Path, default=tl16.DEFAULT_VALIDATION_MANIFEST_PATH)
    parser.add_argument("--validation-fasta-dir", type=Path, default=tl16.DEFAULT_FASTA_DIR)
    parser.add_argument("--tl04-feature-path", type=Path, default=tl13.DEFAULT_TL04_FEATURE_PATH)
    parser.add_argument("--tl04-manifest-path", type=Path, default=tl13.DEFAULT_TL04_MANIFEST_PATH)
    parser.add_argument("--tl17-output-dir", type=Path, default=tl17.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-phage-dir", type=Path, default=tl17.DEFAULT_PHAGE_FASTA_DIR)
    parser.add_argument("--panel-phage-metadata-path", type=Path, default=tl17.DEFAULT_PHAGE_METADATA_PATH)
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


def ensure_tl15_outputs(args: argparse.Namespace) -> Path:
    manifest_path = tl15.DEFAULT_OUTPUT_DIR / "tl15_raw_host_surface_manifest.json"
    if manifest_path.exists():
        return manifest_path
    LOGGER.info("TL15 outputs missing; rebuilding raw host surface projector")
    tl15.main(
        [
            "--input-manifest-path",
            str(args.validation_manifest_path),
            "--fasta-dir",
            str(args.validation_fasta_dir),
            "--output-dir",
            str(tl15.DEFAULT_OUTPUT_DIR),
        ]
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected TL15 manifest at {manifest_path}")
    return manifest_path


def ensure_tl16_outputs(args: argparse.Namespace) -> Path:
    manifest_path = tl16.DEFAULT_OUTPUT_DIR / tl16.MANIFEST_FILENAME
    if manifest_path.exists():
        return manifest_path
    LOGGER.info("TL16 outputs missing; rebuilding host typing projector")
    tl16.main(
        [
            "--fasta-dir",
            str(args.validation_fasta_dir),
            "--validation-manifest-path",
            str(args.validation_manifest_path),
            "--panel-metadata-path",
            str(args.picard_metadata_path),
            "--capsule-definition-dir",
            str(args.abc_capsule_definition_dir),
            "--capsule-profile-dir",
            str(args.abc_capsule_profile_dir),
            "--output-dir",
            str(tl16.DEFAULT_OUTPUT_DIR),
        ]
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected TL16 manifest at {manifest_path}")
    return manifest_path


def ensure_tl17_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    runtime_path = args.tl17_output_dir / tl17.RUNTIME_FILENAME
    projected_feature_path = args.tl17_output_dir / tl17.PROJECTED_FEATURE_FILENAME
    if runtime_path.exists() and projected_feature_path.exists():
        return runtime_path, projected_feature_path
    LOGGER.info("TL17 outputs missing; rebuilding phage compatibility preprocessor")
    tl17.main(
        ["--output-dir", str(args.tl17_output_dir)] + (["--skip-prerequisites"] if args.skip_prerequisites else [])
    )
    if not runtime_path.exists() or not projected_feature_path.exists():
        missing = runtime_path if not runtime_path.exists() else projected_feature_path
        raise FileNotFoundError(f"Expected TL17 artifact at {missing}")
    return runtime_path, projected_feature_path


def build_feature_parity_audit_rows() -> list[dict[str, object]]:
    return [
        {
            "training_block_id": "st04_v0_host_typing_metadata",
            "deployable_status": "replaced_by_deployable_proxy",
            "replacement_block_id": TL16_BLOCK_ID,
            "included_in_tl18_bundle": 1,
            "rationale": (
                "Clermont phylogroup plus O/H typing are raw-genome callable, and capsule metadata gets an explicit "
                "TL16 proxy instead of hidden metadata carry-through."
            ),
        },
        {
            "training_block_id": "st04_v0_non_genome_metadata",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": (
                "Pathotype, origin, collection, mouse lethality, and analogous metadata are not derivable from novel "
                "raw genomes."
            ),
        },
        {
            "training_block_id": "track_c_defense",
            "deployable_status": "included_directly",
            "replacement_block_id": "track_c_defense",
            "included_in_tl18_bundle": 1,
            "rationale": "DefenseFinder-derived host defense features remain directly deployable from raw host assemblies.",
        },
        {
            "training_block_id": "track_c_surface_projectable_subset",
            "deployable_status": "replaced_by_deployable_proxy",
            "replacement_block_id": TL15_BLOCK_ID,
            "included_in_tl18_bundle": 1,
            "rationale": (
                "TL15 supplies a raw-host projector for O-antigen, LPS proxy, capsule proxy, and receptor-presence "
                "families."
            ),
        },
        {
            "training_block_id": "track_c_phylogeny_umap_embedding",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": "Fitted host UMAP coordinates are excluded because the runtime projection contract is not explicit.",
        },
        {
            "training_block_id": "track_c_omp_variant_clusters",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": "The repo still lacks deployable representative sequences for the legacy OMP variant-cluster IDs.",
        },
        {
            "training_block_id": "track_d_phage_genomic_kmers",
            "deployable_status": "included_directly",
            "replacement_block_id": "track_d_phage_genomic_kmers",
            "included_in_tl18_bundle": 1,
            "rationale": "Phage tetranucleotide SVD remains directly deployable from raw phage genomes.",
        },
        {
            "training_block_id": "track_d_viridic_distance_embedding",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": "The repo still does not ship a stable raw-genome projector for the learned phage distance embedding.",
        },
        {
            "training_block_id": "track_e_curated_rbp_receptor_compatibility",
            "deployable_status": "replaced_by_deployable_proxy",
            "replacement_block_id": TL17_BLOCK_ID,
            "included_in_tl18_bundle": 1,
            "rationale": (
                "TL17 replaces the panel-only curated lookup with a raw-phage RBP family projector that keeps "
                "deployable adsorption signal."
            ),
        },
        {
            "training_block_id": "track_e_isolation_host_distance",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": "Isolation-host distance depends on collection metadata that arbitrary novel genomes do not have.",
        },
        {
            "training_block_id": "tl03_rbp_receptor_pairwise",
            "deployable_status": "excluded",
            "replacement_block_id": "",
            "included_in_tl18_bundle": 0,
            "rationale": "TL03 stayed dead-ended for the current lock and still lacks an honest raw-runtime contract.",
        },
        {
            "training_block_id": "tl04_antidef_defense_pairwise",
            "deployable_status": "included_directly",
            "replacement_block_id": TL04_DIRECT_BLOCK_ID,
            "included_in_tl18_bundle": 1,
            "rationale": "TL04 remains deployable because both the anti-defense and defense sides come from raw genomes.",
        },
    ]


def write_parity_audit(output_dir: Path) -> Path:
    rows = build_feature_parity_audit_rows()
    output_path = output_dir / PARITY_AUDIT_FILENAME
    write_csv(
        output_path,
        ["training_block_id", "deployable_status", "replacement_block_id", "included_in_tl18_bundle", "rationale"],
        rows,
    )
    return output_path


def load_validation_hosts(validation_manifest_path: Path, validation_fasta_dir: Path) -> list[dict[str, object]]:
    payload = _load_json(validation_manifest_path)
    hosts: list[dict[str, object]] = []
    for row in payload.get("files", []):
        bacteria = str(row["bacteria"])
        fasta_path = validation_fasta_dir / str(row["filename"])
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing validation FASTA for TL18 round-trip: {fasta_path}")
        hosts.append(
            {
                "host_tax_id": bacteria,
                "host_name": f"Escherichia coli {bacteria}",
                "positive_pair_count": "",
                "unique_phage_count": "",
                "panel_match": bacteria,
                "is_panel_host": 1,
                "assembly_accession": "",
                "assembly_level": "validation_subset",
                "assembly_organism_name": f"Escherichia coli {bacteria}",
                "assembly_ftp_path": "",
                "bacteria": bacteria,
                "fasta_path": str(fasta_path),
                "sha256": str(row["sha256"]),
            }
        )
    if len(hosts) < 3:
        raise ValueError("TL18 requires at least 3 committed round-trip hosts.")
    return sorted(hosts, key=lambda row: str(row["bacteria"]))


def persist_runtime_payloads(bundle_path: Path, runtime_payloads: Mapping[str, Mapping[str, object]]) -> None:
    bundle = joblib.load(bundle_path)
    bundle["deployable_runtime"] = {key: dict(value) for key, value in runtime_payloads.items()}
    joblib.dump(bundle, bundle_path)


def build_roundtrip_reference_predictions(
    *,
    bundle_path: Path,
    host_rows: Sequence[Mapping[str, object]],
    phage_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    runtime = generalized_inference.load_runtime(bundle_path)
    prediction_rows: list[dict[str, object]] = []
    for host_row in host_rows:
        predictions = generalized_inference.score_projected_features(dict(host_row), phage_rows, runtime=runtime)
        predictions["bacteria"] = str(host_row["bacteria"])
        predictions["split_holdout"] = "roundtrip_reference"
        predictions["split_cv5_fold"] = ""
        predictions["label_hard_any_lysis"] = ""
        prediction_rows.extend(
            predictions.rename(
                columns={
                    "p_lysis": "pred_lightgbm_isotonic",
                    "rank": "rank_lightgbm_isotonic",
                }
            ).to_dict("records")
        )
    if len({str(row["bacteria"]) for row in prediction_rows}) < 3:
        raise ValueError("TL18 round-trip reference predictions did not cover 3 validation hosts.")
    ordered_rows = [
        {
            "pair_id": f"{row['bacteria']}__{row['phage']}",
            "bacteria": row["bacteria"],
            "phage": row["phage"],
            "split_holdout": row["split_holdout"],
            "split_cv5_fold": row["split_cv5_fold"],
            "label_hard_any_lysis": row["label_hard_any_lysis"],
            "pred_lightgbm_raw": "",
            "pred_lightgbm_isotonic": row["pred_lightgbm_isotonic"],
            "rank_lightgbm_isotonic": row["rank_lightgbm_isotonic"],
        }
        for row in prediction_rows
    ]
    write_csv(output_path, list(ordered_rows[0].keys()), ordered_rows)
    return output_path


def _host_candidate_from_row(row: Mapping[str, object]) -> tl13.tl09.HostCandidate:
    return tl13.tl09.HostCandidate(
        host_tax_id=str(row["host_tax_id"]),
        host_name=str(row["host_name"]),
        positive_pair_count=0,
        unique_phage_count=0,
        panel_match=str(row["panel_match"]),
        is_panel_host=True,
        assembly_accession=str(row.get("assembly_accession", "")),
        assembly_level=str(row.get("assembly_level", "")),
        assembly_organism_name=str(row.get("assembly_organism_name", row["host_name"])),
        assembly_ftp_path=str(row.get("assembly_ftp_path", "")),
    )


def run_roundtrip_bundle_comparison(
    *,
    bundle_path: Path,
    roundtrip_hosts: Sequence[Mapping[str, object]],
    panel_phage_paths: Sequence[Path],
    reference_predictions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = generalized_inference.load_runtime(bundle_path)
    host_metadata = {
        _host_candidate_from_row(row).host_tax_id: _host_candidate_from_row(row) for row in roundtrip_hosts
    }
    panel_phage_rows = generalized_inference.project_phage_features(panel_phage_paths, runtime=runtime)
    prediction_frames: list[pd.DataFrame] = []
    for host in roundtrip_hosts:
        host_row = generalized_inference.project_host_features(
            Path(str(host["fasta_path"])),
            bacteria_id=str(host["bacteria"]),
            runtime=runtime,
        )
        predictions = generalized_inference.score_projected_features(host_row, panel_phage_rows, runtime=runtime)
        predictions["host_tax_id"] = str(host["host_tax_id"])
        predictions["host_name"] = str(host["host_name"])
        predictions["panel_match"] = str(host["panel_match"])
        prediction_frames.append(predictions)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison = tl13.tl09.build_roundtrip_comparison(
        prediction_frames=prediction_frames,
        host_metadata=host_metadata,
        panel_predictions_path=reference_predictions_path,
    )
    return all_predictions, comparison


def compare_metric_rows(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    baseline_by_metric = {str(row["metric"]): float(row["value"]) for row in baseline_rows}
    candidate_by_metric = {str(row["metric"]): float(row["value"]) for row in candidate_rows}
    output_rows: list[dict[str, object]] = []
    improved_metrics: list[str] = []
    materially_degraded_metrics: list[str] = []
    for metric, direction in PREDECLARED_ROUNDTRIP_METRICS.items():
        baseline_value = baseline_by_metric[metric]
        candidate_value = candidate_by_metric[metric]
        tolerance = MATERIAL_DEGRADATION_TOLERANCES[metric]
        if direction == "lower":
            improved = candidate_value < baseline_value
            materially_degraded = candidate_value > baseline_value + tolerance
        else:
            improved = candidate_value > baseline_value
            materially_degraded = candidate_value < baseline_value - tolerance
        if improved:
            improved_metrics.append(metric)
        if materially_degraded:
            materially_degraded_metrics.append(metric)
        output_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "candidate_improved": int(improved),
                "candidate_materially_degraded": int(materially_degraded),
                "material_degradation_tolerance": tolerance,
            }
        )
    return output_rows, improved_metrics, materially_degraded_metrics


def run_relocation_probe(
    *,
    bundle_dir: Path,
    relocation_probe_dir: Path,
    validation_fasta_dir: Path,
    phage_paths: Sequence[Path],
) -> Path:
    relocated_bundle_dir = relocation_probe_dir / bundle_dir.name
    if relocated_bundle_dir.exists():
        shutil.rmtree(relocated_bundle_dir)
    shutil.copytree(bundle_dir, relocated_bundle_dir)
    bundle_path = relocated_bundle_dir / build_generalized_inference_bundle.BUNDLE_FILENAME
    predictions = generalized_inference.infer(
        validation_fasta_dir / f"{RELOCATION_PROBE_HOST}.fasta",
        list(phage_paths[:RELOCATION_PROBE_PHAGE_COUNT]),
        bundle_path,
    )
    output_path = bundle_dir / RELOCATION_PROBE_FILENAME
    predictions.to_csv(output_path, index=False)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_directory(args.baseline_output_dir)
    ensure_directory(args.relocation_probe_dir)

    LOGGER.info("Starting TL18 generalized inference bundle rebuild")

    if not args.skip_prerequisites:
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
                str(args.output_dir / ".tl18_prereq_probe"),
            ]
        )
        build_generalized_inference_bundle.ensure_prerequisite_outputs(base_args)

    lightgbm_params = build_generalized_inference_bundle.load_locked_lightgbm_params(args.tg01_summary_path)
    ensure_tl15_outputs(args)
    ensure_tl16_outputs(args)
    tl17_runtime_path, tl17_projected_feature_path = ensure_tl17_outputs(args)
    tl13.ensure_tl04_artifacts(
        tl04_feature_path=args.tl04_feature_path,
        tl04_manifest_path=args.tl04_manifest_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
    )
    tl04_contract = tl13.load_tl04_contract(
        tl04_feature_path=args.tl04_feature_path,
        tl04_manifest_path=args.tl04_manifest_path,
        defense_subtypes_path=args.defense_subtypes_path,
    )

    parity_audit_path = write_parity_audit(args.output_dir)
    copied_annotation_dir = tl13.copy_panel_annotation_cache(tl13.CACHED_ANNOTATIONS_DIR, args.output_dir)
    tl13.copy_panel_annotation_cache(tl13.CACHED_ANNOTATIONS_DIR, args.baseline_output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    target_bacteria = sorted({str(row["bacteria"]) for row in st02_rows})
    validation_hosts = load_validation_hosts(args.validation_manifest_path, args.validation_fasta_dir)
    proxy_bacteria = sorted({*target_bacteria, *(str(row["bacteria"]) for row in validation_hosts)})

    tl15_training_rows = build_tl15_panel_training_rows(
        picard_metadata_path=args.picard_metadata_path,
        receptor_cluster_path=args.receptor_cluster_path,
        target_bacteria=proxy_bacteria,
    )
    tl16_training_rows = build_tl16_panel_training_rows(
        picard_metadata_path=args.picard_metadata_path,
        target_bacteria=proxy_bacteria,
    )
    defense_feature_rows, _, _ = tl13.build_generalized_inference_bundle.build_defense_feature_rows(
        tl13.build_generalized_inference_bundle.read_defense_rows(args.defense_subtypes_path)
    )
    defense_by_bacteria = {str(row["bacteria"]): dict(row) for row in defense_feature_rows}
    tl17_training_rows = read_csv_rows(tl17_projected_feature_path)
    tl16_training_by_bacteria = {str(row["bacteria"]): dict(row) for row in tl16_training_rows}
    merged_host_training_rows = [
        {**row, **tl16_training_by_bacteria[str(row["bacteria"])]} for row in tl15_training_rows
    ]
    merged_host_training_by_bacteria = {str(row["bacteria"]): dict(row) for row in merged_host_training_rows}
    tl04_direct_by_phage = {str(row["phage"]): dict(row) for row in tl04_contract["direct_phage_rows"]}
    base_phage_training_rows = read_csv_rows(args.phage_kmer_feature_path)
    base_phage_training_by_name = {str(row["phage"]): dict(row) for row in base_phage_training_rows}
    merged_phage_training_rows = [
        {**base_phage_training_by_name[str(row["phage"])], **row, **tl04_direct_by_phage[str(row["phage"])]}
        for row in tl17_training_rows
    ]
    tl17_runtime_payload = joblib.load(tl17_runtime_path)
    tl15_runtime_payload = build_tl15_runtime_payload(
        output_dir=args.output_dir,
        picard_metadata_path=args.picard_metadata_path,
        o_type_output_path=args.o_type_output_path,
        o_type_allele_path=args.o_type_allele_path,
        o_antigen_override_path=args.o_antigen_override_path,
        abc_capsule_profile_dir=args.abc_capsule_profile_dir,
        abc_capsule_definition_dir=args.abc_capsule_definition_dir,
        omp_reference_path=args.omp_reference_path,
    )
    tl16_runtime_payload = build_tl16_runtime_payload(
        output_dir=args.output_dir,
        capsule_definition_dir=args.abc_capsule_definition_dir,
        capsule_profile_dir=args.abc_capsule_profile_dir,
    )
    tl17_reference_fasta_source = args.tl17_output_dir / tl17.REFERENCE_FASTA_FILENAME
    tl17_reference_fasta_copy = args.output_dir / tl17.REFERENCE_FASTA_FILENAME
    shutil.copy2(tl17_reference_fasta_source, tl17_reference_fasta_copy)

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
        extra_phage_feature_rows=tl04_contract["direct_phage_rows"],
        extra_phage_feature_columns=tl04_contract["direct_columns"],
        pair_feature_rows=tl04_contract["feature_rows"],
        pair_feature_columns=tl04_contract["pairwise_columns"],
        bundle_task_id="TL13_BASELINE",
        bundle_format_version="tl13_current_deployable_baseline_v1",
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
        extra_host_feature_rows=merged_host_training_rows,
        extra_host_feature_columns=[*TL15_NUMERIC_COLUMNS, *TL16_NUMERIC_COLUMNS],
        extra_host_categorical_columns=[*TL15_CATEGORICAL_COLUMNS, *TL16_CATEGORICAL_COLUMNS],
        extra_phage_feature_rows=merged_phage_training_rows,
        extra_phage_feature_columns=[column for column in tl17_training_rows[0].keys() if column != "phage"]
        + list(tl04_contract["direct_columns"]),
        pair_feature_rows=tl04_contract["feature_rows"],
        pair_feature_columns=tl04_contract["pairwise_columns"],
        bundle_task_id="TL18",
        bundle_format_version=DEPLOYABLE_BUNDLE_FORMAT_VERSION,
    )

    persist_runtime_payloads(
        baseline_result["bundle_path"],
        {
            TL04_DIRECT_BLOCK_ID: {
                **tl04_contract["runtime_payload"],
                "panel_annotation_cache_dirname": tl13.PANEL_ANNOTATION_CACHE_DIRNAME,
            }
        },
    )
    # Pre-write the deployable runtime contract so the round-trip comparisons below can load the
    # candidate bundle before the final manifest update adds the remaining metadata fields.
    persist_runtime_payloads(
        candidate_result["bundle_path"],
        {
            TL04_DIRECT_BLOCK_ID: {
                **tl04_contract["runtime_payload"],
                "panel_annotation_cache_dirname": tl13.PANEL_ANNOTATION_CACHE_DIRNAME,
            },
            TL15_BLOCK_ID: tl15_runtime_payload,
            TL16_BLOCK_ID: tl16_runtime_payload,
            TL17_BLOCK_ID: {
                **dict(tl17_runtime_payload),
                "reference_fasta_filename": tl17_reference_fasta_copy.name,
            },
        },
    )

    roundtrip_host_path = write_roundtrip_host_inventory(
        validation_hosts, args.output_dir / ROUNDTRIP_HOST_COHORT_FILENAME
    )
    candidate_reference_host_rows = [
        {
            "bacteria": row["bacteria"],
            **defense_by_bacteria[str(row["bacteria"])],
            **merged_host_training_by_bacteria[str(row["bacteria"])],
        }
        for row in validation_hosts
    ]
    candidate_reference_phage_rows = [dict(row) for row in merged_phage_training_rows]
    baseline_reference_host_rows = [
        {"bacteria": row["bacteria"], **defense_by_bacteria[str(row["bacteria"])]} for row in validation_hosts
    ]
    baseline_reference_phage_rows = [
        {**row, **tl04_direct_by_phage[str(row["phage"])]} for row in base_phage_training_rows
    ]
    roundtrip_reference_path = build_roundtrip_reference_predictions(
        bundle_path=candidate_result["bundle_path"],
        host_rows=candidate_reference_host_rows,
        phage_rows=candidate_reference_phage_rows,
        output_path=args.output_dir / ROUNDTRIP_REFERENCE_FILENAME,
    )
    panel_phages = tl13.tl09.read_panel_phages(args.panel_phage_metadata_path, expected_panel_count=96)
    panel_phage_paths = [args.panel_phage_dir / f"{phage}.fna" for phage in panel_phages]
    baseline_reference_path = build_roundtrip_reference_predictions(
        bundle_path=baseline_result["bundle_path"],
        host_rows=baseline_reference_host_rows,
        phage_rows=baseline_reference_phage_rows,
        output_path=args.baseline_output_dir / ROUNDTRIP_REFERENCE_FILENAME,
    )
    baseline_predictions, baseline_comparison = run_roundtrip_bundle_comparison(
        bundle_path=baseline_result["bundle_path"],
        roundtrip_hosts=validation_hosts,
        panel_phage_paths=panel_phage_paths,
        reference_predictions_path=baseline_reference_path,
    )
    candidate_predictions, candidate_comparison = run_roundtrip_bundle_comparison(
        bundle_path=candidate_result["bundle_path"],
        roundtrip_hosts=validation_hosts,
        panel_phage_paths=panel_phage_paths,
        reference_predictions_path=roundtrip_reference_path,
    )
    baseline_metric_rows = tl13.summarize_roundtrip_metrics(baseline_comparison, bundle_label="current_deployable_tl13")
    candidate_metric_rows = tl13.summarize_roundtrip_metrics(candidate_comparison, bundle_label="candidate_tl18")
    metric_comparison_rows, improved_metrics, materially_degraded_metrics = compare_metric_rows(
        baseline_metric_rows,
        candidate_metric_rows,
    )
    non_improved_metric_count = len(PREDECLARED_ROUNDTRIP_METRICS) - len(improved_metrics)
    roundtrip_gate_cleared = bool(improved_metrics) and not (
        non_improved_metric_count > 0 and len(materially_degraded_metrics) >= non_improved_metric_count
    )
    roundtrip_conclusion = (
        "richer bundle cleared the round-trip gate"
        if roundtrip_gate_cleared
        else "generalized inference remains blocked by the round-trip gate"
    )
    if not roundtrip_gate_cleared:
        LOGGER.warning("TL18 round-trip gate was not cleared: %s", roundtrip_conclusion)

    surface_delta_rows = tl13.build_surface_delta_rows(baseline_predictions, candidate_predictions)
    surface_summary_rows = tl13.summarize_surface_deltas(surface_delta_rows)
    relocation_probe_path = run_relocation_probe(
        bundle_dir=args.output_dir,
        relocation_probe_dir=args.relocation_probe_dir,
        validation_fasta_dir=args.validation_fasta_dir,
        phage_paths=panel_phage_paths,
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
        args.output_dir / ROUNDTRIP_SURFACE_DELTA_FILENAME, list(surface_delta_rows[0].keys()), surface_delta_rows
    )
    write_csv(
        args.output_dir / ROUNDTRIP_SURFACE_SUMMARY_FILENAME, list(surface_summary_rows[0].keys()), surface_summary_rows
    )

    tl13.update_bundle_and_manifest(
        bundle_path=candidate_result["bundle_path"],
        manifest_path=candidate_result["manifest_path"],
        artifact_updates={
            "feature_parity_audit_filename": PARITY_AUDIT_FILENAME,
            "roundtrip_reference_predictions_filename": ROUNDTRIP_REFERENCE_FILENAME,
            "roundtrip_host_cohort_filename": ROUNDTRIP_HOST_COHORT_FILENAME,
            "relocation_probe_predictions_filename": relocation_probe_path.name,
            "tl17_reference_fasta_filename": tl17_reference_fasta_copy.name,
        },
        bundle_updates={
            "deployable_feature_blocks": [
                {"block_id": "track_c_defense", "status": "included_directly", "source": "raw_host_assembly"},
                {
                    "block_id": "track_d_phage_genomic_kmers",
                    "status": "included_directly",
                    "source": "raw_phage_genome",
                },
                {
                    "block_id": TL04_DIRECT_BLOCK_ID,
                    "status": "included_directly",
                    "source": "raw_host_assembly_plus_raw_phage_annotation",
                },
                {"block_id": TL15_BLOCK_ID, "status": "replaced_by_deployable_proxy", "source": "raw_host_assembly"},
                {"block_id": TL16_BLOCK_ID, "status": "replaced_by_deployable_proxy", "source": "raw_host_assembly"},
                {"block_id": TL17_BLOCK_ID, "status": "replaced_by_deployable_proxy", "source": "raw_phage_genome"},
            ],
            "parity_audit": {"path": parity_audit_path.name, "sha256": sha256_file(parity_audit_path)},
            "roundtrip_gate": {
                "predeclared_metrics": PREDECLARED_ROUNDTRIP_METRICS,
                "material_degradation_tolerances": MATERIAL_DEGRADATION_TOLERANCES,
                "improved_metrics": improved_metrics,
                "materially_degraded_metrics": materially_degraded_metrics,
                "gate_cleared": roundtrip_gate_cleared,
                "conclusion": roundtrip_conclusion,
                "metric_comparison_path": ROUNDTRIP_METRIC_COMPARISON_FILENAME,
            },
        },
    )

    LOGGER.info("Completed TL18 generalized inference bundle rebuild")
    LOGGER.info("Round-trip reference predictions: %s", roundtrip_reference_path)
    LOGGER.info("Round-trip cohort manifest: %s", roundtrip_host_path)
    LOGGER.info("Relocated bundle probe predictions: %s", relocation_probe_path)
    LOGGER.info("Bundle-local panel annotation cache: %s", copied_annotation_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
