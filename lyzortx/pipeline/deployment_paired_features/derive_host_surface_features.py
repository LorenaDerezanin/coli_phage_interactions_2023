#!/usr/bin/env python3
"""DEPLOY03: derive continuous host-surface features from raw assemblies."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15
from lyzortx.pipeline.track_l.steps import deployable_tl18_host_runtime as tl18_runtime

LOGGER = logging.getLogger(__name__)

DEFAULT_VALIDATION_FASTAS_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_surface")
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
PER_HOST_FEATURES_FILENAME = "host_surface_features.csv"
VALIDATION_FEATURES_FILENAME = "validation_host_surface_features.csv"
VALIDATION_REPORT_FILENAME = "validation_report.json"
FLOAT_DTYPE = "float64"
STRING_DTYPE = "string"
RECEPTOR_SCORE_COLUMNS: tuple[tuple[str, str], ...] = tuple(
    (receptor_name, present_column.removesuffix("_present") + "_score")
    for receptor_name, present_column, _ in tl15.RECEPTOR_COLUMNS
)
VALIDATION_HOSTS: tuple[str, ...] = ("55989", "EDL933", "LF82")
LEGACY_COLUMNS_DROPPED: tuple[str, ...] = (
    "host_o_type",
    "host_surface_lps_core_type",
    "host_capsule_abc_present",
    "host_o_antigen_present",
    "host_lps_core_present",
    "host_k_antigen_type_source",
    "host_capsule_abc_proxy_present",
    "host_abc_serotype_proxy",
    "host_k_antigen_present",
    "host_k_antigen_type",
    "host_k_antigen_proxy_present",
)


@dataclass(frozen=True)
class SurfaceRuntimeInputs:
    references: tuple[tl15.OAlleleReference, ...]
    o_type_contract: dict[str, dict[str, tuple[str, ...]]]
    o_antigen_query_path: Path
    lps_lookup: dict[str, dict[str, object]]
    capsule_hmm_bundle_path: Path
    capsule_profile_names: tuple[str, ...]
    omp_reference_path: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly_path", nargs="?", type=Path, help="Assembly FASTA for one host strain.")
    parser.add_argument("--bacteria-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--picard-metadata-path", type=Path, default=tl15.DEFAULT_PICARD_METADATA_PATH)
    parser.add_argument("--o-type-output-path", type=Path, default=tl15.DEFAULT_O_TYPE_OUTPUT_PATH)
    parser.add_argument("--o-type-allele-path", type=Path, default=tl15.DEFAULT_O_TYPE_ALLELE_PATH)
    parser.add_argument("--o-antigen-override-path", type=Path, default=tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH)
    parser.add_argument("--abc-capsule-profile-dir", type=Path, default=tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR)
    parser.add_argument("--omp-reference-path", type=Path, default=tl15.DEFAULT_OMP_REFERENCE_PATH)
    parser.add_argument(
        "--run-validation-subset",
        action="store_true",
        help="Run the committed validation hosts (55989, EDL933, LF82) instead of a single assembly.",
    )
    parser.add_argument("--validation-fastas-dir", type=Path, default=DEFAULT_VALIDATION_FASTAS_DIR)
    return parser.parse_args(argv)


def _capsule_score_column_name(profile_name: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", profile_name).strip("_").lower()
    if not token:
        raise ValueError(f"Unable to derive a capsule score column from profile name {profile_name!r}")
    return f"host_capsule_profile_{token}_score"


def build_host_surface_schema(capsule_profile_names: Sequence[str]) -> dict[str, Any]:
    receptor_columns = [{"name": column_name, "dtype": FLOAT_DTYPE} for _, column_name in RECEPTOR_SCORE_COLUMNS]
    capsule_columns = [
        {"name": _capsule_score_column_name(profile_name), "dtype": FLOAT_DTYPE}
        for profile_name in capsule_profile_names
    ]
    columns = [
        {"name": "bacteria", "dtype": STRING_DTYPE},
        {"name": "host_o_antigen_type", "dtype": STRING_DTYPE},
        {"name": "host_o_antigen_score", "dtype": FLOAT_DTYPE},
        {"name": "host_lps_core_type", "dtype": STRING_DTYPE},
        *receptor_columns,
        *capsule_columns,
    ]
    return {
        "feature_block": "host_surface",
        "key_column": "bacteria",
        "column_count": len(columns),
        "columns": columns,
        "categorical_columns": ["host_o_antigen_type", "host_lps_core_type"],
        "receptor_score_columns": [column["name"] for column in receptor_columns],
        "capsule_score_columns": [column["name"] for column in capsule_columns],
        "capsule_profile_names": list(capsule_profile_names),
        "dropped_legacy_columns": list(LEGACY_COLUMNS_DROPPED),
    }


def _column_names_from_schema(schema: Mapping[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _write_single_row_csv(path: Path, row: Mapping[str, object], *, delimiter: str = ",") -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def prepare_host_surface_runtime_inputs(
    *,
    assets_output_dir: Path,
    picard_metadata_path: Path,
    o_type_output_path: Path,
    o_type_allele_path: Path,
    o_antigen_override_path: Path,
    abc_capsule_profile_dir: Path,
    omp_reference_path: Path,
) -> SurfaceRuntimeInputs:
    picard_rows = tl15.read_delimited_rows(picard_metadata_path, delimiter=";")
    o_type_output_rows = tl15.read_delimited_rows(o_type_output_path, delimiter="\t")
    o_type_allele_rows = tl15.read_delimited_rows(o_type_allele_path, delimiter="\t")
    override_references = tl15.load_o_antigen_override_references(o_antigen_override_path)
    references, o_type_contract = tl15.build_o_antigen_reference_contract(
        o_type_output_rows=o_type_output_rows,
        o_type_allele_rows=o_type_allele_rows,
        override_references=override_references,
    )
    ensure_directory(assets_output_dir)
    o_antigen_query_path = assets_output_dir / "o_antigen_reference_queries.fna"
    tl15.write_o_antigen_queries(o_antigen_query_path, references)
    capsule_hmm_bundle_path = tl15.write_capsule_hmm_bundle(abc_capsule_profile_dir, assets_output_dir)
    capsule_profile_names = tuple(path.stem for path in sorted(abc_capsule_profile_dir.glob("*.hmm")))
    if not capsule_profile_names:
        raise FileNotFoundError(f"No capsule profile HMMs found in {abc_capsule_profile_dir}")
    return SurfaceRuntimeInputs(
        references=tuple(references),
        o_type_contract=o_type_contract,
        o_antigen_query_path=o_antigen_query_path,
        lps_lookup=tl15.build_lps_proxy_lookup(picard_rows),
        capsule_hmm_bundle_path=capsule_hmm_bundle_path,
        capsule_profile_names=capsule_profile_names,
        omp_reference_path=omp_reference_path,
    )


def summarize_o_antigen_result(
    *,
    hits: Sequence[tl15.HmmerHit],
    references: Sequence[tl15.OAlleleReference],
    o_type_contract: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, object]:
    reference_lookup = {reference.query_id: reference for reference in references}
    best_hits: dict[str, tl15.HmmerHit] = {}
    for hit in hits:
        if hit.query_name not in reference_lookup or hit.evalue > tl15.NHMMER_EVALUE_THRESHOLD:
            continue
        existing = best_hits.get(hit.query_name)
        if existing is None or (hit.score, -hit.evalue) > (existing.score, -existing.evalue):
            best_hits[hit.query_name] = hit

    candidate_rows: list[dict[str, object]] = []
    for o_type, gene_families in o_type_contract.items():
        matched_families = 0
        total_score = 0.0
        family_scores: dict[str, float] = {}
        evidence_parts: list[str] = []
        for gene_family, query_ids in gene_families.items():
            family_hits = [best_hits[query_id] for query_id in query_ids if query_id in best_hits]
            if not family_hits:
                continue
            best_family_hit = max(family_hits, key=lambda hit: hit.score)
            matched_families += 1
            total_score += best_family_hit.score
            family_scores[gene_family] = best_family_hit.score
            evidence_parts.append(f"{gene_family}:{best_family_hit.query_name}")
        if matched_families:
            candidate_rows.append(
                {
                    "o_type": o_type,
                    "matched_family_count": matched_families,
                    "total_score": total_score,
                    "family_scores": family_scores,
                    "evidence": "|".join(sorted(evidence_parts)),
                }
            )

    if not candidate_rows:
        return {
            "o_type": "",
            "continuous_score": 0.0,
            "matched_family_count": 0,
            "evidence": "no_O_antigen_allele_hits",
        }

    candidate_rows.sort(
        key=lambda row: (
            int(row["matched_family_count"]),
            float(row["total_score"]),
            str(row["o_type"]),
        ),
        reverse=True,
    )
    best = candidate_rows[0]
    o_type = str(best["o_type"])
    if int(best["matched_family_count"]) < 2:
        o_type = ""
    elif (
        len(candidate_rows) > 1
        and int(candidate_rows[1]["matched_family_count"]) == int(best["matched_family_count"])
        and float(candidate_rows[1]["total_score"]) == float(best["total_score"])
    ):
        o_type = ""
    return {
        "o_type": o_type,
        "continuous_score": round(float(best["total_score"]), 6),
        "matched_family_count": int(best["matched_family_count"]),
        "evidence": str(best["evidence"]),
    }


def run_o_antigen_scan(
    *,
    bacteria: str,
    assembly_path: Path,
    runtime_inputs: SurfaceRuntimeInputs,
    output_dir: Path,
) -> dict[str, object]:
    tblout_path = output_dir / f"{bacteria}_o_antigen_nhmmer.tbl"
    tl15._run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "nhmmer",
            "--noali",
            "--tblout",
            str(tblout_path),
            str(runtime_inputs.o_antigen_query_path),
            str(assembly_path),
        ],
        description=f"nhmmer O-antigen scan for {bacteria}",
    )
    return summarize_o_antigen_result(
        hits=tl15.parse_nhmmer_tblout(tblout_path),
        references=runtime_inputs.references,
        o_type_contract=runtime_inputs.o_type_contract,
    )


def summarize_receptor_scores(hits: Sequence[tl15.HmmerHit]) -> dict[str, float]:
    best_scores_by_receptor: dict[str, float] = {}
    valid_receptors = {receptor_name for receptor_name, _ in RECEPTOR_SCORE_COLUMNS}
    for hit in hits:
        match = re.search(r"\|([A-Z0-9]+)_ECOLI\b", hit.query_name)
        if not match:
            continue
        receptor_name = match.group(1)
        if receptor_name == "PQQU":
            receptor_name = "YNCD"
        if receptor_name not in valid_receptors:
            continue
        best_scores_by_receptor[receptor_name] = max(best_scores_by_receptor.get(receptor_name, 0.0), hit.score)
    return {
        receptor_name: round(best_scores_by_receptor.get(receptor_name, 0.0), 6)
        for receptor_name, _ in RECEPTOR_SCORE_COLUMNS
    }


def run_receptor_scan(
    *,
    bacteria: str,
    proteins_path: Path,
    runtime_inputs: SurfaceRuntimeInputs,
    output_dir: Path,
) -> dict[str, float]:
    tblout_path = output_dir / f"{bacteria}_omp_phmmer.tbl"
    tl15._run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "phmmer",
            "--noali",
            "--tblout",
            str(tblout_path),
            str(runtime_inputs.omp_reference_path),
            str(proteins_path),
        ],
        description=f"phmmer receptor scan for {bacteria}",
    )
    return summarize_receptor_scores(tl15.parse_hmmer_tblout(tblout_path))


def summarize_capsule_profile_scores(
    *,
    hits: Sequence[tl15.HmmerHit],
    profile_names: Sequence[str],
) -> dict[str, float]:
    best_scores_by_profile: dict[str, float] = {}
    for hit in hits:
        if hit.evalue > tl15.HMMSCAN_EVALUE_THRESHOLD:
            continue
        best_scores_by_profile[hit.target_name] = max(best_scores_by_profile.get(hit.target_name, 0.0), hit.score)
    return {profile_name: round(best_scores_by_profile.get(profile_name, 0.0), 6) for profile_name in profile_names}


def run_capsule_profile_scan(
    *,
    bacteria: str,
    proteins_path: Path,
    runtime_inputs: SurfaceRuntimeInputs,
    output_dir: Path,
) -> dict[str, float]:
    hits = tl15.run_hmmscan(
        bacteria=bacteria,
        proteins_path=proteins_path,
        hmm_bundle_path=runtime_inputs.capsule_hmm_bundle_path,
        output_dir=output_dir,
    )
    return summarize_capsule_profile_scores(hits=hits, profile_names=runtime_inputs.capsule_profile_names)


def build_host_surface_feature_row(
    *,
    bacteria: str,
    schema: Mapping[str, Any],
    o_antigen_type: str,
    o_antigen_score: float,
    lps_core_type: str,
    receptor_scores: Mapping[str, float],
    capsule_profile_scores: Mapping[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "bacteria": bacteria,
        "host_o_antigen_type": o_antigen_type,
        "host_o_antigen_score": round(float(o_antigen_score), 6),
        "host_lps_core_type": lps_core_type,
    }
    for receptor_name, column_name in RECEPTOR_SCORE_COLUMNS:
        row[column_name] = round(float(receptor_scores.get(receptor_name, 0.0)), 6)
    for profile_name in schema["capsule_profile_names"]:
        row[_capsule_score_column_name(str(profile_name))] = round(
            float(capsule_profile_scores.get(str(profile_name), 0.0)),
            6,
        )
    return row


def derive_host_surface_features(
    assembly_path: Path,
    *,
    bacteria_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    picard_metadata_path: Path = tl15.DEFAULT_PICARD_METADATA_PATH,
    o_type_output_path: Path = tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
    o_type_allele_path: Path = tl15.DEFAULT_O_TYPE_ALLELE_PATH,
    o_antigen_override_path: Path = tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
    abc_capsule_profile_dir: Path = tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
    omp_reference_path: Path = tl15.DEFAULT_OMP_REFERENCE_PATH,
    runtime_inputs: SurfaceRuntimeInputs | None = None,
) -> dict[str, Any]:
    if not assembly_path.exists():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_path}")

    ensure_directory(output_dir)
    resolved_bacteria_id = bacteria_id or assembly_path.stem
    resolved_runtime_inputs = runtime_inputs or prepare_host_surface_runtime_inputs(
        assets_output_dir=output_dir / "assets",
        picard_metadata_path=picard_metadata_path,
        o_type_output_path=o_type_output_path,
        o_type_allele_path=o_type_allele_path,
        o_antigen_override_path=o_antigen_override_path,
        abc_capsule_profile_dir=abc_capsule_profile_dir,
        omp_reference_path=omp_reference_path,
    )
    schema = build_host_surface_schema(resolved_runtime_inputs.capsule_profile_names)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)

    proteins_path = output_dir / "predicted_proteins.faa"
    protein_metadata = tl15.predict_proteins(assembly_path, proteins_path)
    o_antigen_result = run_o_antigen_scan(
        bacteria=resolved_bacteria_id,
        assembly_path=assembly_path,
        runtime_inputs=resolved_runtime_inputs,
        output_dir=output_dir,
    )
    receptor_scores = run_receptor_scan(
        bacteria=resolved_bacteria_id,
        proteins_path=proteins_path,
        runtime_inputs=resolved_runtime_inputs,
        output_dir=output_dir,
    )
    capsule_profile_scores = run_capsule_profile_scan(
        bacteria=resolved_bacteria_id,
        proteins_path=proteins_path,
        runtime_inputs=resolved_runtime_inputs,
        output_dir=output_dir,
    )
    row = build_host_surface_feature_row(
        bacteria=resolved_bacteria_id,
        schema=schema,
        o_antigen_type=str(o_antigen_result["o_type"]),
        o_antigen_score=float(o_antigen_result["continuous_score"]),
        lps_core_type=str(
            resolved_runtime_inputs.lps_lookup.get(str(o_antigen_result["o_type"]), {}).get("proxy_type", "")
        ),
        receptor_scores=receptor_scores,
        capsule_profile_scores=capsule_profile_scores,
    )
    _write_single_row_csv(output_dir / PER_HOST_FEATURES_FILENAME, row)

    manifest = {
        "step_name": "derive_host_surface_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assembly_path": str(assembly_path),
            "picard_metadata_path": str(picard_metadata_path),
            "o_type_output_path": str(o_type_output_path),
            "o_type_allele_path": str(o_type_allele_path),
            "o_antigen_override_path": str(o_antigen_override_path),
            "abc_capsule_profile_dir": str(abc_capsule_profile_dir),
            "omp_reference_path": str(omp_reference_path),
            "schema_manifest_path": str(output_dir / SCHEMA_MANIFEST_FILENAME),
        },
        "outputs": {
            "feature_csv": str(output_dir / PER_HOST_FEATURES_FILENAME),
            "predicted_proteins_faa": str(proteins_path),
        },
        "counts": {
            "receptor_score_column_count": len(schema["receptor_score_columns"]),
            "capsule_score_column_count": len(schema["capsule_score_columns"]),
            "nonzero_receptor_score_count": sum(1 for value in receptor_scores.values() if value > 0),
            "nonzero_capsule_profile_count": sum(1 for value in capsule_profile_scores.values() if value > 0),
            "predicted_protein_count": len(protein_metadata),
            "o_antigen_matched_family_count": int(o_antigen_result["matched_family_count"]),
        },
        "o_antigen_call": {
            "host_o_antigen_type": row["host_o_antigen_type"],
            "host_o_antigen_score": row["host_o_antigen_score"],
            "evidence": str(o_antigen_result["evidence"]),
        },
        "dropped_legacy_columns": list(schema["dropped_legacy_columns"]),
    }
    write_json(output_dir / "manifest.json", manifest)
    return {
        "schema": schema,
        "feature_row": row,
        "manifest": manifest,
    }


def _legacy_receptor_binary(column_name: str, row: Mapping[str, object]) -> int:
    return int(float(row.get(column_name, 0) or 0) > 0)


def build_validation_host_report(
    *,
    derived_row: Mapping[str, object],
    legacy_row: Mapping[str, object],
    schema: Mapping[str, Any],
) -> dict[str, object]:
    receptor_mismatches: list[dict[str, object]] = []
    for receptor_name, score_column in RECEPTOR_SCORE_COLUMNS:
        legacy_present_column = score_column.removesuffix("_score") + "_present"
        legacy_present = _legacy_receptor_binary(legacy_present_column, legacy_row)
        derived_present = int(float(derived_row.get(score_column, 0.0) or 0.0) > 0)
        if legacy_present != derived_present:
            receptor_mismatches.append(
                {
                    "receptor": receptor_name,
                    "legacy_present": legacy_present,
                    "derived_nonzero_score": derived_present,
                    "derived_score": float(derived_row.get(score_column, 0.0) or 0.0),
                }
            )

    capsule_score_columns = [str(column) for column in schema["capsule_score_columns"]]
    top_capsule_profiles = sorted(
        (
            {"column_name": column_name, "score": float(derived_row.get(column_name, 0.0) or 0.0)}
            for column_name in capsule_score_columns
            if float(derived_row.get(column_name, 0.0) or 0.0) > 0
        ),
        key=lambda row: (row["score"], row["column_name"]),
        reverse=True,
    )[:10]

    return {
        "bacteria": str(derived_row["bacteria"]),
        "derived_o_antigen_type": str(derived_row["host_o_antigen_type"]),
        "legacy_o_antigen_type": str(legacy_row.get("host_o_antigen_type", "")),
        "o_antigen_type_match": (
            str(derived_row["host_o_antigen_type"]) == str(legacy_row.get("host_o_antigen_type", ""))
        ),
        "derived_lps_core_type": str(derived_row["host_lps_core_type"]),
        "legacy_lps_core_type": str(legacy_row.get("host_lps_core_type", "")),
        "lps_core_type_match": str(derived_row["host_lps_core_type"]) == str(legacy_row.get("host_lps_core_type", "")),
        "host_o_antigen_score": float(derived_row["host_o_antigen_score"]),
        "nonzero_receptor_score_count": sum(
            1 for _, score_column in RECEPTOR_SCORE_COLUMNS if float(derived_row.get(score_column, 0.0) or 0.0) > 0
        ),
        "receptor_binary_mismatch_count": len(receptor_mismatches),
        "receptor_binary_mismatches": receptor_mismatches,
        "nonzero_capsule_profile_count": sum(
            1 for column_name in capsule_score_columns if float(derived_row.get(column_name, 0.0) or 0.0) > 0
        ),
        "top_capsule_profiles": top_capsule_profiles,
    }


def run_validation_subset(
    *,
    validation_fastas_dir: Path = DEFAULT_VALIDATION_FASTAS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    picard_metadata_path: Path = tl15.DEFAULT_PICARD_METADATA_PATH,
    o_type_output_path: Path = tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
    o_type_allele_path: Path = tl15.DEFAULT_O_TYPE_ALLELE_PATH,
    o_antigen_override_path: Path = tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
    abc_capsule_profile_dir: Path = tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
    omp_reference_path: Path = tl15.DEFAULT_OMP_REFERENCE_PATH,
) -> dict[str, Any]:
    ensure_directory(output_dir)
    runtime_inputs = prepare_host_surface_runtime_inputs(
        assets_output_dir=output_dir / "assets",
        picard_metadata_path=picard_metadata_path,
        o_type_output_path=o_type_output_path,
        o_type_allele_path=o_type_allele_path,
        o_antigen_override_path=o_antigen_override_path,
        abc_capsule_profile_dir=abc_capsule_profile_dir,
        omp_reference_path=omp_reference_path,
    )
    schema = build_host_surface_schema(runtime_inputs.capsule_profile_names)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)
    legacy_rows = tl18_runtime.build_tl15_panel_training_rows(
        picard_metadata_path=picard_metadata_path,
        receptor_cluster_path=tl15.DEFAULT_RECEPTOR_CLUSTER_PATH,
        target_bacteria=VALIDATION_HOSTS,
    )
    legacy_by_bacteria = {str(row["bacteria"]): row for row in legacy_rows}

    derived_rows: list[dict[str, object]] = []
    host_reports: list[dict[str, object]] = []
    for host in VALIDATION_HOSTS:
        assembly_path = validation_fastas_dir / f"{host}.fasta"
        if not assembly_path.exists():
            raise FileNotFoundError(f"Validation FASTA not found: {assembly_path}")
        if host not in legacy_by_bacteria:
            raise ValueError(f"Validation host {host!r} missing from legacy surface rows")
        host_result = derive_host_surface_features(
            assembly_path,
            bacteria_id=host,
            output_dir=output_dir / host,
            picard_metadata_path=picard_metadata_path,
            o_type_output_path=o_type_output_path,
            o_type_allele_path=o_type_allele_path,
            o_antigen_override_path=o_antigen_override_path,
            abc_capsule_profile_dir=abc_capsule_profile_dir,
            omp_reference_path=omp_reference_path,
            runtime_inputs=runtime_inputs,
        )
        derived_row = dict(host_result["feature_row"])
        derived_rows.append(derived_row)
        host_reports.append(
            build_validation_host_report(
                derived_row=derived_row,
                legacy_row=legacy_by_bacteria[host],
                schema=schema,
            )
        )

    write_csv(output_dir / VALIDATION_FEATURES_FILENAME, _column_names_from_schema(schema), derived_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_fastas_dir": str(validation_fastas_dir),
        "schema_manifest_path": str(output_dir / SCHEMA_MANIFEST_FILENAME),
        "comparison_reference": "tl15_panel_training_rows",
        "average_receptor_binary_mismatches_per_host": round(
            sum(int(report["receptor_binary_mismatch_count"]) for report in host_reports) / len(host_reports),
            6,
        ),
        "o_antigen_type_exact_match_count": sum(1 for report in host_reports if bool(report["o_antigen_type_match"])),
        "lps_core_type_exact_match_count": sum(1 for report in host_reports if bool(report["lps_core_type_match"])),
        "host_reports": host_reports,
    }
    write_json(output_dir / VALIDATION_REPORT_FILENAME, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    if args.run_validation_subset:
        summary = run_validation_subset(
            validation_fastas_dir=args.validation_fastas_dir,
            output_dir=args.output_dir,
            picard_metadata_path=args.picard_metadata_path,
            o_type_output_path=args.o_type_output_path,
            o_type_allele_path=args.o_type_allele_path,
            o_antigen_override_path=args.o_antigen_override_path,
            abc_capsule_profile_dir=args.abc_capsule_profile_dir,
            omp_reference_path=args.omp_reference_path,
        )
        LOGGER.info(
            "Validation subset complete: %.3f receptor binary mismatches/host",
            summary["average_receptor_binary_mismatches_per_host"],
        )
        return 0

    if args.assembly_path is None:
        raise SystemExit("assembly_path is required unless --run-validation-subset is set")

    result = derive_host_surface_features(
        args.assembly_path,
        bacteria_id=args.bacteria_id,
        output_dir=args.output_dir,
        picard_metadata_path=args.picard_metadata_path,
        o_type_output_path=args.o_type_output_path,
        o_type_allele_path=args.o_type_allele_path,
        o_antigen_override_path=args.o_antigen_override_path,
        abc_capsule_profile_dir=args.abc_capsule_profile_dir,
        omp_reference_path=args.omp_reference_path,
    )
    LOGGER.info(
        "Derived host-surface features for %s with %d nonzero capsule scores",
        result["feature_row"]["bacteria"],
        result["manifest"]["counts"]["nonzero_capsule_profile_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
