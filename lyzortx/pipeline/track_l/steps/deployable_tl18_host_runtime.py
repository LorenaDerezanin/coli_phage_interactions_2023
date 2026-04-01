"""Helpers for TL15/TL16 host-side deployable runtime and panel feature rows."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv
from lyzortx.pipeline.track_c.steps import build_receptor_surface_feature_block as receptor_surface
from lyzortx.pipeline.track_l.steps import build_host_typing_projector as tl16
from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15

TL15_BLOCK_ID = "tl15_host_surface_projection"
TL16_BLOCK_ID = "tl16_host_typing_projection"

TL15_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "host_o_antigen_type",
    "host_k_antigen_type",
    "host_k_antigen_type_source",
    "host_lps_core_type",
    "host_surface_lps_core_type",
)
TL15_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in tl15.PROJECTED_FEATURE_COLUMNS if column not in {"bacteria", *TL15_CATEGORICAL_COLUMNS}
)

TL16_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "host_clermont_phylo",
    "host_st_warwick",
    "host_o_type",
    "host_h_type",
    "host_serotype",
    "host_abc_serotype_proxy",
)
TL16_NUMERIC_COLUMNS: tuple[str, ...] = ("host_capsule_abc_proxy_present",)


def _read_semicolon_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]


def build_tl15_panel_training_rows(
    *,
    picard_metadata_path: Path,
    lps_primary_path: Path,
    lps_supplemental_path: Path,
    receptor_cluster_path: Path,
    target_bacteria: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    del lps_primary_path, lps_supplemental_path
    picard_rows = _read_semicolon_rows(picard_metadata_path)
    host_metadata = {row["bacteria"]: row for row in picard_rows if row.get("bacteria")}
    receptor_rows = receptor_surface.read_delimited_rows(receptor_cluster_path, "\t")
    receptor_index = {row["bacteria"]: row for row in receptor_rows if row.get("bacteria")}
    lps_lookup = tl15.build_lps_proxy_lookup(picard_rows)
    hosts = sorted(target_bacteria) if target_bacteria is not None else sorted(host_metadata)
    lps_rows_for_surface = {
        bacteria: {
            "LPS_type": str(
                lps_lookup.get(
                    receptor_surface._normalize_category(host_metadata[bacteria].get("O-type", "")),
                    {},
                ).get("proxy_type", "")
            )
        }
        for bacteria in hosts
    }
    training_rows = receptor_surface.build_feature_rows(
        hosts=hosts,
        host_metadata_by_bacteria=host_metadata,
        lps_by_bacteria=lps_rows_for_surface,
        receptor_by_bacteria=receptor_index,
    )
    for row in training_rows:
        row["host_surface_lps_core_type"] = row["host_lps_core_type"]
    return training_rows


def build_tl16_panel_training_rows(
    *,
    picard_metadata_path: Path,
    target_bacteria: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    rows = _read_semicolon_rows(picard_metadata_path)
    allowed = set(target_bacteria) if target_bacteria is not None else None
    projected_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["bacteria"]):
        if allowed is not None and row["bacteria"] not in allowed:
            continue
        o_type = tl16._normalize_text(row.get("O-type", ""))
        h_type = tl16._normalize_text(row.get("H-type", ""))
        projected_rows.append(
            {
                "bacteria": row["bacteria"],
                "host_clermont_phylo": tl16._normalize_text(row.get("Clermont_Phylo", "")),
                "host_st_warwick": tl16._normalize_text(row.get("ST_Warwick", "")),
                "host_o_type": o_type,
                "host_h_type": h_type,
                "host_serotype": tl16.derive_serotype(o_type, h_type),
                "host_capsule_abc_proxy_present": int(float(row.get("Capsule_ABC", "0") or 0) > 0),
                "host_abc_serotype_proxy": tl16.normalize_legacy_abc_serotype(row.get("ABC_serotype", "")),
            }
        )
    if not projected_rows:
        raise ValueError(f"No panel metadata rows found in {picard_metadata_path}")
    return projected_rows


def build_tl15_runtime_payload(
    *,
    output_dir: Path,
    picard_metadata_path: Path,
    o_type_output_path: Path,
    o_type_allele_path: Path,
    o_antigen_override_path: Path,
    abc_capsule_profile_dir: Path,
    abc_capsule_definition_dir: Path,
    omp_reference_path: Path,
) -> dict[str, object]:
    runtime_dir = output_dir / "runtime_tl15"
    ensure_directory(runtime_dir)
    picard_rows = tl15.read_delimited_rows(picard_metadata_path, delimiter=";")
    o_type_output_rows = tl15.read_delimited_rows(o_type_output_path, delimiter="\t")
    o_type_allele_rows = tl15.read_delimited_rows(o_type_allele_path, delimiter="\t")
    override_references = tl15.load_o_antigen_override_references(o_antigen_override_path)
    o_antigen_references, o_type_contract = tl15.build_o_antigen_reference_contract(
        o_type_output_rows=o_type_output_rows,
        o_type_allele_rows=o_type_allele_rows,
        override_references=override_references,
    )
    o_antigen_query_path = runtime_dir / "o_antigen_reference_queries.fna"
    tl15.write_o_antigen_queries(o_antigen_query_path, o_antigen_references)
    capsule_hmm_bundle_path = tl15.write_capsule_hmm_bundle(abc_capsule_profile_dir, runtime_dir)
    omp_reference_copy_path = runtime_dir / omp_reference_path.name
    shutil.copy2(omp_reference_path, omp_reference_copy_path)
    capsule_models = [
        {
            "model_id": model.model_id,
            "mandatory_genes": sorted(model.mandatory_genes),
            "all_genes": sorted(model.all_genes),
            "min_mandatory_genes_required": model.min_mandatory_genes_required,
            "min_genes_required": model.min_genes_required,
            "inter_gene_max_space": model.inter_gene_max_space,
        }
        for model in tl15.load_capsule_models(abc_capsule_definition_dir)
    ]
    return {
        "block_id": TL15_BLOCK_ID,
        "categorical_columns": list(TL15_CATEGORICAL_COLUMNS),
        "numeric_columns": list(TL15_NUMERIC_COLUMNS),
        "o_antigen_query_filename": o_antigen_query_path.name,
        "capsule_hmm_bundle_filename": capsule_hmm_bundle_path.name,
        "omp_reference_filename": omp_reference_copy_path.name,
        "lps_lookup": tl15.build_lps_proxy_lookup(picard_rows),
        "o_type_contract": o_type_contract,
        "o_antigen_references": [
            {
                "query_id": reference.query_id,
                "o_type": reference.o_type,
                "gene_family": reference.gene_family,
                "allele_key": reference.allele_key,
                "sequence": reference.sequence,
            }
            for reference in o_antigen_references
        ],
        "capsule_models": capsule_models,
        "runtime_dirname": runtime_dir.name,
    }


def build_tl16_runtime_payload(
    *,
    output_dir: Path,
    capsule_definition_dir: Path,
    capsule_profile_dir: Path,
) -> dict[str, object]:
    runtime_dir = output_dir / "runtime_tl16"
    definition_copy_dir = runtime_dir / capsule_definition_dir.name
    profile_copy_dir = runtime_dir / capsule_profile_dir.name
    if definition_copy_dir.exists():
        shutil.rmtree(definition_copy_dir)
    if profile_copy_dir.exists():
        shutil.rmtree(profile_copy_dir)
    ensure_directory(runtime_dir)
    shutil.copytree(capsule_definition_dir, definition_copy_dir)
    shutil.copytree(capsule_profile_dir, profile_copy_dir)
    return {
        "block_id": TL16_BLOCK_ID,
        "categorical_columns": list(TL16_CATEGORICAL_COLUMNS),
        "numeric_columns": list(TL16_NUMERIC_COLUMNS),
        "capsule_definition_dirname": definition_copy_dir.name,
        "capsule_profile_dirname": profile_copy_dir.name,
        "micromamba_bin": tl16.MICROMAMBA_BIN,
        "phylogroup_env_name": tl16.PHYLOGROUP_ENV_NAME,
        "serotype_env_name": tl16.SEROTYPE_ENV_NAME,
        "sequence_type_env_name": tl16.SEQUENCE_TYPE_ENV_NAME,
        "mlst_scheme": tl16.MLST_SCHEME,
        "runtime_dirname": runtime_dir.name,
    }


def _reconstruct_tl15_capsule_models(payload: Mapping[str, object]) -> list[tl15.CapsuleModel]:
    return [
        tl15.CapsuleModel(
            model_id=str(row["model_id"]),
            mandatory_genes=frozenset(str(value) for value in row["mandatory_genes"]),
            all_genes=frozenset(str(value) for value in row["all_genes"]),
            min_mandatory_genes_required=int(row["min_mandatory_genes_required"]),
            min_genes_required=int(row["min_genes_required"]),
            inter_gene_max_space=int(row["inter_gene_max_space"]),
        )
        for row in payload.get("capsule_models", [])
    ]


def project_tl15_host_features(
    assembly_path: Path,
    *,
    bacteria: str,
    bundle_dir: Path,
    runtime_payload: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    runtime_dir = bundle_dir / str(runtime_payload["runtime_dirname"])
    o_antigen_query_path = runtime_dir / str(runtime_payload["o_antigen_query_filename"])
    capsule_hmm_bundle_path = runtime_dir / str(runtime_payload["capsule_hmm_bundle_filename"])
    omp_reference_path = runtime_dir / str(runtime_payload["omp_reference_filename"])
    ensure_directory(output_dir)
    proteins_path = output_dir / "tl15_predicted_proteins.faa"
    protein_metadata = tl15.predict_proteins(assembly_path, proteins_path)
    o_antigen_references = [
        tl15.OAlleleReference(
            query_id=str(row["query_id"]),
            o_type=str(row["o_type"]),
            gene_family=str(row["gene_family"]),
            allele_key=str(row["allele_key"]),
            sequence=str(row["sequence"]),
        )
        for row in runtime_payload.get("o_antigen_references", [])
    ]
    o_antigen_type, _ = tl15.call_o_antigen_type(
        bacteria=bacteria,
        assembly_path=assembly_path,
        reference_fasta_path=o_antigen_query_path,
        references=o_antigen_references,
        o_type_contract=dict(runtime_payload["o_type_contract"]),
        output_dir=output_dir,
    )
    hmmscan_hits = tl15.run_hmmscan(
        bacteria=bacteria,
        proteins_path=proteins_path,
        hmm_bundle_path=capsule_hmm_bundle_path,
        output_dir=output_dir,
    )
    capsule_call = tl15.choose_capsule_call(
        models=_reconstruct_tl15_capsule_models(runtime_payload),
        hits=hmmscan_hits,
        protein_metadata=protein_metadata,
    )
    receptor_calls = tl15.call_receptor_presence(
        bacteria=bacteria,
        proteins_path=proteins_path,
        omp_reference_path=omp_reference_path,
        output_dir=output_dir,
    )
    feature_row, _ = tl15.build_projected_feature_row(
        bacteria=bacteria,
        o_antigen_type=o_antigen_type,
        capsule_call=capsule_call,
        lps_lookup=dict(runtime_payload["lps_lookup"]),
        receptor_presence_calls=receptor_calls,
    )
    return feature_row


def project_tl16_host_features(
    assembly_path: Path,
    *,
    bacteria: str,
    bundle_dir: Path,
    runtime_payload: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    runtime_dir = bundle_dir / str(runtime_payload["runtime_dirname"])
    ensure_directory(output_dir)
    definitions = tl16.parse_capsule_definitions(runtime_dir / str(runtime_payload["capsule_definition_dirname"]))
    hmms = tl16.load_capsule_hmms(runtime_dir / str(runtime_payload["capsule_profile_dirname"]))
    phylogroup_report_path = tl16.run_phylogroup_caller(
        bacteria=bacteria,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )
    serotype_output_path, _ = tl16.run_serotype_caller(
        bacteria=bacteria,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )
    mlst_output_path = tl16.run_sequence_type_caller(
        bacteria=bacteria,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )
    capsule_proxy, _ = tl16.scan_capsule_proxy(
        assembly_path=assembly_path,
        output_dir=output_dir / "capsule_proxy",
        definitions=definitions,
        hmms=hmms,
    )
    projected_row = tl16.build_projected_feature_row(
        bacteria=bacteria,
        phylogroup_call=tl16.parse_phylogroup_report(phylogroup_report_path),
        mlst_call=tl16.parse_mlst_legacy_output(mlst_output_path),
        serotype_call=tl16.parse_ectyper_output(serotype_output_path),
        capsule_proxy=capsule_proxy,
    )
    return {column: projected_row[column] for column in ("bacteria", *TL16_CATEGORICAL_COLUMNS, *TL16_NUMERIC_COLUMNS)}


def write_roundtrip_host_inventory(rows: Sequence[Mapping[str, object]], output_path: Path) -> Path:
    if not rows:
        raise ValueError("Round-trip host inventory is empty.")
    write_csv(output_path, list(rows[0].keys()), list(rows))
    return output_path
