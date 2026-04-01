#!/usr/bin/env python3
"""TL16: project deployable host-typing features from raw host assemblies."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pyhmmer

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import sha256_file
from lyzortx.pipeline.track_l.steps.run_novel_host_defense_finder import predict_proteins_with_pyrodigal

LOGGER = logging.getLogger(__name__)

DEFAULT_FASTA_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_VALIDATION_MANIFEST_PATH = Path("data/genomics/bacteria/validation_subset/manifest.json")
DEFAULT_PANEL_METADATA_PATH = Path("data/genomics/bacteria/picard_collection.csv")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/host_typing_projector_tl16")
DEFAULT_CAPSULE_DEFINITION_DIR = Path("data/genomics/bacteria/capsules/ABC_capsules_types/definitions")
DEFAULT_CAPSULE_PROFILE_DIR = Path("data/genomics/bacteria/capsules/ABC_capsules_types/profiles")

PROJECTED_FEATURES_FILENAME = "tl16_projected_host_typing_features.csv"
RAW_CALLS_FILENAME = "tl16_raw_host_typing_calls.csv"
INPUT_INVENTORY_FILENAME = "tl16_validation_input_inventory.csv"
FIELD_STATUS_FILENAME = "tl16_legacy_field_status.csv"
FAMILY_VALIDATION_FILENAME = "tl16_feature_family_validation.csv"
MANIFEST_FILENAME = "tl16_host_typing_projector_manifest.json"

PHYLOGROUP_ENV_NAME = "phylogroup_caller"
SEROTYPE_ENV_NAME = "serotype_caller"
SEQUENCE_TYPE_ENV_NAME = "sequence_type_caller"

CALLER_RUN_DIRNAME = "raw_calls"
PHYLOGROUP_REPORT_SUFFIX = "_phylogroups.txt"
SEROTYPE_OUTPUT_FILENAME = "output.tsv"
SEROTYPE_BLAST_FILENAME = "blastn_output_alleles.txt"
MLST_OUTPUT_FILENAME = "mlst_legacy.tsv"
CAPSULE_PROFILE_HITS_FILENAME = "capsule_profile_hits.csv"
CAPSULE_PROTEIN_FASTA_FILENAME = "capsule_proxy_proteins.faa"

MICROMAMBA_BIN = "micromamba"
MLST_SCHEME = "ecoli_achtman_4"

PROJECTED_FEATURE_COLUMNS = (
    "bacteria",
    "host_clermont_phylo",
    "host_st_warwick",
    "host_o_type",
    "host_h_type",
    "host_serotype",
    "host_capsule_abc_proxy_present",
    "host_abc_serotype_proxy",
    "host_capsule_proxy_top_model",
    "host_capsule_proxy_model_count",
    "host_capsule_proxy_candidate_models",
)


@dataclass(frozen=True)
class CapsuleModelDefinition:
    model_name: str
    min_mandatory_genes_required: int
    min_genes_required: int
    genes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CapsuleProfileHit:
    profile_name: str
    protein_id: str
    score: float
    evalue: float


@dataclass(frozen=True)
class CapsuleModelMatch:
    model_name: str
    mandatory_gene_hits: int
    total_gene_hits: int
    score_sum: float
    matched_genes: tuple[str, ...]


DIRECT_FIELD_SPECS: tuple[dict[str, str], ...] = (
    {
        "legacy_field_name": "host_clermont_phylo",
        "feature_family": "phylogroup",
        "projection_status": "direct",
        "projected_feature_name": "host_clermont_phylo",
        "legacy_source_column": "Clermont_Phylo",
        "rationale": "Clermont typing is called directly from the assembly with the pinned phylogroup env.",
    },
    {
        "legacy_field_name": "host_st_warwick",
        "feature_family": "sequence_type",
        "projection_status": "direct",
        "projected_feature_name": "host_st_warwick",
        "legacy_source_column": "ST_Warwick",
        "rationale": "MLST ST is called directly from the assembly with the pinned Achtman-4 scheme.",
    },
    {
        "legacy_field_name": "host_o_type",
        "feature_family": "serotype",
        "projection_status": "direct",
        "projected_feature_name": "host_o_type",
        "legacy_source_column": "O-type",
        "rationale": "ECTyper reports O-antigen type directly from the assembly.",
    },
    {
        "legacy_field_name": "host_h_type",
        "feature_family": "serotype",
        "projection_status": "direct",
        "projected_feature_name": "host_h_type",
        "legacy_source_column": "H-type",
        "rationale": "ECTyper reports H-antigen type directly from the assembly.",
    },
)

PROXY_FIELD_SPECS: tuple[dict[str, str], ...] = (
    {
        "legacy_field_name": "host_capsule_abc",
        "feature_family": "capsule_presence",
        "projection_status": "deployable_proxy",
        "projected_feature_name": "host_capsule_abc_proxy_present",
        "legacy_source_column": "Capsule_ABC",
        "rationale": (
            "The repo has vendored capsule HMMs but no proven raw-genome reproducer for the legacy Capsule_ABC flag, "
            "so the projector emits an explicit capsule-locus proxy instead of claiming exact parity."
        ),
    },
    {
        "legacy_field_name": "host_abc_serotype",
        "feature_family": "capsule_typed_serotype",
        "projection_status": "deployable_proxy",
        "projected_feature_name": "host_abc_serotype_proxy",
        "legacy_source_column": "ABC_serotype",
        "rationale": (
            "The projector exposes the top HMM-backed capsule model as a proxy, but the current raw-genome capsule "
            "path is too noisy to relabel it as the legacy ABC serotype field."
        ),
    },
)

UNSUPPORTED_FIELD_SPECS: tuple[dict[str, str], ...] = (
    {
        "legacy_field_name": "host_capsule_groupiv_e",
        "feature_family": "capsule_groupiv",
        "projection_status": "unsupported",
        "projected_feature_name": "",
        "legacy_source_column": "Capsule_GroupIV_e",
        "rationale": "No checked-in raw-genome caller or validated proxy exists yet for the legacy Group IV capsule flag.",
    },
    {
        "legacy_field_name": "host_capsule_groupiv_e_stricte",
        "feature_family": "capsule_groupiv",
        "projection_status": "unsupported",
        "projected_feature_name": "",
        "legacy_source_column": "Capsule_GroupIV_e_stricte",
        "rationale": "No checked-in raw-genome caller or validated proxy exists yet for the strict Group IV flag.",
    },
    {
        "legacy_field_name": "host_capsule_groupiv_s",
        "feature_family": "capsule_groupiv",
        "projection_status": "unsupported",
        "projected_feature_name": "",
        "legacy_source_column": "Capsule_GroupIV_s",
        "rationale": "The legacy ordinal Group IV signal has no validated raw-genome reproduction path in this repo.",
    },
    {
        "legacy_field_name": "host_capsule_wzy_stricte",
        "feature_family": "capsule_groupiv",
        "projection_status": "unsupported",
        "projected_feature_name": "",
        "legacy_source_column": "Capsule_Wzy_stricte",
        "rationale": "No validated raw-genome projector exists yet for the strict Wzy-related capsule flag.",
    },
)

NON_DERIVABLE_FIELD_SPECS: tuple[dict[str, str], ...] = (
    {
        "legacy_field_name": "host_origin",
        "feature_family": "non_derivable_metadata",
        "projection_status": "non_derivable",
        "projected_feature_name": "",
        "legacy_source_column": "Origin",
        "rationale": "Origin is collection metadata, not a genome-derived feature.",
    },
    {
        "legacy_field_name": "host_pathotype",
        "feature_family": "non_derivable_metadata",
        "projection_status": "non_derivable",
        "projected_feature_name": "",
        "legacy_source_column": "Pathotype",
        "rationale": "Pathotype is an interpreted metadata label in this repo, not a deployable raw-genome projector output.",
    },
    {
        "legacy_field_name": "host_collection",
        "feature_family": "non_derivable_metadata",
        "projection_status": "non_derivable",
        "projected_feature_name": "",
        "legacy_source_column": "Collection",
        "rationale": "Collection labels are provenance metadata, not genome-derived host typing.",
    },
    {
        "legacy_field_name": "host_mouse_killed_10",
        "feature_family": "non_derivable_metadata",
        "projection_status": "non_derivable",
        "projected_feature_name": "",
        "legacy_source_column": "Mouse_killed_10",
        "rationale": "Mouse lethality is assay metadata and cannot be reconstructed from the assembly alone.",
    },
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-dir", type=Path, default=DEFAULT_FASTA_DIR, help="Directory of host assembly FASTAs.")
    parser.add_argument(
        "--validation-manifest-path",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST_PATH,
        help="Validation subset manifest.json with source checksums.",
    )
    parser.add_argument(
        "--panel-metadata-path",
        type=Path,
        default=DEFAULT_PANEL_METADATA_PATH,
        help="Picard host metadata CSV for parity checks.",
    )
    parser.add_argument(
        "--capsule-definition-dir",
        type=Path,
        default=DEFAULT_CAPSULE_DEFINITION_DIR,
        help="Directory of vendored capsule model XML definitions.",
    )
    parser.add_argument(
        "--capsule-profile-dir",
        type=Path,
        default=DEFAULT_CAPSULE_PROFILE_DIR,
        help="Directory of vendored capsule HMM profiles.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run external callers even if raw output files already exist.",
    )
    return parser.parse_args(argv)


def _run_command(
    command: Sequence[str],
    *,
    description: str,
    cwd: Path | None = None,
    stdout_path: Path | None = None,
) -> None:
    LOGGER.info("Starting %s", description)
    start_time = datetime.now(timezone.utc)
    ensure_directory(stdout_path.parent) if stdout_path is not None else None
    with stdout_path.open("w", encoding="utf-8") if stdout_path is not None else open(os.devnull, "w") as stdout_handle:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=stdout_handle if stdout_path is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    if result.returncode != 0:
        stdout_tail = ""
        if stdout_path is None and isinstance(result.stdout, str):
            stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:] if result.stderr else ""
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode} after {elapsed_seconds:.1f}s\n"
            f"stdout: {stdout_tail}\nstderr: {stderr_tail}"
        )
    LOGGER.info("Completed %s in %.1fs", description, elapsed_seconds)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "-", "Unknown", "nan", "None"}:
        return ""
    return text


def derive_serotype(o_type: object, h_type: object) -> str:
    o_value = _normalize_text(o_type)
    h_value = _normalize_text(h_type)
    if o_value and h_value:
        return f"{o_value}:{h_value}"
    return o_value or h_value


def normalize_legacy_abc_serotype(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    if text.isdigit():
        return f"K{text}"
    if text.upper().startswith("K") and len(text) > 1:
        return f"K{text[1:]}"
    return text


def normalize_capsule_model_to_serotype_proxy(model_name: object) -> str:
    text = _normalize_text(model_name)
    if not text.startswith("K"):
        return ""
    prefix = text.split("_", maxsplit=1)[0]
    if prefix.startswith("K") and prefix[1:].isdigit():
        return prefix
    return text if text.endswith("_like") else ""


def load_validation_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "files" not in payload:
        raise ValueError(f"Validation manifest at {path} is missing the 'files' key.")
    return payload


def build_input_inventory_rows(
    *, fasta_dir: Path, validation_manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    rows = []
    manifest_by_bacteria = {
        str(entry["bacteria"]): dict(entry)
        for entry in validation_manifest.get("files", [])
        if isinstance(entry, Mapping) and "bacteria" in entry
    }
    for fasta_path in sorted(fasta_dir.glob("*.fasta")):
        bacteria = fasta_path.stem
        manifest_entry = manifest_by_bacteria.get(bacteria)
        if manifest_entry is None:
            raise KeyError(f"{fasta_path.name} is missing from validation manifest {DEFAULT_VALIDATION_MANIFEST_PATH}")
        sha256 = sha256_file(fasta_path)
        rows.append(
            {
                "bacteria": bacteria,
                "fasta_path": str(fasta_path),
                "bytes": fasta_path.stat().st_size,
                "sha256": sha256,
                "manifest_sha256": str(manifest_entry["sha256"]),
                "sha256_matches_manifest": int(sha256 == str(manifest_entry["sha256"])),
                "figshare_file_id": manifest_entry["figshare_file_id"],
                "source_md5": manifest_entry["source_md5"],
                "download_url": manifest_entry["download_url"],
            }
        )
    if not rows:
        raise ValueError(f"No FASTA files found in {fasta_dir}")
    return rows


def load_panel_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Panel metadata CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]
    return {row["bacteria"]: row for row in rows if row.get("bacteria")}


def parse_phylogroup_report(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Phylogroup report not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one phylogroup report row in {path}, found {len(lines)}")
    parts = lines[0].split("\t")
    if len(parts) < 6:
        raise ValueError(f"Unexpected phylogroup report format in {path}: {lines[0]}")
    return {
        "assembly_name": parts[0],
        "phylogroup": parts[4].strip(),
        "mash_phylogroup": parts[5].strip(),
    }


def parse_ectyper_output(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"ECTyper output not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one ECTyper output row in {path}, found {len(rows)}")
    row = rows[0]
    return {
        "species": _normalize_text(row.get("Species", "")),
        "o_type": _normalize_text(row.get("O-type", "")),
        "h_type": _normalize_text(row.get("H-type", "")),
        "serotype": _normalize_text(row.get("Serotype", "")),
        "qc": _normalize_text(row.get("QC", "")),
        "warnings": _normalize_text(row.get("Warnings", "")),
    }


def parse_mlst_legacy_output(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"MLST legacy output not found: {path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_lines = [
        line for line in lines if "\t" in line and not line.startswith("FILE\t") and not line.startswith("This is mlst")
    ]
    header = next((line for line in lines if line.startswith("FILE\t")), None)
    if header is None or len(data_lines) != 1:
        raise ValueError(f"Unexpected MLST legacy output format in {path}")
    header_columns = header.split("\t")
    values = data_lines[0].split("\t")
    if len(values) != len(header_columns):
        raise ValueError(f"MLST legacy output row length does not match header in {path}")
    row = dict(zip(header_columns, values, strict=True))
    return {
        "assembly_path": row["FILE"],
        "scheme": row["SCHEME"],
        "st_warwick": row["ST"],
    }


def parse_capsule_definitions(definition_dir: Path) -> list[CapsuleModelDefinition]:
    paths = sorted(definition_dir.glob("*.xml"))
    if not paths:
        raise FileNotFoundError(f"No capsule model definitions found in {definition_dir}")
    definitions = []
    for path in paths:
        root = ET.parse(path).getroot()
        definitions.append(
            CapsuleModelDefinition(
                model_name=path.stem,
                min_mandatory_genes_required=int(root.attrib["min_mandatory_genes_required"]),
                min_genes_required=int(root.attrib["min_genes_required"]),
                genes=tuple((gene.attrib["name"], gene.attrib["presence"]) for gene in root.findall("gene")),
            )
        )
    return definitions


def load_capsule_hmms(profile_dir: Path) -> list[tuple[str, pyhmmer.plan7.HMM, float]]:
    paths = sorted(profile_dir.glob("*.hmm"))
    if not paths:
        raise FileNotFoundError(f"No capsule HMM profiles found in {profile_dir}")
    hmms = []
    for path in paths:
        with path.open("rb") as handle:
            hmm = next(pyhmmer.plan7.HMMFile(handle))
        cutoff = (hmm.cutoffs.gathering or (0.0, 0.0))[0]
        hmms.append((path.stem, hmm, cutoff))
    return hmms


def scan_capsule_proxy(
    *,
    assembly_path: Path,
    output_dir: Path,
    definitions: Sequence[CapsuleModelDefinition],
    hmms: Sequence[tuple[str, pyhmmer.plan7.HMM, float]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    LOGGER.info("Starting capsule proxy scan for %s", assembly_path.name)
    start_time = datetime.now(timezone.utc)
    protein_fasta_path = output_dir / CAPSULE_PROTEIN_FASTA_FILENAME
    protein_metadata = predict_proteins_with_pyrodigal(assembly_path, protein_fasta_path=protein_fasta_path)
    alphabet = pyhmmer.easel.Alphabet.amino()
    with protein_fasta_path.open("rb") as handle:
        proteins = list(pyhmmer.easel.SequenceFile(handle, digital=True, alphabet=alphabet).read_block())

    best_hit_by_profile: dict[str, CapsuleProfileHit] = {}
    for profile_name, hmm, cutoff in hmms:
        top_hits = next(pyhmmer.hmmsearch([hmm], proteins, cpus=1))
        best_hit = max(top_hits, key=lambda hit: hit.score, default=None)
        if best_hit is None or best_hit.score < cutoff:
            continue
        best_hit_by_profile[profile_name] = CapsuleProfileHit(
            profile_name=profile_name,
            protein_id=str(best_hit.name),
            score=float(best_hit.score),
            evalue=float(best_hit.evalue),
        )

    model_matches: list[CapsuleModelMatch] = []
    for definition in definitions:
        matched_genes = tuple(
            sorted(gene_name for gene_name, _ in definition.genes if gene_name in best_hit_by_profile)
        )
        mandatory_gene_hits = sum(
            1
            for gene_name, presence in definition.genes
            if presence == "mandatory" and gene_name in best_hit_by_profile
        )
        total_gene_hits = len(matched_genes)
        if (
            mandatory_gene_hits >= definition.min_mandatory_genes_required
            and total_gene_hits >= definition.min_genes_required
        ):
            score_sum = sum(best_hit_by_profile[gene_name].score for gene_name in matched_genes)
            model_matches.append(
                CapsuleModelMatch(
                    model_name=definition.model_name,
                    mandatory_gene_hits=mandatory_gene_hits,
                    total_gene_hits=total_gene_hits,
                    score_sum=round(score_sum, 6),
                    matched_genes=matched_genes,
                )
            )

    model_matches.sort(
        key=lambda match: (match.mandatory_gene_hits, match.total_gene_hits, match.score_sum, match.model_name),
        reverse=True,
    )
    candidate_models = [match.model_name for match in model_matches]
    top_model = candidate_models[0] if candidate_models else ""
    abc_serotype_proxy = ""
    for candidate in candidate_models:
        normalized = normalize_capsule_model_to_serotype_proxy(candidate)
        if normalized:
            abc_serotype_proxy = normalized
            break
    projection = {
        "host_capsule_abc_proxy_present": int(bool(candidate_models)),
        "host_abc_serotype_proxy": abc_serotype_proxy,
        "host_capsule_proxy_top_model": top_model,
        "host_capsule_proxy_model_count": len(candidate_models),
        "host_capsule_proxy_candidate_models": "|".join(candidate_models[:10]),
        "capsule_proxy_profile_hit_count": len(best_hit_by_profile),
        "capsule_proxy_predicted_cds_count": protein_metadata["predicted_cds_count"],
    }
    hit_rows = [
        {
            "profile_name": hit.profile_name,
            "protein_id": hit.protein_id,
            "score": round(hit.score, 6),
            "evalue": hit.evalue,
        }
        for hit in sorted(best_hit_by_profile.values(), key=lambda item: (item.score, item.profile_name), reverse=True)
    ]
    elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    LOGGER.info(
        "Completed capsule proxy scan for %s in %.1fs (%d candidate models, %d profile hits)",
        assembly_path.name,
        elapsed_seconds,
        len(candidate_models),
        len(hit_rows),
    )
    return projection, hit_rows


def run_phylogroup_caller(*, bacteria: str, assembly_path: Path, output_dir: Path, force: bool) -> Path:
    run_root = output_dir / "phylogroup"
    run_name = "phylogroup_run"
    report_path = run_root / run_name / f"{run_name}{PHYLOGROUP_REPORT_SUFFIX}"
    if report_path.exists() and not force:
        return report_path
    ensure_directory(run_root)
    if (run_root / run_name).exists():
        shutil.rmtree(run_root / run_name)
    _run_command(
        [
            MICROMAMBA_BIN,
            "run",
            "-n",
            PHYLOGROUP_ENV_NAME,
            "clermonTyping.sh",
            "--fasta",
            str(assembly_path.resolve()),
            "--name",
            run_name,
            "--minimal",
        ],
        description=f"Clermont typing for {bacteria}",
        cwd=run_root,
    )
    if not report_path.exists():
        raise FileNotFoundError(f"Clermont typing completed for {bacteria} but did not create {report_path}")
    return report_path


def run_serotype_caller(*, bacteria: str, assembly_path: Path, output_dir: Path, force: bool) -> tuple[Path, Path]:
    run_root = output_dir / "serotype"
    output_path = run_root / SEROTYPE_OUTPUT_FILENAME
    blast_path = run_root / SEROTYPE_BLAST_FILENAME
    if output_path.exists() and blast_path.exists() and not force:
        return output_path, blast_path
    ensure_directory(run_root)
    _run_command(
        [
            MICROMAMBA_BIN,
            "run",
            "-n",
            SEROTYPE_ENV_NAME,
            "ectyper",
            "-i",
            str(assembly_path.resolve()),
            "-o",
            str(run_root.resolve()),
        ],
        description=f"ECTyper serotype call for {bacteria}",
    )
    if not output_path.exists():
        raise FileNotFoundError(f"ECTyper completed for {bacteria} but did not create {output_path}")
    return output_path, blast_path


def run_sequence_type_caller(*, bacteria: str, assembly_path: Path, output_dir: Path, force: bool) -> Path:
    run_root = output_dir / "sequence_type"
    output_path = run_root / MLST_OUTPUT_FILENAME
    if output_path.exists() and not force:
        return output_path
    ensure_directory(run_root)
    _run_command(
        [
            MICROMAMBA_BIN,
            "run",
            "-n",
            SEQUENCE_TYPE_ENV_NAME,
            "mlst",
            "--scheme",
            MLST_SCHEME,
            "--legacy",
            str(assembly_path.resolve()),
        ],
        description=f"MLST call for {bacteria}",
        stdout_path=output_path,
    )
    return output_path


def build_projected_feature_row(
    *,
    bacteria: str,
    phylogroup_call: Mapping[str, str],
    mlst_call: Mapping[str, str],
    serotype_call: Mapping[str, str],
    capsule_proxy: Mapping[str, object],
) -> dict[str, object]:
    o_type = _normalize_text(serotype_call.get("o_type"))
    h_type = _normalize_text(serotype_call.get("h_type"))
    row = {
        "bacteria": bacteria,
        "host_clermont_phylo": _normalize_text(phylogroup_call.get("phylogroup")),
        "host_st_warwick": _normalize_text(mlst_call.get("st_warwick")),
        "host_o_type": o_type,
        "host_h_type": h_type,
        "host_serotype": derive_serotype(o_type, h_type),
        "host_capsule_abc_proxy_present": int(capsule_proxy["host_capsule_abc_proxy_present"]),
        "host_abc_serotype_proxy": _normalize_text(capsule_proxy.get("host_abc_serotype_proxy")),
        "host_capsule_proxy_top_model": _normalize_text(capsule_proxy.get("host_capsule_proxy_top_model")),
        "host_capsule_proxy_model_count": int(capsule_proxy["host_capsule_proxy_model_count"]),
        "host_capsule_proxy_candidate_models": _normalize_text(
            capsule_proxy.get("host_capsule_proxy_candidate_models")
        ),
    }
    return row


def build_raw_call_row(
    *,
    bacteria: str,
    assembly_path: Path,
    phylogroup_call: Mapping[str, str],
    mlst_call: Mapping[str, str],
    serotype_call: Mapping[str, str],
    capsule_proxy: Mapping[str, object],
) -> dict[str, object]:
    return {
        "bacteria": bacteria,
        "assembly_path": str(assembly_path),
        "phylogroup": _normalize_text(phylogroup_call.get("phylogroup")),
        "phylogroup_mash_group": _normalize_text(phylogroup_call.get("mash_phylogroup")),
        "st_warwick": _normalize_text(mlst_call.get("st_warwick")),
        "st_scheme": _normalize_text(mlst_call.get("scheme")),
        "o_type": _normalize_text(serotype_call.get("o_type")),
        "h_type": _normalize_text(serotype_call.get("h_type")),
        "serotype": _normalize_text(serotype_call.get("serotype")),
        "ectyper_species": _normalize_text(serotype_call.get("species")),
        "ectyper_qc": _normalize_text(serotype_call.get("qc")),
        "ectyper_warnings": _normalize_text(serotype_call.get("warnings")),
        "capsule_abc_proxy_present": int(capsule_proxy["host_capsule_abc_proxy_present"]),
        "capsule_top_model": _normalize_text(capsule_proxy.get("host_capsule_proxy_top_model")),
        "capsule_candidate_models": _normalize_text(capsule_proxy.get("host_capsule_proxy_candidate_models")),
        "capsule_candidate_model_count": int(capsule_proxy["host_capsule_proxy_model_count"]),
        "capsule_profile_hit_count": int(capsule_proxy["capsule_proxy_profile_hit_count"]),
        "capsule_predicted_cds_count": int(capsule_proxy["capsule_proxy_predicted_cds_count"]),
    }


def build_legacy_validation_rows(
    *,
    projected_rows: Sequence[Mapping[str, object]],
    panel_metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    projected_by_bacteria = {str(row["bacteria"]): dict(row) for row in projected_rows}
    rows: list[dict[str, object]] = []

    for spec in DIRECT_FIELD_SPECS:
        resolved = 0
        exact = 0
        for bacteria, projected in projected_by_bacteria.items():
            legacy_row = panel_metadata.get(bacteria)
            if legacy_row is None:
                raise KeyError(f"Missing panel metadata row for {bacteria}")
            legacy_value = _normalize_text(legacy_row[spec["legacy_source_column"]])
            projected_value = _normalize_text(projected[spec["projected_feature_name"]])
            if legacy_value:
                resolved += 1
                exact += int(projected_value == legacy_value)
        rows.append(
            {
                **spec,
                "validated_host_count": len(projected_rows),
                "legacy_resolved_host_count": resolved,
                "exact_match_count": exact,
                "agreement_rate_among_resolved": round(exact / resolved, 6) if resolved else "",
                "validation_outcome": "reproduced_directly" if exact == resolved and resolved else "mismatch",
            }
        )

    serotype_resolved = 0
    serotype_exact = 0
    for bacteria, projected in projected_by_bacteria.items():
        legacy_row = panel_metadata[bacteria]
        legacy_serotype = derive_serotype(legacy_row.get("O-type", ""), legacy_row.get("H-type", ""))
        projected_serotype = _normalize_text(projected["host_serotype"])
        if legacy_serotype:
            serotype_resolved += 1
            serotype_exact += int(projected_serotype == legacy_serotype)
    rows.append(
        {
            "legacy_field_name": "host_serotype",
            "feature_family": "serotype",
            "projection_status": "direct",
            "projected_feature_name": "host_serotype",
            "legacy_source_column": "O-type|H-type",
            "rationale": "The projector materializes the combined serotype from the direct ECTyper O/H calls.",
            "validated_host_count": len(projected_rows),
            "legacy_resolved_host_count": serotype_resolved,
            "exact_match_count": serotype_exact,
            "agreement_rate_among_resolved": round(serotype_exact / serotype_resolved, 6) if serotype_resolved else "",
            "validation_outcome": "reproduced_directly"
            if serotype_exact == serotype_resolved and serotype_resolved
            else "mismatch",
        }
    )

    for spec in PROXY_FIELD_SPECS:
        resolved = 0
        exact = 0
        for bacteria, projected in projected_by_bacteria.items():
            legacy_row = panel_metadata[bacteria]
            if spec["legacy_field_name"] == "host_abc_serotype":
                legacy_value = normalize_legacy_abc_serotype(legacy_row[spec["legacy_source_column"]])
            else:
                raw_legacy = _normalize_text(legacy_row[spec["legacy_source_column"]])
                legacy_value = "" if raw_legacy == "" else str(int(float(raw_legacy)))
            projected_value = _normalize_text(projected[spec["projected_feature_name"]])
            if legacy_value:
                resolved += 1
                exact += int(projected_value == legacy_value)
        outcome = "deployable_proxy"
        if resolved and exact < resolved:
            outcome = "noisy_proxy"
        rows.append(
            {
                **spec,
                "validated_host_count": len(projected_rows),
                "legacy_resolved_host_count": resolved,
                "exact_match_count": exact,
                "agreement_rate_among_resolved": round(exact / resolved, 6) if resolved else "",
                "validation_outcome": outcome,
            }
        )

    for spec in UNSUPPORTED_FIELD_SPECS:
        rows.append(
            {
                **spec,
                "validated_host_count": len(projected_rows),
                "legacy_resolved_host_count": "",
                "exact_match_count": "",
                "agreement_rate_among_resolved": "",
                "validation_outcome": "unsupported",
            }
        )
    for spec in NON_DERIVABLE_FIELD_SPECS:
        rows.append(
            {
                **spec,
                "validated_host_count": len(projected_rows),
                "legacy_resolved_host_count": "",
                "exact_match_count": "",
                "agreement_rate_among_resolved": "",
                "validation_outcome": "non_derivable",
            }
        )
    return rows


def build_family_validation_rows(field_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    families = sorted({str(row["feature_family"]) for row in field_rows})
    out = []
    for family in families:
        family_rows = [row for row in field_rows if str(row["feature_family"]) == family]
        projected_columns = sorted(
            {str(row["projected_feature_name"]) for row in family_rows if row["projected_feature_name"]}
        )
        legacy_fields = sorted({str(row["legacy_field_name"]) for row in family_rows})
        statuses = {str(row["validation_outcome"]) for row in family_rows}
        projection_statuses = {str(row["projection_status"]) for row in family_rows}
        if "direct" in projection_statuses:
            if statuses == {"reproduced_directly"}:
                family_outcome = "reproduced_directly"
            else:
                family_outcome = "direct_but_incomplete"
        elif "reproduced_directly" in statuses:
            family_outcome = "reproduced_directly"
        elif "noisy_proxy" in statuses:
            family_outcome = "noisy_proxy"
        elif "deployable_proxy" in statuses:
            family_outcome = "deployable_proxy"
        elif "unsupported" in statuses:
            family_outcome = "unsupported"
        else:
            family_outcome = "non_derivable"
        rationale = " ".join(str(row["rationale"]) for row in family_rows)
        resolved_counts = [
            int(row["legacy_resolved_host_count"]) for row in family_rows if row["legacy_resolved_host_count"] != ""
        ]
        exact_counts = [int(row["exact_match_count"]) for row in family_rows if row["exact_match_count"] != ""]
        out.append(
            {
                "feature_family": family,
                "family_outcome": family_outcome,
                "legacy_fields": "|".join(legacy_fields),
                "projected_feature_names": "|".join(projected_columns),
                "legacy_resolved_host_count": max(resolved_counts) if resolved_counts else "",
                "exact_match_count": min(exact_counts) if exact_counts else "",
                "rationale": rationale,
            }
        )
    return out


def capture_tool_versions() -> dict[str, str]:
    version_commands = {
        "serotype_caller": [MICROMAMBA_BIN, "run", "-n", SEROTYPE_ENV_NAME, "ectyper", "-V"],
        "sequence_type_caller": [MICROMAMBA_BIN, "run", "-n", SEQUENCE_TYPE_ENV_NAME, "mlst", "--version"],
    }
    versions: dict[str, str] = {
        "phylogroup_caller": _capture_micromamba_package_version(PHYLOGROUP_ENV_NAME, "clermontyping"),
    }
    for name, command in version_commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to capture version for {name}: {result.stderr[-1000:]}")
        text = (result.stdout or result.stderr).strip()
        versions[name] = text.splitlines()[-1].strip()
    return versions


def _capture_micromamba_package_version(env_name: str, package_name: str) -> str:
    result = subprocess.run(
        [MICROMAMBA_BIN, "list", "-n", env_name, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list packages for {env_name}: {result.stderr[-1000:]}")
    packages = json.loads(result.stdout)
    for package in packages:
        if str(package.get("name")) == package_name:
            return f"{package_name} {package['version']}"
    raise RuntimeError(f"{package_name} not found in micromamba env {env_name}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    validation_manifest = load_validation_manifest(args.validation_manifest_path)
    panel_metadata = load_panel_metadata(args.panel_metadata_path)
    input_inventory_rows = build_input_inventory_rows(fasta_dir=args.fasta_dir, validation_manifest=validation_manifest)
    if any(int(row["sha256_matches_manifest"]) != 1 for row in input_inventory_rows):
        raise ValueError("Validation FASTA checksum mismatch against manifest.json")

    definitions = parse_capsule_definitions(args.capsule_definition_dir)
    hmms = load_capsule_hmms(args.capsule_profile_dir)

    projected_rows: list[dict[str, object]] = []
    raw_call_rows: list[dict[str, object]] = []
    raw_call_root = args.output_dir / CALLER_RUN_DIRNAME

    for inventory_row in input_inventory_rows:
        bacteria = str(inventory_row["bacteria"])
        assembly_path = Path(str(inventory_row["fasta_path"]))
        host_output_dir = raw_call_root / bacteria
        ensure_directory(host_output_dir)

        phylogroup_report_path = run_phylogroup_caller(
            bacteria=bacteria,
            assembly_path=assembly_path,
            output_dir=host_output_dir,
            force=args.force,
        )
        serotype_output_path, _ = run_serotype_caller(
            bacteria=bacteria,
            assembly_path=assembly_path,
            output_dir=host_output_dir,
            force=args.force,
        )
        mlst_output_path = run_sequence_type_caller(
            bacteria=bacteria,
            assembly_path=assembly_path,
            output_dir=host_output_dir,
            force=args.force,
        )
        capsule_proxy, capsule_hit_rows = scan_capsule_proxy(
            assembly_path=assembly_path,
            output_dir=host_output_dir / "capsule_proxy",
            definitions=definitions,
            hmms=hmms,
        )
        write_csv(
            host_output_dir / "capsule_proxy" / CAPSULE_PROFILE_HITS_FILENAME,
            ("profile_name", "protein_id", "score", "evalue"),
            capsule_hit_rows,
        )

        phylogroup_call = parse_phylogroup_report(phylogroup_report_path)
        serotype_call = parse_ectyper_output(serotype_output_path)
        mlst_call = parse_mlst_legacy_output(mlst_output_path)

        projected_rows.append(
            build_projected_feature_row(
                bacteria=bacteria,
                phylogroup_call=phylogroup_call,
                mlst_call=mlst_call,
                serotype_call=serotype_call,
                capsule_proxy=capsule_proxy,
            )
        )
        raw_call_rows.append(
            build_raw_call_row(
                bacteria=bacteria,
                assembly_path=assembly_path,
                phylogroup_call=phylogroup_call,
                mlst_call=mlst_call,
                serotype_call=serotype_call,
                capsule_proxy=capsule_proxy,
            )
        )

    field_status_rows = build_legacy_validation_rows(projected_rows=projected_rows, panel_metadata=panel_metadata)
    family_validation_rows = build_family_validation_rows(field_status_rows)

    write_csv(args.output_dir / INPUT_INVENTORY_FILENAME, tuple(input_inventory_rows[0].keys()), input_inventory_rows)
    write_csv(args.output_dir / RAW_CALLS_FILENAME, tuple(raw_call_rows[0].keys()), raw_call_rows)
    write_csv(args.output_dir / PROJECTED_FEATURES_FILENAME, PROJECTED_FEATURE_COLUMNS, projected_rows)
    write_csv(args.output_dir / FIELD_STATUS_FILENAME, tuple(field_status_rows[0].keys()), field_status_rows)
    write_csv(
        args.output_dir / FAMILY_VALIDATION_FILENAME, tuple(family_validation_rows[0].keys()), family_validation_rows
    )

    manifest = {
        "task_id": "TL16",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "fasta_dir": str(args.fasta_dir),
            "validation_manifest_path": str(args.validation_manifest_path),
            "panel_metadata_path": str(args.panel_metadata_path),
            "capsule_definition_dir": str(args.capsule_definition_dir),
            "capsule_profile_dir": str(args.capsule_profile_dir),
        },
        "tool_versions": capture_tool_versions(),
        "outputs": {
            "input_inventory_csv": str(args.output_dir / INPUT_INVENTORY_FILENAME),
            "raw_calls_csv": str(args.output_dir / RAW_CALLS_FILENAME),
            "projected_features_csv": str(args.output_dir / PROJECTED_FEATURES_FILENAME),
            "legacy_field_status_csv": str(args.output_dir / FIELD_STATUS_FILENAME),
            "feature_family_validation_csv": str(args.output_dir / FAMILY_VALIDATION_FILENAME),
            "raw_call_root": str(raw_call_root),
        },
        "counts": {
            "validated_host_count": len(projected_rows),
            "direct_feature_count": sum(1 for row in field_status_rows if row["projection_status"] == "direct"),
            "proxy_feature_count": sum(
                1 for row in field_status_rows if row["projection_status"] == "deployable_proxy"
            ),
            "unsupported_feature_count": sum(
                1 for row in field_status_rows if row["projection_status"] == "unsupported"
            ),
            "non_derivable_field_count": sum(
                1 for row in field_status_rows if row["projection_status"] == "non_derivable"
            ),
        },
        "input_inventory": input_inventory_rows,
    }
    write_json(args.output_dir / MANIFEST_FILENAME, manifest)


if __name__ == "__main__":
    main()
