#!/usr/bin/env python3
"""TL15: project deployable host-surface features from raw host assemblies."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from typing import Iterable, Mapping, Optional, Sequence
from xml.etree import ElementTree

import pyrodigal

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_c.steps.build_receptor_surface_feature_block import resolve_k_antigen_type

LOGGER = logging.getLogger(__name__)

DEFAULT_VALIDATION_MANIFEST_PATH = Path("data/genomics/bacteria/validation_subset/manifest.json")
DEFAULT_VALIDATION_FASTA_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_PICARD_METADATA_PATH = Path("data/genomics/bacteria/picard_collection.csv")
DEFAULT_O_TYPE_OUTPUT_PATH = Path("data/genomics/bacteria/o_type/output.tsv")
DEFAULT_O_TYPE_ALLELE_PATH = Path("data/genomics/bacteria/isolation_strains/o_type/blast_output_alleles.txt")
DEFAULT_O_ANTIGEN_OVERRIDE_PATH = Path("lyzortx/pipeline/track_l/assets/tl15_o_antigen_reference_overrides.tsv")
DEFAULT_LPS_PRIMARY_PATH = Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt")
DEFAULT_LPS_SUPPLEMENTAL_PATH = Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_host.txt")
DEFAULT_RECEPTOR_CLUSTER_PATH = Path(
    "data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"
)
DEFAULT_KLEBSIELLA_CAPSULE_PATH = Path(
    "data/genomics/bacteria/capsules/klebsiella_capsules/kaptive_results_high_hits_cured.txt"
)
DEFAULT_ABC_CAPSULE_PROFILE_DIR = Path("data/genomics/bacteria/capsules/ABC_capsules_types/profiles")
DEFAULT_ABC_CAPSULE_DEFINITION_DIR = Path("data/genomics/bacteria/capsules/ABC_capsules_types/definitions")
DEFAULT_OMP_REFERENCE_PATH = Path("lyzortx/pipeline/track_l/assets/tl15_omp_reference_proteins.faa")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/raw_host_surface_projector")

O_ANTIGEN_GENE_FAMILIES = ("wzx", "wzy", "wzm", "wzt")
RECEPTOR_COLUMNS = (
    ("BTUB", "host_receptor_btub_present", "host_receptor_btub_variant"),
    ("FADL", "host_receptor_fadL_present", "host_receptor_fadL_variant"),
    ("FHUA", "host_receptor_fhua_present", "host_receptor_fhua_variant"),
    ("LAMB", "host_receptor_lamB_present", "host_receptor_lamB_variant"),
    ("LPTD", "host_receptor_lptD_present", "host_receptor_lptD_variant"),
    ("NFRA", "host_receptor_nfrA_present", "host_receptor_nfrA_variant"),
    ("OMPA", "host_receptor_ompA_present", "host_receptor_ompA_variant"),
    ("OMPC", "host_receptor_ompC_present", "host_receptor_ompC_variant"),
    ("OMPF", "host_receptor_ompF_present", "host_receptor_ompF_variant"),
    ("TOLC", "host_receptor_tolC_present", "host_receptor_tolC_variant"),
    ("TSX", "host_receptor_tsx_present", "host_receptor_tsx_variant"),
    ("YNCD", "host_receptor_yncD_present", "host_receptor_yncD_variant"),
)
PROJECTED_FEATURE_COLUMNS = (
    "bacteria",
    "host_o_antigen_present",
    "host_o_antigen_type",
    "host_k_antigen_present",
    "host_k_antigen_type",
    "host_k_antigen_type_source",
    "host_k_antigen_proxy_present",
    "host_lps_core_present",
    "host_lps_core_type",
    "host_surface_lps_core_type",
    "host_capsule_abc_present",
    "host_receptor_btub_present",
    "host_receptor_fadL_present",
    "host_receptor_fhua_present",
    "host_receptor_lamB_present",
    "host_receptor_lptD_present",
    "host_receptor_nfrA_present",
    "host_receptor_ompA_present",
    "host_receptor_ompC_present",
    "host_receptor_ompF_present",
    "host_receptor_tolC_present",
    "host_receptor_tsx_present",
    "host_receptor_yncD_present",
)
STATUS_COLUMNS = (
    "bacteria",
    "column_name",
    "projected_value",
    "projection_mode",
    "call_state",
    "evidence",
)
COMPARISON_COLUMNS = (
    "bacteria",
    "column_name",
    "projection_mode",
    "projected_value",
    "expected_value",
    "comparison_status",
    "comparison_note",
)
SUPPORT_TABLE_COLUMNS = (
    "feature_family",
    "projection_status",
    "projected_columns",
    "rationale",
)
ABC_TYPE_SOURCE = "abc_capsule_hmm"
LPS_PROXY_SOURCE = "o_type_majority_lps_lookup"
NOT_CALLABLE = "not_callable"
ABSENT = "absent"
PRESENT = "present"
DIRECT = "direct"
PROXY = "proxy"
UNSUPPORTED = "unsupported"
PHMMER_EVALUE_THRESHOLD = 1e-20
NHMMER_EVALUE_THRESHOLD = 1e-20
HMMSCAN_EVALUE_THRESHOLD = 1e-5
LPS_PROXY_MIN_SUPPORT = 0.75
CAPSULE_CORE_PROFILE_NAMES = ("KpsC", "KpsD", "KpsE", "KpsF", "KpsM", "KpsS", "KpsT", "KpsU")


@dataclass(frozen=True)
class FastaRecord:
    name: str
    sequence: str


@dataclass(frozen=True)
class OAlleleReference:
    query_id: str
    o_type: str
    gene_family: str
    allele_key: str
    sequence: str


@dataclass(frozen=True)
class HmmerHit:
    target_name: str
    query_name: str
    evalue: float
    score: float
    description: str


@dataclass(frozen=True)
class CapsuleModel:
    model_id: str
    mandatory_genes: frozenset[str]
    all_genes: frozenset[str]
    min_mandatory_genes_required: int
    min_genes_required: int
    inter_gene_max_space: int


@dataclass(frozen=True)
class CapsuleCall:
    capsule_type: str
    matched_genes: tuple[str, ...]
    core_profile_hits: tuple[str, ...]
    evidence: str
    projection_mode: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly_paths", nargs="*", type=Path, help="Raw host assembly FASTA paths to project.")
    parser.add_argument(
        "--input-manifest-path",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST_PATH,
        help="Optional FASTA manifest JSON with bacteria/file entries.",
    )
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=DEFAULT_VALIDATION_FASTA_DIR,
        help="Directory used to resolve manifest FASTA filenames.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--picard-metadata-path", type=Path, default=DEFAULT_PICARD_METADATA_PATH)
    parser.add_argument("--o-type-output-path", type=Path, default=DEFAULT_O_TYPE_OUTPUT_PATH)
    parser.add_argument("--o-type-allele-path", type=Path, default=DEFAULT_O_TYPE_ALLELE_PATH)
    parser.add_argument("--o-antigen-override-path", type=Path, default=DEFAULT_O_ANTIGEN_OVERRIDE_PATH)
    parser.add_argument("--lps-primary-path", type=Path, default=DEFAULT_LPS_PRIMARY_PATH)
    parser.add_argument("--lps-supplemental-path", type=Path, default=DEFAULT_LPS_SUPPLEMENTAL_PATH)
    parser.add_argument("--receptor-cluster-path", type=Path, default=DEFAULT_RECEPTOR_CLUSTER_PATH)
    parser.add_argument("--klebsiella-capsule-path", type=Path, default=DEFAULT_KLEBSIELLA_CAPSULE_PATH)
    parser.add_argument("--abc-capsule-profile-dir", type=Path, default=DEFAULT_ABC_CAPSULE_PROFILE_DIR)
    parser.add_argument("--abc-capsule-definition-dir", type=Path, default=DEFAULT_ABC_CAPSULE_DEFINITION_DIR)
    parser.add_argument("--omp-reference-path", type=Path, default=DEFAULT_OMP_REFERENCE_PATH)
    parser.add_argument(
        "--skip-validation-comparison",
        action="store_true",
        help="Skip comparison against existing panel annotations.",
    )
    return parser.parse_args(argv)


def read_fasta_records(path: Path) -> list[FastaRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing FASTA file: {path}")
    records: list[FastaRecord] = []
    name: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequence = "".join(chunks).upper()
                    if sequence:
                        records.append(FastaRecord(name=name, sequence=sequence))
                name = line[1:].split()[0]
                chunks = []
                continue
            chunks.append(line)
    if name is not None:
        sequence = "".join(chunks).upper()
        if sequence:
            records.append(FastaRecord(name=name, sequence=sequence))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def read_delimited_rows(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tabular input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]


def _normalize_category(value: str) -> str:
    normalized = value.strip()
    if normalized in {"", "-", "Unknown"}:
        return ""
    return normalized


def _normalize_dna_sequence(sequence: str) -> str:
    normalized = re.sub(r"[^ACGTNacgtn]", "", sequence).upper()
    if not normalized:
        raise ValueError("Encountered an empty reference sequence after DNA normalization.")
    return normalized


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _parse_binary_flag(value: str) -> int:
    normalized = value.strip()
    if normalized in {"", "0", "0.0", "False", "false"}:
        return 0
    return 1


def _parse_integral_flag(value: str) -> int:
    normalized = value.strip()
    if normalized == "":
        return 0
    parsed = float(normalized)
    rounded = int(parsed)
    if parsed != rounded:
        raise ValueError(f"Expected integral capsule flag, found {value!r}")
    return rounded


def _run_command(command: Sequence[str], *, description: str) -> None:
    LOGGER.info("Starting %s", description)
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    if result.returncode != 0:
        LOGGER.error(
            "%s failed (exit %d, %.1fs)\nstdout: %s\nstderr: %s",
            description,
            result.returncode,
            elapsed,
            result.stdout[-4000:],
            result.stderr[-4000:],
        )
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    LOGGER.info("Completed %s in %.1fs", description, elapsed)


def write_fasta(path: Path, sequences: Iterable[tuple[str, str]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in sequences:
            handle.write(f">{header}\n")
            for chunk in wrap(sequence, 80):
                handle.write(f"{chunk}\n")


def predict_proteins(assembly_path: Path, output_path: Path) -> dict[str, dict[str, object]]:
    records = read_fasta_records(assembly_path)
    train_sequence = max((record.sequence for record in records), key=len)
    finder = pyrodigal.GeneFinder(meta=False)
    finder.train(train_sequence)

    metadata: dict[str, dict[str, object]] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        gene_index = 0
        for record in records:
            genes = finder.find_genes(record.sequence)
            for order, gene in enumerate(genes, start=1):
                protein = str(gene.translate())
                if not protein:
                    continue
                gene_index += 1
                gene_id = f"gene_{gene_index}"
                metadata[gene_id] = {
                    "contig": record.name,
                    "start": int(gene.begin),
                    "end": int(gene.end),
                    "strand": int(gene.strand),
                    "order": order,
                    "length_aa": len(protein),
                }
                header = (
                    f"{gene_id} contig={record.name} start={gene.begin} end={gene.end} "
                    f"strand={gene.strand} order={order}"
                )
                handle.write(f">{header}\n")
                for chunk in wrap(protein, 80):
                    handle.write(f"{chunk}\n")
    if not metadata:
        raise ValueError(f"No protein-coding genes predicted from {assembly_path}")
    return metadata


def parse_hmmer_tblout(path: Path) -> list[HmmerHit]:
    if not path.exists():
        raise FileNotFoundError(f"Missing HMMER output: {path}")
    hits: list[HmmerHit] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=18)
            if len(parts) < 6:
                raise ValueError(f"Unexpected HMMER table row in {path}: {line!r}")
            hits.append(
                HmmerHit(
                    target_name=parts[0],
                    query_name=parts[2],
                    evalue=float(parts[4]),
                    score=float(parts[5]),
                    description=parts[18] if len(parts) > 18 else "",
                )
            )
    return hits


def parse_nhmmer_tblout(path: Path) -> list[HmmerHit]:
    if not path.exists():
        raise FileNotFoundError(f"Missing HMMER output: {path}")
    hits: list[HmmerHit] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=15)
            if len(parts) < 15:
                raise ValueError(f"Unexpected nhmmer table row in {path}: {line!r}")
            hits.append(
                HmmerHit(
                    target_name=parts[0],
                    query_name=parts[2],
                    evalue=float(parts[12]),
                    score=float(parts[13]),
                    description=parts[15] if len(parts) > 15 else "",
                )
            )
    return hits


def build_o_antigen_reference_contract(
    *,
    o_type_output_rows: Sequence[Mapping[str, str]],
    o_type_allele_rows: Sequence[Mapping[str, str]],
    override_references: Optional[Mapping[str, tuple[str, str, str]]] = None,
) -> tuple[list[OAlleleReference], dict[str, dict[str, tuple[str, ...]]]]:
    allele_sequence_by_key: dict[str, tuple[str, str, str]] = {}
    for row in o_type_allele_rows:
        if row.get("type", "") != "O":
            continue
        allele_key = row.get("name", "")
        gene_family = row.get("gene", "").lower()
        sequence = _normalize_dna_sequence(row.get("sseq", ""))
        antigen = row.get("antigen", "")
        if not allele_key or gene_family not in O_ANTIGEN_GENE_FAMILIES or not sequence or not antigen:
            continue
        allele_sequence_by_key[allele_key] = (sequence, gene_family, antigen)
    for allele_key, override_value in (override_references or {}).items():
        allele_sequence_by_key.setdefault(allele_key, override_value)

    references: list[OAlleleReference] = []
    o_type_contract: dict[str, dict[str, tuple[str, ...]]] = {}
    seen_query_ids: set[str] = set()

    for row in o_type_output_rows:
        o_type = _normalize_category(row.get("O-type", ""))
        if not o_type or o_type in {"Oneg", "ONT"}:
            continue
        allele_keys = [token.strip() for token in row.get("AlleleKeys", "").split(";") if token.strip()]
        if not allele_keys:
            continue
        per_family: dict[str, set[str]] = defaultdict(set)
        for allele_key in allele_keys:
            if allele_key not in allele_sequence_by_key:
                continue
            sequence, gene_family, _ = allele_sequence_by_key[allele_key]
            query_id = f"{o_type}__{gene_family}__{allele_key}"
            if query_id not in seen_query_ids:
                references.append(
                    OAlleleReference(
                        query_id=query_id,
                        o_type=o_type,
                        gene_family=gene_family,
                        allele_key=allele_key,
                        sequence=sequence,
                    )
                )
                seen_query_ids.add(query_id)
            per_family[gene_family].add(query_id)
        if per_family:
            existing = o_type_contract.setdefault(o_type, {})
            for gene_family, query_ids in per_family.items():
                combined = set(existing.get(gene_family, ()))
                combined.update(query_ids)
                existing[gene_family] = tuple(sorted(combined))

    if not references:
        raise ValueError("No O-antigen reference alleles could be built from the checked-in ECTyper outputs.")
    return references, o_type_contract


def load_o_antigen_override_references(path: Path) -> dict[str, tuple[str, str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing O-antigen override manifest: {path}")
    override_rows = read_delimited_rows(path, delimiter="\t")
    overrides: dict[str, tuple[str, str, str]] = {}
    fasta_cache: dict[Path, dict[str, str]] = {}
    for row in override_rows:
        allele_key = row.get("allele_key", "")
        o_type = row.get("o_type", "")
        gene_family = row.get("gene_family", "").lower()
        fasta_path = Path(row.get("source_fasta_path", ""))
        record_name = row.get("record_name", "")
        if not allele_key:
            raise ValueError(f"Override row is missing allele_key in {path}")
        if gene_family not in O_ANTIGEN_GENE_FAMILIES:
            raise ValueError(f"Unsupported O-antigen gene family {gene_family!r} in {path}")
        if not fasta_path:
            raise ValueError(f"Override row for {allele_key} is missing source_fasta_path in {path}")
        if fasta_path not in fasta_cache:
            fasta_cache[fasta_path] = {record.name: record.sequence for record in read_fasta_records(fasta_path)}
        sequence_by_record = fasta_cache[fasta_path]
        if record_name not in sequence_by_record:
            raise KeyError(f"Record {record_name!r} not found in override FASTA {fasta_path}")
        start = int(row.get("start", "0"))
        end = int(row.get("end", "0"))
        if start <= 0 or end <= 0:
            raise ValueError(f"Override row for {allele_key} must use positive 1-based coordinates")
        left = min(start, end) - 1
        right = max(start, end)
        sequence = sequence_by_record[record_name][left:right]
        if not sequence:
            raise ValueError(f"Override row for {allele_key} extracted an empty sequence from {fasta_path}")
        strand = row.get("strand", "+").strip()
        if strand == "-":
            sequence = _reverse_complement(sequence)
        elif strand != "+":
            raise ValueError(f"Unsupported strand {strand!r} for {allele_key} in {path}")
        overrides[allele_key] = (_normalize_dna_sequence(sequence), gene_family, o_type)
    return overrides


def write_o_antigen_queries(path: Path, references: Sequence[OAlleleReference]) -> None:
    write_fasta(path, ((reference.query_id, reference.sequence) for reference in references))


def call_o_antigen_type(
    *,
    bacteria: str,
    assembly_path: Path,
    reference_fasta_path: Path,
    references: Sequence[OAlleleReference],
    o_type_contract: Mapping[str, Mapping[str, Sequence[str]]],
    output_dir: Path,
) -> tuple[str, str]:
    tblout_path = output_dir / f"{bacteria}_o_antigen_nhmmer.tbl"
    _run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "nhmmer",
            "--noali",
            "--tblout",
            str(tblout_path),
            str(reference_fasta_path),
            str(assembly_path),
        ],
        description=f"nhmmer O-antigen scan for {bacteria}",
    )
    reference_lookup = {reference.query_id: reference for reference in references}
    best_hits: dict[str, HmmerHit] = {}
    for hit in parse_nhmmer_tblout(tblout_path):
        if hit.query_name not in reference_lookup or hit.evalue > NHMMER_EVALUE_THRESHOLD:
            continue
        existing = best_hits.get(hit.query_name)
        if existing is None or (hit.score, -hit.evalue) > (existing.score, -existing.evalue):
            best_hits[hit.query_name] = hit

    candidate_scores: list[tuple[int, float, str, str]] = []
    for o_type, gene_families in o_type_contract.items():
        matched_families = 0
        total_score = 0.0
        evidence_parts: list[str] = []
        for gene_family, query_ids in gene_families.items():
            family_hits = [best_hits[query_id] for query_id in query_ids if query_id in best_hits]
            if not family_hits:
                continue
            best_family_hit = max(family_hits, key=lambda hit: hit.score)
            matched_families += 1
            total_score += best_family_hit.score
            evidence_parts.append(f"{gene_family}:{best_family_hit.query_name}")
        if matched_families:
            candidate_scores.append((matched_families, total_score, o_type, "|".join(sorted(evidence_parts))))

    if not candidate_scores:
        return "", "no_O_antigen_allele_hits"
    candidate_scores.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best = candidate_scores[0]
    if best[0] < 2:
        return "", f"insufficient_O_antigen_support:{best[3]}"
    if len(candidate_scores) > 1 and candidate_scores[1][0] == best[0] and candidate_scores[1][1] == best[1]:
        return "", f"ambiguous_O_antigen_call:{best[2]}~{candidate_scores[1][2]}"
    return best[2], best[3]


def build_lps_proxy_lookup(picard_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
    counts_by_o_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in picard_rows:
        o_type = _normalize_category(row.get("O-type", ""))
        lps_type = _normalize_category(row.get("LPS_type", ""))
        if not o_type or not lps_type:
            continue
        counts_by_o_type[o_type][lps_type] += 1

    lookup: dict[str, dict[str, object]] = {}
    for o_type, counter in counts_by_o_type.items():
        top_type, top_count = counter.most_common(1)[0]
        support = top_count / sum(counter.values())
        lookup[o_type] = {
            "proxy_type": top_type if support >= LPS_PROXY_MIN_SUPPORT else "",
            "support_fraction": round(support, 6),
            "type_counts": dict(sorted(counter.items())),
        }
    return lookup


def write_capsule_hmm_bundle(profile_dir: Path, output_dir: Path) -> Path:
    bundle_path = output_dir / "abc_capsule_profiles.hmm"
    if bundle_path.exists():
        return bundle_path
    profile_paths = sorted(profile_dir.glob("*.hmm"))
    if not profile_paths:
        raise FileNotFoundError(f"No ABC capsule HMM profiles found in {profile_dir}")
    ensure_directory(output_dir)
    with bundle_path.open("w", encoding="utf-8") as out_handle:
        for profile_path in profile_paths:
            out_handle.write(profile_path.read_text(encoding="utf-8"))
    _run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "hmmpress",
            str(bundle_path),
        ],
        description="hmmpress ABC capsule profile bundle",
    )
    return bundle_path


def load_capsule_models(definition_dir: Path) -> list[CapsuleModel]:
    models: list[CapsuleModel] = []
    for definition_path in sorted(definition_dir.glob("*.xml")):
        root = ElementTree.fromstring(definition_path.read_text(encoding="utf-8"))
        mandatory_genes = []
        all_genes = []
        for gene_element in root.findall("gene"):
            name = gene_element.attrib["name"]
            all_genes.append(name)
            if gene_element.attrib.get("presence") == "mandatory":
                mandatory_genes.append(name)
        models.append(
            CapsuleModel(
                model_id=definition_path.stem,
                mandatory_genes=frozenset(mandatory_genes),
                all_genes=frozenset(all_genes),
                min_mandatory_genes_required=int(root.attrib["min_mandatory_genes_required"]),
                min_genes_required=int(root.attrib["min_genes_required"]),
                inter_gene_max_space=int(root.attrib["inter_gene_max_space"]),
            )
        )
    if not models:
        raise FileNotFoundError(f"No ABC capsule model definitions found in {definition_dir}")
    return models


def run_hmmscan(*, bacteria: str, proteins_path: Path, hmm_bundle_path: Path, output_dir: Path) -> list[HmmerHit]:
    tblout_path = output_dir / f"{bacteria}_abc_capsule_hmmscan.tbl"
    _run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "hmmscan",
            "--noali",
            "--tblout",
            str(tblout_path),
            str(hmm_bundle_path),
            str(proteins_path),
        ],
        description=f"hmmscan ABC capsule profiles for {bacteria}",
    )
    return [hit for hit in parse_hmmer_tblout(tblout_path) if hit.evalue <= HMMSCAN_EVALUE_THRESHOLD]


def choose_capsule_call(
    *,
    models: Sequence[CapsuleModel],
    hits: Sequence[HmmerHit],
    protein_metadata: Mapping[str, Mapping[str, object]],
) -> CapsuleCall:
    best_by_profile_gene: dict[tuple[str, str], HmmerHit] = {}
    for hit in hits:
        key = (hit.target_name, hit.query_name)
        existing = best_by_profile_gene.get(key)
        if existing is None or hit.score > existing.score:
            best_by_profile_gene[key] = hit

    hits_by_profile: dict[str, list[HmmerHit]] = defaultdict(list)
    for hit in best_by_profile_gene.values():
        hits_by_profile[hit.target_name].append(hit)

    core_profiles = sorted(profile for profile in CAPSULE_CORE_PROFILE_NAMES if profile in hits_by_profile)
    best_candidate: tuple[int, int, float, str, CapsuleCall] | None = None
    for model in models:
        hits_by_contig: dict[str, dict[str, HmmerHit]] = defaultdict(dict)
        for gene_name in model.all_genes:
            for hit in hits_by_profile.get(gene_name, []):
                metadata = protein_metadata.get(hit.query_name)
                if metadata is None:
                    continue
                contig = str(metadata["contig"])
                existing = hits_by_contig[contig].get(gene_name)
                if existing is None or hit.score > existing.score:
                    hits_by_contig[contig][gene_name] = hit
        for contig, contig_hits in hits_by_contig.items():
            matched_genes = set(contig_hits)
            mandatory_count = len(matched_genes & model.mandatory_genes)
            total_count = len(matched_genes)
            if mandatory_count < model.min_mandatory_genes_required or total_count < model.min_genes_required:
                continue
            orders = sorted(
                int(protein_metadata[contig_hits[gene_name].query_name]["order"]) for gene_name in matched_genes
            )
            max_span = orders[-1] - orders[0] if orders else 0
            allowed_span = max(0, total_count - 1) * (model.inter_gene_max_space + 1)
            if max_span > allowed_span:
                continue
            total_score = sum(contig_hits[gene_name].score for gene_name in matched_genes)
            evidence = (
                f"model={model.model_id};contig={contig};mandatory={mandatory_count};"
                f"total={total_count};genes={'|'.join(sorted(matched_genes))}"
            )
            capsule_call = CapsuleCall(
                capsule_type=model.model_id,
                matched_genes=tuple(sorted(matched_genes)),
                core_profile_hits=tuple(core_profiles),
                evidence=evidence,
                projection_mode=DIRECT,
            )
            candidate = (mandatory_count, total_count, total_score, model.model_id, capsule_call)
            if best_candidate is None or candidate[:4] > best_candidate[:4]:
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate[4]
    if core_profiles:
        return CapsuleCall(
            capsule_type="",
            matched_genes=(),
            core_profile_hits=tuple(core_profiles),
            evidence=f"core_profiles={'|'.join(core_profiles)}",
            projection_mode=PROXY,
        )
    return CapsuleCall(
        capsule_type="",
        matched_genes=(),
        core_profile_hits=(),
        evidence="no_capsule_profile_hits",
        projection_mode=UNSUPPORTED,
    )


def call_receptor_presence(
    *,
    bacteria: str,
    proteins_path: Path,
    omp_reference_path: Path,
    output_dir: Path,
) -> dict[str, tuple[str, str]]:
    tblout_path = output_dir / f"{bacteria}_omp_phmmer.tbl"
    _run_command(
        [
            "micromamba",
            "run",
            "-n",
            "phage_annotation_tools",
            "phmmer",
            "--noali",
            "--tblout",
            str(tblout_path),
            str(omp_reference_path),
            str(proteins_path),
        ],
        description=f"phmmer receptor scan for {bacteria}",
    )
    best_by_query: dict[str, HmmerHit] = {}
    for hit in parse_hmmer_tblout(tblout_path):
        existing = best_by_query.get(hit.query_name)
        if existing is None or (hit.score, -hit.evalue) > (existing.score, -existing.evalue):
            best_by_query[hit.query_name] = hit

    calls: dict[str, tuple[str, str]] = {}
    for hit in best_by_query.values():
        match = re.search(r"\|([A-Z0-9]+)_ECOLI\b", hit.query_name)
        if not match:
            continue
        receptor_name = match.group(1)
        if receptor_name == "PQQU":
            receptor_name = "YNCD"
        if receptor_name not in {column[0] for column in RECEPTOR_COLUMNS}:
            continue
        if hit.evalue <= PHMMER_EVALUE_THRESHOLD:
            calls[receptor_name] = (
                PRESENT,
                f"best_hit={hit.target_name};score={round(hit.score, 2)};evalue={hit.evalue:g}",
            )
        else:
            calls[receptor_name] = (
                NOT_CALLABLE,
                f"weak_hit={hit.target_name};score={round(hit.score, 2)};evalue={hit.evalue:g}",
            )
    return calls


def build_projected_feature_row(
    *,
    bacteria: str,
    o_antigen_type: str,
    capsule_call: CapsuleCall,
    lps_lookup: Mapping[str, Mapping[str, object]],
    receptor_presence_calls: Mapping[str, tuple[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    row: dict[str, object] = {"bacteria": bacteria}
    status_rows: list[dict[str, object]] = []

    def set_value(column_name: str, value: object, projection_mode: str, call_state: str, evidence: str) -> None:
        if column_name in PROJECTED_FEATURE_COLUMNS:
            row[column_name] = value
        status_rows.append(
            {
                "bacteria": bacteria,
                "column_name": column_name,
                "projected_value": value,
                "projection_mode": projection_mode,
                "call_state": call_state,
                "evidence": evidence,
            }
        )

    if o_antigen_type:
        set_value("host_o_antigen_present", 1, DIRECT, PRESENT, f"O-antigen type={o_antigen_type}")
        set_value("host_o_antigen_type", o_antigen_type, DIRECT, PRESENT, f"O-antigen type={o_antigen_type}")
    else:
        set_value("host_o_antigen_present", "", DIRECT, NOT_CALLABLE, "O-antigen typing unresolved from allele scan")
        set_value("host_o_antigen_type", "", DIRECT, NOT_CALLABLE, "O-antigen typing unresolved from allele scan")

    if capsule_call.capsule_type:
        set_value("host_k_antigen_present", 1, capsule_call.projection_mode, PRESENT, capsule_call.evidence)
        set_value(
            "host_k_antigen_type",
            capsule_call.capsule_type,
            capsule_call.projection_mode,
            PRESENT,
            capsule_call.evidence,
        )
        set_value(
            "host_k_antigen_type_source", ABC_TYPE_SOURCE, capsule_call.projection_mode, PRESENT, capsule_call.evidence
        )
        set_value("host_k_antigen_proxy_present", 1, capsule_call.projection_mode, PRESENT, capsule_call.evidence)
        set_value("host_capsule_abc_present", 1, capsule_call.projection_mode, PRESENT, capsule_call.evidence)
    elif capsule_call.core_profile_hits:
        evidence = capsule_call.evidence
        set_value("host_k_antigen_present", "", PROXY, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_type", "", PROXY, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_type_source", "", PROXY, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_proxy_present", 1, PROXY, PRESENT, evidence)
        set_value("host_capsule_abc_present", 1, PROXY, PRESENT, evidence)
    else:
        evidence = capsule_call.evidence
        set_value("host_k_antigen_present", "", UNSUPPORTED, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_type", "", UNSUPPORTED, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_type_source", "", UNSUPPORTED, NOT_CALLABLE, evidence)
        set_value("host_k_antigen_proxy_present", "", UNSUPPORTED, NOT_CALLABLE, evidence)
        set_value("host_capsule_abc_present", "", UNSUPPORTED, NOT_CALLABLE, evidence)

    lps_proxy = lps_lookup.get(o_antigen_type, {})
    lps_type = str(lps_proxy.get("proxy_type", ""))
    if lps_type:
        evidence = (
            f"{LPS_PROXY_SOURCE};support={lps_proxy['support_fraction']};"
            f"counts={json.dumps(lps_proxy['type_counts'], sort_keys=True)}"
        )
        set_value("host_lps_core_present", 1, PROXY, PRESENT, evidence)
        set_value("host_lps_core_type", lps_type, PROXY, PRESENT, evidence)
        set_value("host_surface_lps_core_type", lps_type, PROXY, PRESENT, evidence)
    else:
        evidence = (
            f"{LPS_PROXY_SOURCE};no_stable_lookup_for_o_type={o_antigen_type or '<missing>'};"
            f"counts={json.dumps(lps_proxy.get('type_counts', {}), sort_keys=True)}"
        )
        set_value("host_lps_core_present", "", PROXY, NOT_CALLABLE, evidence)
        set_value("host_lps_core_type", "", PROXY, NOT_CALLABLE, evidence)
        set_value("host_surface_lps_core_type", "", PROXY, NOT_CALLABLE, evidence)

    for receptor_name, present_column, variant_column in RECEPTOR_COLUMNS:
        call_state, evidence = receptor_presence_calls.get(receptor_name, (NOT_CALLABLE, "no_receptor_hit"))
        if call_state == PRESENT:
            set_value(present_column, 1, DIRECT, PRESENT, evidence)
        elif call_state == ABSENT:
            set_value(present_column, 0, DIRECT, ABSENT, evidence)
        else:
            set_value(present_column, "", DIRECT, NOT_CALLABLE, evidence)
        status_rows.append(
            {
                "bacteria": bacteria,
                "column_name": variant_column,
                "projected_value": "",
                "projection_mode": UNSUPPORTED,
                "call_state": NOT_CALLABLE,
                "evidence": "panel-specific 99% receptor cluster IDs are not reconstructable from the checked-in repo assets",
            }
        )

    for unsupported_column in (
        "host_capsule_groupiv_e_present",
        "host_capsule_groupiv_e_stricte_present",
        "host_capsule_groupiv_s",
        "host_capsule_wzy_stricte_present",
        "host_surface_klebsiella_capsule_type",
        "host_surface_klebsiella_capsule_type_missing",
    ):
        status_rows.append(
            {
                "bacteria": bacteria,
                "column_name": unsupported_column,
                "projected_value": "",
                "projection_mode": UNSUPPORTED,
                "call_state": NOT_CALLABLE,
                "evidence": "no checked-in raw-genome contract in the repo can reproduce this training-time capsule-family field",
            }
        )

    for column in PROJECTED_FEATURE_COLUMNS:
        row.setdefault(column, "")
    return row, status_rows


def build_expected_surface_rows(
    *,
    picard_rows: Sequence[Mapping[str, str]],
    receptor_rows: Sequence[Mapping[str, str]],
    lps_primary_rows: Sequence[Mapping[str, str]],
    lps_supplemental_rows: Sequence[Mapping[str, str]],
    klebsiella_capsule_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, object]]:
    picard_by_bacteria = {row["bacteria"]: row for row in picard_rows if row.get("bacteria", "")}
    receptor_by_bacteria = {row["bacteria"]: row for row in receptor_rows if row.get("bacteria", "")}
    lps_by_bacteria: dict[str, str] = {}
    for row in lps_primary_rows:
        bacteria = row.get("bacteria", "")
        if bacteria:
            lps_by_bacteria[bacteria] = _normalize_category(row.get("LPS_type", ""))
    for row in lps_supplemental_rows:
        bacteria = row.get("Strain", "") or row.get("strain", "")
        if bacteria and bacteria not in lps_by_bacteria:
            lps_by_bacteria[bacteria] = _normalize_category(row.get("LPS_type", ""))
    kleb_by_bacteria = {
        row["bacteria"]: _normalize_category(row.get("Klebs_capsule_type", ""))
        for row in klebsiella_capsule_rows
        if row.get("bacteria", "")
    }

    expected: dict[str, dict[str, object]] = {}
    for bacteria, host_row in picard_by_bacteria.items():
        expected_row: dict[str, object] = {"bacteria": bacteria}
        o_type = _normalize_category(host_row.get("O-type", ""))
        expected_row["host_o_antigen_present"] = 1 if o_type else 0
        expected_row["host_o_antigen_type"] = o_type
        k_type, k_source = resolve_k_antigen_type(host_row)
        expected_row["host_k_antigen_present"] = 1 if k_type else 0
        expected_row["host_k_antigen_type"] = k_type
        expected_row["host_k_antigen_type_source"] = k_source
        capsule_proxy_present = 1 if k_type else 0
        for source_column in (
            "Capsule_ABC",
            "Capsule_GroupIV_e",
            "Capsule_GroupIV_e_stricte",
            "Capsule_GroupIV_s",
            "Capsule_Wzy_stricte",
        ):
            if _parse_binary_flag(host_row.get(source_column, "")):
                capsule_proxy_present = 1
        expected_row["host_k_antigen_proxy_present"] = capsule_proxy_present
        lps_type = _normalize_category(lps_by_bacteria.get(bacteria, host_row.get("LPS_type", "")))
        expected_row["host_lps_core_present"] = 1 if lps_type else 0
        expected_row["host_lps_core_type"] = lps_type
        expected_row["host_surface_lps_core_type"] = lps_type
        expected_row["host_capsule_abc_present"] = _parse_binary_flag(host_row.get("Capsule_ABC", ""))
        receptor_row = receptor_by_bacteria.get(bacteria, {})
        for receptor_name, present_column, variant_column in RECEPTOR_COLUMNS:
            variant = _normalize_category(receptor_row.get(receptor_name, ""))
            expected_row[present_column] = 1 if variant else 0
            expected_row[variant_column] = variant
        expected_row["host_capsule_groupiv_e_present"] = _parse_binary_flag(host_row.get("Capsule_GroupIV_e", ""))
        expected_row["host_capsule_groupiv_e_stricte_present"] = _parse_binary_flag(
            host_row.get("Capsule_GroupIV_e_stricte", "")
        )
        expected_row["host_capsule_groupiv_s"] = _parse_integral_flag(host_row.get("Capsule_GroupIV_s", ""))
        expected_row["host_capsule_wzy_stricte_present"] = _parse_binary_flag(host_row.get("Capsule_Wzy_stricte", ""))
        expected_row["host_surface_klebsiella_capsule_type"] = kleb_by_bacteria.get(bacteria, "")
        expected_row["host_surface_klebsiella_capsule_type_missing"] = int(
            expected_row["host_surface_klebsiella_capsule_type"] == ""
        )
        expected[bacteria] = expected_row
    return expected


def compare_projected_to_expected(
    *,
    projected_rows: Sequence[Mapping[str, object]],
    expected_rows: Mapping[str, Mapping[str, object]],
    status_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    status_index = {
        (str(row["bacteria"]), str(row["column_name"])): dict(row)
        for row in status_rows
        if str(row["column_name"]) in PROJECTED_FEATURE_COLUMNS
    }
    comparison_rows: list[dict[str, object]] = []
    summary_counter: Counter[tuple[str, str]] = Counter()
    for row in projected_rows:
        bacteria = str(row["bacteria"])
        expected_row = expected_rows.get(bacteria)
        if expected_row is None:
            continue
        for column_name in PROJECTED_FEATURE_COLUMNS:
            if column_name == "bacteria":
                continue
            status = status_index[(bacteria, column_name)]
            projected_value = row.get(column_name, "")
            expected_value = expected_row.get(column_name, "")
            if status["call_state"] == NOT_CALLABLE:
                comparison_status = "not_compared_not_callable"
            elif str(projected_value) == str(expected_value):
                comparison_status = "match"
            else:
                comparison_status = "mismatch"
            comparison_rows.append(
                {
                    "bacteria": bacteria,
                    "column_name": column_name,
                    "projection_mode": status["projection_mode"],
                    "projected_value": projected_value,
                    "expected_value": expected_value,
                    "comparison_status": comparison_status,
                    "comparison_note": status["evidence"],
                }
            )
            summary_counter[(column_name, comparison_status)] += 1

    summary_rows: list[dict[str, object]] = []
    for column_name in sorted({column for column, _ in summary_counter}):
        match_count = summary_counter[(column_name, "match")]
        mismatch_count = summary_counter[(column_name, "mismatch")]
        skipped_count = summary_counter[(column_name, "not_compared_not_callable")]
        compared_count = match_count + mismatch_count
        summary_rows.append(
            {
                "column_name": column_name,
                "match_count": match_count,
                "mismatch_count": mismatch_count,
                "not_callable_count": skipped_count,
                "compared_count": compared_count,
                "agreement_rate": round(match_count / compared_count, 6) if compared_count else "",
            }
        )
    return comparison_rows, summary_rows


def build_support_table_rows() -> list[dict[str, object]]:
    return [
        {
            "feature_family": "o_antigen_type",
            "projection_status": DIRECT,
            "projected_columns": "host_o_antigen_present|host_o_antigen_type",
            "rationale": (
                "The repo carries ECTyper-derived O-antigen alleles plus a versioned override manifest for validation-covered "
                "alleles that are referenced in output.tsv but missing from the BLAST export."
            ),
        },
        {
            "feature_family": "lps_core_type",
            "projection_status": PROXY,
            "projected_columns": "host_lps_core_present|host_lps_core_type|host_surface_lps_core_type",
            "rationale": "The repo lacks checked-in waaL reference sequences, so LPS core is proxied through a versioned O-type-to-LPS lookup.",
        },
        {
            "feature_family": "abc_capsule_type_and_proxy",
            "projection_status": DIRECT,
            "projected_columns": (
                "host_k_antigen_present|host_k_antigen_type|host_k_antigen_type_source|"
                "host_k_antigen_proxy_present|host_capsule_abc_present"
            ),
            "rationale": "The repo carries checked-in ABC capsule HMM profiles and model definitions that can be run on raw predicted proteins.",
        },
        {
            "feature_family": "group_iv_capsule_flags",
            "projection_status": UNSUPPORTED,
            "projected_columns": (
                "host_capsule_groupiv_e_present|host_capsule_groupiv_e_stricte_present|"
                "host_capsule_groupiv_s|host_capsule_wzy_stricte_present"
            ),
            "rationale": "No checked-in raw-genome contract in the repo maps those mixed curated capsule flags onto reproducible runtime calls.",
        },
        {
            "feature_family": "omp_receptor_presence",
            "projection_status": DIRECT,
            "projected_columns": "|".join(column for _, column, _ in RECEPTOR_COLUMNS),
            "rationale": "A small checked-in receptor reference protein set is enough to detect the receptor-presence layer from predicted proteins.",
        },
        {
            "feature_family": "omp_receptor_variant_clusters",
            "projection_status": UNSUPPORTED,
            "projected_columns": "|".join(column for _, _, column in RECEPTOR_COLUMNS),
            "rationale": "The panel's 99% receptor cluster IDs are present only as labels, not as representative sequences needed to place new hosts into those exact clusters.",
        },
        {
            "feature_family": "klebsiella_capsule_type",
            "projection_status": UNSUPPORTED,
            "projected_columns": "host_surface_klebsiella_capsule_type|host_surface_klebsiella_capsule_type_missing",
            "rationale": "The checked-in Kaptive output is a panel annotation table, not a reusable raw-genome runtime projector.",
        },
    ]


def resolve_input_hosts(args: argparse.Namespace) -> list[tuple[str, Path]]:
    hosts: list[tuple[str, Path]] = []
    seen: set[str] = set()
    if args.input_manifest_path and args.input_manifest_path.exists():
        manifest = json.loads(args.input_manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("files", []):
            bacteria = str(row["bacteria"])
            fasta_path = args.fasta_dir / str(row["filename"])
            if bacteria in seen:
                continue
            hosts.append((bacteria, fasta_path))
            seen.add(bacteria)
    for assembly_path in args.assembly_paths:
        bacteria = assembly_path.stem
        if bacteria in seen:
            continue
        hosts.append((bacteria, assembly_path))
        seen.add(bacteria)
    if not hosts:
        raise ValueError("No input assemblies resolved. Pass assembly paths or a manifest with files[].")
    return hosts


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging()
    ensure_directory(args.output_dir)

    input_hosts = resolve_input_hosts(args)
    LOGGER.info("Projecting raw host-surface features for %d assemblies", len(input_hosts))

    picard_rows = read_delimited_rows(args.picard_metadata_path, delimiter=";")
    o_type_output_rows = read_delimited_rows(args.o_type_output_path, delimiter="\t")
    o_type_allele_rows = read_delimited_rows(args.o_type_allele_path, delimiter="\t")
    o_antigen_override_references = load_o_antigen_override_references(args.o_antigen_override_path)
    lps_primary_rows = read_delimited_rows(args.lps_primary_path, delimiter="\t")
    lps_supplemental_rows = read_delimited_rows(args.lps_supplemental_path, delimiter="\t")
    receptor_rows = read_delimited_rows(args.receptor_cluster_path, delimiter="\t")
    klebsiella_capsule_rows = read_delimited_rows(args.klebsiella_capsule_path, delimiter="\t")
    abc_models = load_capsule_models(args.abc_capsule_definition_dir)

    o_antigen_references, o_type_contract = build_o_antigen_reference_contract(
        o_type_output_rows=o_type_output_rows,
        o_type_allele_rows=o_type_allele_rows,
        override_references=o_antigen_override_references,
    )
    lps_lookup = build_lps_proxy_lookup(picard_rows)
    expected_rows = build_expected_surface_rows(
        picard_rows=picard_rows,
        receptor_rows=receptor_rows,
        lps_primary_rows=lps_primary_rows,
        lps_supplemental_rows=lps_supplemental_rows,
        klebsiella_capsule_rows=klebsiella_capsule_rows,
    )

    asset_dir = args.output_dir / "runtime_assets"
    ensure_directory(asset_dir)
    o_antigen_query_path = asset_dir / "o_antigen_reference_queries.fna"
    write_o_antigen_queries(o_antigen_query_path, o_antigen_references)
    capsule_hmm_bundle = write_capsule_hmm_bundle(args.abc_capsule_profile_dir, asset_dir)

    projected_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    per_host_input_rows: list[dict[str, object]] = []

    for bacteria, assembly_path in input_hosts:
        host_output_dir = args.output_dir / bacteria
        ensure_directory(host_output_dir)
        proteins_path = host_output_dir / "predicted_proteins.faa"
        protein_metadata = predict_proteins(assembly_path, proteins_path)
        per_host_input_rows.append(
            {
                "bacteria": bacteria,
                "assembly_path": str(assembly_path),
                "assembly_bytes": assembly_path.stat().st_size,
                "predicted_protein_count": len(protein_metadata),
            }
        )

        o_antigen_type, o_antigen_evidence = call_o_antigen_type(
            bacteria=bacteria,
            assembly_path=assembly_path,
            reference_fasta_path=o_antigen_query_path,
            references=o_antigen_references,
            o_type_contract=o_type_contract,
            output_dir=host_output_dir,
        )
        hmmscan_hits = run_hmmscan(
            bacteria=bacteria,
            proteins_path=proteins_path,
            hmm_bundle_path=capsule_hmm_bundle,
            output_dir=host_output_dir,
        )
        capsule_call = choose_capsule_call(models=abc_models, hits=hmmscan_hits, protein_metadata=protein_metadata)
        receptor_calls = call_receptor_presence(
            bacteria=bacteria,
            proteins_path=proteins_path,
            omp_reference_path=args.omp_reference_path,
            output_dir=host_output_dir,
        )

        feature_row, feature_status_rows = build_projected_feature_row(
            bacteria=bacteria,
            o_antigen_type=o_antigen_type,
            capsule_call=capsule_call,
            lps_lookup=lps_lookup,
            receptor_presence_calls=receptor_calls,
        )
        for status_row in feature_status_rows:
            if status_row["column_name"] in {"host_o_antigen_present", "host_o_antigen_type"}:
                status_row["evidence"] = o_antigen_evidence
        projected_rows.append(feature_row)
        status_rows.extend(feature_status_rows)

    projected_rows.sort(key=lambda row: str(row["bacteria"]))
    status_rows.sort(key=lambda row: (str(row["bacteria"]), str(row["column_name"])))
    per_host_input_rows.sort(key=lambda row: str(row["bacteria"]))

    projected_feature_path = args.output_dir / "tl15_projected_host_surface_features.csv"
    status_path = args.output_dir / "tl15_projected_host_surface_status.csv"
    input_audit_path = args.output_dir / "tl15_projected_host_inputs.csv"
    support_table_path = args.output_dir / "tl15_feature_family_support_table.csv"
    write_csv(projected_feature_path, PROJECTED_FEATURE_COLUMNS, projected_rows)
    write_csv(status_path, STATUS_COLUMNS, status_rows)
    write_csv(
        input_audit_path,
        ["bacteria", "assembly_path", "assembly_bytes", "predicted_protein_count"],
        per_host_input_rows,
    )
    write_csv(support_table_path, SUPPORT_TABLE_COLUMNS, build_support_table_rows())

    manifest: dict[str, object] = {
        "task_id": "TL15",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_count": len(input_hosts),
        "inputs": {
            "validation_manifest_path": str(args.input_manifest_path),
            "fasta_dir": str(args.fasta_dir),
            "picard_metadata_path": str(args.picard_metadata_path),
            "o_type_output_path": str(args.o_type_output_path),
            "o_type_allele_path": str(args.o_type_allele_path),
            "o_antigen_override_path": str(args.o_antigen_override_path),
            "lps_primary_path": str(args.lps_primary_path),
            "lps_supplemental_path": str(args.lps_supplemental_path),
            "receptor_cluster_path": str(args.receptor_cluster_path),
            "klebsiella_capsule_path": str(args.klebsiella_capsule_path),
            "abc_capsule_profile_dir": str(args.abc_capsule_profile_dir),
            "abc_capsule_definition_dir": str(args.abc_capsule_definition_dir),
            "omp_reference_path": str(args.omp_reference_path),
        },
        "runtime_contract": {
            "direct_families": ["o_antigen_type", "abc_capsule_type_and_proxy", "omp_receptor_presence"],
            "proxy_families": ["lps_core_type"],
            "unsupported_families": [
                "group_iv_capsule_flags",
                "omp_receptor_variant_clusters",
                "klebsiella_capsule_type",
            ],
            "reference_assets": {
                "o_antigen_queries": str(o_antigen_query_path),
                "o_antigen_override_manifest": str(args.o_antigen_override_path),
                "abc_capsule_hmms": str(capsule_hmm_bundle),
                "omp_reference_fasta": str(args.omp_reference_path),
            },
        },
        "outputs": {
            "projected_feature_csv": str(projected_feature_path),
            "projected_status_csv": str(status_path),
            "input_audit_csv": str(input_audit_path),
            "feature_family_support_csv": str(support_table_path),
        },
        "validation_instructions": {
            "command": (
                "micromamba run -n phage_env python -m lyzortx.pipeline.track_l.steps.build_raw_host_surface_projector "
                f"--input-manifest-path {args.input_manifest_path} --fasta-dir {args.fasta_dir} --output-dir {args.output_dir}"
            )
        },
    }

    if not args.skip_validation_comparison:
        comparison_rows, summary_rows = compare_projected_to_expected(
            projected_rows=projected_rows,
            expected_rows=expected_rows,
            status_rows=status_rows,
        )
        comparison_path = args.output_dir / "tl15_validation_feature_comparison.csv"
        summary_path = args.output_dir / "tl15_validation_feature_summary.csv"
        write_csv(comparison_path, COMPARISON_COLUMNS, comparison_rows)
        write_csv(
            summary_path,
            ["column_name", "match_count", "mismatch_count", "not_callable_count", "compared_count", "agreement_rate"],
            summary_rows,
        )
        manifest["outputs"]["validation_comparison_csv"] = str(comparison_path)
        manifest["outputs"]["validation_summary_csv"] = str(summary_path)

    manifest_path = args.output_dir / "tl15_raw_host_surface_manifest.json"
    write_json(manifest_path, manifest)
    LOGGER.info("Projected feature CSV: %s", projected_feature_path)
    LOGGER.info("Status CSV: %s", status_path)
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
