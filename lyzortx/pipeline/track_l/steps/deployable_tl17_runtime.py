"""Helpers for TL17 deployable phage RBP family projection."""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import (
    call_proteins_with_pyrodigal,
    read_fasta_records,
    read_panel_phages,
)
from lyzortx.pipeline.track_l.steps.parse_annotations import classify_rbp_genes, parse_merged_tsv

LOGGER = logging.getLogger(__name__)

DEFAULT_MMSEQS_COMMAND: tuple[str, ...] = ("micromamba", "run", "-n", "phage_annotation_tools", "mmseqs")
DEFAULT_MMSEQS_MIN_IDENTITY = 0.0
DEFAULT_MMSEQS_MIN_QUERY_COVERAGE = 0.70
DEFAULT_MIN_FAMILY_PHAGE_SUPPORT = 2
DEFAULT_MAX_TARGET_SEQS = 20
STRING_DTYPE = "string"
FLOAT_DTYPE = "float64"
INTEGER_DTYPE = "int64"
MMSEQS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "query",
    "target",
    "pident",
    "alnlen",
    "qstart",
    "qend",
    "qlen",
    "tstart",
    "tend",
    "tlen",
    "evalue",
    "bits",
)
TL17_BLOCK_ID = "tl17_rbp_family_projection"
FAMILY_COLUMN_PREFIX = "tl17_phage_rbp_family"
SUMMARY_HIT_COUNT_COLUMN = "tl17_rbp_reference_hit_count"
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
GENE_INDEX_RE = re.compile(r"_(\d+)$")
PHROG_FAMILY_RE = re.compile(r"RBP_PHROG_(?P<phrog>\S+)")
PROTEIN_COORDS_RE = re.compile(r"start=(?P<start>\d+) end=(?P<end>\d+) strand=(?P<strand>-?1)")


@dataclass(frozen=True)
class Tl17ReferenceProtein:
    reference_id: str
    phage: str
    family_id: str
    gene_name: str
    protein_index: int
    annotation: str
    phrog: str
    protein_sequence: str


@dataclass(frozen=True)
class Tl17FamilyRuntime:
    family_id: str
    column_name: str
    supporting_phage_count: int
    supporting_reference_count: int


@dataclass(frozen=True)
class Tl17MatchRecord:
    query_id: str
    target_id: str
    percent_identity: float
    alignment_length: int
    query_start: int
    query_end: int
    query_length: int
    target_start: int
    target_end: int
    target_length: int
    evalue: str
    bits: float

    @property
    def query_coverage(self) -> float:
        if self.query_length <= 0:
            raise ValueError("query_length must be positive")
        return self.alignment_length / self.query_length


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_family_column_name(family_id: str) -> str:
    match = PHROG_FAMILY_RE.fullmatch(family_id)
    suffix = match.group("phrog") if match else family_id.lower()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", suffix).strip("_").lower()
    return f"{FAMILY_COLUMN_PREFIX}_{cleaned}_percent_identity"


def build_projection_schema(family_rows: Sequence[Tl17FamilyRuntime]) -> dict[str, object]:
    columns = [
        {"name": "phage", "dtype": STRING_DTYPE},
        *({"name": row.column_name, "dtype": FLOAT_DTYPE} for row in family_rows),
        {"name": SUMMARY_HIT_COUNT_COLUMN, "dtype": INTEGER_DTYPE},
    ]
    return {
        "feature_block": TL17_BLOCK_ID,
        "key_column": "phage",
        "column_count": len(columns),
        "columns": columns,
        "family_score_columns": [row.column_name for row in family_rows],
        "reference_hit_count_column": SUMMARY_HIT_COUNT_COLUMN,
        "dropped_legacy_columns": ["tl17_rbp_family_count"],
    }


def write_schema_manifest(family_rows: Sequence[Tl17FamilyRuntime], output_path: Path) -> Path:
    write_json(output_path, build_projection_schema(family_rows))
    return output_path


def parse_gene_index(gene_name: str) -> int:
    match = GENE_INDEX_RE.search(gene_name)
    if match is None:
        raise ValueError(f"Could not parse gene index from {gene_name!r}")
    return int(match.group(1))


def _normalize_coords(start: int, stop: int) -> tuple[int, int]:
    return (min(start, stop), max(start, stop))


def _parse_protein_coords(description: str) -> tuple[tuple[int, int], int] | None:
    match = PROTEIN_COORDS_RE.search(description)
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    strand = int(match.group("strand"))
    return _normalize_coords(start, end), strand


def resolve_reference_protein_sequence(
    *,
    proteins: Sequence[object],
    rbp_record: object,
) -> tuple[int, str]:
    coord_lookup: dict[tuple[tuple[int, int], int], tuple[int, str]] = {}
    for protein_index, protein in enumerate(proteins, start=1):
        description = str(getattr(protein, "description"))
        parsed = _parse_protein_coords(description)
        if parsed is None:
            continue
        coord_lookup[parsed] = (protein_index, str(getattr(protein, "sequence")))

    strand = 1 if str(getattr(rbp_record, "strand")) == "+" else -1
    normalized_coords = _normalize_coords(int(getattr(rbp_record, "start")), int(getattr(rbp_record, "stop")))
    exact_match = coord_lookup.get((normalized_coords, strand))
    if exact_match is not None:
        return exact_match

    protein_index = parse_gene_index(str(getattr(rbp_record, "gene")))
    if protein_index < 1 or protein_index > len(proteins):
        raise ValueError(
            f"RBP record {getattr(rbp_record, 'gene')} points to protein index {protein_index}, but only "
            f"{len(proteins)} proteins were predicted from the raw FASTA and no coordinate match was found."
        )
    fallback_protein = proteins[protein_index - 1]
    return protein_index, str(getattr(fallback_protein, "sequence"))


def build_reference_proteins(
    *,
    phage_metadata_path: Path,
    fna_dir: Path,
    cached_annotations_dir: Path,
    expected_panel_count: int,
    min_family_phage_support: int = DEFAULT_MIN_FAMILY_PHAGE_SUPPORT,
) -> tuple[list[Tl17ReferenceProtein], list[Tl17FamilyRuntime]]:
    panel_phages = read_panel_phages(phage_metadata_path, expected_panel_count=expected_panel_count)
    reference_rows: list[Tl17ReferenceProtein] = []
    family_to_phages: dict[str, set[str]] = {}
    family_to_reference_count: dict[str, int] = {}

    for phage in panel_phages:
        fasta_path = fna_dir / f"{phage}.fna"
        annotation_path = cached_annotations_dir / f"{phage}_cds_final_merged_output.tsv"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing phage FASTA for TL17 reference build: {fasta_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing cached pharokka TSV for TL17 reference build: {annotation_path}")

        genome_records = read_fasta_records(fasta_path, protein=False)
        proteins, _ = call_proteins_with_pyrodigal(phage, genome_records)
        rbp_records = classify_rbp_genes(parse_merged_tsv(annotation_path))
        for rbp_record in rbp_records:
            if rbp_record.phrog == "No_PHROG":
                continue
            protein_index, protein_sequence = resolve_reference_protein_sequence(
                proteins=proteins,
                rbp_record=rbp_record,
            )
            reference_id = f"{phage}|ref_rbp_{protein_index:04d}"
            family_id = f"RBP_PHROG_{rbp_record.phrog}"
            reference_rows.append(
                Tl17ReferenceProtein(
                    reference_id=reference_id,
                    phage=phage,
                    family_id=family_id,
                    gene_name=rbp_record.gene,
                    protein_index=protein_index,
                    annotation=rbp_record.annot,
                    phrog=rbp_record.phrog,
                    protein_sequence=protein_sequence,
                )
            )
            family_to_phages.setdefault(family_id, set()).add(phage)
            family_to_reference_count[family_id] = family_to_reference_count.get(family_id, 0) + 1

    retained_families = {
        family_id for family_id, phages in family_to_phages.items() if len(phages) >= min_family_phage_support
    }
    retained_reference_rows = [row for row in reference_rows if row.family_id in retained_families]
    if not retained_reference_rows:
        raise ValueError("TL17 retained zero reference proteins after applying the minimum family support filter.")

    family_rows = [
        Tl17FamilyRuntime(
            family_id=family_id,
            column_name=build_family_column_name(family_id),
            supporting_phage_count=len(family_to_phages[family_id]),
            supporting_reference_count=family_to_reference_count[family_id],
        )
        for family_id in sorted(retained_families)
    ]
    return retained_reference_rows, family_rows


def write_reference_fasta(reference_rows: Sequence[Tl17ReferenceProtein], output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in reference_rows:
            handle.write(f">{row.reference_id}\n{row.protein_sequence}\n")
    return output_path


def write_family_metadata_csv(family_rows: Sequence[Tl17FamilyRuntime], output_path: Path) -> Path:
    write_csv(
        output_path,
        ["family_id", "column_name", "supporting_phage_count", "supporting_reference_count"],
        [
            {
                "family_id": row.family_id,
                "column_name": row.column_name,
                "supporting_phage_count": row.supporting_phage_count,
                "supporting_reference_count": row.supporting_reference_count,
            }
            for row in family_rows
        ],
    )
    return output_path


def write_reference_metadata_csv(reference_rows: Sequence[Tl17ReferenceProtein], output_path: Path) -> Path:
    write_csv(
        output_path,
        [
            "reference_id",
            "phage",
            "family_id",
            "gene_name",
            "protein_index",
            "annotation",
            "phrog",
            "protein_length_aa",
        ],
        [
            {
                "reference_id": row.reference_id,
                "phage": row.phage,
                "family_id": row.family_id,
                "gene_name": row.gene_name,
                "protein_index": row.protein_index,
                "annotation": row.annotation,
                "phrog": row.phrog,
                "protein_length_aa": len(row.protein_sequence),
            }
            for row in reference_rows
        ],
    )
    return output_path


def write_query_fasta(phage_path: Path, output_path: Path) -> tuple[Path, list[str]]:
    phage = phage_path.stem
    genome_records = read_fasta_records(phage_path, protein=False)
    proteins, _ = call_proteins_with_pyrodigal(phage, genome_records)
    query_ids = [f"{phage}|query_prot_{index:04d}" for index in range(1, len(proteins) + 1)]
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, protein in zip(query_ids, proteins, strict=True):
            handle.write(f">{query_id}\n{protein.sequence}\n")
    return output_path, query_ids


def run_mmseqs_search(
    *,
    query_fasta_path: Path,
    reference_fasta_path: Path,
    output_tsv_path: Path,
    scratch_dir: Path,
    mmseqs_command: Sequence[str] = DEFAULT_MMSEQS_COMMAND,
    max_target_seqs: int = DEFAULT_MAX_TARGET_SEQS,
) -> Path:
    ensure_directory(scratch_dir)
    command = [
        *mmseqs_command,
        "easy-search",
        str(query_fasta_path),
        str(reference_fasta_path),
        str(output_tsv_path),
        str(scratch_dir / "mmseqs_tmp"),
        "--max-seqs",
        str(max_target_seqs),
        "--format-output",
        ",".join(MMSEQS_OUTPUT_COLUMNS),
    ]
    LOGGER.info("Starting TL17 mmseqs search for %s", query_fasta_path.name)
    subprocess.run(command, check=True, capture_output=True, text=True)
    LOGGER.info("Completed TL17 mmseqs search for %s", query_fasta_path.name)
    return output_tsv_path


def read_mmseqs_matches(path: Path) -> list[Tl17MatchRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mmseqs output: {path}")
    matches: list[Tl17MatchRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) != len(MMSEQS_OUTPUT_COLUMNS):
                raise ValueError(f"Unexpected mmseqs column count in {path}: expected {len(MMSEQS_OUTPUT_COLUMNS)}")
            matches.append(
                Tl17MatchRecord(
                    query_id=row[0],
                    target_id=row[1],
                    percent_identity=float(row[2]),
                    alignment_length=int(row[3]),
                    query_start=int(row[4]),
                    query_end=int(row[5]),
                    query_length=int(row[6]),
                    target_start=int(row[7]),
                    target_end=int(row[8]),
                    target_length=int(row[9]),
                    evalue=row[10],
                    bits=float(row[11]),
                )
            )
    return matches


def build_runtime_payload(
    *,
    family_rows: Sequence[Tl17FamilyRuntime],
    reference_rows: Sequence[Tl17ReferenceProtein],
    min_percent_identity: float,
    min_query_coverage: float,
    mmseqs_command: Sequence[str] = DEFAULT_MMSEQS_COMMAND,
) -> dict[str, object]:
    return {
        "block_id": TL17_BLOCK_ID,
        "family_rows": [
            {
                "family_id": row.family_id,
                "column_name": row.column_name,
                "supporting_phage_count": row.supporting_phage_count,
                "supporting_reference_count": row.supporting_reference_count,
            }
            for row in family_rows
        ],
        "reference_rows": [
            {
                "reference_id": row.reference_id,
                "phage": row.phage,
                "family_id": row.family_id,
                "gene_name": row.gene_name,
                "protein_index": row.protein_index,
                "annotation": row.annotation,
                "phrog": row.phrog,
                "protein_sequence": row.protein_sequence,
            }
            for row in reference_rows
        ],
        "matching_policy": {
            "min_percent_identity": min_percent_identity,
            "min_query_coverage": min_query_coverage,
            "mmseqs_command": list(mmseqs_command),
        },
    }


def parse_runtime_payload(
    payload: Mapping[str, object],
) -> tuple[list[Tl17FamilyRuntime], list[Tl17ReferenceProtein], dict[str, object]]:
    family_rows = [
        Tl17FamilyRuntime(
            family_id=str(row["family_id"]),
            column_name=str(row["column_name"]),
            supporting_phage_count=int(row["supporting_phage_count"]),
            supporting_reference_count=int(row["supporting_reference_count"]),
        )
        for row in payload.get("family_rows", [])
    ]
    reference_rows = [
        Tl17ReferenceProtein(
            reference_id=str(row["reference_id"]),
            phage=str(row["phage"]),
            family_id=str(row["family_id"]),
            gene_name=str(row["gene_name"]),
            protein_index=int(row["protein_index"]),
            annotation=str(row["annotation"]),
            phrog=str(row["phrog"]),
            protein_sequence=str(row["protein_sequence"]),
        )
        for row in payload.get("reference_rows", [])
    ]
    matching_policy = dict(payload.get("matching_policy", {}))
    return family_rows, reference_rows, matching_policy


def project_phage_feature_row(
    phage_path: Path,
    *,
    runtime_payload: Mapping[str, object],
    reference_fasta_path: Path,
    scratch_root: Path,
) -> dict[str, object]:
    family_rows, reference_rows, matching_policy = parse_runtime_payload(runtime_payload)
    reference_to_family = {row.reference_id: row.family_id for row in reference_rows}
    phage_scratch_dir = scratch_root / phage_path.stem
    if phage_scratch_dir.exists():
        shutil.rmtree(phage_scratch_dir)
    ensure_directory(phage_scratch_dir)
    query_fasta_path, _ = write_query_fasta(phage_path, phage_scratch_dir / f"{phage_path.stem}.faa")
    match_tsv_path = phage_scratch_dir / "mmseqs_hits.tsv"
    run_mmseqs_search(
        query_fasta_path=query_fasta_path,
        reference_fasta_path=reference_fasta_path,
        output_tsv_path=match_tsv_path,
        scratch_dir=phage_scratch_dir,
        mmseqs_command=tuple(str(token) for token in matching_policy["mmseqs_command"]),
    )
    min_query_coverage = float(matching_policy["min_query_coverage"])
    family_scores = {row.family_id: 0.0 for row in family_rows}
    accepted_hit_count = 0
    for match in read_mmseqs_matches(match_tsv_path):
        family_id = reference_to_family.get(match.target_id)
        if family_id is None:
            continue
        if match.query_coverage < min_query_coverage:
            continue
        accepted_hit_count += 1
        family_scores[family_id] = max(family_scores[family_id], float(match.percent_identity))
    row: dict[str, object] = {"phage": phage_path.stem}
    for family_row in family_rows:
        row[family_row.column_name] = family_scores[family_row.family_id]
    row[SUMMARY_HIT_COUNT_COLUMN] = accepted_hit_count
    return row


def project_panel_feature_rows(
    *,
    phage_metadata_path: Path,
    fna_dir: Path,
    expected_panel_count: int,
    runtime_payload: Mapping[str, object],
    reference_fasta_path: Path,
    scratch_root: Path,
) -> list[dict[str, object]]:
    phages = read_panel_phages(phage_metadata_path, expected_panel_count=expected_panel_count)
    feature_rows = project_phage_feature_rows_batched(
        phage_paths=[fna_dir / f"{phage}.fna" for phage in phages],
        runtime_payload=runtime_payload,
        reference_fasta_path=reference_fasta_path,
        scratch_root=scratch_root,
    )
    if not feature_rows:
        raise ValueError("TL17 panel projection produced zero rows.")
    return feature_rows


def project_phage_feature_rows_batched(
    phage_paths: Sequence[Path],
    *,
    runtime_payload: Mapping[str, object],
    reference_fasta_path: Path,
    scratch_root: Path,
) -> list[dict[str, object]]:
    """Project multiple phages in a single batched mmseqs search instead of one per phage."""
    family_rows, reference_rows, matching_policy = parse_runtime_payload(runtime_payload)
    reference_to_family = {row.reference_id: row.family_id for row in reference_rows}
    min_query_coverage = float(matching_policy["min_query_coverage"])
    mmseqs_command = tuple(str(token) for token in matching_policy["mmseqs_command"])

    batch_scratch_dir = scratch_root / "_batched"
    if batch_scratch_dir.exists():
        shutil.rmtree(batch_scratch_dir)
    ensure_directory(batch_scratch_dir)

    LOGGER.info("Starting batched pyrodigal gene calling for %d phages", len(phage_paths))
    combined_query_path = batch_scratch_dir / "combined_queries.faa"
    phage_names: list[str] = []
    with combined_query_path.open("w", encoding="utf-8") as handle:
        for phage_path in phage_paths:
            phage = phage_path.stem
            phage_names.append(phage)
            genome_records = read_fasta_records(phage_path, protein=False)
            proteins, _ = call_proteins_with_pyrodigal(phage, genome_records)
            for index, protein in enumerate(proteins, 1):
                handle.write(f">{phage}|query_prot_{index:04d}\n{protein.sequence}\n")
    LOGGER.info("Completed batched pyrodigal gene calling; %d phages written to combined query", len(phage_names))

    match_tsv_path = batch_scratch_dir / "mmseqs_hits.tsv"
    run_mmseqs_search(
        query_fasta_path=combined_query_path,
        reference_fasta_path=reference_fasta_path,
        output_tsv_path=match_tsv_path,
        scratch_dir=batch_scratch_dir,
        mmseqs_command=mmseqs_command,
    )

    family_scores_by_phage: dict[str, dict[str, float]] = {
        phage: {row.family_id: 0.0 for row in family_rows} for phage in phage_names
    }
    accepted_hits_by_phage: dict[str, int] = {phage: 0 for phage in phage_names}
    for match in read_mmseqs_matches(match_tsv_path):
        phage = match.query_id.split("|", 1)[0]
        family_id = reference_to_family.get(match.target_id)
        if family_id is None:
            continue
        if match.query_coverage < min_query_coverage:
            continue
        if phage not in family_scores_by_phage:
            continue
        family_scores_by_phage[phage][family_id] = max(
            family_scores_by_phage[phage][family_id], float(match.percent_identity)
        )
        accepted_hits_by_phage[phage] += 1

    feature_rows: list[dict[str, object]] = []
    for phage in phage_names:
        row: dict[str, object] = {"phage": phage}
        for family_row in family_rows:
            row[family_row.column_name] = family_scores_by_phage[phage][family_row.family_id]
        row[SUMMARY_HIT_COUNT_COLUMN] = accepted_hits_by_phage[phage]
        feature_rows.append(row)
    return feature_rows


def build_fasta_inventory_rows(
    *,
    phage_metadata_path: Path,
    fna_dir: Path,
    expected_panel_count: int,
) -> list[dict[str, object]]:
    phages = read_panel_phages(phage_metadata_path, expected_panel_count=expected_panel_count)
    inventory_rows = []
    for phage in phages:
        fasta_path = fna_dir / f"{phage}.fna"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing phage FASTA for TL17 inventory: {fasta_path}")
        inventory_rows.append(
            {
                "phage": phage,
                "fasta_path": str(fasta_path),
                "sha256": _sha256(fasta_path),
            }
        )
    return inventory_rows
