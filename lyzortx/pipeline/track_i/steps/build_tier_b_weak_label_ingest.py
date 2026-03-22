#!/usr/bin/env python3
"""TI06: Ingest Tier B weak labels from Virus-Host DB and NCBI Virus/BioSample metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows

REQUIRED_SOURCE_REGISTRY_COLUMNS = ("source_id", "confidence_tier", "confidence_basis", "notes")

VIRUS_HOST_DB_SOURCE_ID = "virus_host_db"
NCBI_SOURCE_ID = "ncbi_virus_biosample"

VIRUS_HOST_DB_REQUIRED_COLUMNS = (
    "virus tax id",
    "virus name",
    "virus lineage",
    "refseq id",
    "host tax id",
    "host name",
    "host lineage",
    "pmid",
    "evidence",
    "sample type",
    "source organism",
)

OUTPUT_FIELDNAMES = [
    "pair_id",
    "bacteria",
    "bacteria_id",
    "phage",
    "phage_id",
    "label_hard_any_lysis",
    "label_strict_confidence_tier",
    "source_system",
    "source_datasource_id",
    "source_native_record_id",
    "source_source_type",
    "source_relation_type",
    "source_confidence_tier",
    "source_confidence_basis",
    "source_resolution_status",
    "source_disagreement_flag",
    "source_qc_flag",
    "source_virus_tax_id",
    "source_virus_name",
    "source_virus_lineage",
    "source_virus_accession",
    "source_host_tax_id",
    "source_host_name",
    "source_host_lineage",
    "source_biosample_accession",
    "source_biosample_host",
    "source_biosample_isolation_host",
    "source_biosample_isolation_source",
    "source_biosample_title",
    "source_reference_id",
    "source_evidence",
    "source_sample_type",
    "source_source_organism",
]


@dataclass(frozen=True)
class CanonicalResolutionIndex:
    canonical_to_id: Dict[str, str]
    lookup: Dict[str, Tuple[str, str]]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/source_registry.csv"),
    )
    parser.add_argument(
        "--virus-host-db-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/virushostdb.tsv"),
    )
    parser.add_argument(
        "--ncbi-virus-report-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/ncbi_virus_report.jsonl"),
    )
    parser.add_argument(
        "--ncbi-biosample-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/ncbi_biosample.xml"),
    )
    parser.add_argument(
        "--track-a-bacteria-id-map-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/id_map/bacteria_id_map.csv"),
    )
    parser.add_argument(
        "--track-a-phage-id-map-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/id_map/phage_id_map.csv"),
    )
    parser.add_argument(
        "--track-a-bacteria-alias-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/id_map/bacteria_alias_resolution.csv"),
    )
    parser.add_argument(
        "--track-a-phage-alias-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/id_map/phage_alias_resolution.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/tier_b_weak_label_ingest"),
    )
    return parser.parse_args(argv)


def _normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _first_non_blank(row: Mapping[str, str], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return default


def _split_multi_value(value: str) -> List[str]:
    if not value.strip():
        return []
    parts = []
    for chunk in value.replace("|", ",").replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            parts.append(token)
    return parts or [value.strip()]


def _stable_record_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def read_delimited_rows(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]


def require_columns(rows: Sequence[Mapping[str, str]], path: Path, columns: Sequence[str]) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")


def read_jsonl_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
            rows.append({key: "" if value is None else str(value) for key, value in payload.items()})
    return rows


def read_biosample_xml(path: Path) -> List[Dict[str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    rows: List[Dict[str, str]] = []
    for biosample in root.findall(".//BioSample"):
        attributes: Dict[str, str] = {}
        for attribute in biosample.findall("./Attributes/Attribute"):
            key = (
                attribute.get("harmonized_name")
                or attribute.get("attribute_name")
                or attribute.get("display_name")
                or ""
            )
            if key and key not in attributes:
                attributes[key] = (attribute.text or "").strip()

        description = biosample.find("./Description")
        organism = description.find("./Organism") if description is not None else None
        organism_name = ""
        taxonomy_id = ""
        if organism is not None:
            organism_name = (organism.findtext("./OrganismName") or organism.get("taxonomy_name") or "").strip()
            taxonomy_id = organism.get("taxonomy_id", "").strip()

        ids = {
            f"{element.get('db', '')}:{(element.text or '').strip()}"
            for element in biosample.findall("./Ids/Id")
            if (element.text or "").strip()
        }
        rows.append(
            {
                "biosample_accession": biosample.get("accession", "").strip(),
                "biosample_title": (description.findtext("./Title") if description is not None else "") or "",
                "biosample_organism_name": organism_name,
                "biosample_taxonomy_id": taxonomy_id,
                "biosample_host": attributes.get("host", ""),
                "biosample_isolation_host": attributes.get("isolation_host", ""),
                "biosample_isolation_source": attributes.get("isolation_source", ""),
                "biosample_host_disease": attributes.get("host_disease", ""),
                "biosample_source_ids": "|".join(sorted(ids)),
            }
        )
    return rows


def load_source_registry(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path, REQUIRED_SOURCE_REGISTRY_COLUMNS)
    return {row["source_id"]: row for row in rows}


def build_canonical_resolution_index(
    id_map_path: Path,
    alias_path: Path,
    *,
    canonical_name_column: str,
    canonical_id_column: str,
    raw_names_column: str,
    alias_name_column: str = "original_name",
    alias_canonical_name_column: str = "canonical_name",
) -> CanonicalResolutionIndex:
    canonical_to_id: Dict[str, str] = {}
    lookup: Dict[str, Tuple[str, str]] = {}

    if id_map_path.exists():
        for row in read_csv_rows(id_map_path, (canonical_name_column, canonical_id_column, raw_names_column)):
            canonical_name = row[canonical_name_column]
            canonical_id = row[canonical_id_column]
            canonical_to_id[canonical_name] = canonical_id
            lookup[_normalize_key(canonical_name)] = (canonical_name, canonical_id)
            for raw_name in _split_multi_value(row.get(raw_names_column, "")):
                lookup.setdefault(_normalize_key(raw_name), (canonical_name, canonical_id))

    if alias_path.exists():
        for row in read_csv_rows(alias_path, (alias_name_column, alias_canonical_name_column)):
            canonical_name = row[alias_canonical_name_column]
            canonical_id = canonical_to_id.get(canonical_name, "")
            if canonical_name and canonical_name in canonical_to_id:
                lookup[_normalize_key(row[alias_name_column])] = (canonical_name, canonical_id)

    return CanonicalResolutionIndex(canonical_to_id=canonical_to_id, lookup=lookup)


def resolve_canonical_name(name: str, index: CanonicalResolutionIndex) -> Tuple[str, str, str]:
    if not name:
        return "", "", "missing"
    resolved = index.lookup.get(_normalize_key(name))
    if resolved is None:
        return name, "", "unresolved"
    canonical_name, canonical_id = resolved
    if canonical_name == name:
        return canonical_name, canonical_id, "resolved"
    return canonical_name, canonical_id, "resolved_via_alias"


def _common_output_row() -> Dict[str, str]:
    return {field: "" for field in OUTPUT_FIELDNAMES}


def normalize_virus_host_db_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    registry_row: Mapping[str, str],
    bacteria_index: CanonicalResolutionIndex,
    phage_index: CanonicalResolutionIndex,
) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for row in rows:
        refseq_ids = _split_multi_value(row.get("refseq id", ""))
        if not refseq_ids:
            refseq_ids = [
                row.get("virus name", "").strip()
                or _stable_record_id("virus_host_db_refseq", row.get("virus tax id", ""), row.get("host tax id", ""))
            ]
        for refseq_id in refseq_ids:
            bacteria_name, bacteria_id, bacteria_resolution = resolve_canonical_name(
                row.get("host name", ""), bacteria_index
            )
            phage_name, phage_id, phage_resolution = resolve_canonical_name(refseq_id, phage_index)
            output_row = _common_output_row()
            output_row.update(
                {
                    "pair_id": f"{bacteria_name}__{phage_name}",
                    "bacteria": bacteria_name,
                    "bacteria_id": bacteria_id,
                    "phage": phage_name,
                    "phage_id": phage_id,
                    "label_hard_any_lysis": "1",
                    "label_strict_confidence_tier": "",
                    "source_system": VIRUS_HOST_DB_SOURCE_ID,
                    "source_datasource_id": VIRUS_HOST_DB_SOURCE_ID,
                    "source_native_record_id": _stable_record_id(
                        "virushostdb",
                        row.get("virus tax id", ""),
                        row.get("host tax id", ""),
                        row.get("pmid", ""),
                        refseq_id,
                    ),
                    "source_source_type": "metadata_knowledgebase",
                    "source_relation_type": "host_association_non_assay_metadata",
                    "source_confidence_tier": registry_row.get("confidence_tier", "B"),
                    "source_confidence_basis": registry_row.get("confidence_basis", ""),
                    "source_resolution_status": "|".join(sorted({bacteria_resolution, phage_resolution})),
                    "source_disagreement_flag": "0",
                    "source_qc_flag": "ok",
                    "source_virus_tax_id": row.get("virus tax id", ""),
                    "source_virus_name": row.get("virus name", ""),
                    "source_virus_lineage": row.get("virus lineage", ""),
                    "source_virus_accession": refseq_id,
                    "source_host_tax_id": row.get("host tax id", ""),
                    "source_host_name": row.get("host name", ""),
                    "source_host_lineage": row.get("host lineage", ""),
                    "source_reference_id": row.get("pmid", ""),
                    "source_evidence": row.get("evidence", ""),
                    "source_sample_type": row.get("sample type", ""),
                    "source_source_organism": row.get("source organism", ""),
                }
            )
            output.append(output_row)
    return output


def _biosample_lookup_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".xml":
        rows = read_biosample_xml(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        rows = read_jsonl_rows(path)
    else:
        rows = read_delimited_rows(path, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
    lookup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        accession = _first_non_blank(row, ("biosample_accession", "accession", "sample_accession", "BioSample"))
        if accession:
            lookup[accession] = row
    return lookup


def normalize_ncbi_virus_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    registry_row: Mapping[str, str],
    bacteria_index: CanonicalResolutionIndex,
    phage_index: CanonicalResolutionIndex,
    biosample_lookup: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for row in rows:
        virus_accession = _first_non_blank(
            row,
            ("accession", "virus_accession", "refseq_id", "genbank_accession", "sequence_accession"),
            default="",
        )
        virus_name = _first_non_blank(
            row,
            ("virus_name", "organism_name", "title", "name"),
            default="",
        )
        host_name = _first_non_blank(row, ("host", "host_scientific_name", "host_name"), default="")
        biosample_accession = _first_non_blank(
            row,
            ("biosample", "biosample_accession", "sample_accession", "biosample_id"),
            default="",
        )
        biosample_row = biosample_lookup.get(biosample_accession, {})
        biosample_host = _first_non_blank(biosample_row, ("biosample_host", "host", "isolation_host"), default="")
        biosample_isolation_host = _first_non_blank(
            biosample_row,
            ("biosample_isolation_host", "isolation_host"),
            default="",
        )
        biosample_isolation_source = _first_non_blank(
            biosample_row,
            ("biosample_isolation_source", "isolation_source"),
            default="",
        )
        report_host_name, _, report_host_resolution = resolve_canonical_name(host_name, bacteria_index)
        biosample_host_name, _, biosample_host_resolution = resolve_canonical_name(biosample_host, bacteria_index)
        resolved_bacteria_name, bacteria_id, bacteria_resolution = resolve_canonical_name(
            biosample_host or host_name, bacteria_index
        )
        resolved_phage_name, phage_id, phage_resolution = resolve_canonical_name(
            virus_accession or virus_name, phage_index
        )
        qc_flag = "ok"
        if (
            biosample_row
            and report_host_name
            and biosample_host_name
            and _normalize_key(report_host_name) != _normalize_key(biosample_host_name)
        ):
            qc_flag = "host_conflict"
        elif not biosample_row:
            qc_flag = "biosample_missing"

        output_row = _common_output_row()
        output_row.update(
            {
                "pair_id": f"{resolved_bacteria_name}__{resolved_phage_name}",
                "bacteria": resolved_bacteria_name,
                "bacteria_id": bacteria_id,
                "phage": resolved_phage_name,
                "phage_id": phage_id,
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "",
                "source_system": NCBI_SOURCE_ID,
                "source_datasource_id": NCBI_SOURCE_ID,
                "source_native_record_id": _stable_record_id(
                    "ncbi_virus",
                    virus_accession,
                    biosample_accession,
                    host_name,
                    biosample_host,
                ),
                "source_source_type": "metadata_repository",
                "source_relation_type": "host_annotation_and_isolation_metadata",
                "source_confidence_tier": registry_row.get("confidence_tier", "B"),
                "source_confidence_basis": registry_row.get("confidence_basis", ""),
                "source_resolution_status": "|".join(
                    sorted({bacteria_resolution, phage_resolution, report_host_resolution, biosample_host_resolution})
                ),
                "source_disagreement_flag": "1" if qc_flag == "host_conflict" else "0",
                "source_qc_flag": qc_flag,
                "source_virus_tax_id": _first_non_blank(row, ("taxid", "virus_tax_id", "virus_taxon_id"), default=""),
                "source_virus_name": virus_name,
                "source_virus_lineage": _first_non_blank(row, ("virus_lineage", "lineage"), default=""),
                "source_virus_accession": virus_accession,
                "source_host_tax_id": _first_non_blank(row, ("host_taxid", "host_tax_id"), default=""),
                "source_host_name": host_name,
                "source_host_lineage": _first_non_blank(row, ("host_lineage",), default=""),
                "source_biosample_accession": biosample_accession,
                "source_biosample_host": biosample_host,
                "source_biosample_isolation_host": biosample_isolation_host,
                "source_biosample_isolation_source": biosample_isolation_source,
                "source_biosample_title": _first_non_blank(biosample_row, ("biosample_title", "title"), default=""),
                "source_reference_id": biosample_accession or virus_accession,
                "source_evidence": _first_non_blank(row, ("evidence", "source"), default=""),
                "source_sample_type": _first_non_blank(row, ("sample_type",), default=""),
                "source_source_organism": _first_non_blank(row, ("source_organism",), default=""),
            }
        )
        output.append(output_row)
    return output


def read_source_rows(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return read_delimited_rows(path, delimiter="\t")
    if suffix == ".csv":
        return read_delimited_rows(path, delimiter=",")
    if suffix in {".json", ".jsonl"}:
        return read_jsonl_rows(path)
    if suffix == ".xml":
        return read_biosample_xml(path)
    raise ValueError(f"Unsupported file type for {path}")


def build_source_rows(
    *,
    source_id: str,
    registry_rows: Mapping[str, Mapping[str, str]],
    bacteria_index: CanonicalResolutionIndex,
    phage_index: CanonicalResolutionIndex,
    virus_host_db_path: Path,
    ncbi_virus_report_path: Path,
    ncbi_biosample_path: Path,
) -> List[Dict[str, str]]:
    registry_row = registry_rows[source_id]
    if source_id == VIRUS_HOST_DB_SOURCE_ID:
        rows = read_delimited_rows(virus_host_db_path, delimiter="\t")
        require_columns(rows, virus_host_db_path, VIRUS_HOST_DB_REQUIRED_COLUMNS)
        return normalize_virus_host_db_rows(
            rows, registry_row=registry_row, bacteria_index=bacteria_index, phage_index=phage_index
        )

    virus_rows = read_source_rows(ncbi_virus_report_path)
    biosample_lookup = _biosample_lookup_rows(ncbi_biosample_path)
    return normalize_ncbi_virus_rows(
        virus_rows,
        registry_row=registry_row,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
        biosample_lookup=biosample_lookup,
    )


def compute_summary_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    by_source: Dict[str, int] = {}
    by_qc: Dict[str, int] = {}
    for row in rows:
        source_id = row.get("source_system", "")
        by_source[source_id] = by_source.get(source_id, 0) + 1
        qc_flag = row.get("source_qc_flag", "")
        by_qc[qc_flag] = by_qc.get(qc_flag, 0) + 1
    summary: List[Dict[str, object]] = []
    for source_id, count in sorted(by_source.items()):
        summary.append({"slice_type": "source_system", "slice_value": source_id, "row_count": count})
    for qc_flag, count in sorted(by_qc.items()):
        summary.append({"slice_type": "qc_flag", "slice_value": qc_flag, "row_count": count})
    return summary


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    registry_rows = load_source_registry(args.source_registry_path)
    missing_registry = [
        source_id for source_id in (VIRUS_HOST_DB_SOURCE_ID, NCBI_SOURCE_ID) if source_id not in registry_rows
    ]
    if missing_registry:
        raise ValueError(f"Missing Tier B source registry entries: {', '.join(sorted(missing_registry))}")

    bacteria_index = build_canonical_resolution_index(
        args.track_a_bacteria_id_map_path,
        args.track_a_bacteria_alias_path,
        canonical_name_column="canonical_bacteria",
        canonical_id_column="canonical_bacteria_id",
        raw_names_column="raw_names",
    )
    phage_index = build_canonical_resolution_index(
        args.track_a_phage_id_map_path,
        args.track_a_phage_alias_path,
        canonical_name_column="canonical_phage",
        canonical_id_column="canonical_phage_id",
        raw_names_column="raw_names",
    )

    active_sources: List[str] = []
    merged_rows: List[Dict[str, str]] = []
    output_paths: Dict[str, str] = {}
    input_hashes: Dict[str, str] = {}

    if args.virus_host_db_path.exists():
        active_sources.append(VIRUS_HOST_DB_SOURCE_ID)
        source_rows = build_source_rows(
            source_id=VIRUS_HOST_DB_SOURCE_ID,
            registry_rows=registry_rows,
            bacteria_index=bacteria_index,
            phage_index=phage_index,
            virus_host_db_path=args.virus_host_db_path,
            ncbi_virus_report_path=args.ncbi_virus_report_path,
            ncbi_biosample_path=args.ncbi_biosample_path,
        )
        merged_rows.extend(source_rows)
        output_paths[VIRUS_HOST_DB_SOURCE_ID] = str(args.output_dir / "virus_host_db_ingested_pairs.csv")
        input_hashes[VIRUS_HOST_DB_SOURCE_ID] = _hash_path(args.virus_host_db_path)
        write_csv(
            args.output_dir / "virus_host_db_ingested_pairs.csv",
            fieldnames=OUTPUT_FIELDNAMES,
            rows=source_rows,
        )

    if args.ncbi_virus_report_path.exists():
        active_sources.append(NCBI_SOURCE_ID)
        source_rows = build_source_rows(
            source_id=NCBI_SOURCE_ID,
            registry_rows=registry_rows,
            bacteria_index=bacteria_index,
            phage_index=phage_index,
            virus_host_db_path=args.virus_host_db_path,
            ncbi_virus_report_path=args.ncbi_virus_report_path,
            ncbi_biosample_path=args.ncbi_biosample_path,
        )
        merged_rows.extend(source_rows)
        output_paths[NCBI_SOURCE_ID] = str(args.output_dir / "ncbi_virus_biosample_ingested_pairs.csv")
        input_hashes[NCBI_SOURCE_ID] = _hash_path(args.ncbi_virus_report_path)
        if args.ncbi_biosample_path.exists():
            input_hashes["ncbi_biosample"] = _hash_path(args.ncbi_biosample_path)
        write_csv(
            args.output_dir / "ncbi_virus_biosample_ingested_pairs.csv",
            fieldnames=OUTPUT_FIELDNAMES,
            rows=source_rows,
        )

    if not merged_rows:
        raise ValueError("No Tier B input files were found.")

    combined_path = args.output_dir / "ti06_weak_label_ingested_pairs.csv"
    summary_path = args.output_dir / "ti06_weak_label_summary.csv"
    manifest_path = args.output_dir / "ti06_weak_label_manifest.json"

    write_csv(combined_path, fieldnames=OUTPUT_FIELDNAMES, rows=merged_rows)
    write_csv(
        summary_path, fieldnames=["slice_type", "slice_value", "row_count"], rows=compute_summary_rows(merged_rows)
    )
    write_json(
        manifest_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_tier_b_weak_label_ingest",
            "active_sources": active_sources,
            "input_paths": {
                "source_registry": str(args.source_registry_path),
                "virus_host_db": str(args.virus_host_db_path),
                "ncbi_virus_report": str(args.ncbi_virus_report_path),
                "ncbi_biosample": str(args.ncbi_biosample_path),
            },
            "input_hashes_sha256": {
                "source_registry": _hash_path(args.source_registry_path),
                **input_hashes,
            },
            "output_paths": {
                "combined": str(combined_path),
                "summary": str(summary_path),
                **output_paths,
            },
            "registry_confidence_tiers": {
                source_id: registry_rows[source_id].get("confidence_tier", "") for source_id in active_sources
            },
        },
    )


if __name__ == "__main__":
    main()
