#!/usr/bin/env python3
"""TI06: Download and ingest Tier B weak labels from Virus-Host DB and NCBI Virus/BioSample metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows

LOGGER = logging.getLogger(__name__)

REQUIRED_SOURCE_REGISTRY_COLUMNS = ("source_id", "confidence_tier", "confidence_basis", "notes")

VIRUS_HOST_DB_SOURCE_ID = "virus_host_db"
NCBI_SOURCE_ID = "ncbi_virus_biosample"

DEFAULT_VIRUS_HOST_DB_URL = "https://www.genome.jp/ftp/db/virushostdb/virushostdb.tsv"
DEFAULT_NCBI_QUERY = "viruses[filter] AND phage[TITL] AND srcdb_refseq[PROP] AND biosample[PROP]"
DEFAULT_NCBI_RETMAX = 500
DEFAULT_ENTREZ_BATCH_SIZE = 100
DEFAULT_ENTREZ_TOOL = "codex_ti06_ingest"
DEFAULT_ENTREZ_EMAIL = "codex@example.com"
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_MIN_REQUEST_INTERVAL_SECONDS = 0.34

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

RAW_DOWNLOAD_FILENAMES = {
    "virus_host_db": "virushostdb.tsv",
    "ncbi_search": "ncbi_nuccore_esearch.json",
    "ncbi_nuccore_xml": "ncbi_nuccore.xml",
    "ncbi_virus_report": "ncbi_virus_report.jsonl",
    "ncbi_biosample_xml": "ncbi_biosample.xml",
}

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
    "source_biosample_host_disease",
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
        "--virus-host-db-url",
        default=DEFAULT_VIRUS_HOST_DB_URL,
    )
    parser.add_argument(
        "--ncbi-query",
        default=DEFAULT_NCBI_QUERY,
        help=(
            "Entrez nuccore query used to build the NCBI Virus/BioSample weak-label cohort. "
            "The default stays bounded and reproducible to avoid an unbounded crawl in CI."
        ),
    )
    parser.add_argument(
        "--ncbi-retmax",
        type=int,
        default=DEFAULT_NCBI_RETMAX,
    )
    parser.add_argument(
        "--entrez-batch-size",
        type=int,
        default=DEFAULT_ENTREZ_BATCH_SIZE,
    )
    parser.add_argument(
        "--entrez-tool",
        default=DEFAULT_ENTREZ_TOOL,
    )
    parser.add_argument(
        "--entrez-email",
        default=DEFAULT_ENTREZ_EMAIL,
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


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url_to_path(url: str, path: Path) -> None:
    ensure_directory(path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "Codex TI06 Tier B ingest/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    except Exception:
        if path.exists():
            path.unlink()
        raise


def _build_entrez_url(endpoint: str, params: Mapping[str, Any]) -> str:
    encoded = urllib.parse.urlencode({key: value for key, value in params.items() if value not in {"", None}})
    return f"{ENTREZ_BASE_URL}/{endpoint}?{encoded}"


def _sleep_for_entrez_rate_limit() -> None:
    time.sleep(ENTREZ_MIN_REQUEST_INTERVAL_SECONDS)


def _batched(values: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for index in range(0, len(values), batch_size):
        yield list(values[index : index + batch_size])


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


def _parse_entrez_esearch_ids(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("esearchresult", {})
    id_list = result.get("idlist", [])
    if not isinstance(id_list, list):
        raise ValueError(f"Unexpected Entrez esearch payload shape in {path}")
    ids = [str(value).strip() for value in id_list if str(value).strip()]
    if not ids:
        raise ValueError(f"Entrez esearch returned zero IDs for {path}")
    return ids


def _source_feature_qualifier_values(gbseq: ET.Element) -> Dict[str, List[str]]:
    qualifier_values: Dict[str, List[str]] = {}
    for feature in gbseq.findall("./GBSeq_feature-table/GBFeature"):
        feature_key = (feature.findtext("./GBFeature_key") or "").strip()
        if feature_key != "source":
            continue
        for qualifier in feature.findall("./GBFeature_quals/GBQualifier"):
            key = (qualifier.findtext("./GBQualifier_name") or "").strip()
            value = (qualifier.findtext("./GBQualifier_value") or "").strip()
            if key and value:
                qualifier_values.setdefault(key, []).append(value)
    return qualifier_values


def _first_qualifier_value(values: Mapping[str, Sequence[str]], *keys: str) -> str:
    for key in keys:
        entries = values.get(key, [])
        for entry in entries:
            if entry:
                return entry
    return ""


def _extract_taxid(gbseq: ET.Element, qualifier_values: Mapping[str, Sequence[str]]) -> str:
    for value in qualifier_values.get("db_xref", []):
        if value.startswith("taxon:"):
            return value.split(":", maxsplit=1)[1]
    for xref in gbseq.findall("./GBSeq_xrefs/GBXref"):
        if (xref.findtext("./GBXref_dbname") or "").strip() == "taxon":
            return (xref.findtext("./GBXref_id") or "").strip()
    return ""


def _extract_biosample_accession(gbseq: ET.Element) -> str:
    for xref in gbseq.findall("./GBSeq_xrefs/GBXref"):
        if (xref.findtext("./GBXref_dbname") or "").strip() == "BioSample":
            return (xref.findtext("./GBXref_id") or "").strip()
    return ""


def read_nuccore_xml(path: Path) -> List[Dict[str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    rows: List[Dict[str, str]] = []
    for gbseq in root.findall("./GBSeq"):
        qualifier_values = _source_feature_qualifier_values(gbseq)
        accession = (
            gbseq.findtext("./GBSeq_accession-version") or gbseq.findtext("./GBSeq_primary-accession") or ""
        ).strip()
        virus_name = (gbseq.findtext("./GBSeq_organism") or "").strip()
        if not accession or not virus_name:
            continue
        rows.append(
            {
                "accession": accession,
                "virus_name": virus_name,
                "title": (gbseq.findtext("./GBSeq_definition") or "").strip(),
                "host": _first_qualifier_value(qualifier_values, "host", "lab_host"),
                "biosample": _extract_biosample_accession(gbseq),
                "taxid": _extract_taxid(gbseq, qualifier_values),
                "virus_lineage": (gbseq.findtext("./GBSeq_taxonomy") or "").strip(),
                "sample_type": _first_qualifier_value(qualifier_values, "isolation_source"),
                "source_organism": _first_qualifier_value(qualifier_values, "lab_host"),
                "host_lineage": "",
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_combined_xml(output_path: Path, root_tag: str, xml_fragments: Sequence[Path]) -> None:
    combined_root = ET.Element(root_tag)
    total_children = 0
    for fragment_path in xml_fragments:
        fragment_root = ET.parse(fragment_path).getroot()
        children = list(fragment_root)
        if not children:
            raise ValueError(f"Entrez XML response was empty for {fragment_path}")
        total_children += len(children)
        for child in children:
            combined_root.append(child)
    if total_children == 0:
        raise ValueError(f"Combined Entrez XML was empty for {output_path}")
    ET.ElementTree(combined_root).write(output_path, encoding="utf-8", xml_declaration=True)


def _download_entrez_xml_batches(
    *,
    db: str,
    ids: Sequence[str],
    output_path: Path,
    batch_size: int,
    tool: str,
    email: str,
    downloader: Any = _download_url_to_path,
) -> None:
    fragment_paths: List[Path] = []
    root_tag = "GBSet" if db == "nuccore" else "BioSampleSet"
    for batch_index, batch_ids in enumerate(_batched(list(ids), batch_size)):
        fragment_path = output_path.parent / f"{output_path.stem}.batch{batch_index:04d}.xml"
        params = {
            "db": db,
            "id": ",".join(batch_ids),
            "retmode": "xml",
            "tool": tool,
            "email": email,
        }
        if db == "nuccore":
            params["rettype"] = "gb"
        url = _build_entrez_url("efetch.fcgi", params)
        LOGGER.info("Starting Entrez %s efetch batch %s (%s records)", db, batch_index + 1, len(batch_ids))
        downloader(url, fragment_path)
        LOGGER.info("Finished Entrez %s efetch batch %s -> %s", db, batch_index + 1, fragment_path)
        fragment_paths.append(fragment_path)
        _sleep_for_entrez_rate_limit()
    _write_combined_xml(output_path, root_tag, fragment_paths)
    for fragment_path in fragment_paths:
        fragment_path.unlink(missing_ok=True)


def download_virus_host_db_artifact(
    raw_download_dir: Path,
    *,
    url: str = DEFAULT_VIRUS_HOST_DB_URL,
    downloader: Any = _download_url_to_path,
) -> Path:
    ensure_directory(raw_download_dir)
    output_path = raw_download_dir / RAW_DOWNLOAD_FILENAMES["virus_host_db"]
    LOGGER.info("Starting Virus-Host DB download: %s", url)
    downloader(url, output_path)
    LOGGER.info("Finished Virus-Host DB download: %s", output_path)
    rows = read_delimited_rows(output_path, delimiter="\t")
    require_columns(rows, output_path, VIRUS_HOST_DB_REQUIRED_COLUMNS)
    return output_path


def download_ncbi_artifacts(
    raw_download_dir: Path,
    *,
    query: str = DEFAULT_NCBI_QUERY,
    retmax: int = DEFAULT_NCBI_RETMAX,
    batch_size: int = DEFAULT_ENTREZ_BATCH_SIZE,
    tool: str = DEFAULT_ENTREZ_TOOL,
    email: str = DEFAULT_ENTREZ_EMAIL,
    downloader: Any = _download_url_to_path,
) -> Dict[str, Path]:
    if retmax <= 0:
        raise ValueError("NCBI retmax must be positive")

    ensure_directory(raw_download_dir)
    search_path = raw_download_dir / RAW_DOWNLOAD_FILENAMES["ncbi_search"]
    nuccore_xml_path = raw_download_dir / RAW_DOWNLOAD_FILENAMES["ncbi_nuccore_xml"]
    virus_report_path = raw_download_dir / RAW_DOWNLOAD_FILENAMES["ncbi_virus_report"]
    biosample_xml_path = raw_download_dir / RAW_DOWNLOAD_FILENAMES["ncbi_biosample_xml"]

    search_url = _build_entrez_url(
        "esearch.fcgi",
        {
            "db": "nuccore",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "tool": tool,
            "email": email,
        },
    )
    LOGGER.info("Starting Entrez nuccore esearch: %s", query)
    downloader(search_url, search_path)
    LOGGER.info("Finished Entrez nuccore esearch: %s", search_path)

    nuccore_ids = _parse_entrez_esearch_ids(search_path)
    _download_entrez_xml_batches(
        db="nuccore",
        ids=nuccore_ids,
        output_path=nuccore_xml_path,
        batch_size=batch_size,
        tool=tool,
        email=email,
        downloader=downloader,
    )

    virus_rows = read_nuccore_xml(nuccore_xml_path)
    if not virus_rows:
        raise ValueError("Entrez nuccore efetch produced zero virus metadata rows")
    write_jsonl(virus_report_path, virus_rows)

    biosample_accessions = sorted({row["biosample"] for row in virus_rows if row.get("biosample")})
    if not biosample_accessions:
        raise ValueError("Entrez nuccore metadata contained zero BioSample accessions")

    _download_entrez_xml_batches(
        db="biosample",
        ids=biosample_accessions,
        output_path=biosample_xml_path,
        batch_size=batch_size,
        tool=tool,
        email=email,
        downloader=downloader,
    )

    biosample_rows = read_biosample_xml(biosample_xml_path)
    if not biosample_rows:
        raise ValueError("Entrez BioSample efetch produced zero BioSample rows")

    return {
        "ncbi_search": search_path,
        "ncbi_nuccore_xml": nuccore_xml_path,
        "ncbi_virus_report": virus_report_path,
        "ncbi_biosample_xml": biosample_xml_path,
    }


def download_ti06_artifacts(
    raw_download_dir: Path,
    *,
    virus_host_db_url: str = DEFAULT_VIRUS_HOST_DB_URL,
    ncbi_query: str = DEFAULT_NCBI_QUERY,
    ncbi_retmax: int = DEFAULT_NCBI_RETMAX,
    entrez_batch_size: int = DEFAULT_ENTREZ_BATCH_SIZE,
    entrez_tool: str = DEFAULT_ENTREZ_TOOL,
    entrez_email: str = DEFAULT_ENTREZ_EMAIL,
    downloader: Any = _download_url_to_path,
) -> Dict[str, Path]:
    virus_host_db_path = download_virus_host_db_artifact(
        raw_download_dir,
        url=virus_host_db_url,
        downloader=downloader,
    )
    ncbi_paths = download_ncbi_artifacts(
        raw_download_dir,
        query=ncbi_query,
        retmax=ncbi_retmax,
        batch_size=entrez_batch_size,
        tool=entrez_tool,
        email=entrez_email,
        downloader=downloader,
    )
    return {
        "virus_host_db": virus_host_db_path,
        **ncbi_paths,
    }


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
                "source_biosample_host_disease": _first_non_blank(
                    biosample_row,
                    ("biosample_host_disease", "host_disease"),
                    default="",
                ),
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
        normalized = normalize_virus_host_db_rows(
            rows, registry_row=registry_row, bacteria_index=bacteria_index, phage_index=phage_index
        )
        if not normalized:
            raise ValueError("Virus-Host DB ingest produced zero rows")
        return normalized

    virus_rows = read_source_rows(ncbi_virus_report_path)
    if not virus_rows:
        raise ValueError("NCBI virus metadata report was empty")
    biosample_lookup = _biosample_lookup_rows(ncbi_biosample_path)
    if not biosample_lookup:
        raise ValueError("NCBI BioSample metadata was empty")
    normalized = normalize_ncbi_virus_rows(
        virus_rows,
        registry_row=registry_row,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
        biosample_lookup=biosample_lookup,
    )
    if not normalized:
        raise ValueError("NCBI Virus/BioSample ingest produced zero rows")
    return normalized


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

    raw_download_dir = args.output_dir / "raw_ti06_downloads"
    downloaded_paths = download_ti06_artifacts(
        raw_download_dir,
        virus_host_db_url=args.virus_host_db_url,
        ncbi_query=args.ncbi_query,
        ncbi_retmax=args.ncbi_retmax,
        entrez_batch_size=args.entrez_batch_size,
        entrez_tool=args.entrez_tool,
        entrez_email=args.entrez_email,
    )

    merged_rows: List[Dict[str, str]] = []

    virus_host_db_rows = build_source_rows(
        source_id=VIRUS_HOST_DB_SOURCE_ID,
        registry_rows=registry_rows,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
        virus_host_db_path=downloaded_paths["virus_host_db"],
        ncbi_virus_report_path=downloaded_paths["ncbi_virus_report"],
        ncbi_biosample_path=downloaded_paths["ncbi_biosample_xml"],
    )
    merged_rows.extend(virus_host_db_rows)

    ncbi_rows = build_source_rows(
        source_id=NCBI_SOURCE_ID,
        registry_rows=registry_rows,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
        virus_host_db_path=downloaded_paths["virus_host_db"],
        ncbi_virus_report_path=downloaded_paths["ncbi_virus_report"],
        ncbi_biosample_path=downloaded_paths["ncbi_biosample_xml"],
    )
    merged_rows.extend(ncbi_rows)

    combined_path = args.output_dir / "ti06_weak_label_ingested_pairs.csv"
    summary_path = args.output_dir / "ti06_weak_label_summary.csv"
    manifest_path = args.output_dir / "ti06_weak_label_manifest.json"
    virus_host_db_output_path = args.output_dir / "virus_host_db_ingested_pairs.csv"
    ncbi_output_path = args.output_dir / "ncbi_virus_biosample_ingested_pairs.csv"

    write_csv(virus_host_db_output_path, fieldnames=OUTPUT_FIELDNAMES, rows=virus_host_db_rows)
    write_csv(ncbi_output_path, fieldnames=OUTPUT_FIELDNAMES, rows=ncbi_rows)
    write_csv(combined_path, fieldnames=OUTPUT_FIELDNAMES, rows=merged_rows)
    write_csv(
        summary_path, fieldnames=["slice_type", "slice_value", "row_count"], rows=compute_summary_rows(merged_rows)
    )
    write_json(
        manifest_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_tier_b_weak_label_ingest",
            "active_sources": [VIRUS_HOST_DB_SOURCE_ID, NCBI_SOURCE_ID],
            "input_paths": {
                "source_registry": str(args.source_registry_path),
                **{key: str(path) for key, path in downloaded_paths.items()},
            },
            "input_hashes_sha256": {
                "source_registry": _hash_path(args.source_registry_path),
                **{key: _hash_path(path) for key, path in downloaded_paths.items()},
            },
            "output_paths": {
                "combined": str(combined_path),
                "summary": str(summary_path),
                VIRUS_HOST_DB_SOURCE_ID: str(virus_host_db_output_path),
                NCBI_SOURCE_ID: str(ncbi_output_path),
            },
            "registry_confidence_tiers": {
                source_id: registry_rows[source_id].get("confidence_tier", "")
                for source_id in (VIRUS_HOST_DB_SOURCE_ID, NCBI_SOURCE_ID)
            },
            "ncbi_query": args.ncbi_query,
            "ncbi_retmax": args.ncbi_retmax,
            "row_counts": {
                VIRUS_HOST_DB_SOURCE_ID: len(virus_host_db_rows),
                NCBI_SOURCE_ID: len(ncbi_rows),
                "combined": len(merged_rows),
            },
        },
    )


if __name__ == "__main__":
    main()
