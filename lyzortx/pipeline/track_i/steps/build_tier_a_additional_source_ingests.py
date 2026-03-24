#!/usr/bin/env python3
"""TI04: Download and ingest BASEL, KlebPhaCol, and GPB Tier A sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

SOURCE_CONFIDENCE_TIER = "A"
BASEL_SOURCE_SYSTEM = "basel"
KLEBPHACOL_SOURCE_SYSTEM = "klebphacol"
GPB_SOURCE_SYSTEM = "gpb"
BASEL_DOWNLOAD_URL = (
    "https://journals.plos.org/plosbiology/article/file?type=supplementary&id=10.1371/journal.pbio.3001424.s007"
)
KLEBPHACOL_DOWNLOAD_URL = "https://phage.klebphacol.soton.ac.uk/site-data.json"
GPB_DOWNLOAD_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-61946-0/MediaObjects/"
    "41467_2025_61946_MOESM10_ESM.xlsx"
)
RAW_DOWNLOAD_FILENAMES = {
    BASEL_SOURCE_SYSTEM: "basel_s1_data.xlsx",
    KLEBPHACOL_SOURCE_SYSTEM: "klebphacol_site_data.json",
    GPB_SOURCE_SYSTEM: "gpb_moesm10.xlsx",
}
DOWNLOAD_URLS = {
    BASEL_SOURCE_SYSTEM: BASEL_DOWNLOAD_URL,
    KLEBPHACOL_SOURCE_SYSTEM: KLEBPHACOL_DOWNLOAD_URL,
    GPB_SOURCE_SYSTEM: GPB_DOWNLOAD_URL,
}
OUTPUT_FIELDNAMES = [
    "pair_id",
    "bacteria",
    "phage",
    "label_hard_any_lysis",
    "label_strict_confidence_tier",
    "source_system",
    "global_response",
    "datasource_response",
    "source_datasource_id",
    "source_native_record_id",
    "source_disagreement_flag",
    "source_uncertainty",
    "source_strength_label",
    "source_download_url",
    "source_assay_context",
    "source_condition",
    "source_host_species",
    "source_host_strain_id",
    "source_response_code",
]
SUMMARY_FIELDNAMES = ["slice_type", "slice_value", "row_count"]
XLSX_MAIN_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
XLSX_PACKAGE_NS = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
XLSX_RELATIONSHIP_NS = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
BASEL_QUALITATIVE_HOST_COLUMNS = {
    "Y": "E. coli UTI89",
    "Z": "E. coli CFT073",
    "AA": "E. coli 55989",
    "AB": "S. e. Typhimurium 12023s",
    "AC": "S. e. Typhimurium SL1344",
    "AD": "E. coli B REL606",
}
BASEL_PHAGE_PREFIX = "Escherichia phage "
BASEL_BLOCK_PREFIX = "Escherichia phage"
BASEL_DATASOURCE_ID = "plos_s1_data"
BASEL_RESPONSE_NAME_LYSIS = "lysis_observed"
BASEL_RESPONSE_NAME_NO_LYSIS = "no_lysis_observed"
KLEB_FIELD_CONFIG = {
    "host_range_in_lb_media_lysis": ("lb_media", "lysis", "1", ""),
    "host_range_in_lb_media_no_lysis": ("lb_media", "no_lysis", "0", ""),
    "host_range_in_lb_media_undetermined_lysis": ("lb_media", "undetermined_lysis", "", "undetermined"),
    "host_range_in_tsb_media_lysis": ("tsb_media", "lysis", "1", ""),
    "host_range_in_tsb_media_no_lysis": ("tsb_media", "no_lysis", "0", ""),
    "host_range_in_tsb_media_undetermined_lysis": ("tsb_media", "undetermined_lysis", "", "undetermined"),
}
GPB_RESPONSE_BY_CODE = {
    "0": "no_infection",
    "1": "infect_anoxic_only",
    "2": "infect_oxic_only",
    "3": "infect_both_conditions",
}
GPB_DATASOURCE_ID = "figure3abfh"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/tier_a_ingest"),
    )
    return parser.parse_args(argv)


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url_to_path(url: str, path: Path) -> None:
    ensure_directory(path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "Codex TI04 Tier A ingest/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    except Exception:
        if path.exists():
            path.unlink()
        raise


def download_ti04_artifacts(
    output_dir: Path,
    *,
    urls: Mapping[str, str] = DOWNLOAD_URLS,
    downloader: Any = _download_url_to_path,
) -> Dict[str, Path]:
    raw_dir = output_dir / "raw_ti04_downloads"
    ensure_directory(raw_dir)
    downloaded_paths: Dict[str, Path] = {}
    for source_system, filename in RAW_DOWNLOAD_FILENAMES.items():
        if source_system not in urls:
            raise ValueError(f"Missing download URL for {source_system}")
        destination = raw_dir / filename
        LOGGER.info("Starting TI04 download for %s: %s", source_system, urls[source_system])
        downloader(urls[source_system], destination)
        LOGGER.info("Finished TI04 download for %s: %s", source_system, destination)
        downloaded_paths[source_system] = destination
    return downloaded_paths


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _column_letters(cell_reference: str) -> str:
    letters = []
    for character in cell_reference:
        if character.isalpha():
            letters.append(character)
            continue
        break
    return "".join(letters)


def _sheet_name_to_target(path: Path, sheet_name: str) -> str:
    with zipfile.ZipFile(path) as workbook:
        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        workbook_rels = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationship_target_by_id = {
            rel.attrib["Id"]: rel.attrib["Target"] for rel in workbook_rels.findall("p:Relationship", XLSX_PACKAGE_NS)
        }
        sheets = workbook_xml.find("a:sheets", XLSX_MAIN_NS)
        if sheets is None:
            raise ValueError(f"Workbook {path} is missing the sheets collection")
        for sheet in sheets:
            if sheet.attrib.get("name") != sheet_name:
                continue
            relationship_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            return "xl/" + relationship_target_by_id[relationship_id]
    raise ValueError(f"Workbook {path} is missing expected sheet {sheet_name!r}")


def load_xlsx_sheet_rows(path: Path, sheet_name: str) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required XLSX artifact: {path}")

    shared_strings: List[str] = []
    sheet_target = _sheet_name_to_target(path, sheet_name)
    rows: List[Dict[str, str]] = []
    with zipfile.ZipFile(path) as workbook:
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for shared_string in shared_root.findall("a:si", XLSX_MAIN_NS):
                shared_strings.append(
                    "".join(node.text or "" for node in shared_string.iter() if node.tag.endswith("}t"))
                )

        worksheet_root = ET.fromstring(workbook.read(sheet_target))
        sheet_data = worksheet_root.find("a:sheetData", XLSX_MAIN_NS)
        if sheet_data is None:
            raise ValueError(f"Worksheet {sheet_name!r} in {path} has no sheetData")

        for row in sheet_data:
            row_number = row.attrib["r"]
            parsed_row: Dict[str, str] = {"_row_number": row_number}
            for cell in row.findall("a:c", XLSX_MAIN_NS):
                column = _column_letters(cell.attrib["r"])
                value_node = cell.find("a:v", XLSX_MAIN_NS)
                if value_node is None:
                    parsed_row[column] = ""
                    continue
                cell_type = cell.attrib.get("t")
                if cell_type == "s":
                    parsed_row[column] = shared_strings[int(value_node.text)]
                else:
                    parsed_row[column] = value_node.text or ""
            rows.append(parsed_row)

    if not rows:
        raise ValueError(f"Worksheet {sheet_name!r} in {path} produced zero rows")
    return rows


def _normalize_phage_name(value: str) -> str:
    return value.removeprefix(BASEL_PHAGE_PREFIX).strip()


def build_basel_rows(sheet_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    if not sheet_rows:
        raise ValueError("BASEL sheet rows were empty")

    header_row = next((row for row in sheet_rows if row.get("_row_number") == "10"), None)
    if header_row is None:
        raise ValueError("BASEL raw worksheet is missing expected header row 10")
    for column, expected_host in BASEL_QUALITATIVE_HOST_COLUMNS.items():
        observed_host = header_row.get(column, "").strip()
        if observed_host != expected_host:
            raise ValueError(
                f"BASEL qualitative host header mismatch for {column}: expected {expected_host!r}, got {observed_host!r}"
            )

    rows: List[Dict[str, str]] = []
    current_phage_name = ""
    current_experiment_rows: List[Mapping[str, str]] = []

    def flush_current_block() -> None:
        if not current_phage_name:
            return
        phage_name = _normalize_phage_name(current_phage_name)
        if not current_experiment_rows:
            raise ValueError(f"BASEL phage block {current_phage_name!r} had no experiment rows")
        for column, host_name in BASEL_QUALITATIVE_HOST_COLUMNS.items():
            observed_values = [row.get(column, "") for row in current_experiment_rows if row.get(column, "") != ""]
            if not observed_values:
                raise ValueError(f"BASEL block {current_phage_name!r} had no qualitative values for host {host_name!r}")
            if any(value == "1" for value in observed_values):
                label = "1"
                response_name = BASEL_RESPONSE_NAME_LYSIS
            elif all(value == "0" for value in observed_values):
                label = "0"
                response_name = BASEL_RESPONSE_NAME_NO_LYSIS
            else:
                raise ValueError(
                    f"BASEL block {current_phage_name!r} has inconsistent qualitative values for host {host_name!r}: "
                    f"{observed_values}"
                )
            rows.append(
                {
                    "pair_id": f"{host_name}__{phage_name}",
                    "bacteria": host_name,
                    "phage": phage_name,
                    "label_hard_any_lysis": label,
                    "label_strict_confidence_tier": SOURCE_CONFIDENCE_TIER,
                    "source_system": BASEL_SOURCE_SYSTEM,
                    "global_response": response_name,
                    "datasource_response": response_name,
                    "source_datasource_id": BASEL_DATASOURCE_ID,
                    "source_native_record_id": f"{phage_name}:{host_name}",
                    "source_disagreement_flag": "0",
                    "source_uncertainty": "",
                    "source_strength_label": response_name,
                    "source_download_url": BASEL_DOWNLOAD_URL,
                    "source_assay_context": "qualitative_top_agar_enterobacterial_host_range",
                    "source_condition": "not_reported",
                    "source_host_species": host_name,
                    "source_host_strain_id": host_name,
                    "source_response_code": label,
                }
            )

    for row in sheet_rows:
        phage_header = row.get("A", "")
        if phage_header.startswith(BASEL_BLOCK_PREFIX):
            flush_current_block()
            current_phage_name = phage_header
            current_experiment_rows = []
            continue
        if current_phage_name and row.get("B", "").isdigit():
            current_experiment_rows.append(row)

    flush_current_block()
    if not rows:
        raise ValueError("BASEL ingest produced zero rows")
    return rows


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_klebphacol_records(path: Path) -> List[Mapping[str, Any]]:
    payload = _load_json(path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("KlebPhaCol site-data.json was missing the non-empty records list")
    return records


def _apply_pair_disagreement_flags(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    responses_by_pair: Dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["bacteria"], row["phage"])
        response_name = row.get("global_response", "").strip()
        if response_name:
            responses_by_pair[key].add(response_name)

    flagged_rows: List[Dict[str, str]] = []
    for row in rows:
        updated_row = dict(row)
        updated_row["source_disagreement_flag"] = (
            "1" if len(responses_by_pair[(row["bacteria"], row["phage"])]) > 1 else "0"
        )
        flagged_rows.append(updated_row)
    return flagged_rows


def build_klebphacol_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    if not records:
        raise ValueError("KlebPhaCol records were empty")

    rows: List[Dict[str, str]] = []
    for record in records:
        phage_name = str(record.get("phage_name", "")).strip()
        if not phage_name:
            raise ValueError(f"KlebPhaCol record missing phage_name: {record}")
        accession = str(record.get("genbank_accession", "")).strip()
        for field_name, (media, response_name, binary_label, uncertainty) in KLEB_FIELD_CONFIG.items():
            if field_name not in record:
                raise ValueError(f"KlebPhaCol record for {phage_name!r} is missing required field {field_name!r}")
            host_names = _unique_preserving_order(str(host) for host in record.get(field_name) or [])
            for host_name in host_names:
                rows.append(
                    {
                        "pair_id": f"{host_name}__{phage_name}",
                        "bacteria": host_name,
                        "phage": phage_name,
                        "label_hard_any_lysis": binary_label,
                        "label_strict_confidence_tier": SOURCE_CONFIDENCE_TIER,
                        "source_system": KLEBPHACOL_SOURCE_SYSTEM,
                        "global_response": response_name,
                        "datasource_response": response_name,
                        "source_datasource_id": media,
                        "source_native_record_id": (
                            f"{accession}:{phage_name}:{host_name}:{media}:{response_name}"
                            if accession
                            else f"{phage_name}:{host_name}:{media}:{response_name}"
                        ),
                        "source_disagreement_flag": "0",
                        "source_uncertainty": uncertainty,
                        "source_strength_label": response_name,
                        "source_download_url": KLEBPHACOL_DOWNLOAD_URL,
                        "source_assay_context": "host_range_json",
                        "source_condition": media,
                        "source_host_species": "Klebsiella spp.",
                        "source_host_strain_id": host_name,
                        "source_response_code": response_name,
                    }
                )

    if not rows:
        raise ValueError("KlebPhaCol ingest produced zero rows")
    return _apply_pair_disagreement_flags(rows)


def build_gpb_rows(sheet_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    if not sheet_rows:
        raise ValueError("GPB sheet rows were empty")

    row_by_number = {row["_row_number"]: row for row in sheet_rows if "_row_number" in row}
    species_row = row_by_number.get("2")
    strain_row = row_by_number.get("3")
    if species_row is None or strain_row is None:
        raise ValueError("GPB sheet is missing expected host header rows 2 and 3")

    host_columns: List[tuple[str, str, str]] = []
    current_species = ""
    for column, strain_id in strain_row.items():
        if column in {"_row_number", "A", "B"}:
            continue
        species_value = species_row.get(column, "").strip()
        if species_value:
            current_species = species_value
        strain_id = strain_id.strip()
        if not strain_id:
            continue
        if not current_species:
            raise ValueError(f"GPB host column {column} has a strain id without a species header")
        host_columns.append((column, current_species, strain_id))

    if not host_columns:
        raise ValueError("GPB host-range matrix produced zero host columns")

    rows: List[Dict[str, str]] = []
    current_phage_group = ""
    for row in sheet_rows:
        row_number = row.get("_row_number", "")
        if not row_number.isdigit() or int(row_number) < 4:
            continue
        phage_group = row.get("A", "").strip()
        if phage_group:
            current_phage_group = phage_group
        phage_id = row.get("B", "").strip()
        if not phage_id:
            continue
        if not current_phage_group:
            raise ValueError(f"GPB phage row {row_number} is missing both a phage group and a phage id")
        for column, species_name, strain_id in host_columns:
            response_code = row.get(column, "").strip()
            if response_code == "":
                continue
            if response_code not in GPB_RESPONSE_BY_CODE:
                raise ValueError(f"GPB row {row_number} has unknown response code {response_code!r} in column {column}")
            response_name = GPB_RESPONSE_BY_CODE[response_code]
            bacteria_name = f"{species_name} {strain_id}".strip()
            rows.append(
                {
                    "pair_id": f"{bacteria_name}__{phage_id}",
                    "bacteria": bacteria_name,
                    "phage": phage_id,
                    "label_hard_any_lysis": "0" if response_code == "0" else "1",
                    "label_strict_confidence_tier": SOURCE_CONFIDENCE_TIER,
                    "source_system": GPB_SOURCE_SYSTEM,
                    "global_response": response_name,
                    "datasource_response": response_name,
                    "source_datasource_id": GPB_DATASOURCE_ID,
                    "source_native_record_id": f"{phage_id}:{strain_id}",
                    "source_disagreement_flag": "0",
                    "source_uncertainty": "",
                    "source_strength_label": response_name,
                    "source_download_url": GPB_DOWNLOAD_URL,
                    "source_assay_context": current_phage_group,
                    "source_condition": "oxic_anoxic_matrix",
                    "source_host_species": species_name,
                    "source_host_strain_id": strain_id,
                    "source_response_code": response_code,
                }
            )

    if not rows:
        raise ValueError("GPB ingest produced zero rows")
    return rows


def build_summary_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    by_datasource = Counter(row["source_datasource_id"] for row in rows)
    by_response = Counter(row["global_response"] for row in rows)
    by_disagreement = Counter(row["source_disagreement_flag"] for row in rows)

    summary_rows: List[Dict[str, object]] = []
    for datasource_id, count in sorted(by_datasource.items()):
        summary_rows.append({"slice_type": "source_datasource_id", "slice_value": datasource_id, "row_count": count})
    for response_name, count in sorted(by_response.items()):
        summary_rows.append({"slice_type": "global_response", "slice_value": response_name, "row_count": count})
    for disagreement_flag, count in sorted(by_disagreement.items()):
        summary_rows.append(
            {
                "slice_type": "source_disagreement_flag",
                "slice_value": disagreement_flag,
                "row_count": count,
            }
        )
    return summary_rows


def _write_source_outputs(
    *,
    source_system: str,
    rows: Sequence[Dict[str, str]],
    raw_path: Path,
    output_dir: Path,
) -> None:
    if not rows:
        raise ValueError(f"{source_system} ingest produced zero rows")

    pairs_output_path = output_dir / f"ti04_{source_system}_ingested_pairs.csv"
    summary_output_path = output_dir / f"ti04_{source_system}_ingest_summary.csv"
    manifest_output_path = output_dir / f"ti04_{source_system}_ingest_manifest.json"
    summary_rows = build_summary_rows(rows)

    write_csv(pairs_output_path, fieldnames=OUTPUT_FIELDNAMES, rows=rows)
    write_csv(summary_output_path, fieldnames=SUMMARY_FIELDNAMES, rows=summary_rows)
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_tier_a_additional_source_ingests",
            "source_system": source_system,
            "download_url": DOWNLOAD_URLS[source_system],
            "raw_download_path": str(raw_path),
            "raw_download_hash_sha256": _hash_path(raw_path),
            "output_paths": {
                "pairs": str(pairs_output_path),
                "summary": str(summary_output_path),
            },
            "row_count": len(rows),
            "pair_count": len({row["pair_id"] for row in rows}),
        },
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    LOGGER.info("Starting TI04 additional Tier A source download and ingest")
    raw_paths = download_ti04_artifacts(args.output_dir)

    LOGGER.info("Loading BASEL worksheet")
    basel_rows = build_basel_rows(load_xlsx_sheet_rows(raw_paths[BASEL_SOURCE_SYSTEM], "raw data and calculations"))
    LOGGER.info("Loading KlebPhaCol site-data records")
    klebphacol_rows = build_klebphacol_rows(load_klebphacol_records(raw_paths[KLEBPHACOL_SOURCE_SYSTEM]))
    LOGGER.info("Loading GPB host-range matrix")
    gpb_rows = build_gpb_rows(load_xlsx_sheet_rows(raw_paths[GPB_SOURCE_SYSTEM], "figure3abfh"))

    _write_source_outputs(
        source_system=BASEL_SOURCE_SYSTEM,
        rows=basel_rows,
        raw_path=raw_paths[BASEL_SOURCE_SYSTEM],
        output_dir=args.output_dir,
    )
    _write_source_outputs(
        source_system=KLEBPHACOL_SOURCE_SYSTEM,
        rows=klebphacol_rows,
        raw_path=raw_paths[KLEBPHACOL_SOURCE_SYSTEM],
        output_dir=args.output_dir,
    )
    _write_source_outputs(
        source_system=GPB_SOURCE_SYSTEM,
        rows=gpb_rows,
        raw_path=raw_paths[GPB_SOURCE_SYSTEM],
        output_dir=args.output_dir,
    )
    LOGGER.info(
        "Finished TI04 additional Tier A ingests with BASEL=%s, KlebPhaCol=%s, GPB=%s rows",
        len(basel_rows),
        len(klebphacol_rows),
        len(gpb_rows),
    )


if __name__ == "__main__":
    main()
