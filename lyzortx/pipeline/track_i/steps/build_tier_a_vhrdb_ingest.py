#!/usr/bin/env python3
"""TI03: Download and ingest VHRdb pairs with provenance-preserving source fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

SOURCE_SYSTEM = "vhrdb"
SOURCE_CONFIDENCE_TIER = "A"
RAW_DOWNLOAD_FILENAMES = {
    "global_response": "vhrdb_global_response.json",
    "data_source": "vhrdb_data_source.json",
    "virus": "vhrdb_virus.json",
    "host": "vhrdb_host.json",
    "responses": "vhrdb_responses.json",
    "aggregated_responses": "vhrdb_aggregated_responses.json",
}
DEFAULT_ENDPOINT_URLS = {
    "global_response": "https://viralhostrangedb.pasteur.cloud/api/global-response/?format=json",
    "data_source": "https://viralhostrangedb.pasteur.cloud/api/data-source/?format=json",
    "virus": "https://viralhostrangedb.pasteur.cloud/api/virus/?format=json",
    "host": "https://viralhostrangedb.pasteur.cloud/api/host/?format=json",
    "responses": "https://viralhostrangedb.pasteur.cloud/api/responses/?allow_overflow=true&format=json",
    "aggregated_responses": "https://viralhostrangedb.pasteur.cloud/api/aggregated-responses/?allow_overflow=true&format=json",
}
RESPONSE_NAME_FALLBACK = "UNKNOWN_RESPONSE"
UNKNOWN_RESPONSE_NAME = "NOT MAPPED YET"
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
    "source_host_id",
    "source_virus_id",
    "source_datasource_name",
    "source_host_identifier",
    "source_virus_identifier",
    "source_global_response_value",
    "source_datasource_response_value",
]


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


def _normalize_response_value(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required VHRdb artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_name(entity: Mapping[str, Any]) -> str:
    name = str(entity.get("name", "")).strip()
    if not name:
        raise ValueError(f"Entity is missing required name: {entity}")
    return name


def _lookup_identifier(entity: Mapping[str, Any]) -> str:
    return str(entity.get("identifier", "") or "").strip()


def _binary_label_from_global_value(global_value: Any) -> str:
    return "1" if float(global_value) > 0 else "0"


def _download_url_to_path(url: str, path: Path) -> None:
    ensure_directory(path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "Codex TI03 VHRdb ingest/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    except Exception:
        if path.exists():
            path.unlink()
        raise


def download_vhrdb_artifacts(
    output_dir: Path,
    *,
    endpoint_urls: Mapping[str, str] = DEFAULT_ENDPOINT_URLS,
    downloader: Any = _download_url_to_path,
) -> Dict[str, Path]:
    raw_dir = output_dir / "raw_vhrdb_downloads"
    ensure_directory(raw_dir)
    downloaded_paths: Dict[str, Path] = {}
    for endpoint_name, filename in RAW_DOWNLOAD_FILENAMES.items():
        if endpoint_name not in endpoint_urls:
            raise ValueError(f"Missing endpoint URL for {endpoint_name}")
        url = endpoint_urls[endpoint_name]
        destination = raw_dir / filename
        LOGGER.info("Starting VHRdb download: %s", url)
        downloader(url, destination)
        LOGGER.info("Finished VHRdb download: %s", destination)
        downloaded_paths[endpoint_name] = destination
    return downloaded_paths


def load_vhrdb_payloads(raw_paths: Mapping[str, Path]) -> Dict[str, Any]:
    payloads = {name: _load_json(path) for name, path in raw_paths.items()}
    if not payloads["aggregated_responses"]:
        raise ValueError("VHRdb aggregated responses download was empty")
    if not payloads["responses"]:
        raise ValueError("VHRdb per-datasource responses download was empty")
    return payloads


def _build_id_lookup(rows: Iterable[Mapping[str, Any]], *, label: str) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"{label} row is missing id: {row}")
        lookup[row_id] = row
    return lookup


def build_vhrdb_rows(payloads: Mapping[str, Any]) -> List[Dict[str, str]]:
    response_value_to_name = {
        _normalize_response_value(row["value"]): str(row["name"]) for row in payloads["global_response"]
    }
    data_sources = _build_id_lookup(payloads["data_source"], label="Data source")
    viruses = _build_id_lookup(payloads["virus"], label="Virus")
    hosts = _build_id_lookup(payloads["host"], label="Host")
    responses = payloads["responses"]
    aggregated = payloads["aggregated_responses"]

    rows: List[Dict[str, str]] = []
    excluded_not_mapped_count = 0

    for virus_id, host_mapping in aggregated.items():
        if virus_id not in viruses:
            raise ValueError(f"Virus id {virus_id} referenced in VHRdb responses but missing from metadata")
        if virus_id not in responses:
            raise ValueError(f"Virus id {virus_id} referenced in aggregated responses but missing per-source responses")

        virus = viruses[virus_id]
        per_source_host_mapping = responses[virus_id]

        for host_id, aggregate_payload in host_mapping.items():
            if host_id not in hosts:
                raise ValueError(f"Host id {host_id} referenced in VHRdb responses but missing from metadata")
            if host_id not in per_source_host_mapping:
                raise ValueError(
                    f"Host id {host_id} for virus {virus_id} missing from per-source responses despite aggregation"
                )

            host = hosts[host_id]
            global_value = _normalize_response_value(aggregate_payload["val"])
            global_response = response_value_to_name.get(global_value, RESPONSE_NAME_FALLBACK)
            if global_response == UNKNOWN_RESPONSE_NAME:
                excluded_not_mapped_count += 1
                continue

            datasource_mapping = per_source_host_mapping[host_id]
            public_datasource_ids = [
                str(datasource_id)
                for datasource_id in datasource_mapping
                if str(datasource_id) in data_sources and bool(data_sources[str(datasource_id)].get("public", False))
            ]
            if not public_datasource_ids:
                continue

            disagreement_flag = (
                "1"
                if len(
                    {
                        _normalize_response_value(datasource_mapping[datasource_id])
                        for datasource_id in public_datasource_ids
                    }
                )
                > 1
                else "0"
            )

            for datasource_id in public_datasource_ids:
                datasource_value = datasource_mapping[datasource_id]
                if datasource_id not in data_sources:
                    raise ValueError(
                        f"Datasource id {datasource_id} for virus {virus_id} host {host_id} missing from metadata"
                    )
                datasource = data_sources[datasource_id]

                datasource_value_text = _normalize_response_value(datasource_value)
                datasource_response = response_value_to_name.get(datasource_value_text, RESPONSE_NAME_FALLBACK)
                rows.append(
                    {
                        "pair_id": f"{_lookup_name(host)}__{_lookup_name(virus)}",
                        "bacteria": _lookup_name(host),
                        "phage": _lookup_name(virus),
                        "label_hard_any_lysis": _binary_label_from_global_value(global_value),
                        "label_strict_confidence_tier": SOURCE_CONFIDENCE_TIER,
                        "source_system": SOURCE_SYSTEM,
                        "global_response": global_response,
                        "datasource_response": datasource_response,
                        "source_datasource_id": str(datasource_id),
                        "source_native_record_id": f"{virus_id}:{host_id}:{datasource_id}",
                        "source_disagreement_flag": disagreement_flag,
                        "source_uncertainty": "",
                        "source_host_id": str(host_id),
                        "source_virus_id": str(virus_id),
                        "source_datasource_name": _lookup_name(datasource),
                        "source_host_identifier": _lookup_identifier(host),
                        "source_virus_identifier": _lookup_identifier(virus),
                        "source_global_response_value": global_value,
                        "source_datasource_response_value": datasource_value_text,
                    }
                )

    if not rows:
        raise ValueError("VHRdb ingest produced zero rows")

    LOGGER.info("Excluded %s VHRdb virus-host pairs with NOT MAPPED YET global response", excluded_not_mapped_count)
    return sorted(
        rows,
        key=lambda row: (
            row["bacteria"],
            row["phage"],
            row["source_datasource_id"],
            row["source_native_record_id"],
        ),
    )


def build_summary_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    by_source = Counter(row["source_datasource_id"] for row in rows)
    by_global_response = Counter(row["global_response"] for row in rows)
    by_disagreement = Counter(row["source_disagreement_flag"] for row in rows)
    summary_rows: List[Dict[str, object]] = []
    for datasource_id, count in sorted(by_source.items()):
        summary_rows.append({"slice_type": "source_datasource_id", "slice_value": datasource_id, "row_count": count})
    for response_name, count in sorted(by_global_response.items()):
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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    LOGGER.info("Starting TI03 VHRdb download and ingest")
    raw_paths = download_vhrdb_artifacts(args.output_dir)

    LOGGER.info("Loading downloaded VHRdb artifacts")
    payloads = load_vhrdb_payloads(raw_paths)

    LOGGER.info("Normalizing VHRdb responses into Track I Tier A schema")
    rows = build_vhrdb_rows(payloads)
    summary_rows = build_summary_rows(rows)

    pairs_output_path = args.output_dir / "ti03_vhrdb_ingested_pairs.csv"
    summary_output_path = args.output_dir / "ti03_vhrdb_ingest_summary.csv"
    manifest_output_path = args.output_dir / "ti03_vhrdb_ingest_manifest.json"

    write_csv(pairs_output_path, fieldnames=OUTPUT_FIELDNAMES, rows=rows)
    write_csv(
        summary_output_path,
        fieldnames=["slice_type", "slice_value", "row_count"],
        rows=summary_rows,
    )
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_tier_a_vhrdb_ingest",
            "download_urls": DEFAULT_ENDPOINT_URLS,
            "raw_download_paths": {name: str(path) for name, path in raw_paths.items()},
            "raw_download_hashes_sha256": {name: _hash_path(path) for name, path in raw_paths.items()},
            "output_paths": {
                "pairs": str(pairs_output_path),
                "summary": str(summary_output_path),
            },
            "row_count": len(rows),
            "pair_count": len({row["pair_id"] for row in rows}),
        },
    )
    LOGGER.info("Finished TI03 VHRdb ingest with %s rows", len(rows))


if __name__ == "__main__":
    main()
