"""Unit tests for TI03 VHRdb download and ingest."""

from __future__ import annotations

import csv
import json
from urllib.error import URLError

import pytest

from lyzortx.pipeline.track_i.steps.build_tier_a_vhrdb_ingest import build_summary_rows
from lyzortx.pipeline.track_i.steps.build_tier_a_vhrdb_ingest import build_vhrdb_rows
from lyzortx.pipeline.track_i.steps.build_tier_a_vhrdb_ingest import main


def _sample_payloads() -> dict[str, object]:
    return {
        "global_response": [
            {"name": "NOT MAPPED YET", "value": -1000000.0},
            {"name": "No infection", "value": 0.0},
            {"name": "Intermediate", "value": 1.0},
            {"name": "Infection", "value": 2.0},
        ],
        "data_source": [
            {"id": 70, "name": "Source 70", "public": True},
            {"id": 71, "name": "Source 71", "public": True},
            {"id": 72, "name": "Private Source", "public": False},
        ],
        "virus": [
            {"id": 10, "name": "T4", "identifier": "NC_000866"},
        ],
        "host": [
            {"id": 20, "name": "Escherichia coli CFT073", "identifier": "83334"},
            {"id": 21, "name": "Escherichia coli K-12", "identifier": "562"},
        ],
        "responses": {
            "10": {
                "20": {"70": 2.0, "71": 0.0, "72": 1.0},
                "21": {"70": 2.0},
            }
        },
        "aggregated_responses": {
            "10": {
                "20": {"val": 1.0, "diff": 2},
                "21": {"val": -1000000.0, "diff": 1},
            }
        },
    }


def test_build_vhrdb_rows_preserves_source_fidelity_fields() -> None:
    rows = build_vhrdb_rows(_sample_payloads())

    assert len(rows) == 2
    assert {row["source_datasource_id"] for row in rows} == {"70", "71"}
    assert {row["datasource_response"] for row in rows} == {"Infection", "No infection"}

    first_row = rows[0]
    assert first_row["pair_id"] == "Escherichia coli CFT073__T4"
    assert first_row["bacteria"] == "Escherichia coli CFT073"
    assert first_row["phage"] == "T4"
    assert first_row["label_hard_any_lysis"] == "1"
    assert first_row["label_strict_confidence_tier"] == "A"
    assert first_row["source_system"] == "vhrdb"
    assert first_row["global_response"] == "Intermediate"
    assert first_row["source_disagreement_flag"] == "1"
    assert first_row["source_native_record_id"] == "10:20:70"
    assert first_row["source_host_id"] == "20"
    assert first_row["source_virus_id"] == "10"
    assert first_row["source_datasource_name"] == "Source 70"
    assert first_row["source_host_identifier"] == "83334"
    assert first_row["source_virus_identifier"] == "NC_000866"
    assert first_row["source_global_response_value"] == "1"
    assert first_row["source_datasource_response_value"] == "2"


def test_build_vhrdb_rows_skips_not_mapped_global_pairs_and_private_sources() -> None:
    rows = build_vhrdb_rows(_sample_payloads())

    assert all(row["source_native_record_id"].startswith("10:20:") for row in rows)
    assert all(row["source_datasource_name"] != "Private Source" for row in rows)


def test_build_summary_rows_counts_rows_by_slice() -> None:
    summary_rows = build_summary_rows(build_vhrdb_rows(_sample_payloads()))

    disagreement_summary = next(
        row for row in summary_rows if row["slice_type"] == "source_disagreement_flag" and row["slice_value"] == "1"
    )
    assert disagreement_summary["row_count"] == 2


def test_main_writes_vhrdb_artifacts(tmp_path, monkeypatch) -> None:
    payloads = _sample_payloads()

    def fake_download(output_dir, *, endpoint_urls=None, downloader=None):
        raw_dir = output_dir / "raw_vhrdb_downloads"
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, payload in payloads.items():
            path = raw_dir / f"{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[name] = path
        return paths

    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_vhrdb_ingest.download_vhrdb_artifacts",
        fake_download,
    )

    main(["--output-dir", str(tmp_path)])

    pairs_path = tmp_path / "ti03_vhrdb_ingested_pairs.csv"
    summary_path = tmp_path / "ti03_vhrdb_ingest_summary.csv"
    manifest_path = tmp_path / "ti03_vhrdb_ingest_manifest.json"

    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["global_response"] == "Intermediate"
    assert rows[0]["datasource_response"] in {"Infection", "No infection"}

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert any(row["slice_type"] == "global_response" for row in summary_rows)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_tier_a_vhrdb_ingest"
    assert manifest["row_count"] == 2
    assert manifest["pair_count"] == 1


def test_main_propagates_download_failures(tmp_path, monkeypatch) -> None:
    def fake_download(output_dir, *, endpoint_urls=None, downloader=None):
        raise URLError("boom")

    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_vhrdb_ingest.download_vhrdb_artifacts",
        fake_download,
    )

    with pytest.raises(URLError, match="boom"):
        main(["--output-dir", str(tmp_path)])
