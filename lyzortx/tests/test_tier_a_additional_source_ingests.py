"""Unit tests for TI04 BASEL, KlebPhaCol, and GPB Tier A ingests."""

from __future__ import annotations

import csv
import json
from urllib.error import URLError

import pytest

from lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests import build_basel_rows
from lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests import build_gpb_rows
from lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests import build_klebphacol_rows
from lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests import main


def test_build_basel_rows_aggregates_binary_host_range_by_phage_block() -> None:
    sheet_rows = [
        {
            "_row_number": "10",
            "Y": "E. coli UTI89",
            "Z": "E. coli CFT073",
            "AA": "E. coli 55989",
            "AB": "S. e. Typhimurium 12023s",
            "AC": "S. e. Typhimurium SL1344",
            "AD": "E. coli B REL606",
        },
        {"_row_number": "11", "A": "Escherichia phage AugustePiccard (Bas01)"},
        {"_row_number": "12", "B": "1", "Y": "0", "Z": "0", "AA": "0", "AB": "0", "AC": "0", "AD": "1"},
        {"_row_number": "13", "B": "2", "Y": "0", "Z": "0", "AA": "0", "AB": "0", "AC": "0", "AD": "1"},
    ]

    rows = build_basel_rows(sheet_rows)

    assert len(rows) == 6
    assert {row["source_system"] for row in rows} == {"basel"}
    rel606_row = next(row for row in rows if row["bacteria"] == "E. coli B REL606")
    assert rel606_row["phage"] == "AugustePiccard (Bas01)"
    assert rel606_row["label_hard_any_lysis"] == "1"
    assert rel606_row["global_response"] == "lysis_observed"

    uti89_row = next(row for row in rows if row["bacteria"] == "E. coli UTI89")
    assert uti89_row["label_hard_any_lysis"] == "0"
    assert uti89_row["datasource_response"] == "no_lysis_observed"


def test_build_klebphacol_rows_preserves_media_and_pair_disagreement() -> None:
    rows = build_klebphacol_rows(
        [
            {
                "phage_name": "Roth37",
                "genbank_accession": "PQ657803",
                "host_range_in_lb_media_lysis": ["KLEB12"],
                "host_range_in_lb_media_no_lysis": ["KLEB11"],
                "host_range_in_lb_media_undetermined_lysis": [],
                "host_range_in_tsb_media_lysis": [],
                "host_range_in_tsb_media_no_lysis": ["KLEB12"],
                "host_range_in_tsb_media_undetermined_lysis": ["MDRT1"],
            }
        ]
    )

    assert len(rows) == 4
    assert {row["source_system"] for row in rows} == {"klebphacol"}
    lb_positive = next(
        row
        for row in rows
        if row["bacteria"] == "KLEB12"
        and row["source_datasource_id"] == "lb_media"
        and row["global_response"] == "lysis"
    )
    assert lb_positive["label_hard_any_lysis"] == "1"
    assert lb_positive["source_disagreement_flag"] == "1"
    assert lb_positive["source_native_record_id"].startswith("PQ657803:Roth37:KLEB12:lb_media")

    tsb_undetermined = next(row for row in rows if row["bacteria"] == "MDRT1")
    assert tsb_undetermined["label_hard_any_lysis"] == ""
    assert tsb_undetermined["source_uncertainty"] == "undetermined"


def test_build_gpb_rows_expands_host_range_matrix() -> None:
    sheet_rows = [
        {"_row_number": "1", "A": "note"},
        {"_row_number": "2", "C": "Enterococcus faecalis", "D": "", "E": "Klebsiella pneumoniae"},
        {"_row_number": "3", "C": "J LDX001", "D": "J LDX005", "E": "CSXY0169"},
        {"_row_number": "4", "A": "Enterococcus faecalis phage", "B": "CPB0764", "C": "2", "D": "0", "E": "3"},
        {"_row_number": "5", "B": "CPB0867", "C": "0", "D": "0", "E": "0"},
    ]

    rows = build_gpb_rows(sheet_rows)

    assert len(rows) == 6
    assert {row["source_system"] for row in rows} == {"gpb"}
    cpb0764_kleb = next(row for row in rows if row["phage"] == "CPB0764" and "CSXY0169" in row["bacteria"])
    assert cpb0764_kleb["label_hard_any_lysis"] == "1"
    assert cpb0764_kleb["global_response"] == "infect_both_conditions"
    assert cpb0764_kleb["source_assay_context"] == "Enterococcus faecalis phage"

    cpb0867_enterococcus = next(row for row in rows if row["phage"] == "CPB0867" and "J LDX001" in row["bacteria"])
    assert cpb0867_enterococcus["label_hard_any_lysis"] == "0"
    assert cpb0867_enterococcus["source_response_code"] == "0"


def test_main_writes_ti04_source_artifacts(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw_ti04_downloads"

    def fake_download(output_dir, *, urls=None, downloader=None):
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "basel": raw_dir / "basel_s1_data.xlsx",
            "klebphacol": raw_dir / "klebphacol_site_data.json",
            "gpb": raw_dir / "gpb_moesm10.xlsx",
        }
        for path in paths.values():
            path.write_text("placeholder", encoding="utf-8")
        return paths

    def fake_load_xlsx_sheet_rows(path, sheet_name):
        if sheet_name == "raw data and calculations":
            return [
                {
                    "_row_number": "10",
                    "Y": "E. coli UTI89",
                    "Z": "E. coli CFT073",
                    "AA": "E. coli 55989",
                    "AB": "S. e. Typhimurium 12023s",
                    "AC": "S. e. Typhimurium SL1344",
                    "AD": "E. coli B REL606",
                },
                {"_row_number": "11", "A": "Escherichia phage AugustePiccard (Bas01)"},
                {"_row_number": "12", "B": "1", "Y": "0", "Z": "0", "AA": "0", "AB": "0", "AC": "0", "AD": "1"},
            ]
        if sheet_name == "figure3abfh":
            return [
                {"_row_number": "2", "C": "Enterococcus faecalis"},
                {"_row_number": "3", "C": "J LDX001"},
                {"_row_number": "4", "A": "Enterococcus faecalis phage", "B": "CPB0764", "C": "2"},
            ]
        raise AssertionError(f"Unexpected sheet request {sheet_name}")

    def fake_load_kleb_records(path):
        return [
            {
                "phage_name": "Roth37",
                "genbank_accession": "PQ657803",
                "host_range_in_lb_media_lysis": ["KLEB12"],
                "host_range_in_lb_media_no_lysis": [],
                "host_range_in_lb_media_undetermined_lysis": [],
                "host_range_in_tsb_media_lysis": [],
                "host_range_in_tsb_media_no_lysis": [],
                "host_range_in_tsb_media_undetermined_lysis": [],
            }
        ]

    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.download_ti04_artifacts",
        fake_download,
    )
    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.load_xlsx_sheet_rows",
        fake_load_xlsx_sheet_rows,
    )
    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.load_klebphacol_records",
        fake_load_kleb_records,
    )

    main(["--output-dir", str(tmp_path)])

    for source_system in ("basel", "klebphacol", "gpb"):
        pairs_path = tmp_path / f"ti04_{source_system}_ingested_pairs.csv"
        summary_path = tmp_path / f"ti04_{source_system}_ingest_summary.csv"
        manifest_path = tmp_path / f"ti04_{source_system}_ingest_manifest.json"

        with pairs_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert all(row["source_system"] == source_system for row in rows)

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        assert any(row["slice_type"] == "global_response" for row in summary_rows)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["step_name"] == "build_tier_a_additional_source_ingests"
        assert manifest["source_system"] == source_system
        assert manifest["row_count"] > 0


def test_main_propagates_download_failures(tmp_path, monkeypatch) -> None:
    def fake_download(output_dir, *, urls=None, downloader=None):
        raise URLError("boom")

    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.download_ti04_artifacts",
        fake_download,
    )

    with pytest.raises(URLError, match="boom"):
        main(["--output-dir", str(tmp_path)])


def test_main_raises_when_any_source_produces_zero_rows(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw_ti04_downloads"

    def fake_download(output_dir, *, urls=None, downloader=None):
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "basel": raw_dir / "basel_s1_data.xlsx",
            "klebphacol": raw_dir / "klebphacol_site_data.json",
            "gpb": raw_dir / "gpb_moesm10.xlsx",
        }
        for path in paths.values():
            path.write_text("placeholder", encoding="utf-8")
        return paths

    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.download_ti04_artifacts",
        fake_download,
    )
    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.load_xlsx_sheet_rows",
        lambda path, sheet_name: [],
    )
    monkeypatch.setattr(
        "lyzortx.pipeline.track_i.steps.build_tier_a_additional_source_ingests.load_klebphacol_records",
        lambda path: [],
    )

    with pytest.raises(ValueError):
        main(["--output-dir", str(tmp_path)])
