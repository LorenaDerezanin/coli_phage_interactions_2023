"""Unit tests for TI05 Tier A harmonization."""

from __future__ import annotations

import csv
import json

import pytest

from lyzortx.pipeline.track_i.steps.build_tier_a_harmonized_pairs import (
    PANEL_NOVEL_VALUE,
    PANEL_OVERLAP_VALUE,
    compute_summary_rows,
    harmonize_tier_a_rows,
    main,
)
from lyzortx.pipeline.track_i.steps.build_tier_b_weak_label_ingest import build_canonical_resolution_index


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_id_map(path, canonical_name_column, canonical_id_column, raw_names_column, rows) -> None:
    _write_csv(path, [canonical_name_column, canonical_id_column, raw_names_column], rows)


def _write_alias_map(path, rows) -> None:
    _write_csv(path, ["original_name", "canonical_name", "reason"], rows)


def _build_indexes(tmp_path):
    bacteria_id_map = tmp_path / "bacteria_id_map.csv"
    _write_id_map(
        bacteria_id_map,
        "canonical_bacteria",
        "canonical_bacteria_id",
        "raw_names",
        [
            {
                "canonical_bacteria": "Escherichia coli CFT073",
                "canonical_bacteria_id": "BAC0001",
                "raw_names": "E. coli CFT073|CFT073",
            },
            {
                "canonical_bacteria": "Escherichia coli UTI89",
                "canonical_bacteria_id": "BAC0002",
                "raw_names": "UTI89",
            },
        ],
    )
    bacteria_alias_map = tmp_path / "bacteria_alias_resolution.csv"
    _write_alias_map(
        bacteria_alias_map,
        [{"original_name": "E. coli CFT073", "canonical_name": "Escherichia coli CFT073", "reason": "manual_alias"}],
    )
    phage_id_map = tmp_path / "phage_id_map.csv"
    _write_id_map(
        phage_id_map,
        "canonical_phage",
        "canonical_phage_id",
        "raw_names",
        [
            {"canonical_phage": "T4", "canonical_phage_id": "PHG0001", "raw_names": "Enterobacteria phage T4"},
            {"canonical_phage": "AugustePiccard (Bas01)", "canonical_phage_id": "PHG0002", "raw_names": "Bas01"},
        ],
    )
    phage_alias_map = tmp_path / "phage_alias_resolution.csv"
    _write_alias_map(
        phage_alias_map,
        [{"original_name": "Enterobacteria phage T4", "canonical_name": "T4", "reason": "manual_alias"}],
    )
    bacteria_index = build_canonical_resolution_index(
        bacteria_id_map,
        bacteria_alias_map,
        canonical_name_column="canonical_bacteria",
        canonical_id_column="canonical_bacteria_id",
        raw_names_column="raw_names",
    )
    phage_index = build_canonical_resolution_index(
        phage_id_map,
        phage_alias_map,
        canonical_name_column="canonical_phage",
        canonical_id_column="canonical_phage_id",
        raw_names_column="raw_names",
    )
    return bacteria_index, phage_index


def test_harmonize_tier_a_rows_resolves_aliases_and_flags_panel_membership(tmp_path) -> None:
    bacteria_index, phage_index = _build_indexes(tmp_path)

    rows = [
        {
            "pair_id": "E. coli CFT073__Enterobacteria phage T4",
            "bacteria": "E. coli CFT073",
            "phage": "Enterobacteria phage T4",
            "label_hard_any_lysis": "1",
            "label_strict_confidence_tier": "A",
            "source_system": "vhrdb",
            "source_native_record_id": "v1",
        },
        {
            "pair_id": "Unknown host__Mystery phage",
            "bacteria": "Unknown host",
            "phage": "Mystery phage",
            "label_hard_any_lysis": "0",
            "label_strict_confidence_tier": "A",
            "source_system": "gpb",
            "source_native_record_id": "g1",
        },
    ]

    harmonized = harmonize_tier_a_rows(rows, bacteria_index=bacteria_index, phage_index=phage_index)

    overlap_row = next(row for row in harmonized if row["source_system"] == "vhrdb")
    assert overlap_row["pair_id"] == "Escherichia coli CFT073__T4"
    assert overlap_row["bacteria_id"] == "BAC0001"
    assert overlap_row["phage_id"] == "PHG0001"
    assert overlap_row["source_bacteria_raw"] == "E. coli CFT073"
    assert overlap_row["source_phage_raw"] == "Enterobacteria phage T4"
    assert overlap_row["source_resolution_status"] == (
        "bacteria_resolved_via_alias|phage_resolved_via_alias|resolved_via_alias"
    )
    assert overlap_row["internal_panel_pair_flag"] == "1"
    assert overlap_row["panel_membership"] == PANEL_OVERLAP_VALUE

    novel_row = next(row for row in harmonized if row["source_system"] == "gpb")
    assert novel_row["bacteria"] == "Unknown host"
    assert novel_row["phage"] == "Mystery phage"
    assert novel_row["bacteria_id"] == ""
    assert novel_row["phage_id"] == ""
    assert novel_row["internal_panel_pair_flag"] == "0"
    assert novel_row["panel_membership"] == PANEL_NOVEL_VALUE
    assert novel_row["source_resolution_status"] == "bacteria_unresolved|phage_unresolved|unresolved"


def test_compute_summary_rows_counts_overlap_and_novel_unique_pairs() -> None:
    summary_rows = compute_summary_rows(
        [
            {
                "pair_id": "b1__p1",
                "source_system": "vhrdb",
                "panel_membership": PANEL_OVERLAP_VALUE,
                "source_resolution_status": "resolved",
            },
            {
                "pair_id": "b1__p1",
                "source_system": "basel",
                "panel_membership": PANEL_OVERLAP_VALUE,
                "source_resolution_status": "resolved",
            },
            {
                "pair_id": "novel__p9",
                "source_system": "gpb",
                "panel_membership": PANEL_NOVEL_VALUE,
                "source_resolution_status": "unresolved",
            },
        ]
    )

    overlap = next(
        row
        for row in summary_rows
        if row["slice_type"] == "panel_membership" and row["slice_value"] == PANEL_OVERLAP_VALUE
    )
    novel = next(
        row
        for row in summary_rows
        if row["slice_type"] == "panel_membership" and row["slice_value"] == PANEL_NOVEL_VALUE
    )
    assert overlap["row_count"] == 2
    assert overlap["pair_count"] == 1
    assert novel["row_count"] == 1
    assert novel["pair_count"] == 1


def test_main_writes_harmonized_outputs(tmp_path) -> None:
    source_registry = tmp_path / "source_registry.csv"
    _write_csv(
        source_registry,
        ["source_id", "confidence_tier"],
        [
            {"source_id": "vhrdb", "confidence_tier": "A"},
            {"source_id": "basel", "confidence_tier": "A"},
            {"source_id": "klebphacol", "confidence_tier": "A"},
            {"source_id": "gpb", "confidence_tier": "A"},
        ],
    )
    bacteria_id_map = tmp_path / "bacteria_id_map.csv"
    _write_id_map(
        bacteria_id_map,
        "canonical_bacteria",
        "canonical_bacteria_id",
        "raw_names",
        [
            {
                "canonical_bacteria": "Escherichia coli CFT073",
                "canonical_bacteria_id": "BAC0001",
                "raw_names": "E. coli CFT073",
            }
        ],
    )
    phage_id_map = tmp_path / "phage_id_map.csv"
    _write_id_map(
        phage_id_map,
        "canonical_phage",
        "canonical_phage_id",
        "raw_names",
        [{"canonical_phage": "T4", "canonical_phage_id": "PHG0001", "raw_names": "Enterobacteria phage T4"}],
    )
    _write_alias_map(
        tmp_path / "bacteria_alias_resolution.csv",
        [{"original_name": "E. coli CFT073", "canonical_name": "Escherichia coli CFT073", "reason": "manual_alias"}],
    )
    _write_alias_map(
        tmp_path / "phage_alias_resolution.csv",
        [{"original_name": "Enterobacteria phage T4", "canonical_name": "T4", "reason": "manual_alias"}],
    )

    tier_a_fieldnames = [
        "pair_id",
        "bacteria",
        "phage",
        "label_hard_any_lysis",
        "label_strict_confidence_tier",
        "source_system",
        "source_native_record_id",
        "source_datasource_id",
    ]
    _write_csv(
        tmp_path / "ti03_vhrdb_ingested_pairs.csv",
        tier_a_fieldnames,
        [
            {
                "pair_id": "E. coli CFT073__Enterobacteria phage T4",
                "bacteria": "E. coli CFT073",
                "phage": "Enterobacteria phage T4",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "vhrdb",
                "source_native_record_id": "v1",
                "source_datasource_id": "d1",
            }
        ],
    )
    _write_csv(
        tmp_path / "ti04_basel_ingested_pairs.csv",
        tier_a_fieldnames,
        [
            {
                "pair_id": "Escherichia coli CFT073__T4",
                "bacteria": "Escherichia coli CFT073",
                "phage": "T4",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "basel",
                "source_native_record_id": "b1",
                "source_datasource_id": "d2",
            }
        ],
    )
    _write_csv(
        tmp_path / "ti04_klebphacol_ingested_pairs.csv",
        tier_a_fieldnames,
        [
            {
                "pair_id": "Novel host__T4",
                "bacteria": "Novel host",
                "phage": "T4",
                "label_hard_any_lysis": "0",
                "label_strict_confidence_tier": "A",
                "source_system": "klebphacol",
                "source_native_record_id": "k1",
                "source_datasource_id": "d3",
            }
        ],
    )
    _write_csv(
        tmp_path / "ti04_gpb_ingested_pairs.csv",
        tier_a_fieldnames,
        [
            {
                "pair_id": "Escherichia coli CFT073__Novel phage",
                "bacteria": "Escherichia coli CFT073",
                "phage": "Novel phage",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "gpb",
                "source_native_record_id": "g1",
                "source_datasource_id": "d4",
            }
        ],
    )

    output_dir = tmp_path / "out"
    main(
        [
            "--source-registry-path",
            str(source_registry),
            "--vhrdb-path",
            str(tmp_path / "ti03_vhrdb_ingested_pairs.csv"),
            "--basel-path",
            str(tmp_path / "ti04_basel_ingested_pairs.csv"),
            "--klebphacol-path",
            str(tmp_path / "ti04_klebphacol_ingested_pairs.csv"),
            "--gpb-path",
            str(tmp_path / "ti04_gpb_ingested_pairs.csv"),
            "--track-a-bacteria-id-map-path",
            str(bacteria_id_map),
            "--track-a-bacteria-alias-path",
            str(tmp_path / "bacteria_alias_resolution.csv"),
            "--track-a-phage-id-map-path",
            str(phage_id_map),
            "--track-a-phage-alias-path",
            str(tmp_path / "phage_alias_resolution.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti05_tier_a_harmonized_pairs.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert sum(row["internal_panel_pair_flag"] == "1" for row in rows) == 2
    assert any(row["panel_membership"] == PANEL_NOVEL_VALUE for row in rows)

    with (output_dir / "ti05_tier_a_harmonization_summary.csv").open("r", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert any(
        row["slice_type"] == "panel_membership" and row["slice_value"] == PANEL_OVERLAP_VALUE for row in summary_rows
    )

    manifest = json.loads((output_dir / "ti05_tier_a_harmonization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_tier_a_harmonized_pairs"
    assert manifest["joinable_row_count"] == 2
    assert manifest["novel_row_count"] == 2


def test_main_raises_when_harmonization_produces_zero_joinable_rows(tmp_path) -> None:
    source_registry = tmp_path / "source_registry.csv"
    _write_csv(
        source_registry,
        ["source_id", "confidence_tier"],
        [
            {"source_id": "vhrdb", "confidence_tier": "A"},
            {"source_id": "basel", "confidence_tier": "A"},
            {"source_id": "klebphacol", "confidence_tier": "A"},
            {"source_id": "gpb", "confidence_tier": "A"},
        ],
    )
    _write_id_map(
        tmp_path / "bacteria_id_map.csv",
        "canonical_bacteria",
        "canonical_bacteria_id",
        "raw_names",
        [{"canonical_bacteria": "Internal host", "canonical_bacteria_id": "BAC1", "raw_names": "Internal host"}],
    )
    _write_id_map(
        tmp_path / "phage_id_map.csv",
        "canonical_phage",
        "canonical_phage_id",
        "raw_names",
        [{"canonical_phage": "Internal phage", "canonical_phage_id": "PHG1", "raw_names": "Internal phage"}],
    )
    _write_alias_map(tmp_path / "bacteria_alias_resolution.csv", [])
    _write_alias_map(tmp_path / "phage_alias_resolution.csv", [])

    tier_a_fieldnames = [
        "pair_id",
        "bacteria",
        "phage",
        "label_hard_any_lysis",
        "label_strict_confidence_tier",
        "source_system",
    ]
    for filename, source_system in [
        ("ti03_vhrdb_ingested_pairs.csv", "vhrdb"),
        ("ti04_basel_ingested_pairs.csv", "basel"),
        ("ti04_klebphacol_ingested_pairs.csv", "klebphacol"),
        ("ti04_gpb_ingested_pairs.csv", "gpb"),
    ]:
        _write_csv(
            tmp_path / filename,
            tier_a_fieldnames,
            [
                {
                    "pair_id": "Novel host__Novel phage",
                    "bacteria": "Novel host",
                    "phage": "Novel phage",
                    "label_hard_any_lysis": "1",
                    "label_strict_confidence_tier": "A",
                    "source_system": source_system,
                }
            ],
        )

    with pytest.raises(ValueError, match="zero joinable rows"):
        main(
            [
                "--source-registry-path",
                str(source_registry),
                "--vhrdb-path",
                str(tmp_path / "ti03_vhrdb_ingested_pairs.csv"),
                "--basel-path",
                str(tmp_path / "ti04_basel_ingested_pairs.csv"),
                "--klebphacol-path",
                str(tmp_path / "ti04_klebphacol_ingested_pairs.csv"),
                "--gpb-path",
                str(tmp_path / "ti04_gpb_ingested_pairs.csv"),
                "--track-a-bacteria-id-map-path",
                str(tmp_path / "bacteria_id_map.csv"),
                "--track-a-bacteria-alias-path",
                str(tmp_path / "bacteria_alias_resolution.csv"),
                "--track-a-phage-id-map-path",
                str(tmp_path / "phage_id_map.csv"),
                "--track-a-phage-alias-path",
                str(tmp_path / "phage_alias_resolution.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
