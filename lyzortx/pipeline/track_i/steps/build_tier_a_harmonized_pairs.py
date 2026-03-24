#!/usr/bin/env python3
"""TI05: Harmonize Tier A external datasets to the internal canonical schema."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.track_i.steps.build_tier_b_weak_label_ingest import CanonicalResolutionIndex
from lyzortx.pipeline.track_i.steps.build_tier_b_weak_label_ingest import build_canonical_resolution_index
from lyzortx.pipeline.track_i.steps.build_tier_b_weak_label_ingest import resolve_canonical_name

LOGGER = logging.getLogger(__name__)

REQUIRED_SOURCE_REGISTRY_COLUMNS = ("source_id", "confidence_tier")
REQUIRED_TIER_A_COLUMNS = (
    "pair_id",
    "bacteria",
    "phage",
    "label_hard_any_lysis",
    "label_strict_confidence_tier",
    "source_system",
)
TIER_A_SOURCE_ORDER = ("vhrdb", "basel", "klebphacol", "gpb")
DEFAULT_SOURCE_PATHS = {
    "vhrdb": Path("lyzortx/generated_outputs/track_i/tier_a_ingest/ti03_vhrdb_ingested_pairs.csv"),
    "basel": Path("lyzortx/generated_outputs/track_i/tier_a_ingest/ti04_basel_ingested_pairs.csv"),
    "klebphacol": Path("lyzortx/generated_outputs/track_i/tier_a_ingest/ti04_klebphacol_ingested_pairs.csv"),
    "gpb": Path("lyzortx/generated_outputs/track_i/tier_a_ingest/ti04_gpb_ingested_pairs.csv"),
}
SUMMARY_FIELDNAMES = ["slice_type", "slice_value", "row_count", "pair_count"]
OUTPUT_APPEND_COLUMNS = [
    "bacteria_id",
    "phage_id",
    "source_pair_id_raw",
    "source_bacteria_raw",
    "source_phage_raw",
    "source_bacteria_resolution",
    "source_phage_resolution",
    "source_resolution_status",
    "internal_panel_bacteria_flag",
    "internal_panel_phage_flag",
    "internal_panel_pair_flag",
    "panel_membership",
]
PANEL_OVERLAP_VALUE = "overlap_internal_panel"
PANEL_NOVEL_VALUE = "novel_to_internal_panel"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/source_registry.csv"),
    )
    parser.add_argument(
        "--vhrdb-path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["vhrdb"],
    )
    parser.add_argument(
        "--basel-path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["basel"],
    )
    parser.add_argument(
        "--klebphacol-path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["klebphacol"],
    )
    parser.add_argument(
        "--gpb-path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["gpb"],
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
        default=Path("lyzortx/generated_outputs/track_i/tier_a_harmonization"),
    )
    return parser.parse_args(argv)


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_registry(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path, REQUIRED_SOURCE_REGISTRY_COLUMNS)
    return {row["source_id"]: row for row in rows}


def required_source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    source_paths = {
        "vhrdb": args.vhrdb_path,
        "basel": args.basel_path,
        "klebphacol": args.klebphacol_path,
        "gpb": args.gpb_path,
    }
    missing_sources = [source_id for source_id, path in source_paths.items() if not path.exists()]
    if missing_sources:
        missing_details = ", ".join(f"{source_id}={source_paths[source_id]}" for source_id in missing_sources)
        raise FileNotFoundError(
            f"TI05 requires all Tier A ingest outputs from TI03/TI04 before harmonization. Missing: {missing_details}"
        )
    return source_paths


def load_tier_a_rows(
    source_paths: Mapping[str, Path],
    *,
    source_registry_rows: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for source_id in TIER_A_SOURCE_ORDER:
        if source_id not in source_registry_rows:
            raise ValueError(f"Tier A source {source_id!r} is missing from the source registry")
        if source_registry_rows[source_id].get("confidence_tier", "") != "A":
            raise ValueError(f"Source {source_id!r} must be confidence_tier=A for TI05 harmonization")
        source_rows = read_csv_rows(source_paths[source_id], REQUIRED_TIER_A_COLUMNS)
        for row in source_rows:
            if row["source_system"] != source_id:
                raise ValueError(
                    f"Expected source_system={source_id!r} in {source_paths[source_id]}, got {row['source_system']!r}"
                )
        rows.extend(source_rows)
    if not rows:
        raise ValueError("Tier A harmonization received zero input rows")
    return rows


def _combined_resolution_status(bacteria_status: str, phage_status: str) -> str:
    tokens = [
        f"bacteria_{bacteria_status}",
        f"phage_{phage_status}",
    ]
    if "missing" in {bacteria_status, phage_status}:
        tokens.append("missing")
    elif "unresolved" in {bacteria_status, phage_status}:
        tokens.append("unresolved")
    elif "resolved_via_alias" in {bacteria_status, phage_status}:
        tokens.append("resolved_via_alias")
    else:
        tokens.append("resolved")
    return "|".join(tokens)


def harmonize_tier_a_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    bacteria_index: CanonicalResolutionIndex,
    phage_index: CanonicalResolutionIndex,
) -> List[Dict[str, str]]:
    if not bacteria_index.canonical_to_id:
        raise ValueError("Track A bacteria id map produced zero canonical bacteria")
    if not phage_index.canonical_to_id:
        raise ValueError("Track A phage id map produced zero canonical phages")

    harmonized_rows: List[Dict[str, str]] = []
    joinable_count = 0
    for row in rows:
        bacteria_raw = row["bacteria"]
        phage_raw = row["phage"]
        bacteria_name, bacteria_id, bacteria_status = resolve_canonical_name(bacteria_raw, bacteria_index)
        phage_name, phage_id, phage_status = resolve_canonical_name(phage_raw, phage_index)
        harmonized_bacteria = bacteria_name or bacteria_raw
        harmonized_phage = phage_name or phage_raw
        internal_panel_bacteria_flag = "1" if bacteria_id else "0"
        internal_panel_phage_flag = "1" if phage_id else "0"
        internal_panel_pair_flag = (
            "1" if internal_panel_bacteria_flag == "1" and internal_panel_phage_flag == "1" else "0"
        )
        if internal_panel_pair_flag == "1":
            joinable_count += 1

        harmonized_row = dict(row)
        harmonized_row.update(
            {
                "pair_id": f"{harmonized_bacteria}__{harmonized_phage}",
                "bacteria": harmonized_bacteria,
                "bacteria_id": bacteria_id,
                "phage": harmonized_phage,
                "phage_id": phage_id,
                "source_pair_id_raw": row.get("pair_id", ""),
                "source_bacteria_raw": bacteria_raw,
                "source_phage_raw": phage_raw,
                "source_bacteria_resolution": bacteria_status,
                "source_phage_resolution": phage_status,
                "source_resolution_status": _combined_resolution_status(bacteria_status, phage_status),
                "internal_panel_bacteria_flag": internal_panel_bacteria_flag,
                "internal_panel_phage_flag": internal_panel_phage_flag,
                "internal_panel_pair_flag": internal_panel_pair_flag,
                "panel_membership": (PANEL_OVERLAP_VALUE if internal_panel_pair_flag == "1" else PANEL_NOVEL_VALUE),
            }
        )
        harmonized_rows.append(harmonized_row)

    if joinable_count == 0:
        raise ValueError("Tier A harmonization produced zero joinable rows on the internal canonical pair_id grid")
    return sorted(
        harmonized_rows,
        key=lambda row: (
            row["panel_membership"],
            row["source_system"],
            row["pair_id"],
            row.get("source_native_record_id", ""),
        ),
    )


def compute_summary_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    slice_definitions = {
        "panel_membership": lambda row: row["panel_membership"],
        "source_system_and_panel_membership": lambda row: f"{row['source_system']}:{row['panel_membership']}",
        "resolution_status": lambda row: row["source_resolution_status"],
    }
    for slice_type, value_getter in slice_definitions.items():
        row_counter: Counter[str] = Counter()
        pair_counter: Dict[str, set[str]] = {}
        for row in rows:
            slice_value = value_getter(row)
            row_counter[slice_value] += 1
            pair_counter.setdefault(slice_value, set()).add(row["pair_id"])
        for slice_value, row_count in sorted(row_counter.items()):
            summary_rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "row_count": row_count,
                    "pair_count": len(pair_counter[slice_value]),
                }
            )
    return summary_rows


def ordered_fieldnames(rows: Sequence[Mapping[str, str]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for column in OUTPUT_APPEND_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    return fieldnames


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    LOGGER.info("Starting TI05 Tier A harmonization")
    source_registry_rows = read_source_registry(args.source_registry_path)
    source_paths = required_source_paths(args)
    LOGGER.info("Loading Tier A ingest outputs")
    tier_a_rows = load_tier_a_rows(source_paths, source_registry_rows=source_registry_rows)

    LOGGER.info("Loading Track A canonical bacteria and phage maps")
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

    LOGGER.info("Resolving Tier A names onto the internal canonical panel")
    harmonized_rows = harmonize_tier_a_rows(
        tier_a_rows,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
    )
    summary_rows = compute_summary_rows(harmonized_rows)

    pairs_output_path = args.output_dir / "ti05_tier_a_harmonized_pairs.csv"
    summary_output_path = args.output_dir / "ti05_tier_a_harmonization_summary.csv"
    manifest_output_path = args.output_dir / "ti05_tier_a_harmonization_manifest.json"

    write_csv(pairs_output_path, fieldnames=ordered_fieldnames(harmonized_rows), rows=harmonized_rows)
    write_csv(summary_output_path, fieldnames=SUMMARY_FIELDNAMES, rows=summary_rows)
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_tier_a_harmonized_pairs",
            "input_paths": {
                "source_registry": str(args.source_registry_path),
                **{source_id: str(path) for source_id, path in source_paths.items()},
                "track_a_bacteria_id_map": str(args.track_a_bacteria_id_map_path),
                "track_a_bacteria_alias_resolution": str(args.track_a_bacteria_alias_path),
                "track_a_phage_id_map": str(args.track_a_phage_id_map_path),
                "track_a_phage_alias_resolution": str(args.track_a_phage_alias_path),
            },
            "input_hashes_sha256": {
                "source_registry": _hash_path(args.source_registry_path),
                **{source_id: _hash_path(path) for source_id, path in source_paths.items()},
                "track_a_bacteria_id_map": _hash_path(args.track_a_bacteria_id_map_path),
                "track_a_bacteria_alias_resolution": _hash_path(args.track_a_bacteria_alias_path),
                "track_a_phage_id_map": _hash_path(args.track_a_phage_id_map_path),
                "track_a_phage_alias_resolution": _hash_path(args.track_a_phage_alias_path),
            },
            "output_paths": {
                "pairs": str(pairs_output_path),
                "summary": str(summary_output_path),
            },
            "internal_panel_bacteria_count": len(bacteria_index.canonical_to_id),
            "internal_panel_phage_count": len(phage_index.canonical_to_id),
            "row_count": len(harmonized_rows),
            "pair_count": len({row["pair_id"] for row in harmonized_rows}),
            "joinable_row_count": sum(row["internal_panel_pair_flag"] == "1" for row in harmonized_rows),
            "joinable_pair_count": len(
                {row["pair_id"] for row in harmonized_rows if row["internal_panel_pair_flag"] == "1"}
            ),
            "novel_row_count": sum(row["panel_membership"] == PANEL_NOVEL_VALUE for row in harmonized_rows),
            "novel_pair_count": len(
                {row["pair_id"] for row in harmonized_rows if row["panel_membership"] == PANEL_NOVEL_VALUE}
            ),
        },
    )
    LOGGER.info(
        "Finished TI05 Tier A harmonization with %s rows and %s joinable internal-panel rows",
        len(harmonized_rows),
        sum(row["internal_panel_pair_flag"] == "1" for row in harmonized_rows),
    )


if __name__ == "__main__":
    main()
