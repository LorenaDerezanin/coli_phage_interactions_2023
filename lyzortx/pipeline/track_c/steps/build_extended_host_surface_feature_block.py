#!/usr/bin/env python3
"""Build extended Track C host surface features from capsule, LPS, and UMAP inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

UMAP_COLUMNS: Sequence[str] = tuple(f"UMAP{index}" for index in range(8))
FEATURE_COLUMNS: Sequence[str] = (
    "bacteria",
    "host_surface_klebsiella_capsule_type",
    "host_surface_klebsiella_capsule_type_missing",
    "host_surface_lps_core_type",
    "host_phylogeny_umap_00",
    "host_phylogeny_umap_01",
    "host_phylogeny_umap_02",
    "host_phylogeny_umap_03",
    "host_phylogeny_umap_04",
    "host_phylogeny_umap_05",
    "host_phylogeny_umap_06",
    "host_phylogeny_umap_07",
)
METADATA_COLUMNS: Sequence[str] = (
    "column_name",
    "feature_group",
    "feature_type",
    "source_path",
    "source_column",
    "transform",
    "missing_count",
    "missing_rate",
    "provenance_note",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--umap-path",
        type=Path,
        default=Path("data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv"),
        help="Tab-delimited 8D host UMAP embedding table.",
    )
    parser.add_argument(
        "--capsule-path",
        type=Path,
        default=Path("data/genomics/bacteria/capsules/klebsiella_capsules/kaptive_results_high_hits_cured.txt"),
        help="Tab-delimited Klebsiella capsule typing table.",
    )
    parser.add_argument(
        "--lps-primary-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt"),
        help="Primary tab-delimited LPS core annotation table.",
    )
    parser.add_argument(
        "--lps-supplemental-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_host.txt"),
        help="Supplemental tab-delimited LPS core annotation table for the non-370 host subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/extended_host_surface_feature_block"),
        help="Directory for generated Track C extended host surface artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and the manifest.",
    )
    parser.add_argument(
        "--expected-host-count",
        type=int,
        default=404,
        help="Expected number of strains in the output host feature matrix.",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_category(value: str) -> str:
    normalized = value.strip()
    if normalized in {"", "-", "Unknown"}:
        return ""
    return normalized


def _index_unique(rows: Sequence[Mapping[str, str]], key: str, *, path: Path) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            continue
        if value in index:
            raise ValueError(f"Duplicate {key} value {value!r} in {path}")
        index[value] = dict(row)
    return index


def build_capsule_index(capsule_rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    capsule_index = _index_unique(capsule_rows, "bacteria", path=Path("<in-memory capsule rows>"))
    return {
        bacteria: {"klebsiella_capsule_type": _normalize_category(row.get("Klebs_capsule_type", ""))}
        for bacteria, row in capsule_index.items()
    }


def build_lps_core_index(
    primary_rows: Sequence[Mapping[str, str]],
    supplemental_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for row in primary_rows:
        bacteria = row.get("bacteria", "").strip()
        if not bacteria:
            continue
        lps_core_type = _normalize_category(row.get("LPS_type", ""))
        if not lps_core_type:
            continue
        index[bacteria] = {"lps_core_type": lps_core_type, "source_table": "primary"}

    for row in supplemental_rows:
        bacteria = row.get("Strain", "").strip()
        if not bacteria:
            continue
        lps_core_type = _normalize_category(row.get("LPS_type", ""))
        if not lps_core_type:
            continue
        existing = index.get(bacteria)
        if existing is not None and existing["lps_core_type"] != lps_core_type:
            raise ValueError(
                f"Conflicting LPS core annotations for {bacteria!r}: {existing['lps_core_type']} vs {lps_core_type}"
            )
        if existing is None:
            index[bacteria] = {"lps_core_type": lps_core_type, "source_table": "supplemental"}
    return index


def build_feature_rows(
    *,
    umap_rows: Sequence[Mapping[str, str]],
    capsule_index: Mapping[str, Mapping[str, str]],
    lps_index: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, object]]:
    ordered_rows = sorted(umap_rows, key=lambda row: row["bacteria"])
    feature_rows: List[Dict[str, object]] = []

    for row in ordered_rows:
        bacteria = row["bacteria"]
        lps_record = lps_index.get(bacteria)
        if lps_record is None:
            raise KeyError(f"Missing LPS core annotation for {bacteria}")

        capsule_type = ""
        if bacteria in capsule_index:
            capsule_type = _normalize_category(capsule_index[bacteria].get("klebsiella_capsule_type", ""))

        feature_row: Dict[str, object] = {
            "bacteria": bacteria,
            "host_surface_klebsiella_capsule_type": capsule_type,
            "host_surface_klebsiella_capsule_type_missing": int(not capsule_type),
            "host_surface_lps_core_type": lps_record["lps_core_type"],
        }

        for index, source_column in enumerate(UMAP_COLUMNS):
            try:
                feature_row[f"host_phylogeny_umap_{index:02d}"] = float(row[source_column])
            except KeyError as exc:
                raise KeyError(f"Missing {source_column} column in UMAP row for {bacteria}") from exc
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {source_column} value for {bacteria}: {row.get(source_column, '')!r}"
                ) from exc

        feature_rows.append(feature_row)

    return feature_rows


def build_metadata_rows(
    *,
    feature_rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    host_count = len(feature_rows)
    metadata_rows: List[Dict[str, object]] = []

    definitions = {
        "host_surface_klebsiella_capsule_type": {
            "feature_group": "surface_antigen",
            "feature_type": "categorical",
            "source_path": str(args.capsule_path),
            "source_column": "Klebs_capsule_type",
            "transform": "Copy Klebsiella capsule type from the Kaptive-derived call table; normalize blank/Unknown to empty.",
            "provenance_note": "Sparse Klebsiella-type capsule calls; paired missingness indicator retained for downstream joins.",
        },
        "host_surface_klebsiella_capsule_type_missing": {
            "feature_group": "surface_antigen",
            "feature_type": "binary",
            "source_path": str(args.capsule_path),
            "source_column": "Klebs_capsule_type",
            "transform": "1 when host_surface_klebsiella_capsule_type is empty, else 0.",
            "provenance_note": "Explicit missingness flag for incomplete Klebsiella capsule coverage.",
        },
        "host_surface_lps_core_type": {
            "feature_group": "surface_antigen",
            "feature_type": "categorical",
            "source_path": f"{args.lps_primary_path};{args.lps_supplemental_path}",
            "source_column": "LPS_type",
            "transform": "Prefer the primary waaL 370 table; backfill missing hosts from the supplemental host table after conflict checks.",
            "provenance_note": "Combined curated waaL tables provide complete coverage for the 404-host UMAP panel.",
        },
    }
    for index in range(8):
        definitions[f"host_phylogeny_umap_{index:02d}"] = {
            "feature_group": "phylogeny_embedding",
            "feature_type": "continuous",
            "source_path": str(args.umap_path),
            "source_column": f"UMAP{index}",
            "transform": f"Cast UMAP{index} to float and retain the original 8D phylogenomic embedding coordinate.",
            "provenance_note": "The output host contract is anchored to the complete UMAP panel.",
        }

    for column_name in FEATURE_COLUMNS[1:]:
        missing_count = sum(
            1
            for row in feature_rows
            if row[column_name] in {"", None} or (column_name.endswith("_missing") and int(row[column_name]) == 1)
        )
        if column_name.endswith("_missing"):
            missing_count = sum(int(row[column_name]) for row in feature_rows)
        metadata_rows.append(
            {
                "column_name": column_name,
                **definitions[column_name],
                "missing_count": missing_count,
                "missing_rate": round(missing_count / host_count, 6) if host_count else 0.0,
            }
        )
    return metadata_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    umap_rows = read_delimited_rows(args.umap_path, "\t")
    capsule_rows = read_delimited_rows(args.capsule_path, "\t")
    lps_primary_rows = read_delimited_rows(args.lps_primary_path, "\t")
    lps_supplemental_rows = read_delimited_rows(args.lps_supplemental_path, "\t")

    if len(umap_rows) != args.expected_host_count:
        raise ValueError(f"Expected {args.expected_host_count} rows in {args.umap_path}, found {len(umap_rows)}")

    capsule_index = build_capsule_index(capsule_rows)
    lps_index = build_lps_core_index(lps_primary_rows, lps_supplemental_rows)
    feature_rows = build_feature_rows(
        umap_rows=umap_rows,
        capsule_index=capsule_index,
        lps_index=lps_index,
    )
    metadata_rows = build_metadata_rows(feature_rows=feature_rows, args=args)

    host_count = len(feature_rows)
    capsule_observed_hosts = sum(1 for row in feature_rows if row["host_surface_klebsiella_capsule_type"])
    source_breakdown = Counter(lps_index[row["bacteria"]]["source_table"] for row in feature_rows)
    capsule_type_counts = Counter(
        str(row["host_surface_klebsiella_capsule_type"])
        for row in feature_rows
        if row["host_surface_klebsiella_capsule_type"]
    )

    manifest = {
        "version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_count": host_count,
        "feature_count": len(FEATURE_COLUMNS) - 1,
        "output_schema": list(FEATURE_COLUMNS),
        "coverage": {
            "klebsiella_capsule_type": {
                "observed_hosts": capsule_observed_hosts,
                "missing_hosts": host_count - capsule_observed_hosts,
                "missing_rate": round((host_count - capsule_observed_hosts) / host_count, 6) if host_count else 0.0,
                "observed_type_counts": dict(sorted(capsule_type_counts.items())),
            },
            "lps_core_type": {
                "observed_hosts": host_count,
                "missing_hosts": 0,
                "missing_rate": 0.0,
                "source_breakdown": dict(sorted(source_breakdown.items())),
            },
            "umap_phylogeny_8d": {
                "observed_hosts": host_count,
                "missing_hosts": 0,
                "missing_rate": 0.0,
            },
        },
        "sources": {
            "umap_path": str(args.umap_path),
            "umap_sha256": _sha256(args.umap_path),
            "capsule_path": str(args.capsule_path),
            "capsule_sha256": _sha256(args.capsule_path),
            "lps_primary_path": str(args.lps_primary_path),
            "lps_primary_sha256": _sha256(args.lps_primary_path),
            "lps_supplemental_path": str(args.lps_supplemental_path),
            "lps_supplemental_sha256": _sha256(args.lps_supplemental_path),
        },
    }

    ensure_directory(args.output_dir)
    write_csv(args.output_dir / f"host_extended_surface_features_{args.version}.csv", FEATURE_COLUMNS, feature_rows)
    write_csv(
        args.output_dir / f"host_extended_surface_feature_metadata_{args.version}.csv",
        METADATA_COLUMNS,
        metadata_rows,
    )
    write_json(
        args.output_dir / f"host_extended_surface_feature_manifest_{args.version}.json",
        manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
