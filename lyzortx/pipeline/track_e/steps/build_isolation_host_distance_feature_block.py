#!/usr/bin/env python3
"""TE03: Build target-host to isolation-host distance features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import slugify_token

OUTPUT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "isolation_host_umap_euclidean_distance",
    "isolation_host_defense_jaccard_distance",
    "isolation_host_feature_available",
)
FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "isolation_host_umap_euclidean_distance": {
        "type": "float",
        "transform": (
            "Euclidean distance between the pair's target-host 8D phylogeny UMAP coordinates and the phage isolation "
            "host coordinates. Imputed to 0.0 when the isolation host lacks a usable feature profile."
        ),
    },
    "isolation_host_defense_jaccard_distance": {
        "type": "float",
        "transform": (
            "Jaccard distance between the pair's target-host retained defense-subtype vector and the phage isolation "
            "host vector. Imputed to 0.0 when the isolation host lacks a usable feature profile."
        ),
    },
    "isolation_host_feature_available": {
        "type": "binary",
        "transform": (
            "1 when the phage isolation host has both UMAP and retained defense-subtype features available, else 0."
        ),
    },
}


@dataclass(frozen=True)
class IsolationHostFeatureProfile:
    host_name: str
    umap_vector: Tuple[float, ...]
    defense_vector: Tuple[int, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
        help="Input Track C v1 pair table containing phage_host, host_phylogeny_umap_*, and host_defense_subtype_*.",
    )
    parser.add_argument(
        "--umap-path",
        type=Path,
        default=Path("data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv"),
        help="Tab-delimited host UMAP table used to source isolation-host coordinates.",
    )
    parser.add_argument(
        "--defense-subtypes-path",
        type=Path,
        default=Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"),
        help="Semicolon-delimited host defense subtype table used to source isolation-host defense vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_e/isolation_host_distance_feature_block"),
        help="Directory for generated TE03 outputs.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and the manifest.",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(rows: Sequence[Mapping[str, str]], path: Path, columns: Iterable[str]) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")


def detect_target_umap_columns(rows: Sequence[Mapping[str, str]], path: Path) -> List[str]:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    columns = sorted(column for column in rows[0] if column.startswith("host_phylogeny_umap_"))
    if not columns:
        raise ValueError(f"No host_phylogeny_umap_* columns found in {path}")
    return columns


def detect_target_defense_columns(rows: Sequence[Mapping[str, str]], path: Path) -> List[str]:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    columns = sorted(column for column in rows[0] if column.startswith("host_defense_subtype_"))
    if not columns:
        raise ValueError(f"No host_defense_subtype_* columns found in {path}")
    return columns


def _parse_float(value: str, *, column_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {column_name}: {value!r}") from exc


def _parse_binary_flag(value: str) -> int:
    normalized = value.strip()
    if normalized in {"", "0", "0.0"}:
        return 0
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Expected binary/numeric flag, found {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Expected non-negative flag, found {value!r}")
    return 1 if parsed > 0 else 0


def _target_to_source_defense_column(target_column: str) -> str:
    if not target_column.startswith("host_defense_subtype_"):
        raise ValueError(f"Unsupported target defense column {target_column!r}")
    return target_column.removeprefix("host_defense_subtype_")


def resolve_defense_source_columns(
    target_defense_columns: Sequence[str],
    defense_rows: Sequence[Mapping[str, str]],
    path: Path,
) -> Dict[str, str]:
    _require_columns(defense_rows, path, ("bacteria",))
    available_source_columns = {column: column for column in defense_rows[0] if column != "bacteria"}
    by_slug = {slugify_token(column): column for column in available_source_columns}

    resolved: Dict[str, str] = {}
    for target_column in target_defense_columns:
        source_slug = _target_to_source_defense_column(target_column)
        source_column = by_slug.get(source_slug)
        if source_column is None:
            raise ValueError(
                f"Could not map retained defense column {target_column!r} back to a source subtype column in {path}"
            )
        resolved[target_column] = source_column
    return resolved


def build_umap_index(
    umap_rows: Sequence[Mapping[str, str]],
    target_umap_columns: Sequence[str],
) -> Dict[str, Tuple[float, ...]]:
    required_columns = ["bacteria", *[f"UMAP{dim}" for dim in range(len(target_umap_columns))]]
    _require_columns(umap_rows, Path("<umap_rows>"), required_columns)

    index: Dict[str, Tuple[float, ...]] = {}
    for row in umap_rows:
        bacteria = row.get("bacteria", "")
        if not bacteria:
            continue
        if bacteria in index:
            raise ValueError(f"Duplicate bacteria value {bacteria!r} in UMAP rows")
        index[bacteria] = tuple(
            _parse_float(row[f"UMAP{dim}"], column_name=f"UMAP{dim}") for dim in range(len(target_umap_columns))
        )
    return index


def build_isolation_host_feature_index(
    *,
    umap_rows: Sequence[Mapping[str, str]],
    defense_rows: Sequence[Mapping[str, str]],
    target_umap_columns: Sequence[str],
    target_defense_columns: Sequence[str],
    requested_hosts: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, IsolationHostFeatureProfile], List[Dict[str, object]]]:
    umap_index = build_umap_index(umap_rows, target_umap_columns)
    defense_source_columns = resolve_defense_source_columns(
        target_defense_columns, defense_rows, Path("<defense_rows>")
    )
    defense_index = {row["bacteria"]: row for row in defense_rows if row.get("bacteria", "")}

    if requested_hosts is not None:
        all_hosts = sorted(set(requested_hosts))
    else:
        all_hosts = sorted(set(umap_index) | set(defense_index))
    profile_index: Dict[str, IsolationHostFeatureProfile] = {}
    summary_rows: List[Dict[str, object]] = []
    for host_name in all_hosts:
        umap_vector = umap_index.get(host_name)
        defense_row = defense_index.get(host_name)
        has_umap = int(umap_vector is not None)
        has_defense = int(defense_row is not None)
        available = int(has_umap == 1 and has_defense == 1)

        if available == 1 and umap_vector is not None and defense_row is not None:
            defense_vector = tuple(
                _parse_binary_flag(defense_row.get(defense_source_columns[column], ""))
                for column in target_defense_columns
            )
            profile_index[host_name] = IsolationHostFeatureProfile(
                host_name=host_name,
                umap_vector=umap_vector,
                defense_vector=defense_vector,
            )

        summary_rows.append(
            {
                "isolation_host": host_name,
                "has_umap_profile": has_umap,
                "has_defense_profile": has_defense,
                "isolation_host_feature_available": available,
            }
        )

    return profile_index, summary_rows


def compute_jaccard_distance(left: Sequence[int], right: Sequence[int]) -> float:
    intersection = sum(1 for left_value, right_value in zip(left, right) if left_value == 1 and right_value == 1)
    union = sum(1 for left_value, right_value in zip(left, right) if left_value == 1 or right_value == 1)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


def build_feature_rows(
    pair_rows: Sequence[Mapping[str, str]],
    *,
    target_umap_columns: Sequence[str],
    target_defense_columns: Sequence[str],
    isolation_host_index: Mapping[str, IsolationHostFeatureProfile],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    feature_rows: List[Dict[str, object]] = []
    phage_summary: Dict[str, Dict[str, object]] = {}

    for row in pair_rows:
        pair_id = row.get("pair_id", "")
        if not pair_id:
            raise ValueError("Pair-table row is missing pair_id")

        target_umap = tuple(_parse_float(row[column], column_name=column) for column in target_umap_columns)
        target_defense = tuple(_parse_binary_flag(row.get(column, "")) for column in target_defense_columns)
        phage = row.get("phage", "")
        isolation_host = row.get("phage_host", "")
        isolation_profile = isolation_host_index.get(isolation_host)
        available = int(isolation_profile is not None and isolation_host != "")

        umap_distance = 0.0
        defense_distance = 0.0
        if isolation_profile is not None:
            umap_distance = math.dist(target_umap, isolation_profile.umap_vector)
            defense_distance = compute_jaccard_distance(target_defense, isolation_profile.defense_vector)

        feature_rows.append(
            {
                "pair_id": pair_id,
                "bacteria": row.get("bacteria", ""),
                "phage": phage,
                "isolation_host_umap_euclidean_distance": round(umap_distance, 6),
                "isolation_host_defense_jaccard_distance": round(defense_distance, 6),
                "isolation_host_feature_available": available,
            }
        )

        if phage and phage not in phage_summary:
            phage_summary[phage] = {
                "phage": phage,
                "isolation_host": isolation_host,
                "isolation_host_feature_available": available,
            }

    feature_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    phage_summary_rows = sorted(phage_summary.values(), key=lambda item: str(item["phage"]))
    return feature_rows, phage_summary_rows


def build_feature_metadata(pair_table_path: Path) -> List[Dict[str, object]]:
    return [
        {
            "column_name": column,
            "type": FEATURE_METADATA[column]["type"],
            "source_path": str(pair_table_path),
            "transform": FEATURE_METADATA[column]["transform"],
        }
        for column in OUTPUT_FEATURE_COLUMNS
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    pair_rows = read_delimited_rows(args.pair_table_path)
    umap_rows = read_delimited_rows(args.umap_path, delimiter="\t")
    defense_rows = read_delimited_rows(args.defense_subtypes_path, delimiter=";")

    _require_columns(pair_rows, args.pair_table_path, ("pair_id", "bacteria", "phage", "phage_host"))
    target_umap_columns = detect_target_umap_columns(pair_rows, args.pair_table_path)
    target_defense_columns = detect_target_defense_columns(pair_rows, args.pair_table_path)
    isolation_host_index, isolation_host_summary_rows = build_isolation_host_feature_index(
        umap_rows=umap_rows,
        defense_rows=defense_rows,
        target_umap_columns=target_umap_columns,
        target_defense_columns=target_defense_columns,
        requested_hosts={row.get("phage_host", "") for row in pair_rows if row.get("phage_host", "")},
    )
    feature_rows, phage_summary_rows = build_feature_rows(
        pair_rows,
        target_umap_columns=target_umap_columns,
        target_defense_columns=target_defense_columns,
        isolation_host_index=isolation_host_index,
    )
    metadata_rows = build_feature_metadata(args.pair_table_path)

    feature_output_path = args.output_dir / f"isolation_host_distance_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"isolation_host_distance_feature_metadata_{args.version}.csv"
    phage_summary_output_path = args.output_dir / f"phage_isolation_host_coverage_{args.version}.csv"
    host_summary_output_path = args.output_dir / f"isolation_host_feature_coverage_{args.version}.csv"
    manifest_output_path = args.output_dir / f"isolation_host_distance_manifest_{args.version}.json"

    write_csv(feature_output_path, ["pair_id", "bacteria", "phage", *OUTPUT_FEATURE_COLUMNS], feature_rows)
    write_csv(metadata_output_path, ["column_name", "type", "source_path", "transform"], metadata_rows)
    write_csv(
        phage_summary_output_path,
        ["phage", "isolation_host", "isolation_host_feature_available"],
        phage_summary_rows,
    )
    write_csv(
        host_summary_output_path,
        ["isolation_host", "has_umap_profile", "has_defense_profile", "isolation_host_feature_available"],
        isolation_host_summary_rows,
    )

    available_pair_count = sum(int(row["isolation_host_feature_available"]) for row in feature_rows)
    available_phages = [row["phage"] for row in phage_summary_rows if int(row["isolation_host_feature_available"]) == 1]
    unavailable_phages = [
        row["phage"] for row in phage_summary_rows if int(row["isolation_host_feature_available"]) == 0
    ]
    manifest = {
        "step_name": "build_isolation_host_distance_feature_block",
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_count": len(feature_rows),
        "distinct_bacteria_count": len({row["bacteria"] for row in feature_rows}),
        "distinct_phage_count": len({row["phage"] for row in feature_rows}),
        "feature_count": len(OUTPUT_FEATURE_COLUMNS),
        "target_umap_dimension_count": len(target_umap_columns),
        "target_defense_subtype_count": len(target_defense_columns),
        "coverage": {
            "available_pair_count": available_pair_count,
            "available_pair_fraction": round(available_pair_count / len(feature_rows), 6) if feature_rows else 0.0,
            "available_phage_count": len(available_phages),
            "unavailable_phage_count": len(unavailable_phages),
            "unavailable_phages": unavailable_phages,
        },
        "inputs": {
            "pair_table": {"path": str(args.pair_table_path), "sha256": _sha256(args.pair_table_path)},
            "umap": {"path": str(args.umap_path), "sha256": _sha256(args.umap_path)},
            "defense_subtypes": {
                "path": str(args.defense_subtypes_path),
                "sha256": _sha256(args.defense_subtypes_path),
            },
        },
        "policies": {
            "feature_contract": [
                "UMAP Euclidean distance between target host and phage isolation host.",
                "Defense Jaccard distance between target host and phage isolation host.",
                "Binary availability flag for isolation hosts with both feature types present.",
            ],
            "missing_data_policy": (
                "When the phage isolation host is missing from either source table, both distance features are imputed "
                "to 0.0 and isolation_host_feature_available is set to 0."
            ),
            "defense_alignment": (
                "Isolation-host defense vectors are restricted to the retained Track C host_defense_subtype_* columns "
                "and mapped back to raw defense-finder subtype names using the shared slugification contract."
            ),
        },
        "outputs": {
            "feature_csv": str(feature_output_path),
            "feature_metadata_csv": str(metadata_output_path),
            "phage_coverage_csv": str(phage_summary_output_path),
            "isolation_host_coverage_csv": str(host_summary_output_path),
        },
    }
    write_json(manifest_output_path, manifest)

    print(f"Wrote TE03 isolation-host distance features to {feature_output_path}")
    print(f"- Pair rows: {len(feature_rows)}")
    print(f"- Target UMAP dimensions: {len(target_umap_columns)}")
    print(f"- Retained defense subtype columns: {len(target_defense_columns)}")
    print(f"- Phages with available isolation-host profiles: {len(available_phages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
