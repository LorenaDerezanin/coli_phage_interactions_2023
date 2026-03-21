#!/usr/bin/env python3
"""Build a bounded OMP receptor variant feature block from BLAST cluster assignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

RECEPTOR_COLUMNS: Tuple[str, ...] = (
    "BTUB",
    "FADL",
    "FHUA",
    "LAMB",
    "LPTD",
    "NFRA",
    "OMPA",
    "OMPC",
    "OMPF",
    "TOLC",
    "TSX",
    "YNCD",
)

METADATA_COLUMNS: Tuple[str, ...] = (
    "column_name",
    "receptor",
    "grouped_category",
    "global_selection_rank",
    "receptor_selection_rank",
    "support_count",
    "support_rate",
    "indicator_variance",
    "grouped_source_clusters",
    "source_path",
    "source_column",
    "transform",
    "provenance_note",
)


@dataclass(frozen=True)
class CategoryStat:
    """Grouped category summary for one receptor."""

    receptor: str
    category: str
    count: int
    prevalence: float
    variance: float
    grouped_source_clusters: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receptor-clusters-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"),
        help="Tab-delimited BLAST cluster assignments for 12 OMP receptors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/omp_receptor_variant_feature_block"),
        help="Directory for generated Track C OMP receptor artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and the manifest.",
    )
    parser.add_argument(
        "--min-cluster-count",
        type=int,
        default=5,
        help="Clusters observed fewer than this many times are grouped into a receptor-specific rare bucket.",
    )
    parser.add_argument(
        "--max-feature-count",
        type=int,
        default=22,
        help="Maximum number of one-hot receptor-variant features to emit, excluding the bacteria key.",
    )
    parser.add_argument(
        "--expected-host-count",
        type=int,
        default=404,
        help="Expected number of bacterial strains in the receptor cluster table.",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def _index_unique(rows: Sequence[Mapping[str, str]], key: str, *, path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            continue
        if value in out:
            raise ValueError(f"Duplicate {key} value {value!r} in {path}")
        out[value] = dict(row)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_cluster(value: str) -> str:
    return value.strip()


def _feature_column_name(receptor: str, category: str) -> str:
    receptor_label = receptor.lower()
    if category == "rare":
        suffix = "cluster_rare"
    elif category == "missing":
        suffix = "cluster_missing"
    else:
        suffix = f"cluster_{category.lower()}"
    return f"host_omp_receptor_{receptor_label}_{suffix}"


def _grouped_category(value: str, *, kept_clusters: Iterable[str]) -> str:
    normalized = _normalize_cluster(value)
    kept = set(kept_clusters)
    if not normalized:
        return "missing"
    if normalized in kept:
        return normalized
    return "rare"


def summarize_receptor_categories(
    rows: Sequence[Mapping[str, str]],
    *,
    receptor_columns: Sequence[str] = RECEPTOR_COLUMNS,
    min_cluster_count: int = 5,
) -> Dict[str, Dict[str, object]]:
    if min_cluster_count < 1:
        raise ValueError("min_cluster_count must be >= 1")

    row_count = len(rows)
    summaries: Dict[str, Dict[str, object]] = {}
    for receptor in receptor_columns:
        cluster_counts = Counter(
            _normalize_cluster(row.get(receptor, "")) for row in rows if _normalize_cluster(row.get(receptor, ""))
        )
        kept_cluster_counts = {
            cluster: count
            for cluster, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_cluster_count
        }
        rare_clusters = tuple(
            cluster
            for cluster, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))
            if count < min_cluster_count
        )

        grouped_counts: Counter[str] = Counter()
        for row in rows:
            grouped_counts[_grouped_category(row.get(receptor, ""), kept_clusters=kept_cluster_counts)] += 1

        category_stats: List[CategoryStat] = []
        for category, count in sorted(grouped_counts.items(), key=lambda item: (-item[1], item[0])):
            prevalence = count / row_count if row_count else 0.0
            grouped_source_clusters = (
                rare_clusters if category == "rare" else ((category,) if category != "missing" else ())
            )
            category_stats.append(
                CategoryStat(
                    receptor=receptor,
                    category=category,
                    count=count,
                    prevalence=prevalence,
                    variance=prevalence * (1.0 - prevalence),
                    grouped_source_clusters=grouped_source_clusters,
                )
            )

        summaries[receptor] = {
            "receptor": receptor,
            "observed_cluster_count": len(cluster_counts),
            "missing_count": grouped_counts.get("missing", 0),
            "kept_cluster_counts": kept_cluster_counts,
            "rare_clusters": rare_clusters,
            "category_stats": category_stats,
        }

    return summaries


def select_feature_categories(
    summaries: Mapping[str, Mapping[str, object]],
    *,
    receptor_columns: Sequence[str] = RECEPTOR_COLUMNS,
    max_feature_count: int = 22,
) -> List[CategoryStat]:
    receptor_count = len(receptor_columns)
    if max_feature_count < receptor_count:
        raise ValueError(
            f"max_feature_count must be >= the number of receptors ({receptor_count}), got {max_feature_count}"
        )

    def sort_key(stat: CategoryStat) -> Tuple[float, int, str, str]:
        return (-stat.variance, -stat.count, stat.receptor, stat.category)

    selected: Dict[Tuple[str, str], CategoryStat] = {}
    for receptor in receptor_columns:
        category_stats = list(summaries[receptor]["category_stats"])
        non_missing = [stat for stat in category_stats if stat.category != "missing"]
        candidates = non_missing if non_missing else category_stats
        mandatory = sorted(candidates, key=sort_key)[0]
        selected[(mandatory.receptor, mandatory.category)] = mandatory

    extra_candidates: List[CategoryStat] = []
    for receptor in receptor_columns:
        for stat in summaries[receptor]["category_stats"]:
            key = (stat.receptor, stat.category)
            if key in selected:
                continue
            extra_candidates.append(stat)

    remaining_slots = max_feature_count - len(selected)
    for stat in sorted(extra_candidates, key=sort_key)[:remaining_slots]:
        selected[(stat.receptor, stat.category)] = stat

    selected_by_receptor: List[CategoryStat] = []
    for receptor in receptor_columns:
        receptor_stats = [stat for stat in selected.values() if stat.receptor == receptor]
        receptor_stats.sort(key=sort_key)
        selected_by_receptor.extend(receptor_stats)
    return selected_by_receptor


def build_feature_rows(
    receptor_rows: Sequence[Mapping[str, str]],
    summaries: Mapping[str, Mapping[str, object]],
    selected_categories: Sequence[CategoryStat],
) -> List[Dict[str, object]]:
    field_order = ["bacteria"] + [_feature_column_name(stat.receptor, stat.category) for stat in selected_categories]
    output_rows: List[Dict[str, object]] = []

    bacteria_index = _index_unique(receptor_rows, "bacteria", path=Path("<in-memory receptor rows>"))
    for bacteria in sorted(bacteria_index):
        receptor_row = bacteria_index[bacteria]
        encoded_row: Dict[str, object] = {"bacteria": bacteria}
        for stat in selected_categories:
            grouped_value = _grouped_category(
                receptor_row.get(stat.receptor, ""),
                kept_clusters=summaries[stat.receptor]["kept_cluster_counts"].keys(),
            )
            encoded_row[_feature_column_name(stat.receptor, stat.category)] = int(grouped_value == stat.category)
        output_rows.append({column: encoded_row[column] for column in field_order})

    return output_rows


def build_metadata_rows(
    selected_categories: Sequence[CategoryStat],
    *,
    source_path: Path,
    min_cluster_count: int,
) -> List[Dict[str, object]]:
    global_rank_lookup = {
        (stat.receptor, stat.category): rank for rank, stat in enumerate(selected_categories, start=1)
    }
    receptor_rank_lookup: Dict[Tuple[str, str], int] = {}
    for receptor in RECEPTOR_COLUMNS:
        receptor_stats = [stat for stat in selected_categories if stat.receptor == receptor]
        for rank, stat in enumerate(receptor_stats, start=1):
            receptor_rank_lookup[(stat.receptor, stat.category)] = rank

    metadata_rows: List[Dict[str, object]] = []
    for stat in selected_categories:
        if stat.category == "rare":
            transform = (
                "1 when the receptor cluster assignment falls into the receptor-specific grouped rare bucket "
                f"(support < {min_cluster_count}), else 0."
            )
            note = "Rare bucket groups low-support source clusters before feature selection."
        elif stat.category == "missing":
            transform = "1 when the receptor cluster assignment is missing, else 0."
            note = "Missingness is encoded only when it survives the global feature budget."
        else:
            transform = f"1 when the receptor cluster assignment equals {stat.category}, else 0."
            note = "Cluster IDs come from 99% identity BLAST clusters."

        metadata_rows.append(
            {
                "column_name": _feature_column_name(stat.receptor, stat.category),
                "receptor": stat.receptor,
                "grouped_category": stat.category,
                "global_selection_rank": global_rank_lookup[(stat.receptor, stat.category)],
                "receptor_selection_rank": receptor_rank_lookup[(stat.receptor, stat.category)],
                "support_count": stat.count,
                "support_rate": round(stat.prevalence, 6),
                "indicator_variance": round(stat.variance, 6),
                "grouped_source_clusters": json.dumps(list(stat.grouped_source_clusters)),
                "source_path": str(source_path),
                "source_column": stat.receptor,
                "transform": transform,
                "provenance_note": note,
            }
        )
    return metadata_rows


def build_manifest(
    *,
    version: str,
    receptor_rows: Sequence[Mapping[str, str]],
    summaries: Mapping[str, Mapping[str, object]],
    selected_categories: Sequence[CategoryStat],
    receptor_clusters_path: Path,
    matrix_output_path: Path,
    metadata_output_path: Path,
    min_cluster_count: int,
    max_feature_count: int,
) -> Dict[str, object]:
    selected_by_receptor: Dict[str, List[str]] = {receptor: [] for receptor in RECEPTOR_COLUMNS}
    for stat in selected_categories:
        selected_by_receptor[stat.receptor].append(stat.category)

    receptor_summary = {}
    for receptor in RECEPTOR_COLUMNS:
        summary = summaries[receptor]
        receptor_summary[receptor] = {
            "observed_cluster_count": summary["observed_cluster_count"],
            "missing_count": summary["missing_count"],
            "retained_clusters": list(summary["kept_cluster_counts"].keys()),
            "rare_clusters": list(summary["rare_clusters"]),
            "selected_grouped_categories": selected_by_receptor[receptor],
            "selected_feature_columns": [
                _feature_column_name(receptor, category) for category in selected_by_receptor[receptor]
            ],
        }

    return {
        "version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_count": len(receptor_rows),
        "receptor_count": len(RECEPTOR_COLUMNS),
        "feature_count": len(selected_categories),
        "encoding_policy": {
            "source": "blast_results_cured_clusters=99_wide.tsv",
            "rare_cluster_min_count": min_cluster_count,
            "max_feature_count": max_feature_count,
            "selection_rule": (
                "Keep one non-missing grouped category per receptor, then add the highest-variance remaining grouped "
                "categories globally until the feature budget is exhausted."
            ),
            "grouping_rule": (
                "Cluster IDs with support below the rare_cluster_min_count threshold are mapped to receptor-specific "
                "rare buckets before one-hot selection."
            ),
        },
        "schema": {
            "matrix_columns": ["bacteria"]
            + [_feature_column_name(stat.receptor, stat.category) for stat in selected_categories],
            "receptor_columns": list(RECEPTOR_COLUMNS),
        },
        "inputs": {
            "receptor_clusters": {"path": str(receptor_clusters_path), "sha256": _sha256(receptor_clusters_path)},
        },
        "outputs": {
            "matrix_csv": str(matrix_output_path),
            "column_metadata_csv": str(metadata_output_path),
        },
        "receptors": receptor_summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    receptor_rows = read_delimited_rows(args.receptor_clusters_path, delimiter="\t")

    if len(receptor_rows) != args.expected_host_count:
        raise ValueError(
            f"Expected {args.expected_host_count} hosts in {args.receptor_clusters_path}, found {len(receptor_rows)}"
        )

    summaries = summarize_receptor_categories(
        receptor_rows,
        receptor_columns=RECEPTOR_COLUMNS,
        min_cluster_count=args.min_cluster_count,
    )
    selected_categories = select_feature_categories(
        summaries,
        receptor_columns=RECEPTOR_COLUMNS,
        max_feature_count=args.max_feature_count,
    )

    feature_rows = build_feature_rows(receptor_rows, summaries, selected_categories)

    ensure_directory(args.output_dir)
    matrix_output_path = args.output_dir / f"host_omp_receptor_variant_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"host_omp_receptor_variant_feature_metadata_{args.version}.csv"
    manifest_output_path = args.output_dir / f"host_omp_receptor_variant_feature_manifest_{args.version}.json"

    write_csv(
        matrix_output_path,
        ["bacteria"] + [_feature_column_name(stat.receptor, stat.category) for stat in selected_categories],
        feature_rows,
    )
    write_csv(
        metadata_output_path,
        METADATA_COLUMNS,
        build_metadata_rows(
            selected_categories,
            source_path=args.receptor_clusters_path,
            min_cluster_count=args.min_cluster_count,
        ),
    )
    write_json(
        manifest_output_path,
        build_manifest(
            version=args.version,
            receptor_rows=receptor_rows,
            summaries=summaries,
            selected_categories=selected_categories,
            receptor_clusters_path=args.receptor_clusters_path,
            matrix_output_path=matrix_output_path,
            metadata_output_path=metadata_output_path,
            min_cluster_count=args.min_cluster_count,
            max_feature_count=args.max_feature_count,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
