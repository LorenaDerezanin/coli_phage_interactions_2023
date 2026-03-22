#!/usr/bin/env python3
"""TF01/ST0.3c: Build a fixed split protocol with host clusters and phage clades."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps.st03b_build_split_suite import select_holdout_items
from lyzortx.pipeline.steel_thread_v0.steps.st02_build_pair_table import read_delimited_rows

ST02_REQUIRED_COLUMNS: Sequence[str] = (
    "pair_id",
    "bacteria",
    "phage",
    "cv_group",
    "phage_family",
    "phage_subfamily",
    "phage_genus",
    "phage_old_family",
    "phage_old_genus",
)

PROTOCOL_VERSION = "v1"
PROTOCOL_ID = f"tf01_fixed_split_protocol_{PROTOCOL_VERSION}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 canonical pair table path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for TF01/ST0.3c artifacts.",
    )
    parser.add_argument(
        "--host-holdout-fraction",
        type=float,
        default=0.2,
        help="Target fraction of host CV groups routed to holdout.",
    )
    parser.add_argument(
        "--phage-clade-holdout-fraction",
        type=float,
        default=0.2,
        help="Target fraction of phage clades routed to holdout.",
    )
    parser.add_argument(
        "--host-split-salt",
        type=str,
        default="steel_thread_v0_tf01_host_cluster_v1",
        help="Deterministic salt for host-cluster holdout selection.",
    )
    parser.add_argument(
        "--phage-split-salt",
        type=str,
        default="steel_thread_v0_tf01_phage_clade_v1",
        help="Deterministic salt for phage-clade holdout selection.",
    )
    return parser.parse_args(argv)


def _normalized_taxon_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.lower() in {"other", "na", "n/a", "none", "missing"}:
        return ""
    return cleaned


def phage_clade_key(row: Mapping[str, str]) -> str:
    """Derive a stable phage clade label from the available taxonomy columns."""
    for column in ("phage_family", "phage_subfamily", "phage_genus", "phage_old_family", "phage_old_genus"):
        label = _normalized_taxon_label(row.get(column, ""))
        if label:
            return label
    return "__MISSING_PHAGE_CLADE__"


def build_fixed_split_assignments(
    rows: Sequence[Mapping[str, str]],
    *,
    host_holdout_fraction: float,
    phage_clade_holdout_fraction: float,
    host_split_salt: str,
    phage_split_salt: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    if not rows:
        raise ValueError("No rows were provided.")
    if not 0 < host_holdout_fraction < 1:
        raise ValueError("host_holdout_fraction must be between 0 and 1.")
    if not 0 < phage_clade_holdout_fraction < 1:
        raise ValueError("phage_clade_holdout_fraction must be between 0 and 1.")

    host_groups = sorted(
        {(row.get("cv_group", "") or "").strip() for row in rows if (row.get("cv_group", "") or "").strip()}
    )
    if not host_groups:
        raise ValueError("No non-empty cv_group values were found.")
    host_holdout_groups = select_holdout_items(host_groups, host_holdout_fraction, host_split_salt)

    phage_clades = sorted({phage_clade_key(row) for row in rows})
    phage_holdout_clades = select_holdout_items(phage_clades, phage_clade_holdout_fraction, phage_split_salt)

    assignments: List[Dict[str, object]] = []
    host_split_counts = Counter()
    phage_split_counts = Counter()
    dual_split_counts = Counter()

    train_host_groups: Set[str] = set()
    holdout_host_groups_seen: Set[str] = set()
    train_phage_clades: Set[str] = set()
    holdout_phage_clades_seen: Set[str] = set()

    train_host_groups_dual: Set[str] = set()
    dual_holdout_host_groups: Set[str] = set()
    train_phage_clades_dual: Set[str] = set()
    dual_holdout_phage_clades: Set[str] = set()

    for row in rows:
        bacteria = (row.get("bacteria", "") or "").strip()
        phage = (row.get("phage", "") or "").strip()
        pair_id = (row.get("pair_id", "") or "").strip()
        cv_group = (row.get("cv_group", "") or "").strip()
        phage_clade = phage_clade_key(row)

        if not pair_id:
            raise ValueError("Encountered a row with an empty pair_id.")
        if not cv_group:
            raise ValueError(f"Encountered an empty cv_group for pair {pair_id}.")

        host_split = "holdout_test" if cv_group in host_holdout_groups else "train_non_holdout"
        phage_split = "holdout_test" if phage_clade in phage_holdout_clades else "train_non_holdout"
        if host_split == "holdout_test" and phage_split == "holdout_test":
            dual_split = "dual_holdout_test"
        elif host_split == "holdout_test":
            dual_split = "host_only_holdout"
        elif phage_split == "holdout_test":
            dual_split = "phage_only_holdout"
        else:
            dual_split = "train_non_holdout"

        host_split_counts[host_split] += 1
        phage_split_counts[phage_split] += 1
        dual_split_counts[dual_split] += 1

        if host_split == "train_non_holdout":
            train_host_groups.add(cv_group)
        else:
            holdout_host_groups_seen.add(cv_group)

        if phage_split == "train_non_holdout":
            train_phage_clades.add(phage_clade)
        else:
            holdout_phage_clades_seen.add(phage_clade)

        if dual_split == "train_non_holdout":
            train_host_groups_dual.add(cv_group)
            train_phage_clades_dual.add(phage_clade)
        elif dual_split == "dual_holdout_test":
            dual_holdout_host_groups.add(cv_group)
            dual_holdout_phage_clades.add(phage_clade)

        assignments.append(
            {
                "pair_id": pair_id,
                "bacteria": bacteria,
                "phage": phage,
                "cv_group": cv_group,
                "phage_clade": phage_clade,
                "split_protocol_id": PROTOCOL_ID,
                "split_host_cluster_holdout": host_split,
                "split_phage_clade_holdout": phage_split,
                "split_dual_axis": dual_split,
                "split_dual_axis_is_eval": 0 if dual_split == "train_non_holdout" else 1,
            }
        )

    assignments.sort(key=lambda row: (str(row["bacteria"]), str(row["phage"])))

    protocol = {
        "step_name": "st03c_build_fixed_split_protocol",
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "split_type": "leave_cluster_out_host_plus_phage_clade_holdout",
        "host_axis": {
            "group_key": "cv_group",
            "assignment": "hash-based deterministic assignment by cv_group",
            "holdout_fraction": host_holdout_fraction,
            "split_salt": host_split_salt,
        },
        "phage_axis": {
            "group_key": "phage_clade",
            "assignment": "hash-based deterministic assignment by phage_clade",
            "holdout_fraction": phage_clade_holdout_fraction,
            "split_salt": phage_split_salt,
            "clade_definition": {
                "preferred_columns": [
                    "phage_family",
                    "phage_subfamily",
                    "phage_genus",
                    "phage_old_family",
                    "phage_old_genus",
                ],
                "missing_value_tokens": ["", "Other", "NA", "N/A", "none", "missing"],
            },
        },
        "dual_axis": {
            "evaluation_membership": [
                "dual_holdout_test",
                "host_only_holdout",
                "phage_only_holdout",
            ]
        },
    }

    audit = {
        "row_count": len(assignments),
        "host_cluster_count": len(host_groups),
        "host_cluster_holdout_count": len(host_holdout_groups),
        "host_cluster_holdout_fraction_actual": round(len(host_holdout_groups) / len(host_groups), 6),
        "phage_clade_count": len(phage_clades),
        "phage_clade_holdout_count": len(phage_holdout_clades),
        "phage_clade_holdout_fraction_actual": round(len(phage_holdout_clades) / len(phage_clades), 6),
        "split_counts": {
            "host_cluster_holdout": dict(sorted(host_split_counts.items())),
            "phage_clade_holdout": dict(sorted(phage_split_counts.items())),
            "dual_axis": dict(sorted(dual_split_counts.items())),
        },
        "leakage_checks": {
            "host_cluster_holdout_cv_group_overlap_count": len(train_host_groups & holdout_host_groups_seen),
            "phage_clade_holdout_clade_overlap_count": len(train_phage_clades & holdout_phage_clades_seen),
            "dual_axis_train_vs_dual_holdout_cv_group_overlap_count": len(
                train_host_groups_dual & dual_holdout_host_groups
            ),
            "dual_axis_train_vs_dual_holdout_clade_overlap_count": len(
                train_phage_clades_dual & dual_holdout_phage_clades
            ),
        },
        "holdout_membership": {
            "host_cluster_holdout_ids_sorted": sorted(host_holdout_groups),
            "phage_clade_holdout_ids_sorted": sorted(phage_holdout_clades),
        },
    }

    return assignments, protocol, audit


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_delimited_rows(args.st02_pair_table_path, delimiter=",")
    if not st02_rows:
        raise ValueError(f"No rows found in {args.st02_pair_table_path}.")
    missing = [column for column in ST02_REQUIRED_COLUMNS if column not in st02_rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {args.st02_pair_table_path}: {', '.join(sorted(missing))}")

    assignments, protocol, audit = build_fixed_split_assignments(
        st02_rows,
        host_holdout_fraction=args.host_holdout_fraction,
        phage_clade_holdout_fraction=args.phage_clade_holdout_fraction,
        host_split_salt=args.host_split_salt,
        phage_split_salt=args.phage_split_salt,
    )

    assignments_path = args.output_dir / f"st03c_fixed_split_protocol_{PROTOCOL_VERSION}_assignments.csv"
    protocol_path = args.output_dir / f"st03c_fixed_split_protocol_{PROTOCOL_VERSION}_protocol.json"
    audit_path = args.output_dir / f"st03c_fixed_split_protocol_{PROTOCOL_VERSION}_audit.json"

    write_csv(assignments_path, fieldnames=list(assignments[0].keys()), rows=assignments)
    write_json(protocol_path, protocol)
    write_json(audit_path, audit)

    print("TF01/ST0.3c completed.")
    print(f"- Rows: {len(assignments)}")
    print(f"- Host clusters: {audit['host_cluster_count']}")
    print(f"- Host holdouts: {audit['host_cluster_holdout_count']}")
    print(f"- Phage clades: {audit['phage_clade_count']}")
    print(f"- Phage holdouts: {audit['phage_clade_holdout_count']}")
    print(f"- Output assignments: {assignments_path}")


if __name__ == "__main__":
    main()
