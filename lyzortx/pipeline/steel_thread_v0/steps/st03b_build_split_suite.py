#!/usr/bin/env python3
"""ST0.3b: Build split-suite artifacts for host, phage-family, and dual-axis stress tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

ST02_REQUIRED_COLUMNS: Sequence[str] = ("pair_id", "bacteria", "phage", "cv_group", "phage_family")
ST03_REQUIRED_COLUMNS: Sequence[str] = ("pair_id", "split_holdout")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 canonical pair table path.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.3b artifacts.",
    )
    parser.add_argument(
        "--phage-family-holdout-fraction",
        type=float,
        default=0.2,
        help="Target fraction of phage families routed to holdout in split-suite modes.",
    )
    parser.add_argument(
        "--split-salt",
        type=str,
        default="steel_thread_v0_st03b_split_suite_v1",
        help="Deterministic salt for hash-based phage-family assignment.",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def select_holdout_items(items: Sequence[str], holdout_fraction: float, split_salt: str) -> Set[str]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("phage_family_holdout_fraction must be between 0 and 1.")
    scored: List[Tuple[float, str]] = []
    for item in sorted(items):
        score = int(hashlib.sha256(f"{split_salt}|{item}".encode("utf-8")).hexdigest(), 16)
        scored.append((score / float(2**256 - 1), item))
    scored.sort()
    n_holdout = max(1, int(round(len(scored) * holdout_fraction)))
    return {item for _, item in scored[:n_holdout]}


def family_key(raw_family: str) -> str:
    return raw_family if raw_family else "__MISSING_PHAGE_FAMILY__"


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path, ST02_REQUIRED_COLUMNS)
    st03_rows = read_csv_rows(args.st03_split_assignments_path, ST03_REQUIRED_COLUMNS)
    if not st02_rows:
        raise ValueError("ST0.2 table has no rows.")

    st03_holdout_by_pair = {row["pair_id"]: row["split_holdout"] for row in st03_rows}
    missing_pairs = sorted({row["pair_id"] for row in st02_rows} - set(st03_holdout_by_pair.keys()))
    if missing_pairs:
        raise ValueError(f"ST0.3 split assignments missing pair_id values, e.g. {missing_pairs[:3]}")

    all_families = sorted({family_key(row["phage_family"]) for row in st02_rows})
    holdout_families = select_holdout_items(
        items=all_families,
        holdout_fraction=args.phage_family_holdout_fraction,
        split_salt=args.split_salt,
    )

    assignments: List[Dict[str, object]] = []
    host_holdout_counter = Counter()
    family_holdout_counter = Counter()
    dual_axis_counter = Counter()

    train_host_groups: Set[str] = set()
    holdout_host_groups: Set[str] = set()
    train_phage_families: Set[str] = set()
    holdout_phage_families: Set[str] = set()

    train_host_groups_dual: Set[str] = set()
    dual_holdout_host_groups: Set[str] = set()
    train_families_dual: Set[str] = set()
    dual_holdout_families: Set[str] = set()

    for row in st02_rows:
        host_split = st03_holdout_by_pair[row["pair_id"]]
        is_host_holdout = host_split == "holdout_test"
        phage_family = family_key(row["phage_family"])
        is_family_holdout = phage_family in holdout_families

        family_split = "holdout_test" if is_family_holdout else "train_non_holdout"
        if is_host_holdout and is_family_holdout:
            dual_axis_split = "dual_holdout_test"
        elif is_host_holdout:
            dual_axis_split = "host_only_holdout"
        elif is_family_holdout:
            dual_axis_split = "phage_only_holdout"
        else:
            dual_axis_split = "train_non_holdout"

        host_holdout_counter[host_split] += 1
        family_holdout_counter[family_split] += 1
        dual_axis_counter[dual_axis_split] += 1

        if host_split == "train_non_holdout":
            train_host_groups.add(row["cv_group"])
        else:
            holdout_host_groups.add(row["cv_group"])

        if family_split == "train_non_holdout":
            train_phage_families.add(phage_family)
        else:
            holdout_phage_families.add(phage_family)

        if dual_axis_split == "train_non_holdout":
            train_host_groups_dual.add(row["cv_group"])
            train_families_dual.add(phage_family)
        elif dual_axis_split == "dual_holdout_test":
            dual_holdout_host_groups.add(row["cv_group"])
            dual_holdout_families.add(phage_family)

        assignments.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "cv_group": row["cv_group"],
                "phage_family": phage_family,
                "split_host_holdout": host_split,
                "split_phage_family_holdout": family_split,
                "split_dual_axis": dual_axis_split,
                "split_dual_axis_is_eval": 0 if dual_axis_split == "train_non_holdout" else 1,
            }
        )

    assignments.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))

    protocol = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st03b_build_split_suite",
        "split_modes": {
            "host_group_holdout": {
                "group_key": "cv_group",
                "source": "st03_split_assignments.csv::split_holdout",
            },
            "phage_family_holdout": {
                "group_key": "phage_family",
                "holdout_assignment": "hash-based deterministic assignment by phage_family",
                "holdout_fraction": args.phage_family_holdout_fraction,
                "split_salt": args.split_salt,
            },
            "dual_axis_host_phage": {
                "host_axis": "st03 host-group holdout",
                "phage_axis": "st03b phage-family holdout",
                "evaluation_membership": [
                    "dual_holdout_test",
                    "host_only_holdout",
                    "phage_only_holdout",
                ],
            },
        },
        "input_paths": {
            "st02_pair_table": str(args.st02_pair_table_path),
            "st03_split_assignments": str(args.st03_split_assignments_path),
        },
    }

    audit = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "row_count": len(assignments),
        "split_counts": {
            "host_group_holdout": dict(sorted(host_holdout_counter.items())),
            "phage_family_holdout": dict(sorted(family_holdout_counter.items())),
            "dual_axis_host_phage": dict(sorted(dual_axis_counter.items())),
        },
        "phage_family_holdout_count": len(holdout_families),
        "phage_family_total_count": len(all_families),
        "phage_family_holdout_fraction_actual": round(len(holdout_families) / len(all_families), 6),
        "leakage_checks": {
            "host_group_holdout_cv_group_overlap_count": len(train_host_groups & holdout_host_groups),
            "phage_family_holdout_family_overlap_count": len(train_phage_families & holdout_phage_families),
            "dual_axis_train_vs_dual_holdout_cv_group_overlap_count": len(
                train_host_groups_dual & dual_holdout_host_groups
            ),
            "dual_axis_train_vs_dual_holdout_family_overlap_count": len(train_families_dual & dual_holdout_families),
        },
        "holdout_membership": {
            "phage_family_holdout_ids_sorted": sorted(holdout_families),
        },
    }

    write_csv(
        args.output_dir / "st03b_split_suite_assignments.csv", fieldnames=list(assignments[0].keys()), rows=assignments
    )
    write_json(args.output_dir / "st03b_split_suite_protocol.json", protocol)
    write_json(args.output_dir / "st03b_split_suite_audit.json", audit)

    print("ST0.3b completed.")
    print(f"- Rows: {len(assignments)}")
    print(f"- Holdout phage families: {len(holdout_families)} / {len(all_families)}")


if __name__ == "__main__":
    main()
