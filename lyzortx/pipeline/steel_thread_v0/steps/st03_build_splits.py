#!/usr/bin/env python3
"""ST0.3: Build leakage-safe split assignments from ST0.2 canonical pair table."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

ST02_REQUIRED_COLUMNS: Sequence[str] = (
    "pair_id",
    "bacteria",
    "phage",
    "cv_group",
    "label_hard_include_in_training",
    "label_strict_include_in_training",
    "label_strict_confidence_tier",
)


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
        help="Output directory for ST0.3 artifacts.",
    )
    parser.add_argument(
        "--holdout-group-fraction",
        type=float,
        default=0.2,
        help="Target fraction of cv_groups to route to fixed holdout.",
    )
    parser.add_argument(
        "--n-cv-folds",
        type=int,
        default=5,
        help="Number of grouped CV folds on non-holdout groups.",
    )
    parser.add_argument(
        "--split-salt",
        type=str,
        default="steel_thread_v0_st03_split_v1",
        help="Deterministic salt for hash-based group assignment.",
    )
    return parser.parse_args(argv)


def read_st02_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        missing = [c for c in ST02_REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        out: List[Dict[str, str]] = []
        for row in reader:
            out.append({k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()})
        return out


def normalized_hash_01(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def assign_group_splits(
    groups: Sequence[str], holdout_group_fraction: float, n_cv_folds: int, split_salt: str
) -> Tuple[Set[str], Dict[str, int]]:
    if not 0 < holdout_group_fraction < 1:
        raise ValueError("holdout_group_fraction must be between 0 and 1.")
    if n_cv_folds < 2:
        raise ValueError("n_cv_folds must be >= 2.")

    scored = []
    for group in sorted(groups):
        holdout_score = normalized_hash_01(f"{split_salt}|holdout|{group}")
        scored.append((holdout_score, group))
    scored.sort()

    n_holdout = max(1, int(round(len(scored) * holdout_group_fraction)))
    holdout_groups = {group for _, group in scored[:n_holdout]}

    cv_fold_by_group: Dict[str, int] = {}
    for group in sorted(groups):
        if group in holdout_groups:
            continue
        fold_hash = int(
            hashlib.sha256(f"{split_salt}|cvfold|{group}".encode("utf-8")).hexdigest(),
            16,
        )
        cv_fold_by_group[group] = fold_hash % n_cv_folds

    return holdout_groups, cv_fold_by_group


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    rows = read_st02_rows(args.st02_pair_table_path)
    if not rows:
        raise ValueError(f"No rows found in {args.st02_pair_table_path}.")

    groups = sorted({row["cv_group"] for row in rows if row["cv_group"] != ""})
    if not groups:
        raise ValueError("No non-empty cv_group values found in ST0.2 table.")

    holdout_groups, cv_fold_by_group = assign_group_splits(
        groups=groups,
        holdout_group_fraction=args.holdout_group_fraction,
        n_cv_folds=args.n_cv_folds,
        split_salt=args.split_salt,
    )

    split_rows: List[Dict[str, object]] = []
    holdout_counter = Counter()
    cv_fold_counter = Counter()
    strict_holdout_counter = Counter()
    strict_cv_fold_counter = Counter()
    hard_trainable_counter = Counter()
    strict_trainable_counter = Counter()

    bacteria_by_holdout: Dict[str, Set[str]] = defaultdict(set)
    groups_by_holdout: Dict[str, Set[str]] = defaultdict(set)
    bacteria_by_cv_fold: Dict[int, Set[str]] = defaultdict(set)
    groups_by_cv_fold: Dict[int, Set[str]] = defaultdict(set)

    for row in rows:
        cv_group = row["cv_group"]
        if cv_group == "":
            raise ValueError("Encountered empty cv_group in ST0.2 table.")

        holdout_split = "holdout_test" if cv_group in holdout_groups else "train_non_holdout"
        cv_fold = -1 if holdout_split == "holdout_test" else cv_fold_by_group[cv_group]

        hard_trainable = row["label_hard_include_in_training"] == "1"
        strict_trainable = row["label_strict_include_in_training"] == "1"

        holdout_counter[holdout_split] += 1
        if cv_fold >= 0:
            cv_fold_counter[str(cv_fold)] += 1
        if hard_trainable:
            hard_trainable_counter[holdout_split] += 1
        if strict_trainable:
            strict_holdout_counter[holdout_split] += 1
            if cv_fold >= 0:
                strict_cv_fold_counter[str(cv_fold)] += 1
            strict_trainable_counter[row["label_strict_confidence_tier"]] += 1

        bacteria = row["bacteria"]
        bacteria_by_holdout[holdout_split].add(bacteria)
        groups_by_holdout[holdout_split].add(cv_group)
        if cv_fold >= 0:
            bacteria_by_cv_fold[cv_fold].add(bacteria)
            groups_by_cv_fold[cv_fold].add(cv_group)

        split_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": bacteria,
                "phage": row["phage"],
                "cv_group": cv_group,
                "split_holdout": holdout_split,
                "split_cv5_fold": cv_fold,
                "split_is_holdout": 1 if holdout_split == "holdout_test" else 0,
                "is_hard_trainable": 1 if hard_trainable else 0,
                "is_strict_trainable": 1 if strict_trainable else 0,
                "strict_confidence_tier": row["label_strict_confidence_tier"],
            }
        )

    split_rows.sort(key=lambda x: (str(x["bacteria"]), str(x["phage"])))
    output_path = args.output_dir / "st03_split_assignments.csv"
    write_csv(output_path, fieldnames=list(split_rows[0].keys()), rows=split_rows)

    train_b = bacteria_by_holdout["train_non_holdout"]
    test_b = bacteria_by_holdout["holdout_test"]
    train_g = groups_by_holdout["train_non_holdout"]
    test_g = groups_by_holdout["holdout_test"]

    holdout_bacteria_overlap = len(train_b & test_b)
    holdout_group_overlap = len(train_g & test_g)
    cv_group_cross_fold_overlap = 0
    for i in range(args.n_cv_folds):
        for j in range(i + 1, args.n_cv_folds):
            cv_group_cross_fold_overlap += len(groups_by_cv_fold[i] & groups_by_cv_fold[j])

    protocol = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st03_build_splits",
        "split_protocol_id": "steel_thread_v0_st03_split_v1",
        "split_type": "grouped_host_split",
        "split_rules": {
            "group_key": "cv_group",
            "holdout_assignment": "hash-based deterministic assignment by cv_group",
            "cv_assignment": "hash-based deterministic fold assignment on non-holdout cv_groups",
            "n_cv_folds": args.n_cv_folds,
            "holdout_group_fraction": args.holdout_group_fraction,
            "split_salt": args.split_salt,
        },
        "input_st02_pair_table_path": str(args.st02_pair_table_path),
    }

    audit = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "row_count": len(split_rows),
        "distinct_bacteria_count": len({row["bacteria"] for row in split_rows}),
        "distinct_phage_count": len({row["phage"] for row in split_rows}),
        "cv_group_count": len(groups),
        "holdout_group_count": len(holdout_groups),
        "holdout_group_fraction_actual": round(len(holdout_groups) / len(groups), 6),
        "holdout_row_counts": dict(sorted(holdout_counter.items())),
        "cv_fold_row_counts": dict(sorted(cv_fold_counter.items())),
        "hard_trainable_holdout_counts": dict(sorted(hard_trainable_counter.items())),
        "strict_trainable_holdout_counts": dict(sorted(strict_holdout_counter.items())),
        "strict_trainable_cv_fold_counts": dict(sorted(strict_cv_fold_counter.items())),
        "strict_trainable_tier_counts": dict(sorted(strict_trainable_counter.items())),
        "leakage_checks": {
            "holdout_bacteria_overlap_count": holdout_bacteria_overlap,
            "holdout_cv_group_overlap_count": holdout_group_overlap,
            "cv_group_cross_fold_overlap_count": cv_group_cross_fold_overlap,
        },
        "holdout_group_ids_sorted": sorted(holdout_groups),
    }

    write_json(args.output_dir / "st03_split_protocol.json", protocol)
    write_json(args.output_dir / "st03_split_audit.json", audit)

    print("ST0.3 completed.")
    print(f"- Rows: {len(split_rows)}")
    print(f"- CV groups: {len(groups)}")
    print(f"- Holdout groups: {len(holdout_groups)}")
    print(f"- Holdout row count: {holdout_counter['holdout_test']}")
    print(f"- Output assignments: {output_path}")


if __name__ == "__main__":
    main()
