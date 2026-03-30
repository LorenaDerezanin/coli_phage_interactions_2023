"""Compare holdout-clean and leaked TL03/TL04 rebuilds.

This script is intended for notebook-backed one-off analysis only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nonzero_row_count(rows: list[dict[str, str]], columns: list[str]) -> int:
    return sum(any(float(row[column]) != 0.0 for column in columns) for row in rows)


def _compare(
    task_name: str,
    clean_dir: Path,
    leaked_dir: Path,
    feature_csv_name: str,
    metadata_csv_name: str,
) -> dict[str, object]:
    clean_rows = _load_csv(clean_dir / feature_csv_name)
    leaked_rows = _load_csv(leaked_dir / feature_csv_name)
    clean_meta = _load_csv(clean_dir / metadata_csv_name)
    leaked_meta = _load_csv(leaked_dir / metadata_csv_name)

    clean_feature_cols = [column for column in clean_rows[0] if column not in {"pair_id", "bacteria", "phage"}]
    leaked_feature_cols = [column for column in leaked_rows[0] if column not in {"pair_id", "bacteria", "phage"}]
    clean_pairwise_cols = [row["column_name"] for row in clean_meta if row["block_type"].startswith("pairwise")]
    leaked_pairwise_cols = [row["column_name"] for row in leaked_meta if row["block_type"].startswith("pairwise")]
    clean_direct_cols = [row["column_name"] for row in clean_meta if not row["block_type"].startswith("pairwise")]
    leaked_direct_cols = [row["column_name"] for row in leaked_meta if not row["block_type"].startswith("pairwise")]

    clean_weights = {
        row["column_name"]: float(row["weight"]) for row in clean_meta if row["block_type"].startswith("pairwise")
    }
    leaked_weights = {
        row["column_name"]: float(row["weight"]) for row in leaked_meta if row["block_type"].startswith("pairwise")
    }
    deltas = []
    for column in sorted(set(clean_weights) | set(leaked_weights)):
        clean_weight = clean_weights.get(column, 0.0)
        leaked_weight = leaked_weights.get(column, 0.0)
        delta = clean_weight - leaked_weight
        if abs(delta) > 1e-9:
            deltas.append(
                {
                    "column_name": column,
                    "clean_weight": clean_weight,
                    "leaked_weight": leaked_weight,
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
            )
    deltas.sort(key=lambda row: (-row["abs_delta"], row["column_name"]))

    return {
        "task": task_name,
        "feature_columns_clean": len(clean_feature_cols),
        "feature_columns_leaked": len(leaked_feature_cols),
        "direct_profile_columns_clean": len(clean_direct_cols),
        "direct_profile_columns_leaked": len(leaked_direct_cols),
        "pairwise_columns_clean": len(clean_pairwise_cols),
        "pairwise_columns_leaked": len(leaked_pairwise_cols),
        "nonzero_rows_clean": _nonzero_row_count(clean_rows, clean_feature_cols),
        "nonzero_rows_leaked": _nonzero_row_count(leaked_rows, leaked_feature_cols),
        "pairwise_nonzero_rows_clean": _nonzero_row_count(clean_rows, clean_pairwise_cols),
        "pairwise_nonzero_rows_leaked": _nonzero_row_count(leaked_rows, leaked_pairwise_cols),
        "top_changed_associations": deltas[:8],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-tl03-dir", type=Path, required=True)
    parser.add_argument("--clean-tl04-dir", type=Path, required=True)
    parser.add_argument("--leaked-tl03-dir", type=Path, required=True)
    parser.add_argument("--leaked-tl04-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = {
        "TL03": _compare(
            "TL03",
            args.clean_tl03_dir,
            args.leaked_tl03_dir,
            "mechanistic_rbp_receptor_features_v1.csv",
            "mechanistic_rbp_receptor_feature_metadata_v1.csv",
        ),
        "TL04": _compare(
            "TL04",
            args.clean_tl04_dir,
            args.leaked_tl04_dir,
            "mechanistic_defense_evasion_features_v1.csv",
            "mechanistic_defense_evasion_feature_metadata_v1.csv",
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
