#!/usr/bin/env python3
"""TE02: Build leakage-safe defense-evasion proxy features from family-by-subtype training rates."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps.st03b_build_split_suite import family_key

OUTPUT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "defense_evasion_expected_score",
    "defense_evasion_mean_score",
    "defense_evasion_supported_subtype_count",
    "defense_evasion_family_training_pair_count",
)

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "defense_evasion_expected_score": {
        "type": "float",
        "transform": (
            "Sum of leakage-safe phage-family lysis-rate estimates across all host defense subtypes "
            "present on the pair."
        ),
    },
    "defense_evasion_mean_score": {
        "type": "float",
        "transform": (
            "defense_evasion_expected_score divided by the host's present defense subtype count; "
            "0 when none are present."
        ),
    },
    "defense_evasion_supported_subtype_count": {
        "type": "integer",
        "transform": (
            "Count of host defense subtypes whose phage-family lysis-rate estimate was observed in the leakage-safe "
            "training view for this row."
        ),
    },
    "defense_evasion_family_training_pair_count": {
        "type": "integer",
        "transform": (
            "Number of leakage-safe training pairs available for this phage family in the row's fold/holdout scenario."
        ),
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
        help="Input pair table containing phage_family and host_defense_subtype_* columns.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments used for leakage-safe training views.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_e/defense_evasion_proxy_feature_block"),
        help="Directory for generated TE02 outputs.",
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


def resolve_training_flag_column(rows: Sequence[Mapping[str, str]], path: Path) -> str:
    candidates = ("include_in_training", "label_hard_include_in_training")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise ValueError(f"Missing training-flag column in {path}; expected one of: {', '.join(candidates)}")


def detect_defense_subtype_columns(rows: Sequence[Mapping[str, str]], path: Path) -> List[str]:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    columns = sorted(column for column in rows[0] if column.startswith("host_defense_subtype_"))
    if not columns:
        raise ValueError(f"No host_defense_subtype_* columns found in {path}")
    return columns


def merge_pair_rows(
    pair_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
    training_flag_column: str,
    defense_subtype_columns: Sequence[str],
) -> List[Dict[str, object]]:
    split_by_pair_id = {row["pair_id"]: row for row in split_rows}

    merged_rows: List[Dict[str, object]] = []
    for row in pair_rows:
        pair_id = row.get("pair_id", "")
        if not pair_id:
            raise ValueError("Pair-table row is missing pair_id")
        split_row = split_by_pair_id.get(pair_id)
        if split_row is None:
            raise ValueError(f"Missing ST0.3 split assignment for pair_id {pair_id!r}")

        split_fold_raw = split_row.get("split_cv5_fold", "")
        split_fold = int(split_fold_raw) if split_fold_raw else -1
        subtype_state = {column: _parse_binary_flag(row.get(column, "")) for column in defense_subtype_columns}

        merged_rows.append(
            {
                "pair_id": pair_id,
                "bacteria": row.get("bacteria", ""),
                "phage": row.get("phage", ""),
                "phage_family": family_key(row.get("phage_family", "")),
                "label_hard_any_lysis": row.get("label_hard_any_lysis", ""),
                "include_in_training": row.get(training_flag_column, ""),
                "split_holdout": split_row.get("split_holdout", ""),
                "split_cv5_fold": split_fold,
                "host_defense_subtypes": subtype_state,
            }
        )

    merged_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    return merged_rows


def _is_training_example(row: Mapping[str, object]) -> bool:
    return str(row.get("include_in_training", "")) == "1" and str(row.get("split_holdout", "")) == "train_non_holdout"


def _scenario_key(row: Mapping[str, object]) -> str:
    if str(row.get("split_holdout", "")) == "holdout_test":
        return "holdout"
    return f"cv_fold_{int(row['split_cv5_fold'])}"


def build_training_family_defense_profiles(
    merged_rows: Sequence[Mapping[str, object]],
    defense_subtype_columns: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    fold_ids = sorted(
        {
            int(row["split_cv5_fold"])
            for row in merged_rows
            if str(row.get("split_holdout", "")) == "train_non_holdout" and int(row["split_cv5_fold"]) >= 0
        }
    )

    training_rows_by_scenario: Dict[str, List[Mapping[str, object]]] = {
        "holdout": [row for row in merged_rows if _is_training_example(row)]
    }
    for fold_id in fold_ids:
        training_rows_by_scenario[f"cv_fold_{fold_id}"] = [
            row for row in merged_rows if _is_training_example(row) and int(row["split_cv5_fold"]) != fold_id
        ]

    scenario_profiles: Dict[str, Dict[str, object]] = {}
    for scenario, training_rows in training_rows_by_scenario.items():
        family_pair_counts: Counter[str] = Counter()
        subtype_denominators: Counter[Tuple[str, str]] = Counter()
        subtype_positive_sums: Dict[Tuple[str, str], float] = defaultdict(float)

        for row in training_rows:
            family = str(row["phage_family"])
            family_pair_counts[family] += 1
            label = float(_parse_binary_flag(str(row.get("label_hard_any_lysis", ""))))
            subtype_state = row["host_defense_subtypes"]
            for subtype in defense_subtype_columns:
                if subtype_state[subtype] != 1:
                    continue
                subtype_denominators[(family, subtype)] += 1
                subtype_positive_sums[(family, subtype)] += label

        subtype_rates = {
            key: subtype_positive_sums[key] / count for key, count in subtype_denominators.items() if count > 0
        }
        scenario_profiles[scenario] = {
            "family_pair_counts": dict(family_pair_counts),
            "subtype_denominators": dict(subtype_denominators),
            "subtype_rates": subtype_rates,
        }

    return scenario_profiles


def build_feature_rows(
    merged_rows: Sequence[Mapping[str, object]],
    scenario_profiles: Mapping[str, Mapping[str, object]],
    defense_subtype_columns: Sequence[str],
) -> List[Dict[str, object]]:
    feature_rows: List[Dict[str, object]] = []
    for row in merged_rows:
        scenario = _scenario_key(row)
        scenario_profile = scenario_profiles[scenario]
        family = str(row["phage_family"])
        family_pair_counts = scenario_profile["family_pair_counts"]
        subtype_rates = scenario_profile["subtype_rates"]
        subtype_state = row["host_defense_subtypes"]

        present_subtypes = [subtype for subtype in defense_subtype_columns if subtype_state[subtype] == 1]
        expected_score = 0.0
        supported_subtype_count = 0
        for subtype in present_subtypes:
            key = (family, subtype)
            if key in subtype_rates:
                supported_subtype_count += 1
            expected_score += float(subtype_rates.get(key, 0.0))

        mean_score = expected_score / len(present_subtypes) if present_subtypes else 0.0
        feature_rows.append(
            {
                "pair_id": row["pair_id"],
                "bacteria": row["bacteria"],
                "phage": row["phage"],
                "defense_evasion_expected_score": round(expected_score, 6),
                "defense_evasion_mean_score": round(mean_score, 6),
                "defense_evasion_supported_subtype_count": supported_subtype_count,
                "defense_evasion_family_training_pair_count": int(family_pair_counts.get(family, 0)),
            }
        )

    feature_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    return feature_rows


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


def build_family_defense_rate_rows(
    scenario_profiles: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for scenario, profile in sorted(scenario_profiles.items()):
        family_pair_counts = profile["family_pair_counts"]
        subtype_denominators = profile["subtype_denominators"]
        subtype_rates = profile["subtype_rates"]
        for (family, subtype), denominator in sorted(subtype_denominators.items()):
            output_rows.append(
                {
                    "scenario": scenario,
                    "phage_family": family,
                    "defense_subtype": subtype,
                    "family_training_pair_count": int(family_pair_counts.get(family, 0)),
                    "subtype_training_pair_count": int(denominator),
                    "average_lysis_rate": round(float(subtype_rates[(family, subtype)]), 6),
                }
            )
    return output_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    pair_rows = read_delimited_rows(args.pair_table_path)
    split_rows = read_delimited_rows(args.st03_split_assignments_path)
    _require_columns(
        pair_rows,
        args.pair_table_path,
        ("pair_id", "bacteria", "phage", "phage_family", "label_hard_any_lysis"),
    )
    _require_columns(split_rows, args.st03_split_assignments_path, ("pair_id", "split_holdout", "split_cv5_fold"))

    training_flag_column = resolve_training_flag_column(pair_rows, args.pair_table_path)
    defense_subtype_columns = detect_defense_subtype_columns(pair_rows, args.pair_table_path)
    merged_rows = merge_pair_rows(pair_rows, split_rows, training_flag_column, defense_subtype_columns)
    scenario_profiles = build_training_family_defense_profiles(merged_rows, defense_subtype_columns)
    feature_rows = build_feature_rows(merged_rows, scenario_profiles, defense_subtype_columns)
    metadata_rows = build_feature_metadata(args.pair_table_path)
    family_rate_rows = build_family_defense_rate_rows(scenario_profiles)

    feature_output_path = args.output_dir / f"defense_evasion_proxy_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"defense_evasion_proxy_feature_metadata_{args.version}.csv"
    family_rates_output_path = args.output_dir / f"family_defense_lysis_rates_{args.version}.csv"
    manifest_output_path = args.output_dir / f"defense_evasion_proxy_manifest_{args.version}.json"

    write_csv(feature_output_path, ["pair_id", "bacteria", "phage", *OUTPUT_FEATURE_COLUMNS], feature_rows)
    write_csv(metadata_output_path, ["column_name", "type", "source_path", "transform"], metadata_rows)
    write_csv(
        family_rates_output_path,
        [
            "scenario",
            "phage_family",
            "defense_subtype",
            "family_training_pair_count",
            "subtype_training_pair_count",
            "average_lysis_rate",
        ],
        family_rate_rows,
    )

    holdout_profile = scenario_profiles["holdout"]
    manifest = {
        "step_name": "build_defense_evasion_proxy_feature_block",
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_count": len(feature_rows),
        "distinct_bacteria_count": len({row["bacteria"] for row in feature_rows}),
        "distinct_phage_count": len({row["phage"] for row in feature_rows}),
        "distinct_family_count": len({family_key(row["phage_family"]) for row in pair_rows}),
        "feature_count": len(OUTPUT_FEATURE_COLUMNS),
        "defense_subtype_count": len(defense_subtype_columns),
        "training_profiles": {
            "scenario_count": len(scenario_profiles),
            "holdout_training_pair_count": sum(int(value) for value in holdout_profile["family_pair_counts"].values()),
            "holdout_family_subtype_rate_count": len(holdout_profile["subtype_rates"]),
        },
        "inputs": {
            "pair_table": {"path": str(args.pair_table_path), "sha256": _sha256(args.pair_table_path)},
            "st03_split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
            },
        },
        "policies": {
            "training_flag_column": training_flag_column,
            "phage_family_missing_token": "__MISSING_PHAGE_FAMILY__",
            "training_views": {
                "holdout_rows": "Use only train_non_holdout rows with training_flag=1.",
                "cv_rows": "Use only train_non_holdout rows with training_flag=1 from all other CV folds.",
            },
            "rate_definition": (
                "Average label_hard_any_lysis among leakage-safe training pairs where the defense subtype is present."
            ),
            "score_definition": "Sum family-by-subtype lysis rates across the host's present defense subtype columns.",
        },
        "outputs": {
            "feature_csv": str(feature_output_path),
            "feature_metadata_csv": str(metadata_output_path),
            "family_defense_rate_csv": str(family_rates_output_path),
        },
    }
    write_json(manifest_output_path, manifest)

    print(f"Wrote TE02 defense-evasion proxy features to {feature_output_path}")
    print(f"- Pair rows: {len(feature_rows)}")
    print(f"- Defense subtype columns: {len(defense_subtype_columns)}")
    print(f"- Holdout training pairs: {manifest['training_profiles']['holdout_training_pair_count']}")
    print(f"- Holdout family-subtype rates: {manifest['training_profiles']['holdout_family_subtype_rate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
