#!/usr/bin/env python3
"""TI09: Run the strict external-data ablation sequence in the planned source order."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_i.steps.build_external_training_cohorts import TRAINING_ARM_INDEX, TRAINING_ARM_ORDER

REQUIRED_COHORT_COLUMNS = (
    "pair_id",
    "source_system",
    "first_training_arm",
    "first_training_arm_index",
    "effective_training_weight",
)

STRICT_ABLATION_SOURCE_ADDITIONS: Dict[str, Tuple[str, ...]] = {
    "internal_only": ("internal",),
    "plus_vhrdb": ("vhrdb",),
    "plus_basel": ("basel",),
    "plus_klebphacol": ("klebphacol",),
    "plus_gpb": ("gpb",),
    "plus_tier_b": ("virus_host_db", "ncbi_virus_biosample"),
}

STRICT_SOURCE_ORDER = (
    "internal",
    "vhrdb",
    "basel",
    "klebphacol",
    "gpb",
    "virus_host_db",
    "ncbi_virus_biosample",
)
STRICT_SOURCE_INDEX = {source: idx for idx, source in enumerate(STRICT_SOURCE_ORDER)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-cohort-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/training_cohort_integration/ti08_training_cohort_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/strict_ablation_sequence"),
    )
    return parser.parse_args(argv)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_sources(sources: Sequence[str]) -> str:
    return "|".join(sources)


def _sorted_unique_sources(sources: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for source in sources:
        if source not in STRICT_SOURCE_INDEX:
            raise ValueError(f"Unsupported source_system in strict ablation sequence: {source!r}")
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return sorted(ordered, key=lambda source: STRICT_SOURCE_INDEX[source])


def _planned_sources_for_arm(arm: str) -> Tuple[str, ...]:
    if arm not in STRICT_ABLATION_SOURCE_ADDITIONS:
        raise ValueError(f"Unsupported training arm in strict ablation sequence: {arm!r}")
    return STRICT_ABLATION_SOURCE_ADDITIONS[arm]


def _cumulative_planned_sources_for_arm(arm: str) -> Tuple[str, ...]:
    arm_index = TRAINING_ARM_INDEX[arm]
    cumulative: List[str] = []
    for prior_arm in TRAINING_ARM_ORDER[: arm_index + 1]:
        cumulative.extend(_planned_sources_for_arm(prior_arm))
    return tuple(cumulative)


def compute_strict_ablation_summary(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    previous_pair_ids: set[str] = set()
    previous_row_count = 0

    normalized_rows = [_normalize_row(row) for row in rows]
    for row in normalized_rows:
        source_system = row.get("source_system", "")
        if source_system not in STRICT_SOURCE_INDEX:
            raise ValueError(f"Unsupported source_system in strict ablation sequence: {source_system!r}")

    for arm in TRAINING_ARM_ORDER:
        arm_index = TRAINING_ARM_INDEX[arm]
        arm_rows = [row for row in normalized_rows if int(row["first_training_arm_index"]) <= arm_index]
        pair_ids = {str(row["pair_id"]) for row in arm_rows}
        observed_sources = _sorted_unique_sources([str(row["source_system"]) for row in arm_rows])
        current_rows = [row for row in normalized_rows if int(row["first_training_arm_index"]) == arm_index]
        current_pair_ids = {str(row["pair_id"]) for row in current_rows}
        current_sources = _sorted_unique_sources([str(row["source_system"]) for row in current_rows])
        external_rows = [row for row in arm_rows if str(row["source_system"]) != "internal"]
        external_pair_ids = {str(row["pair_id"]) for row in external_rows}

        summary_rows.append(
            {
                "arm": arm,
                "arm_index": arm_index,
                "planned_source_systems_added": _join_sources(_planned_sources_for_arm(arm)),
                "observed_source_systems_added": _join_sources(current_sources),
                "cumulative_source_systems": _join_sources(observed_sources),
                "cumulative_row_count": len(arm_rows),
                "cumulative_pair_count": len(pair_ids),
                "cumulative_external_row_count": len(external_rows),
                "cumulative_external_pair_count": len(external_pair_ids),
                "new_rows_vs_previous_arm": len(arm_rows) - previous_row_count,
                "new_pairs_vs_previous_arm": len(pair_ids - previous_pair_ids),
                "new_observed_pairs_vs_previous_arm": len(current_pair_ids - previous_pair_ids),
                "cumulative_training_weight": safe_round(
                    sum(float(row["effective_training_weight"]) for row in arm_rows)
                ),
                "cumulative_planned_source_count": len(_cumulative_planned_sources_for_arm(arm)),
            }
        )
        previous_pair_ids = pair_ids
        previous_row_count = len(arm_rows)

    return summary_rows


def ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    rows = read_csv_rows(args.training_cohort_path, REQUIRED_COHORT_COLUMNS)
    summary_rows = compute_strict_ablation_summary(rows)

    summary_output_path = args.output_dir / "ti09_strict_ablation_summary.csv"
    manifest_output_path = args.output_dir / "ti09_strict_ablation_manifest.json"

    write_csv(summary_output_path, fieldnames=ordered_fieldnames(summary_rows), rows=summary_rows)
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_strict_ablation_sequence",
            "strict_ablation_order": list(TRAINING_ARM_ORDER),
            "input_paths": {
                "training_cohort_rows": str(args.training_cohort_path),
            },
            "input_hashes_sha256": {
                "training_cohort_rows": _hash_path(args.training_cohort_path),
            },
            "output_paths": {
                "summary": str(summary_output_path),
            },
        },
    )


if __name__ == "__main__":
    main()
