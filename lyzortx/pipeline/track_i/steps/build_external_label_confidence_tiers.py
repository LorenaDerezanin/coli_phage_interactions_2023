#!/usr/bin/env python3
"""TI07: Assign unified confidence tiers and training weights for external labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows

REQUIRED_SOURCE_REGISTRY_COLUMNS = (
    "source_id",
    "source_type",
    "confidence_tier",
    "confidence_basis",
    "host_resolution",
    "notes",
)
REQUIRED_EXTERNAL_COLUMNS = ("pair_id", "source_system")

OUTPUT_APPEND_COLUMNS = [
    "external_label_confidence_tier",
    "external_label_confidence_score",
    "external_label_training_weight",
    "external_label_confidence_reason",
    "external_label_include_in_training",
]


@dataclass(frozen=True)
class ExternalConfidenceConfig:
    """Tier scores and training weights for external labels."""

    high_score: int = 3
    medium_score: int = 2
    low_score: int = 1
    high_weight: float = 1.0
    medium_weight: float = 0.5
    low_weight: float = 0.2
    exclude_weight: float = 0.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry-path",
        type=Path,
        default=Path("lyzortx/research_notes/external_data/source_registry.csv"),
    )
    parser.add_argument(
        "--tier-a-ingest-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/tier_a_harmonization/ti05_tier_a_harmonized_pairs.csv"),
    )
    parser.add_argument(
        "--tier-b-ingest-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/tier_b_weak_label_ingest/ti06_weak_label_ingested_pairs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_i/external_label_confidence_tiers"),
    )
    return parser.parse_args(argv)


def _normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}


def read_external_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [column for column in REQUIRED_EXTERNAL_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [_normalize_row(row) for row in reader if row.get("source_system", "").strip() != "internal"]


def load_source_registry(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path, REQUIRED_SOURCE_REGISTRY_COLUMNS)
    return {row["source_id"]: row for row in rows}


def _split_pipe_values(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def base_confidence_score(source_registry_row: Mapping[str, str]) -> Tuple[int, str]:
    confidence_tier = source_registry_row.get("confidence_tier", "")
    source_type = source_registry_row.get("source_type", "")
    if confidence_tier == "A":
        return 3, "assay_backed_external_source"
    if source_type == "metadata_knowledgebase":
        return 2, "curated_metadata_association"
    return 1, "repository_or_submitter_metadata"


def penalty_reasons(row: Mapping[str, str]) -> List[str]:
    penalties: List[str] = []
    if row.get("source_disagreement_flag", "") == "1":
        penalties.append("source_disagreement")

    qc_flag = row.get("source_qc_flag", "")
    if qc_flag and qc_flag != "ok":
        penalties.append(f"qc_{qc_flag}")

    resolution_status = set(_split_pipe_values(row.get("source_resolution_status", "")))
    if resolution_status & {"unresolved", "missing"}:
        penalties.append("unresolved_entity_mapping")

    return penalties


def score_to_tier(score: int, config: ExternalConfidenceConfig) -> Tuple[str, float, int]:
    if score >= config.high_score:
        return "high", config.high_weight, 1
    if score >= config.medium_score:
        return "medium", config.medium_weight, 1
    if score >= config.low_score:
        return "low", config.low_weight, 1
    return "exclude", config.exclude_weight, 0


def assign_external_confidence(
    row: Mapping[str, str],
    *,
    source_registry_row: Mapping[str, str],
    config: ExternalConfidenceConfig,
) -> Dict[str, object]:
    base_score, base_reason = base_confidence_score(source_registry_row)
    penalties = penalty_reasons(row)
    adjusted_score = max(base_score - len(penalties), 0)
    confidence_tier, training_weight, include_in_training = score_to_tier(adjusted_score, config)
    reason_parts = [f"base_{base_reason}", *penalties]
    return {
        "external_label_confidence_tier": confidence_tier,
        "external_label_confidence_score": adjusted_score,
        "external_label_training_weight": training_weight,
        "external_label_confidence_reason": "|".join(reason_parts),
        "external_label_include_in_training": include_in_training,
    }


def apply_external_confidence_policy(
    rows: Sequence[Mapping[str, str]],
    *,
    source_registry_rows: Mapping[str, Mapping[str, str]],
    config: ExternalConfidenceConfig,
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for row in rows:
        source_system = row.get("source_system", "")
        if source_system not in source_registry_rows:
            raise ValueError(f"Source {source_system!r} is missing from source registry")
        assignment = assign_external_confidence(
            row,
            source_registry_row=source_registry_rows[source_system],
            config=config,
        )
        output_row = dict(row)
        output_row.update(assignment)
        output_rows.append(output_row)
    return output_rows


def compute_summary_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_tier: Counter[str] = Counter()
    by_source: Counter[Tuple[str, str]] = Counter()
    by_include: Counter[str] = Counter()

    for row in rows:
        tier = str(row["external_label_confidence_tier"])
        source_system = str(row.get("source_system", ""))
        include = str(row["external_label_include_in_training"])
        by_tier[tier] += 1
        by_source[(source_system, tier)] += 1
        by_include[include] += 1

    summary_rows: List[Dict[str, object]] = []
    for tier, count in sorted(by_tier.items()):
        summary_rows.append({"slice_type": "confidence_tier", "slice_value": tier, "row_count": count})
    for (source_system, tier), count in sorted(by_source.items()):
        summary_rows.append(
            {
                "slice_type": "source_and_tier",
                "slice_value": f"{source_system}:{tier}",
                "row_count": count,
            }
        )
    for include, count in sorted(by_include.items()):
        summary_rows.append({"slice_type": "include_in_training", "slice_value": include, "row_count": count})
    return summary_rows


def build_policy_definition(config: ExternalConfidenceConfig) -> Dict[str, object]:
    return {
        "policy_name": "track_i_external_label_confidence_tiers",
        "policy_version": "v1",
        "base_score_rules": {
            "3": "Tier A direct assay-backed external sources from source_registry confidence_tier=A",
            "2": "Curated metadata knowledgebases such as Virus-Host DB",
            "1": "Repository or submitter metadata such as NCBI Virus/BioSample",
        },
        "downgrade_rules": {
            "source_disagreement": "subtract 1 when source_disagreement_flag=1",
            "non_ok_qc": "subtract 1 when source_qc_flag is present and not ok",
            "unresolved_entity_mapping": ("subtract 1 when source_resolution_status contains unresolved or missing"),
        },
        "tier_mapping": {
            "high": "score >= 3",
            "medium": "score == 2",
            "low": "score == 1",
            "exclude": "score <= 0",
        },
        "training_weights": {
            "high": config.high_weight,
            "medium": config.medium_weight,
            "low": config.low_weight,
            "exclude": config.exclude_weight,
        },
        "thresholds": asdict(config),
    }


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
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
    config = ExternalConfidenceConfig()
    source_registry_rows = load_source_registry(args.source_registry_path)

    input_paths: Dict[str, Path] = {}
    input_rows: List[Dict[str, str]] = []
    if args.tier_a_ingest_path.exists():
        input_paths["tier_a_ingest"] = args.tier_a_ingest_path
        input_rows.extend(read_external_rows(args.tier_a_ingest_path))
    if args.tier_b_ingest_path.exists():
        input_paths["tier_b_ingest"] = args.tier_b_ingest_path
        input_rows.extend(read_external_rows(args.tier_b_ingest_path))

    if not input_rows:
        raise ValueError("No external label inputs were found. Expected Tier A and/or Tier B ingest outputs.")

    output_rows = apply_external_confidence_policy(
        input_rows,
        source_registry_rows=source_registry_rows,
        config=config,
    )
    summary_rows = compute_summary_rows(output_rows)
    policy_definition = build_policy_definition(config)

    combined_output_path = args.output_dir / "ti07_external_label_confidence_pairs.csv"
    summary_output_path = args.output_dir / "ti07_external_label_confidence_summary.csv"
    policy_output_path = args.output_dir / "ti07_external_label_confidence_policy.json"
    manifest_output_path = args.output_dir / "ti07_external_label_confidence_manifest.json"

    write_csv(combined_output_path, fieldnames=ordered_fieldnames(output_rows), rows=output_rows)
    write_csv(
        summary_output_path,
        fieldnames=["slice_type", "slice_value", "row_count"],
        rows=summary_rows,
    )

    policy_definition["generated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    policy_definition["input_paths"] = {
        "source_registry": str(args.source_registry_path),
        **{name: str(path) for name, path in input_paths.items()},
    }
    write_json(policy_output_path, policy_definition)
    write_json(
        manifest_output_path,
        {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "step_name": "build_external_label_confidence_tiers",
            "input_paths": {
                "source_registry": str(args.source_registry_path),
                **{name: str(path) for name, path in input_paths.items()},
            },
            "input_hashes_sha256": {
                "source_registry": _hash_path(args.source_registry_path),
                **{name: _hash_path(path) for name, path in input_paths.items()},
            },
            "output_paths": {
                "pairs": str(combined_output_path),
                "summary": str(summary_output_path),
                "policy": str(policy_output_path),
            },
        },
    )


if __name__ == "__main__":
    main()
