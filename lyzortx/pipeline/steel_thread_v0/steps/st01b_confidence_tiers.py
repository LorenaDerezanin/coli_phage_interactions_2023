#!/usr/bin/env python3
"""ST0.1b: Build strict confidence tiers from ST0.1 pair-level outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

INPUT_REQUIRED_COLUMNS = (
    "bacteria",
    "phage",
    "score_1_count",
    "score_0_count",
    "score_n_count",
    "interpretable_count",
    "hard_label_any_lysis",
)


@dataclass(frozen=True)
class StrictTierConfig:
    """Thresholds for strict confidence tiering."""

    min_positive_lysis_obs: int = 2
    min_positive_fraction: float = 0.4
    min_negative_no_lysis_obs: int = 7
    max_uninterpretable_obs: int = 1


def _read_st01_pair_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")

        missing = [c for c in INPUT_REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")

        rows: List[Dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            normalized = {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            for count_col in ("score_1_count", "score_0_count", "score_n_count", "interpretable_count"):
                try:
                    int(normalized[count_col])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid integer in {path}:{line_no} column={count_col}: {normalized[count_col]!r}"
                    ) from exc
            if normalized["hard_label_any_lysis"] not in {"", "0", "1"}:
                raise ValueError(
                    f"Unexpected hard_label_any_lysis at {path}:{line_no}: {normalized['hard_label_any_lysis']!r}"
                )
            rows.append(normalized)
        return rows


def _positive_fraction(interpretable_count: int, score_1_count: int) -> float:
    if interpretable_count <= 0:
        return 0.0
    return score_1_count / interpretable_count


def assign_strict_tier(row: Dict[str, str], config: StrictTierConfig) -> Tuple[str, str, str, int]:
    """Assign strict confidence tier and machine-readable reason."""
    score_1_count = int(row["score_1_count"])
    score_0_count = int(row["score_0_count"])
    score_n_count = int(row["score_n_count"])
    interpretable_count = int(row["interpretable_count"])
    hard_label = row["hard_label_any_lysis"]
    positive_fraction = _positive_fraction(
        interpretable_count=interpretable_count,
        score_1_count=score_1_count,
    )

    if (
        hard_label == "1"
        and score_1_count >= config.min_positive_lysis_obs
        and positive_fraction >= config.min_positive_fraction
        and score_n_count <= config.max_uninterpretable_obs
    ):
        return "high_conf_pos", "high_conf_positive_rule", "1", 1

    if (
        hard_label == "0"
        and score_0_count >= config.min_negative_no_lysis_obs
        and score_n_count <= config.max_uninterpretable_obs
    ):
        return "high_conf_neg", "high_conf_negative_rule", "0", 1

    return "ambiguous", "does_not_meet_strict_rules", "", 0


def build_policy_definition(config: StrictTierConfig) -> Dict[str, object]:
    return {
        "policy_name": "steel_thread_v0_st01b_confidence_tiers",
        "policy_version": "v1",
        "strict_tier_rules": {
            "high_conf_pos": (
                "hard_label_any_lysis=1 and score_1_count >= min_positive_lysis_obs and "
                "score_1_count / interpretable_count >= min_positive_fraction and "
                "score_n_count <= max_uninterpretable_obs"
            ),
            "high_conf_neg": (
                "hard_label_any_lysis=0 and score_0_count >= min_negative_no_lysis_obs and "
                "score_n_count <= max_uninterpretable_obs"
            ),
            "ambiguous": "all remaining pairs",
        },
        "thresholds": asdict(config),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st01-pair-audit-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st01_pair_label_audit.csv"),
        help="Input ST0.1 pair-level audit CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.1b artifacts.",
    )
    parser.add_argument(
        "--min-positive-lysis-obs",
        type=int,
        default=2,
        help="Minimum score_1_count to assign high_conf_pos.",
    )
    parser.add_argument(
        "--min-positive-fraction",
        type=float,
        default=0.4,
        help="Minimum score_1_count / interpretable_count for high_conf_pos.",
    )
    parser.add_argument(
        "--min-negative-no-lysis-obs",
        type=int,
        default=7,
        help="Minimum score_0_count for high_conf_neg when hard label is 0.",
    )
    parser.add_argument(
        "--max-uninterpretable-obs",
        type=int,
        default=1,
        help="Maximum score_n_count allowed for high-confidence tiers.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = StrictTierConfig(
        min_positive_lysis_obs=args.min_positive_lysis_obs,
        min_positive_fraction=args.min_positive_fraction,
        min_negative_no_lysis_obs=args.min_negative_no_lysis_obs,
        max_uninterpretable_obs=args.max_uninterpretable_obs,
    )
    output_dir = args.output_dir
    ensure_directory(output_dir)

    input_rows = _read_st01_pair_rows(args.st01_pair_audit_path)
    total_pairs = len(input_rows)
    if total_pairs == 0:
        raise ValueError(f"No rows found in {args.st01_pair_audit_path}.")

    tier_counter: Counter[str] = Counter()
    strict_label_counter: Counter[str] = Counter()
    strict_conflicting_counter: Counter[str] = Counter()
    strict_with_n_counter: Counter[str] = Counter()

    output_rows: List[Dict[str, object]] = []
    for row in input_rows:
        strict_tier, strict_tier_reason, strict_label, strict_include = assign_strict_tier(
            row=row,
            config=config,
        )
        tier_counter[strict_tier] += 1
        if strict_label:
            strict_label_counter[strict_label] += 1

        uncertainty_flags = row.get("uncertainty_flags", "")
        has_conflict = "conflicting_interpretable_observations" in uncertainty_flags
        has_uninterpretable = "has_uninterpretable" in uncertainty_flags
        if strict_include and has_conflict:
            strict_conflicting_counter[strict_tier] += 1
        if strict_include and has_uninterpretable:
            strict_with_n_counter[strict_tier] += 1

        score_1_count = int(row["score_1_count"])
        interpretable_count = int(row["interpretable_count"])
        positive_fraction = _positive_fraction(
            interpretable_count=interpretable_count,
            score_1_count=score_1_count,
        )

        output_row = dict(row)
        output_row["positive_fraction_interpretable"] = round(positive_fraction, 6)
        output_row["strict_confidence_tier"] = strict_tier
        output_row["strict_tier_reason"] = strict_tier_reason
        output_row["strict_label"] = strict_label
        output_row["strict_include_in_training"] = strict_include
        output_rows.append(output_row)

    strict_total = tier_counter["high_conf_pos"] + tier_counter["high_conf_neg"]
    strict_coverage_fraction = strict_total / total_pairs
    strict_pos_fraction = strict_label_counter["1"] / strict_total if strict_total else 0.0

    policy_definition = build_policy_definition(config)
    policy_definition["generated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    policy_definition["input_st01_pair_audit_path"] = str(args.st01_pair_audit_path)

    audit = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_st01_pair_audit_path": str(args.st01_pair_audit_path),
        "total_pairs": total_pairs,
        "strict_tier_counts": dict(sorted(tier_counter.items())),
        "strict_label_counts": {
            "positive_1": strict_label_counter["1"],
            "negative_0": strict_label_counter["0"],
            "ambiguous": tier_counter["ambiguous"],
        },
        "strict_coverage_fraction": round(strict_coverage_fraction, 6),
        "strict_positive_fraction_within_strict_slice": round(strict_pos_fraction, 6),
        "strict_conflicting_counts": dict(sorted(strict_conflicting_counter.items())),
        "strict_has_uninterpretable_counts": dict(sorted(strict_with_n_counter.items())),
        "strict_thresholds": asdict(config),
    }

    output_path_policy = output_dir / "st01b_confidence_policy_definition.json"
    output_path_audit = output_dir / "st01b_confidence_audit.json"
    output_path_pairs = output_dir / "st01b_pair_confidence_audit.csv"

    write_json(output_path_policy, policy_definition)
    write_json(output_path_audit, audit)

    ordered_columns = list(input_rows[0].keys()) + [
        "positive_fraction_interpretable",
        "strict_confidence_tier",
        "strict_tier_reason",
        "strict_label",
        "strict_include_in_training",
    ]
    write_csv(
        output_path_pairs,
        fieldnames=ordered_columns,
        rows=output_rows,
    )

    print("ST0.1b completed.")
    print(f"- Input pair rows: {total_pairs}")
    print(f"- High-confidence positive: {tier_counter['high_conf_pos']}")
    print(f"- High-confidence negative: {tier_counter['high_conf_neg']}")
    print(f"- Ambiguous: {tier_counter['ambiguous']}")
    print(f"- Strict coverage fraction: {strict_coverage_fraction:.6f}")
    print(f"- Output directory: {output_dir}")


if __name__ == "__main__":
    main()
