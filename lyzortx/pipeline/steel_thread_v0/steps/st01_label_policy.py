#!/usr/bin/env python3
"""ST0.1: Define label policy and uncertainty flags from raw interactions."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lyzortx.pipeline.steel_thread_v0.io.load_inputs import iter_raw_interactions
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json


@dataclass(frozen=True)
class LabelPolicyConfig:
    """Policy thresholds for hard labels and uncertainty flags."""

    expected_observations_per_pair: int = 9
    min_interpretable_obs_for_hard_negative: int = 5
    high_uninterpretable_fraction_threshold: float = 0.25


@dataclass
class PairAccumulator:
    """Raw observation counts for a bacteria-phage pair."""

    total_obs: int = 0
    score_1_count: int = 0
    score_0_count: int = 0
    score_n_count: int = 0
    replicates: set[str] = field(default_factory=set)
    dilutions: set[str] = field(default_factory=set)

    def add(self, row: Dict[str, str]) -> None:
        self.total_obs += 1
        score = row["score"]
        if score == "1":
            self.score_1_count += 1
        elif score == "0":
            self.score_0_count += 1
        else:
            self.score_n_count += 1
        self.replicates.add(row["replicate"])
        self.dilutions.add(row["log_dilution"])

    @property
    def interpretable_count(self) -> int:
        return self.score_1_count + self.score_0_count

    @property
    def uninterpretable_fraction(self) -> float:
        if self.total_obs == 0:
            return 1.0
        return self.score_n_count / self.total_obs


def evaluate_pair_policy(counts: PairAccumulator, policy: LabelPolicyConfig) -> Tuple[Optional[int], str, List[str]]:
    """Apply hard-label and uncertainty policy for one bacteria-phage pair."""
    hard_label: Optional[int]
    label_reason: str

    if counts.score_1_count > 0:
        hard_label = 1
        label_reason = "at_least_one_lysis_observed"
    elif counts.score_0_count >= policy.min_interpretable_obs_for_hard_negative:
        hard_label = 0
        label_reason = "sufficient_interpretable_no_lysis_observed"
    else:
        hard_label = None
        label_reason = "insufficient_interpretable_support_for_hard_negative"

    flags: List[str] = []
    if counts.score_n_count > 0:
        flags.append("has_uninterpretable")
    if counts.uninterpretable_fraction >= policy.high_uninterpretable_fraction_threshold:
        flags.append("high_uninterpretable_fraction")
    if counts.score_1_count > 0 and counts.score_0_count > 0:
        flags.append("conflicting_interpretable_observations")
    if counts.total_obs != policy.expected_observations_per_pair:
        flags.append("incomplete_observation_grid")
    if counts.interpretable_count < policy.min_interpretable_obs_for_hard_negative:
        flags.append("low_interpretable_support")
    if hard_label is None:
        flags.append("unresolved_label")

    return hard_label, label_reason, flags


def build_policy_definition(policy: LabelPolicyConfig) -> Dict[str, object]:
    """Build machine-readable policy metadata."""
    return {
        "policy_name": "steel_thread_v0_st01_label_policy",
        "policy_version": "v1",
        "hard_label_rules": {
            "positive": "hard_label=1 if at least one interpretable score is 1",
            "negative": ("hard_label=0 if no score is 1 and score_0_count >= min_interpretable_obs_for_hard_negative"),
            "unresolved": "hard_label is null otherwise",
        },
        "uncertainty_flags": {
            "has_uninterpretable": "pair contains one or more score='n' observations",
            "high_uninterpretable_fraction": ("score_n_count / total_obs >= high_uninterpretable_fraction_threshold"),
            "conflicting_interpretable_observations": "pair contains both score='0' and score='1'",
            "incomplete_observation_grid": "total_obs differs from expected_observations_per_pair",
            "low_interpretable_support": ("interpretable_count < min_interpretable_obs_for_hard_negative"),
            "unresolved_label": "pair does not satisfy hard positive or hard negative rule",
        },
        "thresholds": asdict(policy),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-interactions-path",
        type=Path,
        default=Path("data/interactions/raw/raw_interactions.csv"),
        help="Input raw interactions CSV path (semicolon delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for policy and audit artifacts.",
    )
    parser.add_argument(
        "--expected-observations-per-pair",
        type=int,
        default=9,
        help="Expected raw observation count per (bacteria, phage) pair.",
    )
    parser.add_argument(
        "--min-interpretable-obs-for-hard-negative",
        type=int,
        default=5,
        help="Minimum score='0' count to allow a hard negative label when score='1' is absent.",
    )
    parser.add_argument(
        "--high-uninterpretable-fraction-threshold",
        type=float,
        default=0.25,
        help="Threshold for flagging high uninterpretable fractions.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    policy = LabelPolicyConfig(
        expected_observations_per_pair=args.expected_observations_per_pair,
        min_interpretable_obs_for_hard_negative=args.min_interpretable_obs_for_hard_negative,
        high_uninterpretable_fraction_threshold=args.high_uninterpretable_fraction_threshold,
    )

    output_dir = args.output_dir
    ensure_directory(output_dir)

    pair_counts: Dict[Tuple[str, str], PairAccumulator] = {}
    bacteria_ids: set[str] = set()
    phage_ids: set[str] = set()
    raw_row_count = 0

    for row in iter_raw_interactions(args.raw_interactions_path):
        raw_row_count += 1
        bacteria = row["bacteria"]
        phage = row["phage"]
        bacteria_ids.add(bacteria)
        phage_ids.add(phage)
        key = (bacteria, phage)
        pair_counts.setdefault(key, PairAccumulator()).add(row)

    flag_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()

    audit_rows: List[Dict[str, object]] = []
    for bacteria, phage in sorted(pair_counts):
        counts = pair_counts[(bacteria, phage)]
        hard_label, label_reason, flags = evaluate_pair_policy(counts, policy)
        for flag in flags:
            flag_counter[flag] += 1

        label_key = "unresolved" if hard_label is None else str(hard_label)
        label_counter[label_key] += 1

        audit_rows.append(
            {
                "bacteria": bacteria,
                "phage": phage,
                "total_obs": counts.total_obs,
                "score_1_count": counts.score_1_count,
                "score_0_count": counts.score_0_count,
                "score_n_count": counts.score_n_count,
                "interpretable_count": counts.interpretable_count,
                "uninterpretable_fraction": round(counts.uninterpretable_fraction, 6),
                "hard_label_any_lysis": "" if hard_label is None else hard_label,
                "label_reason": label_reason,
                "include_in_training": int(hard_label is not None),
                "uncertainty_flags": "|".join(flags),
                "uncertainty_flag_count": len(flags),
                "n_distinct_replicates": len(counts.replicates),
                "n_distinct_dilutions": len(counts.dilutions),
                "missing_obs_count": max(policy.expected_observations_per_pair - counts.total_obs, 0),
            }
        )

    total_pairs = len(pair_counts)
    labeled_pairs = label_counter["0"] + label_counter["1"]
    expected_full_grid_pairs = len(bacteria_ids) * len(phage_ids)

    policy_definition = build_policy_definition(policy)
    policy_definition["generated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    policy_definition["input_raw_interactions_path"] = str(args.raw_interactions_path)

    policy_audit = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_raw_interactions_path": str(args.raw_interactions_path),
        "raw_row_count": raw_row_count,
        "distinct_bacteria_count": len(bacteria_ids),
        "distinct_phage_count": len(phage_ids),
        "observed_pair_count": total_pairs,
        "expected_full_grid_pair_count": expected_full_grid_pairs,
        "observed_vs_full_grid_pair_fraction": round(total_pairs / expected_full_grid_pairs, 6)
        if expected_full_grid_pairs
        else 0.0,
        "hard_label_counts": {
            "positive_1": label_counter["1"],
            "negative_0": label_counter["0"],
            "unresolved": label_counter["unresolved"],
        },
        "hard_label_coverage_fraction": round(labeled_pairs / total_pairs, 6) if total_pairs else 0.0,
        "uncertainty_flag_counts": dict(sorted(flag_counter.items())),
        "policy_thresholds": asdict(policy),
    }

    write_json(output_dir / "st01_label_policy_definition.json", policy_definition)
    write_json(output_dir / "st01_label_policy_audit.json", policy_audit)

    write_csv(
        output_dir / "st01_pair_label_audit.csv",
        fieldnames=[
            "bacteria",
            "phage",
            "total_obs",
            "score_1_count",
            "score_0_count",
            "score_n_count",
            "interpretable_count",
            "uninterpretable_fraction",
            "hard_label_any_lysis",
            "label_reason",
            "include_in_training",
            "uncertainty_flags",
            "uncertainty_flag_count",
            "n_distinct_replicates",
            "n_distinct_dilutions",
            "missing_obs_count",
        ],
        rows=audit_rows,
    )

    print("ST0.1 completed.")
    print(f"- Input rows: {raw_row_count}")
    print(f"- Pair count: {total_pairs}")
    print(f"- Hard positive pairs: {label_counter['1']}")
    print(f"- Hard negative pairs: {label_counter['0']}")
    print(f"- Unresolved pairs: {label_counter['unresolved']}")
    print(f"- Output directory: {output_dir}")


if __name__ == "__main__":
    main()
