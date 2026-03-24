#!/usr/bin/env python3
"""Entry point for Track I external-data ingestion helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_i.steps import (
    build_external_label_confidence_tiers,
    build_external_training_cohorts,
    build_incremental_lift_failure_analysis,
    build_strict_ablation_sequence,
    build_tier_b_weak_label_ingest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=[
            "weak-label-ingest",
            "external-confidence-tiers",
            "training-cohorts",
            "strict-ablation-sequence",
            "incremental-lift-failure-analysis",
            "all",
        ],
        default="all",
        help=(
            "Track I step to run. 'all' runs the implemented weak-label ingest, confidence-tier, cohort, strict-"
            "ablation, and lift-analysis steps."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    if args.step in {"weak-label-ingest", "all"}:
        build_tier_b_weak_label_ingest.main([])
    if args.step in {"external-confidence-tiers", "all"}:
        build_external_label_confidence_tiers.main([])
    if args.step in {"training-cohorts", "all"}:
        build_external_training_cohorts.main([])
    if args.step in {"strict-ablation-sequence", "all"}:
        build_strict_ablation_sequence.main([])
    if args.step in {"incremental-lift-failure-analysis", "all"}:
        build_incremental_lift_failure_analysis.main([])


if __name__ == "__main__":
    main()
