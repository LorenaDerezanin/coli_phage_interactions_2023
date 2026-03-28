#!/usr/bin/env python3
"""Entry point for Track I external-data download and ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_i.steps import (
    build_tier_a_additional_source_ingests,
    build_tier_a_harmonized_pairs,
    build_tier_a_vhrdb_ingest,
    build_tier_b_weak_label_ingest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=[
            "tier-a-ingest",
            "tier-a-harmonization",
            "weak-label-ingest",
            "all",
        ],
        default="all",
        help="Track I step to run. 'all' runs Tier A ingest + harmonization + Tier B weak-label ingest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    if args.step in {"tier-a-ingest", "all"}:
        build_tier_a_vhrdb_ingest.main([])
        build_tier_a_additional_source_ingests.main([])
    if args.step in {"tier-a-harmonization", "all"}:
        build_tier_a_harmonized_pairs.main([])
    if args.step in {"weak-label-ingest", "all"}:
        build_tier_b_weak_label_ingest.main([])


if __name__ == "__main__":
    main()
