#!/usr/bin/env python3
"""Entry point for Track I external-data ingestion helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_i.steps import build_tier_b_weak_label_ingest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["weak-label-ingest", "all"],
        default="all",
        help="Track I step to run. 'all' runs the implemented weak-label ingest step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"weak-label-ingest", "all"}:
        build_tier_b_weak_label_ingest.main([])
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
