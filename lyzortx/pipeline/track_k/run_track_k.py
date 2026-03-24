#!/usr/bin/env python3
"""Entry point for Track K external-data lift measurement tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_k.steps import build_basel_lift_report
from lyzortx.pipeline.track_k.steps import build_vhrdb_lift_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["vhrdb-lift", "basel-lift", "all"],
        default="all",
        help="Track K step to run. 'all' runs the implemented lift-measurement steps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    if args.step in {"vhrdb-lift", "all"}:
        build_vhrdb_lift_report.main([])
    if args.step in {"basel-lift", "all"}:
        build_basel_lift_report.main([])


if __name__ == "__main__":
    main()
