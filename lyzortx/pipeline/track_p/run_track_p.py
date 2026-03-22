#!/usr/bin/env python3
"""Entry point for Track P presentation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_p.steps import build_digital_phagogram


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["digital-phagogram", "all"],
        default="all",
        help="Track P step to run. 'all' runs the implemented presentation artifact step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"digital-phagogram", "all"}:
        build_digital_phagogram.main([])


if __name__ == "__main__":
    main()
