#!/usr/bin/env python3
"""Entry point for Track H in-silico cocktail recommendation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_h.steps import build_explained_recommendations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["explained-recommendations", "all"],
        default="all",
        help="Track H step to run. 'all' runs the implemented recommendation-report step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"explained-recommendations", "all"}:
        build_explained_recommendations.main([])


if __name__ == "__main__":
    main()
