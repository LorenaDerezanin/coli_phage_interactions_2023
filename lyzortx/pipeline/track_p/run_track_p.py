#!/usr/bin/env python3
"""Entry point for Track P presentation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_p.steps import build_digital_phagogram, build_feature_lift_visualization
from lyzortx.pipeline.track_p.steps import build_panel_coverage_heatmap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["digital-phagogram", "feature-lift-visualization", "panel-coverage-heatmap", "all"],
        default="all",
        help="Track P step to run. 'all' runs all implemented presentation artifact steps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"digital-phagogram", "all"}:
        build_digital_phagogram.main([])
    if args.step in {"feature-lift-visualization", "all"}:
        build_feature_lift_visualization.main([])
    if args.step in {"panel-coverage-heatmap", "all"}:
        build_panel_coverage_heatmap.main([])


if __name__ == "__main__":
    main()
