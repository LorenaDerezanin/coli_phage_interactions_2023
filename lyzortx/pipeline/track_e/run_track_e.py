#!/usr/bin/env python3
"""Entry point for Track E pairwise compatibility feature builders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_e.steps import (
    build_defense_evasion_proxy_feature_block,
    build_rbp_receptor_compatibility_feature_block,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["rbp-receptor-compatibility", "defense-evasion-proxy", "all"],
        default="all",
        help="Track E step to run. 'all' runs the implemented Track E feature builders.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"rbp-receptor-compatibility", "all"}:
        build_rbp_receptor_compatibility_feature_block.main([])
    if args.step in {"defense-evasion-proxy", "all"}:
        build_defense_evasion_proxy_feature_block.main([])


if __name__ == "__main__":
    main()
