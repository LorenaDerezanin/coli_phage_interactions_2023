#!/usr/bin/env python3
"""Entry point for Track D phage sequence processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_d.steps import build_phage_protein_sets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["protein-sets", "all"],
        default="all",
        help="Track D step to run. 'all' currently runs the phage protein-set builder.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step in {"protein-sets", "all"}:
        build_phage_protein_sets.main([])
        return
    raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
