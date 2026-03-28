#!/usr/bin/env python3
"""Entry point for Track L: Mechanistic Features from Pharokka Annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_l.steps import parse_annotations, run_pharokka


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["annotate", "parse", "all"],
        default="all",
        help=(
            "Track L step to run. "
            "'annotate' runs pharokka on all FNA files. "
            "'parse' parses pharokka outputs into summary tables. "
            "'all' runs both steps sequentially."
        ),
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        required=True,
        help="Path to pharokka database directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Threads per pharokka invocation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run pharokka even if output already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.step in {"annotate", "all"}:
        run_pharokka.main(
            [
                "--database-dir",
                str(args.database_dir),
                "--threads",
                str(args.threads),
                *(["--force"] if args.force else []),
            ]
        )
    if args.step in {"parse", "all"}:
        parse_annotations.main([])


if __name__ == "__main__":
    main()
