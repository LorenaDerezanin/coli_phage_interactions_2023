#!/usr/bin/env python3
"""Entry point for Track A data integrity and label generation."""

from __future__ import annotations

import argparse

from lyzortx.pipeline.track_a.checks import check_track_a_integrity
from lyzortx.pipeline.track_a.steps import build_track_a_foundation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["build", "check", "all"],
        default="all",
        help="Track A step to run. 'all' runs build then checks.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="If set, fail checks when warning-level integrity checks fail.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step == "build":
        build_track_a_foundation.main([])
    elif args.step == "check":
        check_args: list[str] = []
        if args.fail_on_warnings:
            check_args.append("--fail-on-warnings")
        check_track_a_integrity.main(check_args)
    elif args.step == "all":
        check_args = ["--run-build"]
        if args.fail_on_warnings:
            check_args.append("--fail-on-warnings")
        check_track_a_integrity.main(check_args)
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
