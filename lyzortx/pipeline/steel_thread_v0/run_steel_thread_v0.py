#!/usr/bin/env python3
"""Entry point for steel-thread v0 orchestration."""

from __future__ import annotations

import argparse

from lyzortx.pipeline.steel_thread_v0.checks import check_st01_regression
from lyzortx.pipeline.steel_thread_v0.steps import st01_label_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["st01", "check-st01"],
        default="st01",
        help="Steel-thread step to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step == "st01":
        st01_label_policy.main([])
    elif args.step == "check-st01":
        check_st01_regression.main([])
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
