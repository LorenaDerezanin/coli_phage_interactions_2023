#!/usr/bin/env python3
"""Entry point for steel-thread v0 orchestration."""

from __future__ import annotations

import argparse

from lyzortx.pipeline.steel_thread_v0.checks import check_st01_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st01b_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st02_regression
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["st01", "st01b", "st02", "check-st01", "check-st01b", "check-st02"],
        default="st01",
        help="Steel-thread step to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step == "st01":
        st01_label_policy.main([])
    elif args.step == "st01b":
        st01b_confidence_tiers.main([])
    elif args.step == "st02":
        st02_build_pair_table.main([])
    elif args.step == "check-st01":
        check_st01_regression.main([])
    elif args.step == "check-st01b":
        check_st01b_regression.main([])
    elif args.step == "check-st02":
        check_st02_regression.main([])
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
