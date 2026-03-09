#!/usr/bin/env python3
"""Entry point for steel-thread v0 orchestration."""

from __future__ import annotations

import argparse

from lyzortx.pipeline.steel_thread_v0.checks import check_st01_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st01b_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st02_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st03_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st03b_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st04_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st05_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st06_regression
from lyzortx.pipeline.steel_thread_v0.checks import check_st07_regression
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
    st03b_build_split_suite,
    st04_train_baselines,
    st05_calibrate_rank,
    st06_recommend_top3,
    st06b_compare_ranking_policies,
    st07_build_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=[
            "st01",
            "st01b",
            "st02",
            "st03",
            "st03b",
            "st04",
            "st05",
            "st06",
            "st06b",
            "st07",
            "check-st01",
            "check-st01b",
            "check-st02",
            "check-st03",
            "check-st03b",
            "check-st04",
            "check-st05",
            "check-st06",
            "check-st07",
        ],
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
    elif args.step == "st03":
        st03_build_splits.main([])
    elif args.step == "st03b":
        st03b_build_split_suite.main([])
    elif args.step == "st04":
        st04_train_baselines.main([])
    elif args.step == "st05":
        st05_calibrate_rank.main([])
    elif args.step == "st06":
        st06_recommend_top3.main([])
    elif args.step == "st06b":
        st06b_compare_ranking_policies.main([])
    elif args.step == "st07":
        st07_build_report.main([])
    elif args.step == "check-st01":
        check_st01_regression.main([])
    elif args.step == "check-st01b":
        check_st01b_regression.main([])
    elif args.step == "check-st02":
        check_st02_regression.main([])
    elif args.step == "check-st03":
        check_st03_regression.main([])
    elif args.step == "check-st03b":
        check_st03b_regression.main([])
    elif args.step == "check-st04":
        check_st04_regression.main([])
    elif args.step == "check-st05":
        check_st05_regression.main([])
    elif args.step == "check-st06":
        check_st06_regression.main([])
    elif args.step == "check-st07":
        check_st07_regression.main([])
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
