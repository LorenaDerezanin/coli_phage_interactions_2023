#!/usr/bin/env python3
"""Entry point for Track J v1 release regeneration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.pipeline.track_c.steps import build_extended_host_surface_feature_block
from lyzortx.pipeline.track_c.steps import build_omp_receptor_variant_feature_block
from lyzortx.pipeline.track_c.steps import build_receptor_surface_feature_block
from lyzortx.pipeline.track_c.steps import build_v1_host_feature_pair_table
from lyzortx.pipeline.track_d import run_track_d
from lyzortx.pipeline.track_e import run_track_e
from lyzortx.pipeline.track_g import run_track_g
from lyzortx.pipeline.track_h import run_track_h
from lyzortx.pipeline.steel_thread_v0.steps import st01_label_policy
from lyzortx.pipeline.steel_thread_v0.steps import st01b_confidence_tiers
from lyzortx.pipeline.steel_thread_v0.steps import st02_build_pair_table
from lyzortx.pipeline.steel_thread_v0.steps import st03_build_splits

StepRunner = Tuple[str, Callable[[], None]]


def foundation_runners() -> Tuple[StepRunner, ...]:
    return (
        ("st01-label-policy", lambda: st01_label_policy.main([])),
        ("st01b-confidence-tiers", lambda: st01b_confidence_tiers.main([])),
        ("st02-pair-table", lambda: st02_build_pair_table.main([])),
        ("st03-splits", lambda: st03_build_splits.main([])),
    )


def feature_block_runners() -> Tuple[StepRunner, ...]:
    return (
        ("track-c-receptor-surface", lambda: build_receptor_surface_feature_block.main([])),
        ("track-c-omp-variants", lambda: build_omp_receptor_variant_feature_block.main([])),
        ("track-c-extended-surface", lambda: build_extended_host_surface_feature_block.main([])),
        ("track-c-v1-pair-table", lambda: build_v1_host_feature_pair_table.main([])),
    )


def release_runners() -> Tuple[StepRunner, ...]:
    return (
        *foundation_runners(),
        *feature_block_runners(),
        ("track-d", lambda: run_track_d.main(["--step", "all"])),
        ("track-e", lambda: run_track_e.main(["--step", "all"])),
        ("track-g", lambda: run_track_g.main(["--step", "all"])),
        ("track-h", lambda: run_track_h.main(["--step", "all"])),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["foundation", "feature-blocks", "modeling", "recommendations", "all"],
        default="all",
        help="Track J step to run. 'all' runs the full release regeneration sequence.",
    )
    return parser.parse_args(argv)


def _runners_for_step(step: str) -> Iterable[StepRunner]:
    runners = release_runners()
    if step == "foundation":
        return runners[:4]
    if step == "feature-blocks":
        return runners[4:8]
    if step == "modeling":
        return runners[8:11]
    if step == "recommendations":
        return runners[11:]
    if step == "all":
        return runners
    raise ValueError(f"Unsupported step: {step}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for _, runner in _runners_for_step(args.step):
        runner()


if __name__ == "__main__":
    main()
