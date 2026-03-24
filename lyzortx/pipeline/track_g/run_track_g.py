#!/usr/bin/env python3
"""Entry point for Track G modeling tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_g.steps import calibrate_gbm_outputs
from lyzortx.pipeline.track_g.steps import compute_shap_explanations
from lyzortx.pipeline.track_g.steps import investigate_non_leaky_candidate_features
from lyzortx.pipeline.track_g.steps import run_feature_block_ablation_suite
from lyzortx.pipeline.track_g.steps import run_feature_subset_sweep
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=[
            "train-v1-binary",
            "calibrate-gbm",
            "feature-block-ablation",
            "compute-shap",
            "feature-subset-sweep",
            "non-leaky-candidate-search",
            "all",
        ],
        default="all",
        help="Track G step to run. 'all' runs the implemented Track G modeling steps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    if args.step in {"train-v1-binary", "all"}:
        train_v1_binary_classifier.main([])
    if args.step in {"calibrate-gbm", "all"}:
        calibrate_gbm_outputs.main([])
    if args.step in {"feature-block-ablation", "all"}:
        run_feature_block_ablation_suite.main([])
    if args.step in {"compute-shap", "all"}:
        compute_shap_explanations.main([])
    if args.step in {"feature-subset-sweep", "all"}:
        run_feature_subset_sweep.main([])
    if args.step == "non-leaky-candidate-search":
        investigate_non_leaky_candidate_features.main([])


if __name__ == "__main__":
    main()
