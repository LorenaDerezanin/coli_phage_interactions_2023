#!/usr/bin/env python3
"""Entry point for Track L: Mechanistic Features from Pharokka Annotations.

Individual steps and groups (run individually or with 'all'):
  annotate                — Run pharokka on phage genomes and cache key TSVs.
  features (group)        — parse → enrich.
  tl17-phage-compatibility-preprocessor — Build the TL17 deployable phage compatibility block.
  inference (group)       — generalized-inference-bundle → deployable-generalized-inference-bundle →
                            richer-deployable-generalized-inference-bundle.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Callable
from os import cpu_count
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_l.steps import (
    build_generalized_inference_bundle,
    build_tl17_phage_compatibility_preprocessor,
    build_tl13_generalized_inference_bundle,
    build_tl18_generalized_inference_bundle,
    parse_annotations,
    run_enrichment_analysis,
    run_pharokka,
)

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = Path("lyzortx/generated_outputs/track_l/pharokka_annotations")
CACHED_DIR = Path("data/annotations/pharokka")
TSV_SUFFIXES = ("_cds_final_merged_output.tsv", "_cds_functions.tsv")

StepFn = Callable[[argparse.Namespace], object]


def cache_key_tsvs(annotations_dir: Path, cached_dir: Path) -> int:
    """Copy key pharokka TSVs from per-phage output dirs to the flat cached dir.

    Returns the number of files copied.
    """
    cached_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for phage_dir in sorted(annotations_dir.iterdir()):
        if not phage_dir.is_dir():
            continue
        for suffix in TSV_SUFFIXES:
            src = phage_dir / f"{phage_dir.name}{suffix}"
            if src.exists():
                dst = cached_dir / src.name
                shutil.copy2(src, dst)
                copied += 1
    return copied


def _run_annotate(args: argparse.Namespace) -> None:
    if args.database_dir is None:
        msg = "--database-dir is required for the annotate step"
        raise SystemExit(msg)
    run_pharokka.main(
        [
            "--database-dir",
            str(args.database_dir),
            "--threads",
            str(args.threads),
            "--parallel",
            str(args.parallel),
            *(["--force"] if args.force else []),
        ]
    )
    copied = cache_key_tsvs(ANNOTATIONS_DIR, CACHED_DIR)
    logger.info("Cached %d TSV files to %s", copied, CACHED_DIR)


# ---------------------------------------------------------------------------
# Step registry.  Each group is an ordered list of (name, function) pairs.
# "all" runs every group in order.  When adding a step, add it to the
# appropriate group — omitting it is a bug.
# ---------------------------------------------------------------------------

FEATURE_STEPS: list[tuple[str, StepFn]] = [
    ("parse", lambda _args: parse_annotations.main([])),
    ("enrich", lambda _args: run_enrichment_analysis.main([])),
]

INFERENCE_STEPS: list[tuple[str, StepFn]] = [
    ("generalized-inference-bundle", lambda _args: build_generalized_inference_bundle.main([])),
    ("deployable-generalized-inference-bundle", lambda _args: build_tl13_generalized_inference_bundle.main([])),
    (
        "richer-deployable-generalized-inference-bundle",
        lambda _args: build_tl18_generalized_inference_bundle.main([]),
    ),
]

# Groups bundle multiple steps behind a single flag.  Single-step entries
# (annotate, retrain-mechanistic-v1) are reachable by their own name and
# don't need a group wrapper.
GROUPS: list[tuple[str, list[tuple[str, StepFn]]]] = [
    ("features", FEATURE_STEPS),
    ("inference", INFERENCE_STEPS),
]

# Full ordered pipeline — all groups plus standalone steps in execution order.
# "all" runs these top to bottom.
ALL_STEPS: list[tuple[str, StepFn]] = [
    ("annotate", _run_annotate),
    *FEATURE_STEPS,
    (
        "tl17-phage-compatibility-preprocessor",
        lambda _args: build_tl17_phage_compatibility_preprocessor.main([]),
    ),
    *INFERENCE_STEPS,
]

STEP_NAMES = [name for name, _ in ALL_STEPS]
GROUP_NAMES = [name for name, _ in GROUPS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=[*STEP_NAMES, *GROUP_NAMES, "all"],
        default="all",
        help="Individual step, group name, or 'all' (default: all).",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=None,
        help="Path to pharokka database directory (required for annotate step)",
    )
    cores = cpu_count() or 4
    threads_per_phage = 2
    default_parallel = max(1, cores // threads_per_phage)
    parser.add_argument(
        "--threads",
        type=int,
        default=threads_per_phage,
        help=f"Threads per pharokka invocation (default: {threads_per_phage})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=default_parallel,
        help=f"Number of phages in parallel (default: {default_parallel}, from {cores} cores)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run pharokka even if output already exists",
    )
    return parser.parse_args(argv)


def _run_steps(steps: list[tuple[str, StepFn]], args: argparse.Namespace) -> None:
    for name, fn in steps:
        logger.info("=== Track L: running step '%s' ===", name)
        fn(args)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)

    if args.step == "all":
        _run_steps(ALL_STEPS, args)
        return

    # Check group names first, then individual steps.
    group_map = dict(GROUPS)
    if args.step in group_map:
        _run_steps(group_map[args.step], args)
        return

    step_map = dict(ALL_STEPS)
    step_map[args.step](args)


if __name__ == "__main__":
    main()
