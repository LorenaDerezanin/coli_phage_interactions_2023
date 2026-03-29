#!/usr/bin/env python3
"""Entry point for Track L: Mechanistic Features from Pharokka Annotations."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from os import cpu_count
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.track_l.steps import (
    build_mechanistic_rbp_receptor_features,
    parse_annotations,
    run_enrichment_analysis,
    run_pharokka,
)

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = Path("lyzortx/generated_outputs/track_l/pharokka_annotations")
CACHED_DIR = Path("data/annotations/pharokka")
TSV_SUFFIXES = ("_cds_final_merged_output.tsv", "_cds_functions.tsv")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["annotate", "parse", "enrich", "rbp-features", "all"],
        default="all",
        help=(
            "Track L step to run. "
            "'annotate' runs pharokka on all FNA files. "
            "'parse' parses pharokka outputs into summary tables. "
            "'enrich' runs TL02 PHROG x host-feature enrichment analyses. "
            "'rbp-features' runs TL03 mechanistic RBP-receptor feature construction. "
            "'all' runs annotate + parse (not enrich, which depends on Track A outputs)."
        ),
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


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)

    if args.step in {"annotate", "all"}:
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

    if args.step in {"parse", "all"}:
        parse_annotations.main([])

    if args.step == "enrich":
        run_enrichment_analysis.main([])
    if args.step == "rbp-features":
        build_mechanistic_rbp_receptor_features.main([])


if __name__ == "__main__":
    main()
