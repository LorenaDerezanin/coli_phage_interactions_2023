#!/usr/bin/env python3
"""User-facing entry point for short AUTORESEARCH experiment runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch.prepare_cache import (
    CACHE_MANIFEST_FILENAME,
    DISALLOWED_SEARCH_SPLITS,
    SLOT_SPECS,
    SUPPORTED_SEARCH_SPLITS,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Prepared AUTORESEARCH search cache directory from prepare.py.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    cache_manifest_path = args.cache_dir / CACHE_MANIFEST_FILENAME
    if not cache_manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared search cache not found at {cache_manifest_path}. Run lyzortx/autoresearch/prepare.py first."
        )

    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    exported_splits = tuple(sorted(cache_manifest["pair_tables"].keys()))
    if exported_splits != tuple(sorted(SUPPORTED_SEARCH_SPLITS)):
        raise ValueError(
            "AUTORESEARCH cache does not match the frozen split contract: "
            f"expected {SUPPORTED_SEARCH_SPLITS}, got {exported_splits}"
        )
    if any(split in cache_manifest["pair_tables"] for split in DISALLOWED_SEARCH_SPLITS):
        raise ValueError("Sealed holdout split leaked into the AUTORESEARCH search cache.")

    LOGGER.info("AUTORESEARCH train sandbox validated cache at %s", args.cache_dir)
    LOGGER.info("Exported splits: %s", ", ".join(SUPPORTED_SEARCH_SPLITS))
    LOGGER.info("Reserved feature slots: %s", ", ".join(spec.slot_name for spec in SLOT_SPECS))
    LOGGER.info("train.py is the short-loop experiment surface; cache rebuilding belongs in prepare.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
