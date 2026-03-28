#!/usr/bin/env python3
"""Run Pharokka genome annotation on all phage FNA files.

Iterates over all .fna files in the phage genomes directory, runs pharokka
on each, and verifies that every phage produces >0 annotated CDS.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging

logger = logging.getLogger(__name__)

FNA_DIR = Path("data/genomics/phages/FNA")
OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/pharokka_annotations")
EXPECTED_PHAGE_COUNT = 97


def discover_fna_files(fna_dir: Path) -> list[Path]:
    """Return sorted list of .fna files in the given directory.

    Raises FileNotFoundError if the directory does not exist or contains no
    .fna files.
    """
    if not fna_dir.is_dir():
        msg = f"FNA directory does not exist: {fna_dir}"
        raise FileNotFoundError(msg)
    fna_files = sorted(fna_dir.glob("*.fna"))
    if not fna_files:
        msg = f"No .fna files found in {fna_dir}"
        raise FileNotFoundError(msg)
    return fna_files


def run_pharokka_on_file(
    fna_path: Path,
    output_dir: Path,
    database_dir: Path,
    threads: int,
    force: bool,
) -> Path:
    """Run pharokka on a single FNA file.

    Returns the per-phage output directory.
    """
    phage_name = fna_path.stem
    phage_output_dir = output_dir / phage_name

    if phage_output_dir.exists() and not force:
        logger.info("Skipping %s (output already exists)", phage_name)
        return phage_output_dir

    cmd = [
        "pharokka.py",
        "-i",
        str(fna_path),
        "-o",
        str(phage_output_dir),
        "-d",
        str(database_dir),
        "-p",
        phage_name,
        "-t",
        str(threads),
        "-f",
        "--skip_mash",
        "--skip_extra_annotations",
    ]

    logger.info("Running pharokka on %s ...", phage_name)
    start = datetime.now(timezone.utc)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    if result.returncode != 0:
        logger.error(
            "pharokka failed for %s (exit %d, %.1fs)\nstdout: %s\nstderr: %s",
            phage_name,
            result.returncode,
            elapsed,
            result.stdout[-2000:] if result.stdout else "",
            result.stderr[-2000:] if result.stderr else "",
        )
        msg = f"pharokka failed for {phage_name} with exit code {result.returncode}"
        raise RuntimeError(msg)

    logger.info("Finished %s in %.1fs", phage_name, elapsed)
    return phage_output_dir


def verify_annotations(output_dir: Path, phage_name: str) -> int:
    """Verify that a pharokka output directory has >0 annotated CDS.

    Returns the CDS count. Raises ValueError if no CDS found.
    """
    merged_tsv = output_dir / f"{phage_name}_cds_final_merged_output.tsv"
    if not merged_tsv.exists():
        msg = f"Missing merged output for {phage_name}: {merged_tsv}"
        raise FileNotFoundError(msg)

    # Count non-header lines
    with merged_tsv.open(encoding="utf-8") as fh:
        cds_count = sum(1 for _ in fh) - 1  # subtract header

    if cds_count <= 0:
        msg = f"Zero annotated CDS for {phage_name}"
        raise ValueError(msg)

    return cds_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fna-dir",
        type=Path,
        default=FNA_DIR,
        help="Directory containing phage .fna files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root output directory for pharokka results",
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
        default=4,
        help="Threads per pharokka invocation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run pharokka even if output already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    logger.info("Starting pharokka annotation of phage genomes")
    logger.info("FNA dir: %s  Output dir: %s  Database: %s", args.fna_dir, args.output_dir, args.database_dir)

    fna_files = discover_fna_files(args.fna_dir)
    logger.info("Found %d FNA files", len(fna_files))

    if len(fna_files) != EXPECTED_PHAGE_COUNT:
        logger.warning(
            "Expected %d phage genomes but found %d",
            EXPECTED_PHAGE_COUNT,
            len(fna_files),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_cds = 0
    failed: list[str] = []

    for i, fna_path in enumerate(fna_files, 1):
        phage_name = fna_path.stem
        logger.info("[%d/%d] Processing %s", i, len(fna_files), phage_name)
        try:
            phage_output_dir = run_pharokka_on_file(
                fna_path, args.output_dir, args.database_dir, args.threads, args.force
            )
            cds_count = verify_annotations(phage_output_dir, phage_name)
            total_cds += cds_count
            logger.info("  %s: %d CDS annotated", phage_name, cds_count)
        except (RuntimeError, FileNotFoundError, ValueError):
            logger.exception("Failed for %s", phage_name)
            failed.append(phage_name)

    if failed:
        logger.error("Pharokka failed for %d phages: %s", len(failed), ", ".join(failed))
        msg = f"Pharokka annotation failed for {len(failed)} phages"
        raise RuntimeError(msg)

    logger.info(
        "Pharokka annotation complete: %d phages, %d total CDS",
        len(fna_files),
        total_cds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
