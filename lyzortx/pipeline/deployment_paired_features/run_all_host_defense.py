#!/usr/bin/env python3
"""Run DEPLOY02 host-defense feature derivation on all 403 Picard assemblies in parallel.

Usage:
    python -m lyzortx.pipeline.deployment_paired_features.run_all_host_defense [--max-workers N]

Pre-computed per-host outputs are skipped automatically (presence of host_defense_gene_counts.csv).
After all hosts complete, aggregates results into a single checked-in CSV at
``lyzortx/data/deployment_paired_features/403_host_defense_gene_counts.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.deployment_paired_features.derive_host_defense_features import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
    PER_HOST_COUNTS_FILENAME,
    build_host_defense_schema,
    derive_host_defense_features,
)
from lyzortx.pipeline.track_l.steps.run_novel_host_defense_finder import (
    DEFAULT_MODELS_DIR,
    MODEL_INSTALL_MODE_FORBID,
    _read_panel_defense_rows,
    ensure_defense_finder_models,
)

LOGGER = logging.getLogger(__name__)
ASSEMBLIES_DIR = Path("lyzortx/data/assemblies/picard")
EXPECTED_HOST_COUNT = 403
AGGREGATED_CSV_PATH = Path("lyzortx/data/deployment_paired_features/403_host_defense_gene_counts.csv")


def _process_one_host(
    assembly_path: Path,
    bacteria_id: str,
    output_dir: Path,
    panel_path: Path,
    models_dir: Path,
) -> tuple[str, bool, str]:
    """Process a single host. Returns (bacteria_id, success, message)."""
    try:
        host_output_dir = output_dir / bacteria_id
        derive_host_defense_features(
            assembly_path,
            bacteria_id=bacteria_id,
            output_dir=host_output_dir,
            panel_defense_subtypes_path=panel_path,
            models_dir=models_dir,
            workers=1,
            force_model_update=False,
            model_install_mode=MODEL_INSTALL_MODE_FORBID,
            force_run=False,
            preserve_raw=False,
        )
        return bacteria_id, True, "ok"
    except Exception as exc:
        return bacteria_id, False, str(exc)


def load_host_defense_rows(
    per_host_output_dir: Path,
    panel_defense_subtypes_path: Path,
    *,
    bacteria_ids: list[str] | None = None,
) -> tuple[dict[str, object], list[str], list[dict[str, str]]]:
    panel_rows = _read_panel_defense_rows(panel_defense_subtypes_path)
    schema = build_host_defense_schema(panel_rows)
    columns = [col["name"] for col in schema["columns"]]

    requested = None if bacteria_ids is None else set(bacteria_ids)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for host_dir in sorted(per_host_output_dir.iterdir()):
        if not host_dir.is_dir():
            continue
        if requested is not None and host_dir.name not in requested:
            continue
        counts_path = host_dir / PER_HOST_COUNTS_FILENAME
        if not counts_path.exists():
            continue
        with counts_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bacteria = str(row.get("bacteria", "")).strip()
                if not bacteria:
                    raise ValueError(f"Missing bacteria key in {counts_path}")
                seen.add(bacteria)
                rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No per-host output directories with {PER_HOST_COUNTS_FILENAME} found in {per_host_output_dir}"
        )

    if requested is not None:
        missing = sorted(requested - seen)
        if missing:
            raise FileNotFoundError(
                "Missing per-host host-defense outputs for requested bacteria: " + ", ".join(missing)
            )

    rows.sort(key=lambda r: r.get("bacteria", ""))
    return schema, columns, rows


def aggregate_host_defense_csvs(
    per_host_output_dir: Path,
    aggregated_csv_path: Path,
    panel_defense_subtypes_path: Path,
    *,
    bacteria_ids: list[str] | None = None,
) -> int:
    """Collect all per-host gene count CSVs into a single aggregated CSV.

    Returns the number of rows written.
    """
    _, columns, rows = load_host_defense_rows(
        per_host_output_dir,
        panel_defense_subtypes_path,
        bacteria_ids=bacteria_ids,
    )

    aggregated_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, 0) for col in columns})

    LOGGER.info("Wrote %d rows to %s", len(rows), aggregated_csv_path)
    return len(rows)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip derivation and only aggregate existing per-host CSVs.",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        count = aggregate_host_defense_csvs(args.output_dir, AGGREGATED_CSV_PATH, DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH)
        LOGGER.info("Aggregation complete: %d hosts", count)
        return 0

    if not ASSEMBLIES_DIR.exists():
        raise FileNotFoundError(f"Assemblies directory not found: {ASSEMBLIES_DIR}")

    assemblies = sorted(ASSEMBLIES_DIR.glob("*.fasta"))
    if len(assemblies) != EXPECTED_HOST_COUNT:
        LOGGER.warning("Expected %d assemblies, found %d", EXPECTED_HOST_COUNT, len(assemblies))

    # Pre-install models once before fanning out
    LOGGER.info("Ensuring Defense Finder models are installed in %s", args.models_dir)
    ensure_defense_finder_models(models_dir=args.models_dir, force_update=False)

    # Filter to hosts that still need processing
    pending: list[tuple[Path, str]] = []
    skipped = 0
    for assembly_path in assemblies:
        bacteria_id = assembly_path.stem
        counts_path = args.output_dir / bacteria_id / PER_HOST_COUNTS_FILENAME
        if counts_path.exists():
            skipped += 1
            continue
        pending.append((assembly_path, bacteria_id))

    LOGGER.info(
        "Host defense derivation: %d pending, %d already complete, %d workers",
        len(pending),
        skipped,
        args.max_workers,
    )

    if pending:
        start = datetime.now(timezone.utc)
        succeeded = 0
        failed = 0
        failures: list[tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_host,
                    assembly_path,
                    bacteria_id,
                    args.output_dir,
                    DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
                    args.models_dir,
                ): bacteria_id
                for assembly_path, bacteria_id in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                bacteria_id = futures[future]
                bid, ok, msg = future.result()
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    failures.append((bid, msg))
                    LOGGER.error("FAILED %s: %s", bid, msg)
                if i % 10 == 0 or i == len(futures):
                    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                    LOGGER.info(
                        "Progress: %d/%d done (%.0fs elapsed, %d ok, %d failed)",
                        i,
                        len(futures),
                        elapsed,
                        succeeded,
                        failed,
                    )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        LOGGER.info(
            "Finished host defense derivation in %.0fs: %d succeeded, %d failed, %d skipped",
            elapsed,
            succeeded,
            failed,
            skipped,
        )
        if failures:
            LOGGER.error("Failed hosts:")
            for bid, msg in failures:
                LOGGER.error("  %s: %s", bid, msg)
            return 1
    else:
        LOGGER.info("All hosts already processed — skipping derivation")

    # Aggregate into a single checked-in CSV
    count = aggregate_host_defense_csvs(args.output_dir, AGGREGATED_CSV_PATH, DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH)
    if count != EXPECTED_HOST_COUNT:
        LOGGER.warning("Expected %d hosts in aggregated CSV, got %d", EXPECTED_HOST_COUNT, count)
    LOGGER.info("Aggregated CSV written to %s (%d rows)", AGGREGATED_CSV_PATH, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
