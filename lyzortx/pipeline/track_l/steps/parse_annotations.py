#!/usr/bin/env python3
"""Parse Pharokka annotation outputs into per-phage summary tables.

Reads the per-phage ``_cds_final_merged_output.tsv`` and
``_cds_functions.tsv`` files produced by Pharokka, then generates:

1. A PHROGs category count table (phage x category).
2. A per-phage RBP (receptor binding protein) gene list with functional
   family annotations.
3. A per-phage anti-defense gene list (anti-restriction, anti-CRISPR, etc.).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = Path("lyzortx/generated_outputs/track_l/pharokka_annotations")
CACHED_ANNOTATIONS_DIR = Path("data/annotations/pharokka")
OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/parsed_summaries")
EXPECTED_PHAGE_COUNT = 97

# Pharokka _cds_final_merged_output.tsv column names
COL_GENE = "gene"
COL_PHROG = "phrog"
COL_ANNOT = "annot"
COL_CATEGORY = "category"
COL_START = "start"
COL_STOP = "stop"
COL_STRAND = "strand"
COL_CONTIG = "contig"

# The 10 PHROG functional categories (as defined in pharokka source)
PHROG_CATEGORIES: tuple[str, ...] = (
    "connector",
    "DNA, RNA and nucleotide metabolism",
    "head and packaging",
    "integration and excision",
    "lysis",
    "moron, auxiliary metabolic gene and host takeover",
    "other",
    "tail",
    "transcription regulation",
    "unknown function",
)

# Patterns for identifying RBP (receptor binding protein) genes
RBP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"receptor.binding", re.IGNORECASE),
    re.compile(r"\bRBP\b", re.IGNORECASE),
    re.compile(r"tail.?fiber", re.IGNORECASE),
    re.compile(r"tail.?spike", re.IGNORECASE),
    re.compile(r"baseplate.?wedge.*binding", re.IGNORECASE),
    re.compile(r"host.specificity", re.IGNORECASE),
    re.compile(r"side.tail.fiber", re.IGNORECASE),
    re.compile(r"long.tail.fiber", re.IGNORECASE),
    re.compile(r"short.tail.fiber", re.IGNORECASE),
    re.compile(r"adhesin", re.IGNORECASE),
    re.compile(r"Dit.*binding", re.IGNORECASE),
)

# Patterns for identifying anti-defense genes
ANTI_DEFENSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"anti.?CRISPR", re.IGNORECASE),
    re.compile(r"anti.?restriction", re.IGNORECASE),
    re.compile(r"anti.?defense", re.IGNORECASE),
    re.compile(r"anti.?RM", re.IGNORECASE),
    re.compile(r"anti.?abortive", re.IGNORECASE),
    re.compile(r"Ocr\b", re.IGNORECASE),
    re.compile(r"\bArd\b"),
    re.compile(r"methyltransferase", re.IGNORECASE),
    re.compile(r"DNA.methylase", re.IGNORECASE),
    re.compile(r"Dam\b.*methyl", re.IGNORECASE),
    re.compile(r"Dcm\b.*methyl", re.IGNORECASE),
)


@dataclass
class CdsRecord:
    """A single CDS record from pharokka merged output."""

    gene: str
    start: int
    stop: int
    strand: str
    contig: str
    phrog: str
    annot: str
    category: str


@dataclass
class PhageSummary:
    """Aggregated annotation summary for one phage."""

    phage_name: str
    total_cds: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    rbp_genes: list[CdsRecord] = field(default_factory=list)
    anti_defense_genes: list[CdsRecord] = field(default_factory=list)


def parse_merged_tsv(tsv_path: Path) -> list[CdsRecord]:
    """Parse a pharokka _cds_final_merged_output.tsv into CdsRecord list.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file has zero data rows.
    """
    if not tsv_path.exists():
        msg = f"Merged TSV not found: {tsv_path}"
        raise FileNotFoundError(msg)

    records: list[CdsRecord] = []
    with tsv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            records.append(
                CdsRecord(
                    gene=row[COL_GENE],
                    start=int(row[COL_START]),
                    stop=int(row[COL_STOP]),
                    strand=row[COL_STRAND],
                    contig=row[COL_CONTIG],
                    phrog=row[COL_PHROG],
                    annot=row[COL_ANNOT],
                    category=row[COL_CATEGORY],
                )
            )

    if not records:
        msg = f"Zero CDS records in {tsv_path}"
        raise ValueError(msg)

    return records


def matches_any_pattern(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Return True if text matches any of the compiled regex patterns."""
    return any(p.search(text) for p in patterns)


def classify_rbp_genes(records: list[CdsRecord]) -> list[CdsRecord]:
    """Identify CDS records that are receptor binding proteins."""
    return [r for r in records if matches_any_pattern(r.annot, RBP_PATTERNS)]


def classify_anti_defense_genes(records: list[CdsRecord]) -> list[CdsRecord]:
    """Identify CDS records that are anti-defense genes."""
    return [r for r in records if matches_any_pattern(r.annot, ANTI_DEFENSE_PATTERNS)]


def count_categories(records: list[CdsRecord]) -> dict[str, int]:
    """Count CDS records by PHROG category."""
    counts: dict[str, int] = {cat: 0 for cat in PHROG_CATEGORIES}
    for r in records:
        if r.category in counts:
            counts[r.category] += 1
        else:
            # Unexpected category — count it but log a warning
            logger.warning("Unexpected PHROG category: %r", r.category)
            counts[r.category] = counts.get(r.category, 0) + 1
    return counts


def summarize_phage(phage_name: str, records: list[CdsRecord]) -> PhageSummary:
    """Build a PhageSummary from a list of CDS records for one phage."""
    return PhageSummary(
        phage_name=phage_name,
        total_cds=len(records),
        category_counts=count_categories(records),
        rbp_genes=classify_rbp_genes(records),
        anti_defense_genes=classify_anti_defense_genes(records),
    )


def discover_phage_dirs(annotations_dir: Path) -> list[Path]:
    """Find per-phage output directories under the annotations root.

    Each subdirectory that contains a *_cds_final_merged_output.tsv is
    treated as a phage annotation directory.
    """
    if not annotations_dir.is_dir():
        msg = f"Annotations directory does not exist: {annotations_dir}"
        raise FileNotFoundError(msg)

    dirs = sorted(d for d in annotations_dir.iterdir() if d.is_dir() and list(d.glob("*_cds_final_merged_output.tsv")))
    if not dirs:
        msg = f"No pharokka output directories found in {annotations_dir}"
        raise FileNotFoundError(msg)
    return dirs


def discover_cached_phages(cached_dir: Path) -> list[tuple[str, Path]]:
    """Find phages from a flat directory of cached TSVs.

    Returns sorted list of (phage_name, tsv_path) tuples.
    """
    if not cached_dir.is_dir():
        msg = f"Cached annotations directory does not exist: {cached_dir}"
        raise FileNotFoundError(msg)

    tsvs = sorted(cached_dir.glob("*_cds_final_merged_output.tsv"))
    if not tsvs:
        msg = f"No cached merged TSVs found in {cached_dir}"
        raise FileNotFoundError(msg)

    results = []
    for tsv in tsvs:
        # Strip _cds_final_merged_output.tsv suffix to get phage name
        phage_name = tsv.name.removesuffix("_cds_final_merged_output.tsv")
        results.append((phage_name, tsv))
    return results


def write_category_summary(summaries: list[PhageSummary], output_dir: Path) -> Path:
    """Write a per-phage PHROGs category count table to CSV."""
    fieldnames = ["phage", "total_cds", *PHROG_CATEGORIES]
    rows = []
    for s in summaries:
        row: dict[str, str | int] = {"phage": s.phage_name, "total_cds": s.total_cds}
        for cat in PHROG_CATEGORIES:
            row[cat] = s.category_counts.get(cat, 0)
        rows.append(row)

    out_path = output_dir / "phrog_category_counts.csv"
    write_csv(out_path, fieldnames, rows)
    logger.info("Wrote PHROG category counts: %s (%d phages)", out_path, len(rows))
    return out_path


def write_rbp_gene_list(summaries: list[PhageSummary], output_dir: Path) -> Path:
    """Write per-phage RBP gene list to CSV."""
    fieldnames = ["phage", "gene", "start", "stop", "phrog", "annot", "category"]
    rows = []
    for s in summaries:
        for g in s.rbp_genes:
            rows.append(
                {
                    "phage": s.phage_name,
                    "gene": g.gene,
                    "start": g.start,
                    "stop": g.stop,
                    "phrog": g.phrog,
                    "annot": g.annot,
                    "category": g.category,
                }
            )

    out_path = output_dir / "rbp_genes.csv"
    write_csv(out_path, fieldnames, rows)
    logger.info("Wrote RBP gene list: %s (%d genes across %d phages)", out_path, len(rows), len(summaries))
    return out_path


def write_anti_defense_gene_list(summaries: list[PhageSummary], output_dir: Path) -> Path:
    """Write per-phage anti-defense gene list to CSV."""
    fieldnames = ["phage", "gene", "start", "stop", "phrog", "annot", "category"]
    rows = []
    for s in summaries:
        for g in s.anti_defense_genes:
            rows.append(
                {
                    "phage": s.phage_name,
                    "gene": g.gene,
                    "start": g.start,
                    "stop": g.stop,
                    "phrog": g.phrog,
                    "annot": g.annot,
                    "category": g.category,
                }
            )

    out_path = output_dir / "anti_defense_genes.csv"
    write_csv(out_path, fieldnames, rows)
    logger.info("Wrote anti-defense gene list: %s (%d genes across %d phages)", out_path, len(rows), len(summaries))
    return out_path


def write_manifest(summaries: list[PhageSummary], output_dir: Path) -> Path:
    """Write a manifest JSON with run metadata."""
    total_rbp = sum(len(s.rbp_genes) for s in summaries)
    total_anti_defense = sum(len(s.anti_defense_genes) for s in summaries)
    manifest = {
        "step": "TL01_parse_annotations",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phage_count": len(summaries),
        "total_cds": sum(s.total_cds for s in summaries),
        "total_rbp_genes": total_rbp,
        "total_anti_defense_genes": total_anti_defense,
        "outputs": [
            "phrog_category_counts.csv",
            "rbp_genes.csv",
            "anti_defense_genes.csv",
        ],
    }
    out_path = output_dir / "manifest.json"
    write_json(out_path, manifest)
    logger.info("Wrote manifest: %s", out_path)
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=ANNOTATIONS_DIR,
        help="Root directory containing per-phage pharokka outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for parsed summary outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    # Try generated outputs first, fall back to cached tracked TSVs
    phage_entries: list[tuple[str, Path]] = []
    if args.annotations_dir.is_dir():
        try:
            phage_dirs = discover_phage_dirs(args.annotations_dir)
            phage_entries = [(d.name, d / f"{d.name}_cds_final_merged_output.tsv") for d in phage_dirs]
            logger.info("Using generated annotations from %s (%d phages)", args.annotations_dir, len(phage_entries))
        except FileNotFoundError:
            pass

    if not phage_entries and CACHED_ANNOTATIONS_DIR.is_dir():
        phage_entries = discover_cached_phages(CACHED_ANNOTATIONS_DIR)
        logger.info("Using cached annotations from %s (%d phages)", CACHED_ANNOTATIONS_DIR, len(phage_entries))

    if not phage_entries:
        msg = (
            f"No annotations found in {args.annotations_dir} or {CACHED_ANNOTATIONS_DIR}. "
            "Run pharokka first or ensure cached TSVs are present."
        )
        raise FileNotFoundError(msg)

    if len(phage_entries) != EXPECTED_PHAGE_COUNT:
        logger.warning(
            "Expected %d phages but found %d",
            EXPECTED_PHAGE_COUNT,
            len(phage_entries),
        )

    summaries: list[PhageSummary] = []
    for phage_name, tsv_path in phage_entries:
        records = parse_merged_tsv(tsv_path)
        summary = summarize_phage(phage_name, records)
        summaries.append(summary)
        logger.info(
            "  %s: %d CDS, %d RBP, %d anti-defense",
            phage_name,
            summary.total_cds,
            len(summary.rbp_genes),
            len(summary.anti_defense_genes),
        )

    ensure_directory(args.output_dir)
    write_category_summary(summaries, args.output_dir)
    write_rbp_gene_list(summaries, args.output_dir)
    write_anti_defense_gene_list(summaries, args.output_dir)
    write_manifest(summaries, args.output_dir)

    total_cds = sum(s.total_cds for s in summaries)
    total_rbp = sum(len(s.rbp_genes) for s in summaries)
    total_anti_def = sum(len(s.anti_defense_genes) for s in summaries)
    logger.info(
        "Parsing complete: %d phages, %d total CDS, %d RBP genes, %d anti-defense genes",
        len(summaries),
        total_cds,
        total_rbp,
        total_anti_def,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
