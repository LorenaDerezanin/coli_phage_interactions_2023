"""Derive per-phage functional gene repertoire features from Pharokka annotations.

Computes PHROG category counts and proportions, anti-defense gene indicators,
and depolymerase presence from Pharokka CDS annotations. These capture the
phage infection strategy beyond RBP family presence.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lyzortx.pipeline.track_l.steps.parse_annotations import (
    PHROG_CATEGORIES,
    classify_anti_defense_genes,
    count_categories,
    matches_any_pattern,
    parse_merged_tsv,
)

LOGGER = logging.getLogger(__name__)

# Depolymerase-associated annotation patterns.
DEPOLYMERASE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"depolymerase", re.IGNORECASE),
    re.compile(r"polysaccharide.?degrading", re.IGNORECASE),
    re.compile(r"endosialidase", re.IGNORECASE),
    re.compile(r"hyaluronidase", re.IGNORECASE),
    re.compile(r"lyase.*polysaccharide", re.IGNORECASE),
    re.compile(r"pectin.*lyase", re.IGNORECASE),
)

# Sanitized PHROG category names for use as feature column suffixes.
_CATEGORY_SLUG: dict[str, str] = {
    "connector": "connector",
    "DNA, RNA and nucleotide metabolism": "dna_rna_metabolism",
    "head and packaging": "head_packaging",
    "integration and excision": "integration_excision",
    "lysis": "lysis",
    "moron, auxiliary metabolic gene and host takeover": "moron_amg",
    "other": "other",
    "tail": "tail",
    "transcription regulation": "transcription_reg",
    "unknown function": "unknown",
}

# Ordered feature names for the per-phage output row.
CATEGORY_COUNT_FEATURES = [f"phrog_count_{_CATEGORY_SLUG[cat]}" for cat in PHROG_CATEGORIES]
CATEGORY_FRAC_FEATURES = [f"phrog_frac_{_CATEGORY_SLUG[cat]}" for cat in PHROG_CATEGORIES]
PHAGE_FUNCTIONAL_FEATURE_NAMES = (
    ["total_cds"]
    + CATEGORY_COUNT_FEATURES
    + CATEGORY_FRAC_FEATURES
    + [
        "anti_defense_count",
        "has_anti_defense",
        "depolymerase_count",
        "has_depolymerase",
    ]
)


def build_phage_functional_feature_row(
    phage_name: str,
    annotation_dir: Path,
) -> dict[str, object]:
    """Build a single feature row for one phage from Pharokka annotations.

    Returns a dict with keys: phage, total_cds, phrog_count_*, phrog_frac_*,
    anti_defense_count, has_anti_defense, depolymerase_count, has_depolymerase.
    """
    row: dict[str, object] = {"phage": phage_name}

    tsv_path = annotation_dir / f"{phage_name}_cds_final_merged_output.tsv"
    if not tsv_path.exists():
        LOGGER.warning("No Pharokka annotation for phage %s, returning zeros", phage_name)
        for name in PHAGE_FUNCTIONAL_FEATURE_NAMES:
            row[name] = 0
        return row

    records = parse_merged_tsv(tsv_path)
    total_cds = len(records)
    row["total_cds"] = total_cds

    # PHROG category counts and fractions.
    category_counts = count_categories(records)
    for cat in PHROG_CATEGORIES:
        slug = _CATEGORY_SLUG[cat]
        count = category_counts.get(cat, 0)
        row[f"phrog_count_{slug}"] = count
        row[f"phrog_frac_{slug}"] = round(count / total_cds, 6) if total_cds > 0 else 0.0

    # Anti-defense genes.
    anti_defense = classify_anti_defense_genes(records)
    row["anti_defense_count"] = len(anti_defense)
    row["has_anti_defense"] = 1 if anti_defense else 0

    # Depolymerases.
    depoly = [r for r in records if matches_any_pattern(r.annot, DEPOLYMERASE_PATTERNS)]
    row["depolymerase_count"] = len(depoly)
    row["has_depolymerase"] = 1 if depoly else 0

    return row


def build_phage_functional_schema() -> list[str]:
    """Return the ordered list of feature column names (without entity key)."""
    return list(PHAGE_FUNCTIONAL_FEATURE_NAMES)
