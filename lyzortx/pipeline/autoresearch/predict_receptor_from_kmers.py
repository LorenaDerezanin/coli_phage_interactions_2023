"""Predict phage receptor class using GenoPHI k-mer features (GT06).

Scans phage proteomes for the 815 receptor-predictive amino acid k-mers
identified by Moriniere 2026 (Dataset S6) and assigns receptor class based
on k-mer hit counts per receptor.

The prediction logic: for each phage, count how many receptor-specific k-mers
are found in its proteome. The receptor class with the most hits wins. This
is a simplified version of GenoPHI's gradient-boosted classifier — using the
same features but a simpler decision rule.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import openpyxl

LOGGER = logging.getLogger(__name__)

# Maps Dataset S6 receptor names to our OMP receptor short names (matching
# derive_pairwise_receptor_omp_features.py conventions).
S6_RECEPTOR_TO_OMP = {
    "tsx": "tsx",
    "ompA": "ompA",
    "ompC": "ompC",
    "ompF": "ompF",
    "fhuA": "fhua",
    "btuB": "btub",
    "lptD": "lptD",
    "lamB": "lamB",
}

# LPS receptor types (not OMP but still useful for the LPS cross-term).
S6_RECEPTOR_LPS = {"Kdo", "HepI", "HepII", "GluI"}

# NGR = "No Glycan Receptor" — protein receptor but unidentified.
S6_RECEPTOR_NGR = {"NGR"}


@dataclass
class ReceptorPrediction:
    phage: str
    predicted_receptor: str  # OMP short name, "lps", "ngr", or "unknown"
    receptor_type: str  # "omp", "lps", "ngr", or "unknown"
    confidence: float  # fraction of total hits from the predicted class
    hit_counts: dict[str, int]  # receptor -> k-mer hit count


def load_receptor_kmers(dataset_path: Path) -> dict[str, set[str]]:
    """Load receptor-predictive k-mers from GenoPHI Dataset S6.

    Returns receptor_name -> set of k-mer sequences.
    """
    wb = openpyxl.load_workbook(dataset_path, read_only=True)
    ws = wb["Dataset S6"]

    receptor_kmers: dict[str, set[str]] = defaultdict(set)
    for row in ws.iter_rows(min_row=3, values_only=True):
        receptor = row[0]
        segment_seq = row[4]
        if receptor and segment_seq:
            receptor_kmers[str(receptor)].add(str(segment_seq))

    wb.close()
    total = sum(len(v) for v in receptor_kmers.values())
    LOGGER.info("Loaded %d receptor-predictive k-mers across %d receptors", total, len(receptor_kmers))
    return dict(receptor_kmers)


def load_phage_proteomes(proteome_path: Path) -> dict[str, list[str]]:
    """Load phage protein sequences from FASTA.

    Returns phage_name -> list of protein sequences.
    """
    phage_proteins: dict[str, list[str]] = defaultdict(list)
    current_phage = None
    current_seq_parts: list[str] = []

    with open(proteome_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_phage and current_seq_parts:
                    phage_proteins[current_phage].append("".join(current_seq_parts))
                # Parse phage name from header: ">409_P1|query_prot_0001"
                header = line[1:]
                current_phage = header.split("|")[0] if "|" in header else header.split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
    if current_phage and current_seq_parts:
        phage_proteins[current_phage].append("".join(current_seq_parts))

    LOGGER.info(
        "Loaded proteomes for %d phages (%d total proteins)",
        len(phage_proteins),
        sum(len(v) for v in phage_proteins.values()),
    )
    return dict(phage_proteins)


def predict_receptor_for_phage(
    proteins: list[str],
    receptor_kmers: dict[str, set[str]],
) -> tuple[str, str, float, dict[str, int]]:
    """Predict receptor class for one phage based on k-mer hits.

    Returns (predicted_receptor, receptor_type, confidence, hit_counts).
    """
    # Concatenate all proteins into a single search string for efficiency.
    proteome = " ".join(proteins)  # space-separated to prevent cross-protein matches

    hit_counts: dict[str, int] = {}
    for receptor, kmers in receptor_kmers.items():
        count = sum(1 for kmer in kmers if kmer in proteome)
        if count > 0:
            hit_counts[receptor] = count

    if not hit_counts:
        return "unknown", "unknown", 0.0, hit_counts

    total_hits = sum(hit_counts.values())
    best_receptor = max(hit_counts, key=hit_counts.get)
    confidence = hit_counts[best_receptor] / total_hits

    # Classify receptor type.
    if best_receptor in S6_RECEPTOR_TO_OMP:
        return S6_RECEPTOR_TO_OMP[best_receptor], "omp", confidence, hit_counts
    elif best_receptor in S6_RECEPTOR_LPS:
        return "lps", "lps", confidence, hit_counts
    elif best_receptor in S6_RECEPTOR_NGR:
        return "ngr", "ngr", confidence, hit_counts
    else:
        return best_receptor, "unknown", confidence, hit_counts


def predict_receptors(
    proteome_path: Path,
    dataset_path: Path,
) -> list[ReceptorPrediction]:
    """Predict receptor class for all phages.

    Returns list of ReceptorPrediction sorted by phage name.
    """
    receptor_kmers = load_receptor_kmers(dataset_path)
    phage_proteomes = load_phage_proteomes(proteome_path)

    predictions: list[ReceptorPrediction] = []
    for phage in sorted(phage_proteomes):
        proteins = phage_proteomes[phage]
        receptor, rtype, confidence, hits = predict_receptor_for_phage(proteins, receptor_kmers)
        predictions.append(
            ReceptorPrediction(
                phage=phage,
                predicted_receptor=receptor,
                receptor_type=rtype,
                confidence=confidence,
                hit_counts=hits,
            )
        )

    # Log summary.
    omp_count = sum(1 for p in predictions if p.receptor_type == "omp")
    lps_count = sum(1 for p in predictions if p.receptor_type == "lps")
    ngr_count = sum(1 for p in predictions if p.receptor_type == "ngr")
    unk_count = sum(1 for p in predictions if p.receptor_type == "unknown")
    LOGGER.info(
        "Receptor predictions: %d OMP, %d LPS, %d NGR, %d unknown (total %d)",
        omp_count,
        lps_count,
        ngr_count,
        unk_count,
        len(predictions),
    )

    # Log per-receptor distribution.
    receptor_dist: dict[str, int] = defaultdict(int)
    for p in predictions:
        receptor_dist[p.predicted_receptor] += 1
    for receptor, count in sorted(receptor_dist.items(), key=lambda x: -x[1]):
        LOGGER.info("  %s: %d phages", receptor, count)

    return predictions


def save_predictions_csv(predictions: list[ReceptorPrediction], output_path: Path) -> None:
    """Save predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phage", "predicted_receptor", "receptor_type", "confidence", "hit_counts"])
        for p in predictions:
            writer.writerow(
                [
                    p.phage,
                    p.predicted_receptor,
                    p.receptor_type,
                    f"{p.confidence:.3f}",
                    ";".join(f"{k}={v}" for k, v in sorted(p.hit_counts.items(), key=lambda x: -x[1])),
                ]
            )
    LOGGER.info("Saved %d predictions to %s", len(predictions), output_path)
