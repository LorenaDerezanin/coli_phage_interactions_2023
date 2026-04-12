#!/usr/bin/env python3
"""Precompute GenoPHI-style binary protein-family features via MMseqs2 (GT08).

Concatenates all host and phage predicted proteins, clusters with MMseqs2 at
40% identity / 80% coverage (GenoPHI parameters), and builds a binary
presence-absence matrix: does genome X contain a member of cluster Y?

Saves the binary matrix to .scratch/gt08_protein_family_features.npz.

Usage:
    python -m lyzortx.pipeline.autoresearch.precompute_protein_family_features
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lyzortx.log_config import setup_logging

LOGGER = logging.getLogger(__name__)

HOST_PROTEIN_DIR = Path("lyzortx/generated_outputs/autoresearch/host_surface_cache_build")
PHAGE_PROTEOME_PATH = Path(
    "lyzortx/generated_outputs/autoresearch/phage_projection_cache_build/_batched/combined_queries.faa"
)
OUTPUT_DIR = Path(".scratch/gt08_protein_families")

# MMseqs2 clustering parameters (GenoPHI: identity 0.4, coverage 0.8).
MMSEQS_MIN_SEQ_ID = 0.4
MMSEQS_COVERAGE = 0.8
MMSEQS_COVERAGE_MODE = 0  # bidirectional coverage

# Filtering: remove clusters present in only 1 genome (singletons).
MIN_GENOME_COUNT = 2


def concatenate_proteomes(output_fasta: Path) -> tuple[list[str], list[str]]:
    """Concatenate all host and phage proteins into one FASTA with genome prefixes.

    Returns (host_genomes, phage_genomes) sorted lists.
    """
    host_genomes: list[str] = []
    phage_genomes: list[str] = []
    total_proteins = 0

    with open(output_fasta, "w", encoding="utf-8") as out:
        # Host proteins.
        host_dirs = sorted(HOST_PROTEIN_DIR.iterdir())
        for i, host_dir in enumerate(host_dirs):
            protein_file = host_dir / "predicted_proteins.faa"
            if not protein_file.exists():
                continue
            genome_id = host_dir.name
            host_genomes.append(genome_id)
            n_proteins = 0
            with open(protein_file, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(">"):
                        # Prefix with genome ID: >HOST__001-023__gene_1
                        header = line[1:].split()[0]
                        out.write(f">HOST__{genome_id}__{header}\n")
                        n_proteins += 1
                    else:
                        # Remove stop codon markers.
                        out.write(line.rstrip().replace("*", "") + "\n")
            total_proteins += n_proteins
            if (i + 1) % 50 == 0:
                LOGGER.info(
                    "Concatenated %d/%d host genomes (%d proteins so far)", i + 1, len(host_dirs), total_proteins
                )

        # Phage proteins.
        if not PHAGE_PROTEOME_PATH.exists():
            raise FileNotFoundError(f"Phage proteome not found at {PHAGE_PROTEOME_PATH}")

        current_phage = None
        with open(PHAGE_PROTEOME_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    header = line[1:].strip()
                    # Parse phage name: "409_P1|query_prot_0001" -> "409_P1"
                    phage_name = header.split("|")[0] if "|" in header else header.split()[0]
                    if phage_name != current_phage:
                        current_phage = phage_name
                        if phage_name not in phage_genomes:
                            phage_genomes.append(phage_name)
                    out.write(f">PHAGE__{phage_name}__{header.replace(' ', '_')}\n")
                    total_proteins += 1
                else:
                    out.write(line.rstrip().replace("*", "") + "\n")

    host_genomes.sort()
    phage_genomes.sort()
    LOGGER.info(
        "Concatenated %d total proteins from %d host genomes + %d phage genomes",
        total_proteins,
        len(host_genomes),
        len(phage_genomes),
    )
    return host_genomes, phage_genomes


def run_mmseqs2_clustering(input_fasta: Path, work_dir: Path) -> Path:
    """Run MMseqs2 clustering and return path to the TSV result."""
    db_path = work_dir / "seqdb"
    cluster_path = work_dir / "clusters"
    tsv_path = work_dir / "clusters.tsv"
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    LOGGER.info("Creating MMseqs2 database...")
    subprocess.run(
        ["mmseqs", "createdb", str(input_fasta), str(db_path)],
        check=True,
        capture_output=True,
    )

    LOGGER.info(
        "Running MMseqs2 clustering (min-seq-id=%.1f, coverage=%.1f)...",
        MMSEQS_MIN_SEQ_ID,
        MMSEQS_COVERAGE,
    )
    subprocess.run(
        [
            "mmseqs",
            "cluster",
            str(db_path),
            str(cluster_path),
            str(tmp_dir),
            "--min-seq-id",
            str(MMSEQS_MIN_SEQ_ID),
            "-c",
            str(MMSEQS_COVERAGE),
            "--cov-mode",
            str(MMSEQS_COVERAGE_MODE),
            "--cluster-mode",
            "0",  # set-cover clustering (greedy)
            "--threads",
            "4",
        ],
        check=True,
        capture_output=True,
    )

    LOGGER.info("Creating TSV output...")
    subprocess.run(
        ["mmseqs", "createtsv", str(db_path), str(db_path), str(cluster_path), str(tsv_path)],
        check=True,
        capture_output=True,
    )

    return tsv_path


def build_presence_absence_matrix(
    cluster_tsv: Path,
    host_genomes: list[str],
    phage_genomes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build binary presence-absence matrices from MMseqs2 cluster TSV.

    Returns (host_matrix, phage_matrix) where rows are genomes and columns are cluster IDs.
    """
    # Parse cluster assignments: representative -> member.
    cluster_members: dict[str, set[str]] = defaultdict(set)
    with open(cluster_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                cluster_members[representative].add(member)

    LOGGER.info("Found %d clusters from MMseqs2", len(cluster_members))

    # Map each protein to its genome.
    genome_to_clusters: dict[str, set[str]] = defaultdict(set)
    for rep, members in cluster_members.items():
        for member in members:
            # Parse genome from header: HOST__001-023__gene_1 or PHAGE__409_P1__...
            parts = member.split("__", 2)
            if len(parts) >= 2:
                genome_id = parts[1]
                genome_to_clusters[genome_id].add(rep)

    # Identify non-singleton clusters (present in >= MIN_GENOME_COUNT genomes).
    all_genomes = set(host_genomes) | set(phage_genomes)
    cluster_genome_counts: dict[str, int] = {}
    for rep in cluster_members:
        genomes_with_cluster = sum(1 for g in all_genomes if rep in genome_to_clusters.get(g, set()))
        cluster_genome_counts[rep] = genomes_with_cluster

    non_singleton_clusters = sorted(rep for rep, count in cluster_genome_counts.items() if count >= MIN_GENOME_COUNT)
    LOGGER.info(
        "After filtering (>=%d genomes): %d / %d clusters retained",
        MIN_GENOME_COUNT,
        len(non_singleton_clusters),
        len(cluster_members),
    )

    # Build host matrix.
    host_data = np.zeros((len(host_genomes), len(non_singleton_clusters)), dtype=np.int8)
    for i, genome in enumerate(host_genomes):
        genome_clusters = genome_to_clusters.get(genome, set())
        for j, cluster in enumerate(non_singleton_clusters):
            if cluster in genome_clusters:
                host_data[i, j] = 1

    # Build phage matrix.
    phage_data = np.zeros((len(phage_genomes), len(non_singleton_clusters)), dtype=np.int8)
    for i, genome in enumerate(phage_genomes):
        genome_clusters = genome_to_clusters.get(genome, set())
        for j, cluster in enumerate(non_singleton_clusters):
            if cluster in genome_clusters:
                phage_data[i, j] = 1

    cluster_names = [f"pf_{i}" for i in range(len(non_singleton_clusters))]
    host_df = pd.DataFrame(host_data, index=host_genomes, columns=cluster_names)
    phage_df = pd.DataFrame(phage_data, index=phage_genomes, columns=cluster_names)

    # Log summary statistics.
    host_nonzero = (host_data.sum(axis=0) > 0).sum()
    phage_nonzero = (phage_data.sum(axis=0) > 0).sum()
    both_nonzero = ((host_data.sum(axis=0) > 0) & (phage_data.sum(axis=0) > 0)).sum()
    LOGGER.info(
        "Matrix: %d host genomes x %d clusters, %d phage genomes x %d clusters",
        len(host_genomes),
        len(cluster_names),
        len(phage_genomes),
        len(cluster_names),
    )
    LOGGER.info(
        "Cluster presence: %d host-only, %d phage-only, %d shared",
        host_nonzero - both_nonzero,
        phage_nonzero - both_nonzero,
        both_nonzero,
    )

    return host_df, phage_df


def main() -> None:
    setup_logging()
    LOGGER.info("GT08 protein family precompute starting at %s", datetime.now(timezone.utc).isoformat())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_fasta = OUTPUT_DIR / "all_proteomes.faa"

    # Step 1: Concatenate all proteomes.
    LOGGER.info("Step 1: Concatenating proteomes...")
    host_genomes, phage_genomes = concatenate_proteomes(combined_fasta)

    # Step 2: Run MMseqs2 clustering.
    LOGGER.info("Step 2: Running MMseqs2 clustering...")
    mmseqs_work = OUTPUT_DIR / "mmseqs_work"
    mmseqs_work.mkdir(exist_ok=True)
    cluster_tsv = run_mmseqs2_clustering(combined_fasta, mmseqs_work)

    # Step 3: Build presence-absence matrices.
    LOGGER.info("Step 3: Building presence-absence matrices...")
    host_matrix, phage_matrix = build_presence_absence_matrix(cluster_tsv, host_genomes, phage_genomes)

    # Step 4: Save outputs.
    host_matrix.to_csv(OUTPUT_DIR / "host_protein_families.csv")
    phage_matrix.to_csv(OUTPUT_DIR / "phage_protein_families.csv")

    # Also save as compressed npz for fast loading.
    np.savez_compressed(
        OUTPUT_DIR / "protein_family_matrices.npz",
        host_data=host_matrix.values,
        phage_data=phage_matrix.values,
        host_genomes=np.array(host_genomes),
        phage_genomes=np.array(phage_genomes),
        cluster_names=np.array(host_matrix.columns.tolist()),
    )

    # Variance pre-flight.
    LOGGER.info("=" * 60)
    LOGGER.info("VARIANCE PRE-FLIGHT")
    LOGGER.info("=" * 60)

    # Host-side variance.
    host_var = host_matrix.var(axis=0)
    host_high_var = (host_var > 0.05).sum()
    LOGGER.info(
        "Host features: %d clusters, %d with variance > 0.05 (%.1f%%)",
        len(host_var),
        host_high_var,
        host_high_var / len(host_var) * 100 if len(host_var) > 0 else 0,
    )

    # Phage-side variance.
    phage_var = phage_matrix.var(axis=0)
    phage_high_var = (phage_var > 0.05).sum()
    LOGGER.info(
        "Phage features: %d clusters, %d with variance > 0.05 (%.1f%%)",
        len(phage_var),
        phage_high_var,
        phage_high_var / len(phage_var) * 100 if len(phage_var) > 0 else 0,
    )

    # Check for degenerate features (>90% zero or >90% one).
    host_mean = host_matrix.mean(axis=0)
    host_degenerate = ((host_mean < 0.1) | (host_mean > 0.9)).sum()
    phage_mean = phage_matrix.mean(axis=0)
    phage_degenerate = ((phage_mean < 0.1) | (phage_mean > 0.9)).sum()
    LOGGER.info(
        "Degenerate features (>90%% same value): host=%d/%d, phage=%d/%d",
        host_degenerate,
        len(host_mean),
        phage_degenerate,
        len(phage_mean),
    )

    LOGGER.info("Outputs saved to %s", OUTPUT_DIR)

    # Clean up large intermediate files.
    if combined_fasta.exists():
        combined_fasta.unlink()
        LOGGER.info("Cleaned up concatenated FASTA (large intermediate)")
    if mmseqs_work.exists():
        shutil.rmtree(mmseqs_work)
        LOGGER.info("Cleaned up MMseqs2 work directory")

    LOGGER.info("Precompute complete")


if __name__ == "__main__":
    main()
