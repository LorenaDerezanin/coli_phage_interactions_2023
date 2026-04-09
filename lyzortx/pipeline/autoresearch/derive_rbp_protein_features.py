"""Derive per-phage RBP features from precomputed PLM embeddings.

The embedding precomputation step (ProstT5→SaProt or ESM-2) runs separately via
``precompute_rbp_plm_embeddings.py`` and caches results to ``.scratch/rbp_plm_embeddings.npz``.
This module loads the cached embeddings, applies PCA dimensionality reduction, and produces
per-phage feature rows for the ``phage_rbp_struct`` slot.

For phages without Pharokka-detected RBPs (16/97 in the current panel), all embedding
features are zero and the ``has_annotated_rbp`` indicator is 0.

RBP protein extraction utilities (``extract_rbp_proteins_for_phage``, ``RbpProtein``, etc.)
are kept here for reuse by the precompute script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Seq import Seq
from sklearn.decomposition import PCA

from lyzortx.pipeline.track_l.steps.parse_annotations import (
    CdsRecord,
    classify_rbp_genes,
    parse_merged_tsv,
)

LOGGER = logging.getLogger(__name__)

# Default PCA dimensionality for PLM embeddings.
DEFAULT_N_COMPONENTS = 32

# Raw PLM embedding dimension (SaProt and ESM-2 both produce 1280-dim).
RAW_EMBEDDING_DIM = 1280

# Per-phage feature names: metadata + PCA components.
# These are set dynamically based on n_components via build_phage_rbp_schema().
METADATA_FEATURES = ["has_annotated_rbp", "rbp_count"]


def build_phage_rbp_schema(n_components: int = DEFAULT_N_COMPONENTS) -> list[str]:
    """Return the ordered list of feature column names (without entity key)."""
    pca_features = [f"rbp_plm_pc{i}" for i in range(n_components)]
    return METADATA_FEATURES + pca_features


# ---------------------------------------------------------------------------
# RBP protein extraction (reused by precompute_rbp_plm_embeddings.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RbpProtein:
    """An extracted RBP protein sequence with metadata."""

    gene_id: str
    protein_seq: str
    annot: str


def read_genome_sequence(fasta_path: Path) -> dict[str, str]:
    """Read a genome FASTA and return {contig_id: sequence} mapping."""
    contigs: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    contigs[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
    if header is not None and chunks:
        contigs[header] = "".join(chunks)
    if not contigs:
        raise ValueError(f"No contigs found in {fasta_path}")
    return contigs


def extract_cds_protein(record: CdsRecord, contigs: dict[str, str]) -> str:
    """Extract CDS nucleotide sequence and translate to protein.

    Uses standard bacterial translation table (11). Handles forward and
    reverse complement strands.
    """
    contig_seq = contigs.get(record.contig)
    if contig_seq is None:
        raise ValueError(
            f"Contig {record.contig!r} for gene {record.gene!r} not found in genome. "
            f"Available contigs: {sorted(contigs)}"
        )
    # Pharokka uses 1-based coordinates.
    start = record.start - 1
    stop = record.stop
    if start < 0 or stop > len(contig_seq):
        raise ValueError(
            f"Gene {record.gene!r} coordinates ({record.start}-{record.stop}) "
            f"exceed contig {record.contig!r} length ({len(contig_seq)})"
        )
    nuc_seq = contig_seq[start:stop]
    if record.strand == "-":
        nuc_seq = str(Seq(nuc_seq).reverse_complement())

    # Translate with bacterial table 11, allowing alternative start codons.
    protein = str(Seq(nuc_seq).translate(table=11))
    # Strip trailing stop codon if present.
    if protein.endswith("*"):
        protein = protein[:-1]
    return protein


def extract_rbp_proteins_for_phage(
    phage_name: str,
    genome_fasta_path: Path,
    annotation_dir: Path,
) -> list[RbpProtein]:
    """Extract RBP protein sequences for a single phage.

    Returns empty list if no annotation file found or no RBPs annotated.
    """
    tsv_path = annotation_dir / f"{phage_name}_cds_final_merged_output.tsv"
    if not tsv_path.exists():
        LOGGER.debug("No Pharokka annotation for phage %s", phage_name)
        return []

    records = parse_merged_tsv(tsv_path)
    rbp_records = classify_rbp_genes(records)
    if not rbp_records:
        LOGGER.debug("No RBP genes annotated for phage %s", phage_name)
        return []

    contigs = read_genome_sequence(genome_fasta_path)
    proteins: list[RbpProtein] = []
    for rec in rbp_records:
        try:
            protein_seq = extract_cds_protein(rec, contigs)
            if len(protein_seq) >= 10:
                proteins.append(
                    RbpProtein(
                        gene_id=rec.gene,
                        protein_seq=protein_seq,
                        annot=rec.annot,
                    )
                )
        except (ValueError, KeyError) as exc:
            LOGGER.warning("Skipping RBP %s in phage %s: %s", rec.gene, phage_name, exc)
    return proteins


# ---------------------------------------------------------------------------
# PLM embedding loading and PCA
# ---------------------------------------------------------------------------


def load_cached_embeddings(cache_path: Path) -> dict:
    """Load precomputed PLM embeddings from .npz cache.

    Returns a dict with keys: phage_names, embeddings, has_rbp, rbp_counts, model_backend.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"PLM embedding cache not found at {cache_path}. "
            "Run: python -m lyzortx.pipeline.autoresearch.precompute_rbp_plm_embeddings"
        )
    data = np.load(cache_path, allow_pickle=True)
    return {
        "phage_names": list(data["phage_names"]),
        "embeddings": data["embeddings"],  # (N, 1280)
        "has_rbp": data["has_rbp"],
        "rbp_counts": data["rbp_counts"],
        "model_backend": str(data["model_backend"]),
    }


def fit_pca_on_embeddings(
    embeddings: np.ndarray,
    has_rbp: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> PCA:
    """Fit PCA on non-zero phage embeddings.

    Only phages with annotated RBPs (has_rbp=True) are used for fitting.
    """
    non_zero_mask = has_rbp.astype(bool)
    n_non_zero = non_zero_mask.sum()

    # Clamp n_components to available samples.
    effective_n = min(n_components, n_non_zero - 1)
    if effective_n < n_components:
        LOGGER.warning(
            "Only %d non-zero phage embeddings; reducing PCA from %d to %d components",
            n_non_zero,
            n_components,
            effective_n,
        )

    pca = PCA(n_components=effective_n, random_state=42)
    pca.fit(embeddings[non_zero_mask])
    LOGGER.info(
        "PCA fit: %d components explain %.1f%% of variance (from %d phages × %d dims)",
        effective_n,
        pca.explained_variance_ratio_.sum() * 100,
        n_non_zero,
        embeddings.shape[1],
    )
    return pca


def build_phage_rbp_plm_rows(
    cache_path: Path,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> list[dict[str, object]]:
    """Build feature rows for all phages from cached PLM embeddings.

    Loads cached embeddings, fits PCA, and produces one row per phage with
    has_annotated_rbp, rbp_count, and PCA component features.
    """
    cache = load_cached_embeddings(cache_path)
    phage_names = cache["phage_names"]
    embeddings = cache["embeddings"]
    has_rbp = cache["has_rbp"]
    rbp_counts = cache["rbp_counts"]

    LOGGER.info(
        "Loaded %d phage embeddings (model: %s) from %s",
        len(phage_names),
        cache["model_backend"],
        cache_path,
    )

    pca = fit_pca_on_embeddings(embeddings, has_rbp, n_components)
    effective_n = pca.n_components_

    # Project all phages (including zero-embedding ones).
    projected = pca.transform(embeddings)  # (N, effective_n)

    # Zero out PCA features for phages without RBPs (their raw embedding is all-zero,
    # but PCA centering would give them non-zero projections).
    for i in range(len(phage_names)):
        if not has_rbp[i]:
            projected[i, :] = 0.0

    rows: list[dict[str, object]] = []
    for i, phage in enumerate(phage_names):
        row: dict[str, object] = {
            "phage": phage,
            "has_annotated_rbp": int(has_rbp[i]),
            "rbp_count": int(rbp_counts[i]),
        }
        for j in range(effective_n):
            row[f"rbp_plm_pc{j}"] = round(float(projected[i, j]), 6)
        # Pad remaining components with zero if effective_n < n_components.
        for j in range(effective_n, n_components):
            row[f"rbp_plm_pc{j}"] = 0.0
        rows.append(row)

    return rows
