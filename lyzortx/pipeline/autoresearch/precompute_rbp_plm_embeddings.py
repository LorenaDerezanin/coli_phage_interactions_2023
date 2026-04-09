#!/usr/bin/env python3
"""Precompute RBP protein language model embeddings for all panel phages.

Supports two backends:
  saprot  — ProstT5 (AA→3Di) + SaProt (AA+3Di→1280-dim embedding). Structure-aware.
  esm2    — ESM-2 650M (AA→1280-dim embedding). Sequence-only, simpler/faster.

Usage:
    python -m lyzortx.pipeline.autoresearch.precompute_rbp_plm_embeddings
    python -m lyzortx.pipeline.autoresearch.precompute_rbp_plm_embeddings --model esm2
    python -m lyzortx.pipeline.autoresearch.precompute_rbp_plm_embeddings --dry-run

Output: .scratch/rbp_plm_embeddings.npz with keys:
  phage_names    — (N,) array of phage name strings
  embeddings     — (N, 1280) array of per-phage mean-pooled embeddings
  has_rbp        — (N,) boolean array
  rbp_counts     — (N,) integer array
  model_backend  — string identifying the model used
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

# torch and transformers are imported lazily inside model-loading functions because:
# (1) torch is ~2 GB and takes several seconds to import — --dry-run and the test suite
#     should not pay this cost;
# (2) derive_rbp_protein_features imports this module's extraction utilities and would
#     transitively pull in torch at top level otherwise.
from lyzortx.pipeline.autoresearch.derive_rbp_protein_features import (
    extract_rbp_proteins_for_phage,
)

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FASTA_DIR = REPO_ROOT / "data" / "genomics" / "phages" / "FNA"
DEFAULT_ANNOTATION_DIR = REPO_ROOT / "data" / "annotations" / "pharokka"
DEFAULT_OUTPUT = REPO_ROOT / ".scratch" / "rbp_plm_embeddings.npz"

# ProstT5 uses these to mark rare residues.
RARE_AA_MAP = str.maketrans("UZOB", "XXXX")

# SaProt/ESM-2 embedding dimension.
EMBEDDING_DIM = 1280

# Maximum protein length (in residues) for model input. Longer sequences are truncated.
# SaProt max position embeddings = 1024 tokens; each residue = 1 token in the interleaved format.
MAX_PROTEIN_LENGTH = 1022  # Leave room for BOS/EOS tokens.

SAPROT_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
PROSTT5_MODEL_ID = "Rostlab/ProstT5_fp16"
ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"


def _select_device() -> str:
    """Select the best available torch device."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _discover_phages(
    fasta_dir: Path,
    annotation_dir: Path,
) -> list[tuple[str, Path]]:
    """Discover phages that have both FASTA and Pharokka annotation files.

    Returns sorted list of (phage_name, fasta_path) tuples.
    """
    phages: list[tuple[str, Path]] = []
    for fasta_path in sorted(fasta_dir.glob("*.fna")):
        phage_name = fasta_path.stem
        anno_path = annotation_dir / f"{phage_name}_cds_final_merged_output.tsv"
        if anno_path.exists():
            phages.append((phage_name, fasta_path))
        else:
            LOGGER.debug("No annotation for %s, skipping", phage_name)
    return phages


# ---------------------------------------------------------------------------
# SaProt backend: ProstT5 → 3Di → SaProt embedding
# ---------------------------------------------------------------------------


def _load_prostt5(device: str):
    """Load ProstT5 model and tokenizer for AA→3Di translation."""
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

    LOGGER.info("Loading ProstT5 from %s ...", PROSTT5_MODEL_ID)
    t0 = time.time()
    tokenizer = T5Tokenizer.from_pretrained(PROSTT5_MODEL_ID, do_lower_case=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(PROSTT5_MODEL_ID)
    model = model.to(device)
    model.eval()
    LOGGER.info("ProstT5 loaded in %.1fs on %s", time.time() - t0, device)
    return model, tokenizer


def _load_saprot(device: str):
    """Load SaProt model and tokenizer for structure-aware embeddings."""
    from transformers import EsmModel, EsmTokenizer

    LOGGER.info("Loading SaProt from %s ...", SAPROT_MODEL_ID)
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_MODEL_ID)
    model = EsmModel.from_pretrained(SAPROT_MODEL_ID)
    model = model.to(device)
    model.eval()
    LOGGER.info("SaProt loaded in %.1fs on %s", time.time() - t0, device)
    return model, tokenizer


def _translate_aa_to_3di(
    aa_seq: str,
    prostt5_model,
    prostt5_tokenizer,
    device: str,
) -> str:
    """Translate an amino acid sequence to 3Di structural tokens using ProstT5."""
    import torch

    # Clean sequence: replace rare residues, truncate.
    clean = aa_seq.translate(RARE_AA_MAP)[:MAX_PROTEIN_LENGTH]

    # ProstT5 input: "<AA2fold>" prefix + space-separated AA characters.
    input_text = "<AA2fold> " + " ".join(list(clean))
    ids = prostt5_tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = prostt5_model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            max_new_tokens=len(clean) + 10,
            do_sample=False,
        )

    decoded = prostt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    threedi = decoded.replace(" ", "").lower()

    # Ensure 3Di length matches AA length (truncate or pad with 'd' if needed).
    if len(threedi) < len(clean):
        threedi = threedi + "d" * (len(clean) - len(threedi))
    elif len(threedi) > len(clean):
        threedi = threedi[: len(clean)]

    return threedi


def _interleave_aa_3di(aa_seq: str, threedi: str) -> str:
    """Interleave AA and 3Di tokens for SaProt input.

    Each position becomes one uppercase AA + one lowercase 3Di character.
    E.g., AA="MEV", 3Di="dvr" → "MdEvVr"
    """
    return "".join(aa + ss for aa, ss in zip(aa_seq, threedi))


def _saprot_embedding(
    interleaved_seq: str,
    saprot_model,
    saprot_tokenizer,
    device: str,
) -> np.ndarray:
    """Compute a 1280-dim SaProt embedding from an interleaved AA+3Di sequence."""
    import torch

    inputs = saprot_tokenizer(interleaved_seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = saprot_model(**inputs, output_hidden_states=True)

    # Mean-pool last hidden layer over sequence positions (excluding BOS/EOS).
    hidden = outputs.hidden_states[-1]  # (1, seq_len, 1280)
    # Exclude first (BOS) and last (EOS) tokens.
    embedding = hidden[:, 1:-1, :].mean(dim=1).squeeze(0)  # (1280,)
    return embedding.cpu().float().numpy()


def compute_saprot_embeddings(
    phage_proteins: dict[str, list[str]],
    device: str,
) -> dict[str, np.ndarray]:
    """Compute SaProt embeddings for all phage RBP proteins.

    Args:
        phage_proteins: {phage_name: [protein_seq, ...]}
        device: torch device string

    Returns:
        {phage_name: 1280-dim mean-pooled embedding}
    """
    prostt5_model, prostt5_tokenizer = _load_prostt5(device)
    saprot_model, saprot_tokenizer = _load_saprot(device)

    # Flatten all proteins for sequential processing.
    all_items: list[tuple[str, str]] = []
    for phage, seqs in sorted(phage_proteins.items()):
        for seq in seqs:
            all_items.append((phage, seq))

    LOGGER.info("Computing SaProt embeddings for %d RBP proteins ...", len(all_items))
    t0 = time.time()

    # Per-protein embeddings, grouped by phage for mean-pooling.
    phage_emb_lists: dict[str, list[np.ndarray]] = {p: [] for p in phage_proteins}

    for i, (phage, aa_seq) in enumerate(all_items, 1):
        # Step 1: ProstT5 AA→3Di (handles cleaning/truncation internally)
        threedi = _translate_aa_to_3di(aa_seq, prostt5_model, prostt5_tokenizer, device)

        # Step 2: Interleave (use cleaned sequence matching 3Di length)
        clean = aa_seq.translate(RARE_AA_MAP)[:MAX_PROTEIN_LENGTH]
        interleaved = _interleave_aa_3di(clean, threedi)

        # Step 3: SaProt embedding
        emb = _saprot_embedding(interleaved, saprot_model, saprot_tokenizer, device)
        phage_emb_lists[phage].append(emb)

        if i % 10 == 0 or i == len(all_items):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            LOGGER.info(
                "  %d/%d proteins (%.1f/s, %.0fs elapsed)",
                i,
                len(all_items),
                rate,
                elapsed,
            )

    # Mean-pool per phage.
    result: dict[str, np.ndarray] = {}
    for phage, emb_list in phage_emb_lists.items():
        if emb_list:
            result[phage] = np.mean(emb_list, axis=0).astype(np.float32)
        else:
            result[phage] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    LOGGER.info("SaProt embeddings computed in %.1fs", time.time() - t0)
    return result


# ---------------------------------------------------------------------------
# ESM-2 backend: sequence-only embeddings
# ---------------------------------------------------------------------------


def _load_esm2(device: str):
    """Load ESM-2 650M model and tokenizer."""
    from transformers import EsmModel, EsmTokenizer

    LOGGER.info("Loading ESM-2 from %s ...", ESM2_MODEL_ID)
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(ESM2_MODEL_ID)
    model = EsmModel.from_pretrained(ESM2_MODEL_ID)
    model = model.to(device)
    model.eval()
    LOGGER.info("ESM-2 loaded in %.1fs on %s", time.time() - t0, device)
    return model, tokenizer


def _esm2_embedding(
    aa_seq: str,
    esm2_model,
    esm2_tokenizer,
    device: str,
) -> np.ndarray:
    """Compute a 1280-dim ESM-2 embedding from an AA sequence."""
    import torch

    clean = aa_seq.translate(RARE_AA_MAP)[:MAX_PROTEIN_LENGTH]
    inputs = esm2_tokenizer(clean, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = esm2_model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]
    embedding = hidden[:, 1:-1, :].mean(dim=1).squeeze(0)
    return embedding.cpu().float().numpy()


def compute_esm2_embeddings(
    phage_proteins: dict[str, list[str]],
    device: str,
) -> dict[str, np.ndarray]:
    """Compute ESM-2 embeddings for all phage RBP proteins."""
    esm2_model, esm2_tokenizer = _load_esm2(device)

    all_items: list[tuple[str, str]] = []
    for phage, seqs in sorted(phage_proteins.items()):
        for seq in seqs:
            all_items.append((phage, seq))

    LOGGER.info("Computing ESM-2 embeddings for %d RBP proteins ...", len(all_items))
    t0 = time.time()

    phage_emb_lists: dict[str, list[np.ndarray]] = {p: [] for p in phage_proteins}

    for i, (phage, aa_seq) in enumerate(all_items, 1):
        emb = _esm2_embedding(aa_seq, esm2_model, esm2_tokenizer, device)
        phage_emb_lists[phage].append(emb)

        if i % 10 == 0 or i == len(all_items):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            LOGGER.info(
                "  %d/%d proteins (%.1f/s, %.0fs elapsed)",
                i,
                len(all_items),
                rate,
                elapsed,
            )

    result: dict[str, np.ndarray] = {}
    for phage, emb_list in phage_emb_lists.items():
        if emb_list:
            result[phage] = np.mean(emb_list, axis=0).astype(np.float32)
        else:
            result[phage] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    LOGGER.info("ESM-2 embeddings computed in %.1fs", time.time() - t0)
    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def extract_all_rbp_proteins(
    fasta_dir: Path,
    annotation_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Extract RBP protein sequences for all discovered phages.

    Returns:
        phage_proteins: {phage_name: [protein_seq, ...]}
        rbp_counts: {phage_name: n_rbps}
    """
    phages = _discover_phages(fasta_dir, annotation_dir)
    LOGGER.info("Discovered %d phages with annotations", len(phages))

    phage_proteins: dict[str, list[str]] = {}
    rbp_counts: dict[str, int] = {}

    for i, (phage_name, fasta_path) in enumerate(phages, 1):
        proteins = extract_rbp_proteins_for_phage(phage_name, fasta_path, annotation_dir)
        phage_proteins[phage_name] = [p.protein_seq for p in proteins]
        rbp_counts[phage_name] = len(proteins)
        if i % 20 == 0 or i == len(phages):
            LOGGER.info("  Extracted RBPs for %d/%d phages", i, len(phages))

    n_with_rbps = sum(1 for seqs in phage_proteins.values() if seqs)
    n_total_rbps = sum(len(seqs) for seqs in phage_proteins.values())
    LOGGER.info(
        "%d/%d phages have annotated RBPs (%d total RBP proteins)",
        n_with_rbps,
        len(phages),
        n_total_rbps,
    )
    return phage_proteins, rbp_counts


def save_embeddings(
    output_path: Path,
    phage_proteins: dict[str, list[str]],
    embeddings: dict[str, np.ndarray],
    rbp_counts: dict[str, int],
    model_backend: str,
) -> None:
    """Save precomputed embeddings to .npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phage_names = sorted(embeddings.keys())
    emb_matrix = np.stack([embeddings[p] for p in phage_names])
    has_rbp = np.array([len(phage_proteins.get(p, [])) > 0 for p in phage_names])
    rbp_count_arr = np.array([rbp_counts.get(p, 0) for p in phage_names])

    np.savez(
        output_path,
        phage_names=np.array(phage_names),
        embeddings=emb_matrix,
        has_rbp=has_rbp,
        rbp_counts=rbp_count_arr,
        model_backend=np.array(model_backend),
    )
    LOGGER.info(
        "Saved %d phage embeddings (%d×%d) to %s",
        len(phage_names),
        emb_matrix.shape[0],
        emb_matrix.shape[1],
        output_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=["saprot", "esm2"],
        default="saprot",
        help="Embedding backend: saprot (ProstT5→SaProt, structure-aware) or esm2 (sequence-only)",
    )
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=DEFAULT_FASTA_DIR,
        help="Directory containing phage FASTA files (*.fna)",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing Pharokka CDS annotation TSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (auto-detected if omitted: mps > cuda > cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract RBP proteins and report counts without running model inference",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from lyzortx.log_config import setup_logging

    setup_logging()
    args = parse_args(argv)

    phage_proteins, rbp_counts = extract_all_rbp_proteins(args.fasta_dir, args.annotation_dir)

    if args.dry_run:
        LOGGER.info("Dry run — skipping model inference")
        return

    device = args.device or _select_device()
    LOGGER.info("Using device: %s, model backend: %s", device, args.model)

    if args.model == "saprot":
        embeddings = compute_saprot_embeddings(phage_proteins, device)
    else:
        embeddings = compute_esm2_embeddings(phage_proteins, device)

    save_embeddings(args.output, phage_proteins, embeddings, rbp_counts, args.model)


if __name__ == "__main__":
    main()
