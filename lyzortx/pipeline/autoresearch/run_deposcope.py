"""Run DepoScope depolymerase prediction on phage protein sets.

Thin wrapper around the DepoScope inference pipeline (Concha-Eloko et al., PLoS Comp Bio 2024).
Reads per-phage protein FASTAs, runs fine-tuned ESM-2 t12 + CNN classifier, outputs a CSV of
depolymerase predictions with scores and per-residue token labels.

Model weights must be pre-downloaded to .scratch/deposcope/:
  - esm2_t12_35M_UR50D__fulltrain__finetuneddepolymerase.2103.4_labels/  (fine-tuned ESM-2)
  - Deposcope.esm2_t12_35M_UR50D.2203.full.model  (CNN classifier)

Usage:
    python -m lyzortx.pipeline.autoresearch.run_deposcope
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lyzortx.log_config import setup_logging

LOGGER = logging.getLogger(__name__)

PROTEIN_FASTA_DIR = Path("lyzortx/generated_outputs/track_d/phage_protein_sets/protein_fastas")
MODEL_DIR = Path(".scratch/deposcope")
ESM2_CHECKPOINT = MODEL_DIR / "esm2_t12_35M_UR50D__fulltrain__finetuneddepolymerase.2103.4_labels" / "checkpoint-2255"
CNN_WEIGHTS = MODEL_DIR / "Deposcope.esm2_t12_35M_UR50D.2203.full.model"
OUTPUT_PATH = Path(".scratch/deposcope/predictions.csv")

MAX_LENGTH = 1024
SCORE_THRESHOLD = 0.5


class DpoClassifier(nn.Module):
    """DepoScope CNN classifier. Takes ESM-2 token class IDs as input."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pretrained_model: AutoModelForTokenClassification,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * (max_length - 8), 32)
        self.classifier = nn.Linear(32, 1)

    def _predict_tokens(self, sequence: str) -> torch.Tensor:
        """Run ESM-2 token classification on a single sequence."""
        input_ids = self.tokenizer.encode(sequence, truncation=True, return_tensors="pt")
        input_ids = input_ids.to(next(self.pretrained_model.parameters()).device)
        with torch.no_grad():
            outputs = self.pretrained_model(input_ids)
        probs = F.softmax(outputs.logits, dim=-1)
        _, token_ids = torch.max(probs, dim=-1)
        return token_ids.view(1, -1)

    def _pad_or_truncate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Force token sequence to exactly max_length."""
        seq_len = tokens.shape[1]
        if seq_len < self.max_length:
            padding = torch.zeros(1, self.max_length - seq_len, dtype=tokens.dtype, device=tokens.device)
            return torch.cat([tokens, padding], dim=1)
        return tokens[:, : self.max_length]

    def forward(self, sequences: list[str]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Classify a batch of sequences."""
        token_outputs = []
        padded = []
        for seq in sequences:
            tokens = self._predict_tokens(seq)
            token_outputs.append(tokens.squeeze(0))
            padded.append(self._pad_or_truncate(tokens))
        x = torch.cat(padded, dim=0).unsqueeze(1).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.classifier(x)
        return logits, token_outputs


def load_model(device: torch.device) -> DpoClassifier:
    """Load the fine-tuned ESM-2 and CNN classifier."""
    if not ESM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"ESM-2 checkpoint not found at {ESM2_CHECKPOINT}")
    if not CNN_WEIGHTS.exists():
        raise FileNotFoundError(f"CNN weights not found at {CNN_WEIGHTS}")

    LOGGER.info("Loading fine-tuned ESM-2 t12 from %s", ESM2_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(str(ESM2_CHECKPOINT))
    esm_model = AutoModelForTokenClassification.from_pretrained(str(ESM2_CHECKPOINT))
    esm_model = esm_model.to(device).eval()

    model = DpoClassifier(tokenizer, esm_model)
    LOGGER.info("Loading CNN classifier from %s", CNN_WEIGHTS)
    state_dict = torch.load(str(CNN_WEIGHTS), map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


def predict_phage(model: DpoClassifier, fasta_path: Path) -> list[dict]:
    """Run DepoScope on all proteins in a phage FASTA. Returns list of per-protein results."""
    results = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        if len(seq) < 10:
            continue
        with torch.no_grad():
            logits, token_outputs = model([seq])
        score = torch.sigmoid(logits).item()
        is_depolymerase = score >= SCORE_THRESHOLD
        results.append(
            {
                "protein_id": record.id,
                "length": len(seq),
                "deposcope_score": round(score, 4),
                "is_depolymerase": is_depolymerase,
            }
        )
    return results


def main() -> None:
    setup_logging()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = load_model(device)

    fasta_files = sorted(PROTEIN_FASTA_DIR.glob("*.faa"))
    if not fasta_files:
        raise FileNotFoundError(f"No .faa files in {PROTEIN_FASTA_DIR}")
    LOGGER.info("Processing %d phage protein FASTAs", len(fasta_files))

    all_results: list[dict] = []
    t0 = time.time()
    total_proteins = 0

    for i, fasta_path in enumerate(fasta_files):
        phage = fasta_path.stem
        proteins = predict_phage(model, fasta_path)
        for p in proteins:
            p["phage"] = phage
        all_results.extend(proteins)
        total_proteins += len(proteins)

        depo_count = sum(1 for p in proteins if p["is_depolymerase"])
        if depo_count > 0:
            LOGGER.info(
                "  %s: %d/%d proteins are depolymerases (max score %.3f)",
                phage,
                depo_count,
                len(proteins),
                max(p["deposcope_score"] for p in proteins),
            )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = total_proteins / elapsed
            LOGGER.info(
                "Progress: %d/%d phages, %d proteins, %.1f prot/s, %.0fs elapsed",
                i + 1,
                len(fasta_files),
                total_proteins,
                rate,
                elapsed,
            )

    elapsed = time.time() - t0
    LOGGER.info(
        "Done: %d proteins across %d phages in %.1fs (%.1f prot/s)",
        total_proteins,
        len(fasta_files),
        elapsed,
        total_proteins / elapsed if elapsed > 0 else 0,
    )

    # Summary
    depo_proteins = [r for r in all_results if r["is_depolymerase"]]
    depo_phages = len({r["phage"] for r in depo_proteins})
    LOGGER.info(
        "Depolymerases found: %d proteins in %d/%d phages",
        len(depo_proteins),
        depo_phages,
        len(fasta_files),
    )

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phage", "protein_id", "length", "deposcope_score", "is_depolymerase"]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    LOGGER.info("Wrote predictions to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
