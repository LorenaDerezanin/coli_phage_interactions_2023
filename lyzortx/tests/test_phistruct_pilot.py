from pathlib import Path

from lyzortx.pipeline.track_a.steps.run_phistruct_rbp_pilot import (
    build_phistruct_style_embeddings,
    expected_calibration_error,
    top_k_hit_rate,
)


def test_build_phistruct_style_embeddings_from_fasta_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "P1.fna").write_text(">P1\nACGTACGTACGTACGT\n", encoding="utf-8")
    (tmp_path / "P2.fna").write_text(">P2\nAAAACCCCGGGGTTTT\n", encoding="utf-8")

    emb_a = build_phistruct_style_embeddings(
        phage_names=["P1", "P2"],
        phage_fna_dir=tmp_path,
        embedding_dim=4,
        kmer_size=2,
        random_state=42,
    )
    emb_b = build_phistruct_style_embeddings(
        phage_names=["P1", "P2"],
        phage_fna_dir=tmp_path,
        embedding_dim=4,
        kmer_size=2,
        random_state=42,
    )

    assert emb_a["P1"] == emb_b["P1"]
    assert emb_a["P2"] == emb_b["P2"]
    assert len(emb_a["P1"]) == 4


def test_build_phistruct_style_embeddings_tracks_missing_genomes(tmp_path: Path) -> None:
    (tmp_path / "P1.fna").write_text(">P1\nACGTACGT\n", encoding="utf-8")

    emb = build_phistruct_style_embeddings(
        phage_names=["P1", "P_MISSING"],
        phage_fna_dir=tmp_path,
        embedding_dim=3,
        kmer_size=2,
        random_state=42,
    )

    assert emb["__metadata__"]["missing_genome_count"] == 1.0


def test_expected_calibration_error_simple_case() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    ece = expected_calibration_error(y_true, y_prob, n_bins=2)

    assert ece == 0.15


def test_top_k_hit_rate() -> None:
    rows = [
        {"bacteria": "B1", "label": 0, "pred_prob": 0.9},
        {"bacteria": "B1", "label": 1, "pred_prob": 0.8},
        {"bacteria": "B2", "label": 0, "pred_prob": 0.7},
        {"bacteria": "B2", "label": 1, "pred_prob": 0.6},
    ]

    assert top_k_hit_rate(rows, top_k=1) == 0.0
    assert top_k_hit_rate(rows, top_k=2) == 1.0
