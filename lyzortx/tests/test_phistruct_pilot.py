from lyzortx.pipeline.track_a.steps.run_phistruct_rbp_pilot import (
    expected_calibration_error,
    phistruct_style_embedding,
    top_k_hit_rate,
)


def test_phistruct_style_embedding_is_deterministic_and_normalized() -> None:
    row = {"Family": "Siphoviridae", "Genus": "Tequintavirus"}
    emb_a = phistruct_style_embedding(row, dim=8)
    emb_b = phistruct_style_embedding(row, dim=8)

    assert emb_a == emb_b
    assert len(emb_a) == 8
    sq_norm = sum(float(v) * float(v) for v in emb_a.values())
    assert 0.99 <= sq_norm <= 1.01


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
