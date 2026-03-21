from lyzortx.pipeline.steel_thread_v0.steps.st06_recommend_top3 import (
    bootstrap_topk_ci,
    evaluate_holdout_slice,
)


def test_evaluate_holdout_slice_handles_dual_slices() -> None:
    holdout_by_bacteria = {
        "b1": [
            {"label_hard_binary": "1", "is_strict_trainable": "1"},
            {"label_hard_binary": "0", "is_strict_trainable": "1"},
        ],
        "b2": [
            {"label_hard_binary": "1", "is_strict_trainable": "0"},
            {"label_hard_binary": "0", "is_strict_trainable": "0"},
        ],
    }
    recs_by_bacteria = {
        "b1": [{"label_hard_binary": "1", "label_strict_confidence_tier": "high_conf_pos"}],
        "b2": [{"label_hard_binary": "0", "label_strict_confidence_tier": "ambiguous"}],
    }

    full_metrics = evaluate_holdout_slice(holdout_by_bacteria, recs_by_bacteria, slice_name="full_label")
    strict_metrics = evaluate_holdout_slice(
        holdout_by_bacteria,
        recs_by_bacteria,
        slice_name="strict_confidence",
    )

    assert full_metrics["holdout_strain_count"] == 2
    assert full_metrics["holdout_hit_count"] == 1
    assert strict_metrics["holdout_strain_count"] == 1
    assert strict_metrics["holdout_hit_count"] == 1


def test_evaluate_holdout_slice_counts_missing_strict_recs_as_miss() -> None:
    holdout_by_bacteria = {
        "b1": [{"label_hard_binary": "1", "is_strict_trainable": "1"}],
    }
    recs_by_bacteria = {
        "b1": [{"label_hard_binary": "1", "label_strict_confidence_tier": "ambiguous"}],
    }

    strict_metrics = evaluate_holdout_slice(
        holdout_by_bacteria,
        recs_by_bacteria,
        slice_name="strict_confidence",
    )

    assert strict_metrics["holdout_strain_count"] == 1
    assert strict_metrics["holdout_hit_count"] == 0


def test_bootstrap_topk_ci_bounds() -> None:
    holdout_by_bacteria = {
        "b1": [{"label_hard_binary": "1", "is_strict_trainable": "1"}],
        "b2": [{"label_hard_binary": "0", "is_strict_trainable": "1"}],
    }
    recs_by_bacteria = {
        "b1": [{"label_hard_binary": "1", "label_strict_confidence_tier": "high_conf_pos"}],
        "b2": [{"label_hard_binary": "0", "label_strict_confidence_tier": "high_conf_neg"}],
    }

    ci = bootstrap_topk_ci(
        holdout_by_bacteria,
        recs_by_bacteria,
        slice_name="full_label",
        bootstrap_samples=100,
        bootstrap_random_state=7,
    )

    assert ci["bootstrap_samples"] == 100
    assert 0.0 <= ci["ci_low_topk_hit_rate_all_strains"] <= ci["ci_high_topk_hit_rate_all_strains"] <= 1.0
    assert 0.0 <= ci["ci_low_topk_hit_rate_susceptible_only"] <= ci["ci_high_topk_hit_rate_susceptible_only"] <= 1.0
