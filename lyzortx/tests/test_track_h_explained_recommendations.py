import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_h.steps.build_explained_recommendations import (
    bootstrap_probability_intervals,
    build_explained_recommendation_rows,
    main,
    render_markdown_report,
    select_top_recommendations,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_top_recommendations_returns_holdout_top_k_per_strain() -> None:
    rows = [
        {
            "pair_id": "a1",
            "bacteria": "A",
            "phage": "p2",
            "pred_lightgbm_isotonic": "0.60",
            "split_holdout": "holdout_test",
        },
        {
            "pair_id": "a2",
            "bacteria": "A",
            "phage": "p1",
            "pred_lightgbm_isotonic": "0.80",
            "split_holdout": "holdout_test",
        },
        {
            "pair_id": "a3",
            "bacteria": "A",
            "phage": "p3",
            "pred_lightgbm_isotonic": "0.50",
            "split_holdout": "holdout_test",
        },
        {
            "pair_id": "b1",
            "bacteria": "B",
            "phage": "p1",
            "pred_lightgbm_isotonic": "0.90",
            "split_holdout": "holdout_test",
        },
    ]

    selected = select_top_recommendations(rows, top_k=2)

    assert [row["phage"] for row in selected if row["bacteria"] == "A"] == ["p1", "p2"]
    assert [row["recommendation_rank"] for row in selected if row["bacteria"] == "A"] == ["1", "2"]
    assert [row["phage"] for row in selected if row["bacteria"] == "B"] == ["p1"]


def test_bootstrap_probability_intervals_return_point_estimate_and_ci() -> None:
    calibration_rows = [
        {"pred_lightgbm_raw": "0.10", "label_hard_any_lysis": "0"},
        {"pred_lightgbm_raw": "0.20", "label_hard_any_lysis": "0"},
        {"pred_lightgbm_raw": "0.80", "label_hard_any_lysis": "1"},
        {"pred_lightgbm_raw": "0.90", "label_hard_any_lysis": "1"},
        {"pred_lightgbm_raw": "0.70", "label_hard_any_lysis": "1"},
        {"pred_lightgbm_raw": "0.30", "label_hard_any_lysis": "0"},
    ]
    candidates = [{"pair_id": "x1", "pred_lightgbm_raw": "0.75"}]

    intervals = bootstrap_probability_intervals(
        calibration_rows,
        candidates,
        bootstrap_samples=64,
        random_state=7,
    )

    assert set(intervals) == {"x1"}
    assert (
        0.0 <= intervals["x1"]["ci_low"] <= intervals["x1"]["calibrated_p_lysis"] <= intervals["x1"]["ci_high"] <= 1.0
    )


def test_build_explained_recommendation_rows_merges_probability_and_shap_features() -> None:
    recommendation_rows = [
        {
            "pair_id": "a1",
            "bacteria": "A",
            "phage": "p1",
            "phage_family": "F1",
            "recommendation_rank": "1",
            "split_holdout": "holdout_test",
            "prediction_context": "holdout",
            "label_hard_any_lysis": "1",
        }
    ]
    shap_rows = [
        {
            "pair_id": "a1",
            "bacteria": "A",
            "phage": "p1",
            "recommendation_rank": "1",
            "top_positive_feature_1": "host_n_infections",
            "top_positive_shap_1": "1.2",
            "top_positive_feature_2": "phage_gc_content",
            "top_positive_shap_2": "0.8",
            "top_positive_feature_3": "defense_evasion_mean_score",
            "top_positive_shap_3": "0.4",
            "top_negative_feature_1": "host_lps_type=R1",
            "top_negative_shap_1": "-0.7",
            "top_negative_feature_2": "",
            "top_negative_shap_2": "",
            "top_negative_feature_3": "",
            "top_negative_shap_3": "",
        }
    ]
    probability_intervals = {
        "a1": {
            "calibrated_p_lysis": 0.81,
            "ci_low": 0.74,
            "ci_high": 0.88,
            "bootstrap_samples_used": 40.0,
        }
    }

    rows = build_explained_recommendation_rows(recommendation_rows, shap_rows, probability_intervals, top_k=3)

    assert rows[0]["calibrated_p_lysis"] == 0.81
    assert rows[0]["calibrated_p_lysis_ci_low"] == 0.74
    assert rows[0]["top_shap_feature_1"] == "host_n_infections"
    assert rows[0]["top_shap_feature_2"] == "phage_gc_content"
    assert rows[0]["top_shap_feature_3"] == "host_lps_type=R1"


def test_render_markdown_report_covers_all_holdout_strains() -> None:
    summary_rows = [
        {
            "bacteria": "A",
            "n_holdout_pairs": 3,
            "n_true_positive_phages_holdout": 1,
            "top_recommended_phage": "p1",
            "top_recommended_calibrated_p_lysis": 0.8,
            "top_recommended_ci_low": 0.7,
            "top_recommended_ci_high": 0.9,
            "top_shap_summary": "host_n_infections (+1.2000)",
            "top3_hit": 1,
        },
        {
            "bacteria": "B",
            "n_holdout_pairs": 3,
            "n_true_positive_phages_holdout": 0,
            "top_recommended_phage": "q1",
            "top_recommended_calibrated_p_lysis": 0.6,
            "top_recommended_ci_low": 0.5,
            "top_recommended_ci_high": 0.7,
            "top_shap_summary": "phage_gc_content (+0.9000)",
            "top3_hit": 0,
        },
    ]
    recommendation_rows = [
        {
            "bacteria": "A",
            "phage": "p1",
            "recommendation_rank": 1,
            "calibrated_p_lysis": 0.8,
            "calibrated_p_lysis_ci_low": 0.7,
            "calibrated_p_lysis_ci_high": 0.9,
            "top_shap_summary": "host_n_infections (+1.2000)",
        },
        {
            "bacteria": "B",
            "phage": "q1",
            "recommendation_rank": 1,
            "calibrated_p_lysis": 0.6,
            "calibrated_p_lysis_ci_low": 0.5,
            "calibrated_p_lysis_ci_high": 0.7,
            "top_shap_summary": "phage_gc_content (+0.9000)",
        },
    ]

    report = render_markdown_report(summary_rows, recommendation_rows)

    assert "### A" in report
    assert "### B" in report
    assert "95% CI" in report
    assert "Top SHAP features" in report


def test_main_writes_track_h_outputs(tmp_path: Path) -> None:
    tg02_path = tmp_path / "tg02.csv"
    tg04_path = tmp_path / "tg04.csv"
    output_dir = tmp_path / "out"

    _write_csv(
        tg02_path,
        fieldnames=[
            "pair_id",
            "bacteria",
            "phage",
            "phage_family",
            "split_holdout",
            "split_cv5_fold",
            "prediction_context",
            "is_strict_trainable",
            "label_hard_any_lysis",
            "pred_lightgbm_raw",
            "pred_lightgbm_isotonic",
            "pred_lightgbm_platt",
        ],
        rows=[
            {
                "pair_id": "a1",
                "bacteria": "A",
                "phage": "p1",
                "phage_family": "F1",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "1",
                "prediction_context": "holdout",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "1",
                "pred_lightgbm_raw": "0.91",
                "pred_lightgbm_isotonic": "0.88",
                "pred_lightgbm_platt": "0.86",
            },
            {
                "pair_id": "a2",
                "bacteria": "A",
                "phage": "p2",
                "phage_family": "F2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "1",
                "prediction_context": "holdout",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "0",
                "pred_lightgbm_raw": "0.50",
                "pred_lightgbm_isotonic": "0.40",
                "pred_lightgbm_platt": "0.42",
            },
            {
                "pair_id": "a3",
                "bacteria": "A",
                "phage": "p3",
                "phage_family": "F2",
                "split_holdout": "holdout_test",
                "split_cv5_fold": "1",
                "prediction_context": "holdout",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "0",
                "pred_lightgbm_raw": "0.30",
                "pred_lightgbm_isotonic": "0.20",
                "pred_lightgbm_platt": "0.23",
            },
            {
                "pair_id": "a4",
                "bacteria": "A",
                "phage": "p4",
                "phage_family": "F2",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "prediction_context": "train",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "1",
                "pred_lightgbm_raw": "0.90",
                "pred_lightgbm_isotonic": "0.89",
                "pred_lightgbm_platt": "0.87",
            },
            {
                "pair_id": "a5",
                "bacteria": "A",
                "phage": "p5",
                "phage_family": "F2",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "prediction_context": "train",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "0",
                "pred_lightgbm_raw": "0.10",
                "pred_lightgbm_isotonic": "0.11",
                "pred_lightgbm_platt": "0.12",
            },
            {
                "pair_id": "a6",
                "bacteria": "A",
                "phage": "p6",
                "phage_family": "F2",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "prediction_context": "train",
                "is_strict_trainable": "1",
                "label_hard_any_lysis": "1",
                "pred_lightgbm_raw": "0.80",
                "pred_lightgbm_isotonic": "0.81",
                "pred_lightgbm_platt": "0.79",
            },
        ],
    )

    _write_csv(
        tg04_path,
        fieldnames=[
            "pair_id",
            "bacteria",
            "phage",
            "recommendation_rank",
            "top_positive_feature_1",
            "top_positive_shap_1",
            "top_positive_feature_2",
            "top_positive_shap_2",
            "top_positive_feature_3",
            "top_positive_shap_3",
            "top_negative_feature_1",
            "top_negative_shap_1",
            "top_negative_feature_2",
            "top_negative_shap_2",
            "top_negative_feature_3",
            "top_negative_shap_3",
        ],
        rows=[
            {
                "pair_id": "a1",
                "bacteria": "A",
                "phage": "p1",
                "recommendation_rank": "1",
                "top_positive_feature_1": "host_n_infections",
                "top_positive_shap_1": "1.2",
                "top_positive_feature_2": "phage_gc_content",
                "top_positive_shap_2": "0.8",
                "top_positive_feature_3": "defense_evasion_mean_score",
                "top_positive_shap_3": "0.4",
                "top_negative_feature_1": "host_lps_type=R1",
                "top_negative_shap_1": "-0.7",
                "top_negative_feature_2": "",
                "top_negative_shap_2": "",
                "top_negative_feature_3": "",
                "top_negative_shap_3": "",
            },
            {
                "pair_id": "a2",
                "bacteria": "A",
                "phage": "p2",
                "recommendation_rank": "2",
                "top_positive_feature_1": "phage_gc_content",
                "top_positive_shap_1": "0.9",
                "top_positive_feature_2": "host_n_infections",
                "top_positive_shap_2": "0.4",
                "top_positive_feature_3": "",
                "top_positive_shap_3": "",
                "top_negative_feature_1": "host_lps_type=R2",
                "top_negative_shap_1": "-0.3",
                "top_negative_feature_2": "",
                "top_negative_shap_2": "",
                "top_negative_feature_3": "",
                "top_negative_shap_3": "",
            },
            {
                "pair_id": "a3",
                "bacteria": "A",
                "phage": "p3",
                "recommendation_rank": "3",
                "top_positive_feature_1": "defense_evasion_mean_score",
                "top_positive_shap_1": "0.6",
                "top_positive_feature_2": "",
                "top_positive_shap_2": "",
                "top_positive_feature_3": "",
                "top_positive_shap_3": "",
                "top_negative_feature_1": "host_n_infections",
                "top_negative_shap_1": "-0.2",
                "top_negative_feature_2": "",
                "top_negative_shap_2": "",
                "top_negative_feature_3": "",
                "top_negative_shap_3": "",
            },
        ],
    )

    main(
        [
            "--tg02-predictions-path",
            str(tg02_path),
            "--tg04-explanations-path",
            str(tg04_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "64",
            "--bootstrap-random-state",
            "7",
        ]
    )

    recommendations_path = output_dir / "th02_explained_recommendations.csv"
    summary_path = output_dir / "th02_holdout_strain_summary.csv"
    report_path = output_dir / "th02_explained_recommendations_report.md"
    manifest_path = output_dir / "th02_explained_recommendations_summary.json"

    assert recommendations_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert manifest_path.exists()

    with recommendations_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[0]["top_shap_feature_1"] == "host_n_infections"
    assert rows[0]["calibrated_p_lysis_ci_low"]
    assert rows[0]["calibrated_p_lysis_ci_high"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["holdout_strain_count"] == 1
    assert manifest["recommendation_row_count"] == 3
