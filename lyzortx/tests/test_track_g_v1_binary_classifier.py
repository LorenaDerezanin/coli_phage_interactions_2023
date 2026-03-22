import csv

from lyzortx.pipeline.track_g import run_track_g
from lyzortx.pipeline.track_g.steps import calibrate_gbm_outputs
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import (
    build_feature_space,
    compute_top3_hit_rate,
    merge_expanded_feature_rows,
    select_best_candidate,
)


def test_merge_expanded_feature_rows_adds_split_phage_and_pair_features() -> None:
    merged = merge_expanded_feature_rows(
        track_c_pair_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": "1",
                "host_surface_lps_core_type": "R1",
            }
        ],
        split_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "cv_group": "G1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            }
        ],
        phage_feature_blocks=[
            [{"phage": "P1", "phage_gc_content": "0.5"}],
            [{"phage": "P1", "phage_viridic_mds_00": "0.2"}],
        ],
        pair_feature_blocks=[
            [{"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "lookup_available": "1"}],
            [{"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "defense_evasion_score": "0.7"}],
            [{"pair_id": "B1__P1", "bacteria": "B1", "phage": "P1", "isolation_host_distance": "0.3"}],
        ],
    )

    assert merged[0]["cv_group"] == "G1"
    assert merged[0]["phage_gc_content"] == "0.5"
    assert merged[0]["lookup_available"] == "1"
    assert merged[0]["defense_evasion_score"] == "0.7"
    assert merged[0]["isolation_host_distance"] == "0.3"


def test_build_feature_space_keeps_v0_columns_and_adds_track_specific_blocks() -> None:
    feature_space = build_feature_space(
        st02_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "host_pathotype": "pt",
                "host_mouse_killed_10": "0",
            }
        ],
        track_c_pair_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "host_pathotype": "pt",
                "host_mouse_killed_10": "0",
                "host_surface_lps_core_type": "R1",
                "host_phylogeny_umap_00": "0.1",
            }
        ],
        track_d_feature_columns=["phage_gc_content", "phage_viridic_mds_00"],
        track_e_feature_columns=["lookup_available", "isolation_host_distance"],
    )

    assert "host_pathotype" in feature_space.categorical_columns
    assert "host_surface_lps_core_type" in feature_space.categorical_columns
    assert "host_phylogeny_umap_00" in feature_space.numeric_columns
    assert "phage_gc_content" in feature_space.numeric_columns
    assert "lookup_available" in feature_space.numeric_columns


def test_compute_top3_hit_rate_reports_all_and_susceptible_denominators() -> None:
    metrics = compute_top3_hit_rate(
        [
            {"bacteria": "B1", "phage": "P1", "label_hard_any_lysis": "1", "predicted_probability": 0.9},
            {"bacteria": "B1", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.8},
            {"bacteria": "B1", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.7},
            {"bacteria": "B2", "phage": "P1", "label_hard_any_lysis": "0", "predicted_probability": 0.9},
            {"bacteria": "B2", "phage": "P2", "label_hard_any_lysis": "0", "predicted_probability": 0.8},
            {"bacteria": "B2", "phage": "P3", "label_hard_any_lysis": "0", "predicted_probability": 0.7},
        ],
        probability_key="predicted_probability",
    )

    assert metrics["strain_count"] == 2
    assert metrics["hit_count"] == 1
    assert metrics["top3_hit_rate_all_strains"] == 0.5
    assert metrics["susceptible_strain_count"] == 1
    assert metrics["top3_hit_rate_susceptible_only"] == 1.0


def test_select_best_candidate_prefers_auc_then_top3_then_brier() -> None:
    best = select_best_candidate(
        [
            {
                "params": {"name": "a"},
                "summary": {
                    "mean_roc_auc": 0.81,
                    "mean_top3_hit_rate_all_strains": 0.80,
                    "mean_brier_score": 0.20,
                },
            },
            {
                "params": {"name": "b"},
                "summary": {
                    "mean_roc_auc": 0.81,
                    "mean_top3_hit_rate_all_strains": 0.85,
                    "mean_brier_score": 0.25,
                },
            },
        ]
    )

    assert best["params"]["name"] == "b"


def test_run_track_g_dispatches_training_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )

    run_track_g.main(["--step", "train-v1-binary"])
    assert calls == ["train-v1-binary"]

    calls.clear()
    run_track_g.main(["--step", "all"])
    assert calls == ["train-v1-binary", "calibrate-gbm"]


def test_run_track_g_dispatches_calibration_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("calibrate-gbm"),
    )

    run_track_g.main(["--step", "calibrate-gbm"])
    assert calls == ["calibrate-gbm"]


def test_tg02_calibration_outputs_expected_files_and_rows(tmp_path) -> None:
    predictions_path = tmp_path / "tg01_pair_predictions.csv"
    st02_path = tmp_path / "st02_pair_table.csv"
    st03_path = tmp_path / "st03_split_assignments.csv"
    output_dir = tmp_path / "tg02"

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "bacteria",
                "phage",
                "split_holdout",
                "split_cv5_fold",
                "label_hard_any_lysis",
                "prediction_context",
                "lightgbm_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "pair_id": "B1__P1",
                    "bacteria": "B1",
                    "phage": "P1",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.9",
                },
                {
                    "pair_id": "B1__P2",
                    "bacteria": "B1",
                    "phage": "P2",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.2",
                },
                {
                    "pair_id": "B2__P1",
                    "bacteria": "B2",
                    "phage": "P1",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.8",
                },
                {
                    "pair_id": "B2__P2",
                    "bacteria": "B2",
                    "phage": "P2",
                    "split_holdout": "train_non_holdout",
                    "split_cv5_fold": "0",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "non_holdout_oof",
                    "lightgbm_probability": "0.1",
                },
                {
                    "pair_id": "B3__P1",
                    "bacteria": "B3",
                    "phage": "P1",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.7",
                },
                {
                    "pair_id": "B3__P2",
                    "bacteria": "B3",
                    "phage": "P2",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.4",
                },
                {
                    "pair_id": "B4__P1",
                    "bacteria": "B4",
                    "phage": "P1",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "1",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.6",
                },
                {
                    "pair_id": "B4__P2",
                    "bacteria": "B4",
                    "phage": "P2",
                    "split_holdout": "holdout_test",
                    "split_cv5_fold": "-1",
                    "label_hard_any_lysis": "0",
                    "prediction_context": "holdout_final",
                    "lightgbm_probability": "0.3",
                },
            ]
        )

    with st02_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "phage_family", "label_strict_confidence_tier"],
        )
        writer.writeheader()
        for pair_id, phage_family, tier in [
            ("B1__P1", "fam1", "high"),
            ("B1__P2", "fam2", "high"),
            ("B2__P1", "fam1", "medium"),
            ("B2__P2", "fam2", "high"),
            ("B3__P1", "fam1", "high"),
            ("B3__P2", "fam2", "high"),
            ("B4__P1", "fam1", "medium"),
            ("B4__P2", "fam2", "high"),
        ]:
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "phage_family": phage_family,
                    "label_strict_confidence_tier": tier,
                }
            )

    with st03_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "is_strict_trainable"])
        writer.writeheader()
        for pair_id, flag in [
            ("B1__P1", "1"),
            ("B1__P2", "1"),
            ("B2__P1", "0"),
            ("B2__P2", "1"),
            ("B3__P1", "1"),
            ("B3__P2", "1"),
            ("B4__P1", "0"),
            ("B4__P2", "1"),
        ]:
            writer.writerow({"pair_id": pair_id, "is_strict_trainable": flag})

    exit_code = calibrate_gbm_outputs.main(
        [
            "--tg01-predictions-path",
            str(predictions_path),
            "--st02-pair-table-path",
            str(st02_path),
            "--st03-split-assignments-path",
            str(st03_path),
            "--output-dir",
            str(output_dir),
            "--skip-prerequisites",
        ]
    )

    assert exit_code == 0

    with (output_dir / "tg02_calibration_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with (output_dir / "tg02_pair_predictions_calibrated.csv").open("r", newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with (output_dir / "tg02_ranked_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        ranking_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 12
    assert {row["variant"] for row in summary_rows} == {"raw", "isotonic", "platt"}
    assert {row["label_slice"] for row in summary_rows} == {"full_label", "strict_confidence"}
    assert len(prediction_rows) == 8
    assert prediction_rows[0]["pred_lightgbm_isotonic"] != ""
    assert prediction_rows[0]["is_strict_trainable"] in {"0", "1"}
    assert len(ranking_rows) == 8
    assert ranking_rows[0]["rank_lightgbm_isotonic"] == "1"
