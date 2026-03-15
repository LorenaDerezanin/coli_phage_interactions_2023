import csv
import json
from pathlib import Path

from lyzortx.pipeline.steel_thread_v0.steps.st07_build_report import main


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_st07_metrics_summary_includes_dual_slice_rows(tmp_path: Path) -> None:
    st04_path = tmp_path / "st04.json"
    st05_summary_path = tmp_path / "st05_summary.csv"
    st05_ranked_path = tmp_path / "st05_ranked.csv"
    st06_top3_path = tmp_path / "st06_top3.csv"
    st06_summary_path = tmp_path / "st06_summary.json"
    output_dir = tmp_path / "out"

    st04_path.write_text(
        json.dumps(
            {
                "models": {
                    "logreg": {
                        "holdout_binary_metrics": {"roc_auc": 0.7},
                        "holdout_top3_metrics": {"topk_hit_rate_all_strains": 0.5},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        st05_summary_path,
        fieldnames=["dataset", "model", "variant", "label_slice", "brier_score", "log_loss", "ece"],
        rows=[
            {
                "dataset": "holdout",
                "model": "logreg",
                "variant": "isotonic",
                "label_slice": "full_label",
                "brier_score": "0.11",
                "log_loss": "0.21",
                "ece": "0.01",
            },
            {
                "dataset": "holdout",
                "model": "logreg",
                "variant": "isotonic",
                "label_slice": "strict_confidence",
                "brier_score": "0.07",
                "log_loss": "0.14",
                "ece": "0.02",
            },
        ],
    )

    _write_csv(
        st05_ranked_path,
        fieldnames=[
            "split_holdout",
            "bacteria",
            "phage",
            "label_hard_binary",
            "score_logreg_isotonic",
            "rank_logreg_isotonic",
        ],
        rows=[
            {
                "split_holdout": "holdout_test",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_binary": "1",
                "score_logreg_isotonic": "0.8",
                "rank_logreg_isotonic": "1",
            }
        ],
    )

    _write_csv(
        st06_top3_path,
        fieldnames=["split_holdout", "bacteria", "phage", "recommendation_rank", "label_hard_binary"],
        rows=[
            {
                "split_holdout": "holdout_test",
                "bacteria": "b1",
                "phage": "p1",
                "recommendation_rank": "1",
                "label_hard_binary": "1",
            }
        ],
    )

    st06_summary_path.write_text(
        json.dumps(
            {
                "holdout_topk_metrics": {
                    "full_label": {"topk_hit_rate_all_strains": 0.61},
                    "strict_confidence": {"topk_hit_rate_all_strains": 0.72},
                },
                "holdout_topk_bootstrap_ci": {},
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "--st04-metrics-path",
            str(st04_path),
            "--st05-calibration-summary-path",
            str(st05_summary_path),
            "--st05-ranked-predictions-path",
            str(st05_ranked_path),
            "--st06-top3-path",
            str(st06_top3_path),
            "--st06-summary-path",
            str(st06_summary_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "metrics_summary.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metrics = {row["metric_name"]: row["value"] for row in rows if row["split"] == "holdout"}

    assert metrics["topk_hit_rate_all_strains__full_label"] == "0.61"
    assert metrics["topk_hit_rate_all_strains__strict_confidence"] == "0.72"
    assert metrics["brier_score__full_label"] == "0.11"
    assert metrics["brier_score__strict_confidence"] == "0.07"
    assert metrics["ece__full_label"] == "0.01"
    assert metrics["ece__strict_confidence"] == "0.02"
