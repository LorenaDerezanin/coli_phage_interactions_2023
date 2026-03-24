#!/usr/bin/env python3
"""TH02: Build clinician-ready explained recommendations for holdout strains."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round

logger = logging.getLogger(__name__)

TG02_REQUIRED_COLUMNS: Tuple[str, ...] = (
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
)

TG04_REQUIRED_COLUMNS: Tuple[str, ...] = (
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
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg02-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_g/tg02_gbm_calibration/tg02_pair_predictions_calibrated.csv"),
        help="TG02 calibrated pair prediction CSV.",
    )
    parser.add_argument(
        "--tg04-explanations-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_g/tg04_shap_explanations/tg04_recommendation_pair_explanations.csv"
        ),
        help="TG04 recommendation explanation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_h/th02_explained_recommendations"),
        help="Directory for Track H artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of recommendations to surface per holdout strain.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=0,
        help="TG02 calibration fold used to fit bootstrap calibrators.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI estimation.",
    )
    parser.add_argument(
        "--bootstrap-random-state",
        type=int,
        default=42,
        help="Random seed for bootstrap CI estimation.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_rows_by_bacteria(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bacteria"])].append(dict(row))
    return grouped


def select_top_recommendations(
    rows: Sequence[Mapping[str, str]],
    *,
    top_k: int,
    score_key: str = "pred_lightgbm_isotonic",
) -> List[Dict[str, str]]:
    grouped = _group_rows_by_bacteria(rows)
    selected: List[Dict[str, str]] = []
    for bacteria in sorted(grouped):
        ranked = sorted(grouped[bacteria], key=lambda row: (-float(row[score_key]), str(row["phage"])))
        for rank, row in enumerate(ranked[:top_k], start=1):
            enriched = dict(row)
            enriched["recommendation_rank"] = str(rank)
            selected.append(enriched)
    return selected


def _fit_isotonic_calibrator(
    calibration_rows: Sequence[Mapping[str, str]],
    *,
    raw_score_key: str,
    label_key: str,
) -> IsotonicRegression:
    x_calib = np.asarray([float(row[raw_score_key]) for row in calibration_rows], dtype=float)
    y_calib = np.asarray([int(row[label_key]) for row in calibration_rows], dtype=int)
    if len(x_calib) == 0:
        raise ValueError("No calibration rows available for Track H.")
    if len(np.unique(y_calib)) < 2:
        raise ValueError("Calibration rows must contain both classes for bootstrap CI estimation.")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(x_calib, y_calib)
    return calibrator


def bootstrap_probability_intervals(
    calibration_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    *,
    raw_score_key: str = "pred_lightgbm_raw",
    label_key: str = "label_hard_any_lysis",
    bootstrap_samples: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    if not candidate_rows:
        return {}
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1.")

    point_estimator = _fit_isotonic_calibrator(calibration_rows, raw_score_key=raw_score_key, label_key=label_key)
    candidate_raw = np.asarray([float(row[raw_score_key]) for row in candidate_rows], dtype=float)
    point_estimates = np.asarray(point_estimator.predict(candidate_raw), dtype=float)

    x_calib = np.asarray([float(row[raw_score_key]) for row in calibration_rows], dtype=float)
    y_calib = np.asarray([int(row[label_key]) for row in calibration_rows], dtype=int)
    rng = np.random.default_rng(random_state)
    bootstrap_predictions: List[np.ndarray] = []
    for _ in range(bootstrap_samples):
        sample_indices = rng.integers(0, len(calibration_rows), size=len(calibration_rows))
        y_sample = y_calib[sample_indices]
        if len(np.unique(y_sample)) < 2:
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(x_calib[sample_indices], y_sample)
        bootstrap_predictions.append(np.asarray(calibrator.predict(candidate_raw), dtype=float))

    if not bootstrap_predictions:
        raise ValueError("Bootstrap CI estimation failed because no valid resamples contained both classes.")

    samples = np.vstack(bootstrap_predictions)
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    intervals: Dict[str, Dict[str, float]] = {}
    for index, row in enumerate(candidate_rows):
        intervals[str(row["pair_id"])] = {
            "calibrated_p_lysis": safe_round(float(point_estimates[index])),
            "ci_low": safe_round(float(low[index])),
            "ci_high": safe_round(float(high[index])),
            "bootstrap_samples_used": float(samples.shape[0]),
        }
    return intervals


def _shap_candidates_from_row(row: Mapping[str, str]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for prefix in ("top_positive", "top_negative"):
        for position in range(1, 4):
            feature_key = f"{prefix}_feature_{position}"
            shap_key = f"{prefix}_shap_{position}"
            feature_name = str(row.get(feature_key, ""))
            shap_raw = str(row.get(shap_key, ""))
            if not feature_name or shap_raw == "":
                continue
            candidates.append(
                {
                    "feature_name": feature_name,
                    "shap_value": float(shap_raw),
                }
            )
    deduped: Dict[str, Dict[str, object]] = {}
    for candidate in candidates:
        name = str(candidate["feature_name"])
        previous = deduped.get(name)
        if previous is None or abs(float(candidate["shap_value"])) > abs(float(previous["shap_value"])):
            deduped[name] = candidate
    return sorted(
        deduped.values(),
        key=lambda item: (-abs(float(item["shap_value"])), str(item["feature_name"])),
    )[:3]


def format_shap_summary(features: Sequence[Mapping[str, object]]) -> str:
    if not features:
        return ""
    return "; ".join(f"{item['feature_name']} ({float(item['shap_value']):+.4f})" for item in features)


def build_explained_recommendation_rows(
    recommendation_rows: Sequence[Mapping[str, str]],
    shap_rows: Sequence[Mapping[str, str]],
    probability_intervals: Mapping[str, Mapping[str, float]],
    *,
    top_k: int,
) -> List[Dict[str, object]]:
    shap_by_pair = {str(row["pair_id"]): dict(row) for row in shap_rows}
    output_rows: List[Dict[str, object]] = []
    for row in recommendation_rows:
        pair_id = str(row["pair_id"])
        shap_row = shap_by_pair.get(pair_id)
        if shap_row is None:
            raise KeyError(f"Missing TG04 SHAP row for pair_id {pair_id}")
        interval = probability_intervals[pair_id]
        top_features = _shap_candidates_from_row(shap_row)
        explained_row: Dict[str, object] = {
            "pair_id": pair_id,
            "bacteria": row["bacteria"],
            "phage": row["phage"],
            "phage_family": row["phage_family"],
            "recommendation_rank": int(row["recommendation_rank"]),
            "split_holdout": row["split_holdout"],
            "prediction_context": row["prediction_context"],
            "label_hard_any_lysis": row["label_hard_any_lysis"],
            "calibrated_p_lysis": interval["calibrated_p_lysis"],
            "calibrated_p_lysis_ci_low": interval["ci_low"],
            "calibrated_p_lysis_ci_high": interval["ci_high"],
            "bootstrap_samples_used": int(interval["bootstrap_samples_used"]),
            "top_shap_summary": format_shap_summary(top_features),
        }
        for position in range(1, top_k + 1):
            feature = top_features[position - 1] if position - 1 < len(top_features) else {}
            explained_row[f"top_shap_feature_{position}"] = feature.get("feature_name", "")
            explained_row[f"top_shap_value_{position}"] = safe_round(float(feature["shap_value"])) if feature else ""
        output_rows.append(explained_row)

    output_rows.sort(key=lambda row: (str(row["bacteria"]), int(row["recommendation_rank"]), str(row["phage"])))
    return output_rows


def build_strain_summary_rows(
    recommendation_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, str]],
    *,
    top_k: int,
) -> List[Dict[str, object]]:
    recommendations_by_bacteria: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
        recommendations_by_bacteria[str(row["bacteria"])].append(row)

    candidates_by_bacteria: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_bacteria[str(row["bacteria"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for bacteria in sorted(candidates_by_bacteria):
        ranked_candidates = sorted(
            candidates_by_bacteria[bacteria],
            key=lambda row: (-float(row["pred_lightgbm_isotonic"]), str(row["phage"])),
        )
        top_recommendations = sorted(
            recommendations_by_bacteria.get(bacteria, []),
            key=lambda row: int(row["recommendation_rank"]),
        )
        top3_hit = any(int(row["label_hard_any_lysis"]) == 1 for row in top_recommendations[:top_k])
        positive_count = sum(int(row["label_hard_any_lysis"]) for row in ranked_candidates)
        top_row = top_recommendations[0]
        summary_rows.append(
            {
                "bacteria": bacteria,
                "n_holdout_pairs": len(ranked_candidates),
                "n_true_positive_phages_holdout": positive_count,
                "top_recommended_phage": top_row["phage"],
                "top_recommended_calibrated_p_lysis": top_row["calibrated_p_lysis"],
                "top_recommended_ci_low": top_row["calibrated_p_lysis_ci_low"],
                "top_recommended_ci_high": top_row["calibrated_p_lysis_ci_high"],
                "top_shap_summary": top_row["top_shap_summary"],
                "top3_hit": int(top3_hit),
            }
        )
    return summary_rows


def render_markdown_report(
    summary_rows: Sequence[Mapping[str, object]],
    recommendation_rows: Sequence[Mapping[str, object]],
) -> str:
    recommendations_by_bacteria: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
        recommendations_by_bacteria[str(row["bacteria"])].append(row)

    lines: List[str] = []
    lines.append("# Track H TH02 Explained Recommendations")
    lines.append("")
    lines.append("This report summarizes the top-3 holdout phage recommendations for each strain using TG02 calibrated")
    lines.append("P(lysis) values, bootstrap 95% confidence intervals, and TG04 SHAP drivers.")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Holdout strains covered: {len(summary_rows)}")
    lines.append(f"- Recommendation rows: {len(recommendation_rows)}")
    lines.append("")
    lines.append("| Strain | Top phage | P(lysis) | 95% CI | Top-3 hit | Top SHAP features |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        lines.append(
            "| {bacteria} | {phage} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {hit} | {shap} |".format(
                bacteria=row["bacteria"],
                phage=row["top_recommended_phage"],
                p=float(row["top_recommended_calibrated_p_lysis"]),
                lo=float(row["top_recommended_ci_low"]),
                hi=float(row["top_recommended_ci_high"]),
                hit="yes" if int(row["top3_hit"]) else "no",
                shap=row["top_shap_summary"],
            )
        )
    lines.append("")
    lines.append("## Per-Strain Detail")
    lines.append("")
    for row in summary_rows:
        bacteria = str(row["bacteria"])
        lines.append(f"### {bacteria}")
        lines.append(
            f"Top recommendation: {row['top_recommended_phage']} at P(lysis) "
            f"{float(row['top_recommended_calibrated_p_lysis']):.3f} "
            f"(95% CI {float(row['top_recommended_ci_low']):.3f}-{float(row['top_recommended_ci_high']):.3f})."
        )
        lines.append(
            f"Holdout positives: {int(row['n_true_positive_phages_holdout'])}; top-3 hit: "
            f"{'yes' if int(row['top3_hit']) else 'no'}."
        )
        lines.append("")
        lines.append("| Rank | Phage | P(lysis) | 95% CI | Top SHAP features |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rec in sorted(recommendations_by_bacteria[bacteria], key=lambda item: int(item["recommendation_rank"])):
            lines.append(
                "| {rank} | {phage} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {shap} |".format(
                    rank=int(rec["recommendation_rank"]),
                    phage=rec["phage"],
                    p=float(rec["calibrated_p_lysis"]),
                    lo=float(rec["calibrated_p_lysis_ci_low"]),
                    hi=float(rec["calibrated_p_lysis_ci_high"]),
                    shap=rec["top_shap_summary"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _required_columns(rows: Sequence[Mapping[str, str]], required: Iterable[str], *, source: str) -> None:
    if not rows:
        raise ValueError(f"{source} input is empty.")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {', '.join(missing)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("TH02 starting: build explained recommendations")
    ensure_directory(args.output_dir)

    tg02_rows = read_csv_rows(args.tg02_predictions_path, required_columns=TG02_REQUIRED_COLUMNS)
    tg04_rows = read_csv_rows(args.tg04_explanations_path, required_columns=TG04_REQUIRED_COLUMNS)
    _required_columns(tg02_rows, TG02_REQUIRED_COLUMNS, source="TG02")
    _required_columns(tg04_rows, TG04_REQUIRED_COLUMNS, source="TG04")

    holdout_rows = [row for row in tg02_rows if row["split_holdout"] == "holdout_test"]
    calibration_rows = [
        row
        for row in tg02_rows
        if row["split_holdout"] == "train_non_holdout"
        and row["split_cv5_fold"] == str(args.calibration_fold)
        and row["label_hard_any_lysis"] != ""
    ]
    if not holdout_rows:
        raise ValueError("No holdout rows found in TG02 predictions.")
    if not calibration_rows:
        raise ValueError("No calibration rows found for TH02.")

    recommendation_source_rows = select_top_recommendations(holdout_rows, top_k=args.top_k)
    probability_intervals = bootstrap_probability_intervals(
        calibration_rows,
        recommendation_source_rows,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.bootstrap_random_state,
    )
    explained_rows = build_explained_recommendation_rows(
        recommendation_source_rows,
        tg04_rows,
        probability_intervals,
        top_k=args.top_k,
    )
    summary_rows = build_strain_summary_rows(explained_rows, holdout_rows, top_k=args.top_k)
    report_text = render_markdown_report(summary_rows, explained_rows)

    output_recommendations = args.output_dir / "th02_explained_recommendations.csv"
    output_summary = args.output_dir / "th02_holdout_strain_summary.csv"
    output_report = args.output_dir / "th02_explained_recommendations_report.md"
    output_manifest = args.output_dir / "th02_explained_recommendations_summary.json"

    write_csv(output_recommendations, fieldnames=list(explained_rows[0].keys()), rows=explained_rows)
    write_csv(output_summary, fieldnames=list(summary_rows[0].keys()), rows=summary_rows)
    output_report.write_text(report_text, encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TH02",
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "holdout_strain_count": len(summary_rows),
        "recommendation_row_count": len(explained_rows),
        "outputs": {
            "recommendations_csv": str(output_recommendations),
            "strain_summary_csv": str(output_summary),
            "report_md": str(output_report),
        },
        "inputs": {
            "tg02_predictions": {
                "path": str(args.tg02_predictions_path),
                "sha256": sha256(args.tg02_predictions_path),
            },
            "tg04_explanations": {
                "path": str(args.tg04_explanations_path),
                "sha256": sha256(args.tg04_explanations_path),
            },
        },
    }
    write_json(output_manifest, summary)

    logger.info("TH02 completed.")
    logger.info("- Holdout strains covered: %d", len(summary_rows))
    logger.info("- Recommendation rows: %d", len(explained_rows))
    logger.info("- Output report: %s", output_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
