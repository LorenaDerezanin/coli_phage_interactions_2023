#!/usr/bin/env python3
"""ST0.6: Generate top-k phage recommendations with configurable ranking policy."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st05-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st05_pair_predictions_calibrated.csv"),
        help="Input ST0.5 calibrated pair predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate"),
        help="Output directory for ST0.6 artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of phages to recommend per strain.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="pred_logreg_platt",
        help="Column used for ranking phages within each strain.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=0,
        help="Maximum recommendations from one phage family. Set to 0 to disable diversity cap.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for holdout top-k CI estimates.",
    )
    parser.add_argument(
        "--bootstrap-random-state",
        type=int,
        default=42,
        help="Random seed for bootstrap CI sampling.",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def safe_round(value: float) -> float:
    return round(float(value), 6)


def _slice_available_rows(rows: List[Dict[str, str]], slice_name: str) -> List[Dict[str, str]]:
    if slice_name == "full_label":
        return [row for row in rows if row["label_hard_binary"] != ""]
    if slice_name == "strict_confidence":
        return [row for row in rows if row["is_strict_trainable"] == "1" and row["label_hard_binary"] != ""]
    raise ValueError(f"Unknown slice: {slice_name}")


def _slice_recommendations(rows: List[Dict[str, object]], slice_name: str) -> List[Dict[str, object]]:
    if slice_name == "full_label":
        return rows
    if slice_name == "strict_confidence":
        return [row for row in rows if str(row["label_strict_confidence_tier"]) in {"high_conf_pos", "high_conf_neg"}]
    raise ValueError(f"Unknown slice: {slice_name}")


def evaluate_holdout_slice(
    holdout_by_bacteria: Dict[str, List[Dict[str, str]]],
    recs_by_bacteria: Dict[str, List[Dict[str, object]]],
    slice_name: str,
) -> Dict[str, float]:
    holdout_total = 0
    holdout_hits = 0
    susceptible_total = 0
    susceptible_hits = 0

    for bacteria in sorted(holdout_by_bacteria.keys()):
        available = _slice_available_rows(holdout_by_bacteria[bacteria], slice_name=slice_name)
        if not available:
            continue

        holdout_total += 1
        recs = _slice_recommendations(recs_by_bacteria.get(bacteria, []), slice_name=slice_name)
        rec_hit = any(str(row["label_hard_binary"]) == "1" for row in recs)
        holdout_hits += 1 if rec_hit else 0

        susceptible = any(row["label_hard_binary"] == "1" for row in available)
        if susceptible:
            susceptible_total += 1
            susceptible_hits += 1 if rec_hit else 0

    return {
        "holdout_strain_count": holdout_total,
        "holdout_hit_count": holdout_hits,
        "topk_hit_rate_all_strains": safe_round(holdout_hits / holdout_total if holdout_total else 0.0),
        "holdout_susceptible_strain_count": susceptible_total,
        "holdout_susceptible_hit_count": susceptible_hits,
        "topk_hit_rate_susceptible_only": safe_round(
            susceptible_hits / susceptible_total if susceptible_total else 0.0
        ),
    }


def bootstrap_topk_ci(
    holdout_by_bacteria: Dict[str, List[Dict[str, str]]],
    recs_by_bacteria: Dict[str, List[Dict[str, object]]],
    slice_name: str,
    bootstrap_samples: int,
    bootstrap_random_state: int,
) -> Dict[str, float]:
    eligible_strains = [
        bacteria
        for bacteria, rows in sorted(holdout_by_bacteria.items())
        if _slice_available_rows(rows, slice_name=slice_name)
    ]
    if not eligible_strains:
        return {
            "bootstrap_samples": bootstrap_samples,
            "ci_low_topk_hit_rate_all_strains": 0.0,
            "ci_high_topk_hit_rate_all_strains": 0.0,
            "ci_low_topk_hit_rate_susceptible_only": 0.0,
            "ci_high_topk_hit_rate_susceptible_only": 0.0,
        }

    rng = np.random.default_rng(bootstrap_random_state)
    all_strain_rates = []
    susceptible_rates = []
    for _ in range(bootstrap_samples):
        sampled_ids = rng.choice(eligible_strains, size=len(eligible_strains), replace=True)
        sampled_holdout = {f"sample_{i}": holdout_by_bacteria[b] for i, b in enumerate(sampled_ids)}
        sampled_recs = {f"sample_{i}": recs_by_bacteria.get(b, []) for i, b in enumerate(sampled_ids)}
        metrics = evaluate_holdout_slice(sampled_holdout, sampled_recs, slice_name=slice_name)
        all_strain_rates.append(float(metrics["topk_hit_rate_all_strains"]))
        susceptible_rates.append(float(metrics["topk_hit_rate_susceptible_only"]))

    all_low, all_high = np.quantile(np.asarray(all_strain_rates, dtype=float), [0.025, 0.975])
    sus_low, sus_high = np.quantile(np.asarray(susceptible_rates, dtype=float), [0.025, 0.975])
    return {
        "bootstrap_samples": bootstrap_samples,
        "ci_low_topk_hit_rate_all_strains": safe_round(float(all_low)),
        "ci_high_topk_hit_rate_all_strains": safe_round(float(all_high)),
        "ci_low_topk_hit_rate_susceptible_only": safe_round(float(sus_low)),
        "ci_high_topk_hit_rate_susceptible_only": safe_round(float(sus_high)),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")
    if args.max_per_family < 0:
        raise ValueError("max-per-family must be >= 0")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be >= 1")

    prediction_rows = read_csv_rows(args.st05_predictions_path)
    if not prediction_rows:
        raise ValueError("ST0.5 calibrated predictions input is empty.")
    if args.score_column not in prediction_rows[0]:
        raise ValueError(f"Configured score column not found: {args.score_column}")
    required_columns = [
        "bacteria",
        "phage",
        "phage_family",
        "split_holdout",
        "label_hard_binary",
        "label_strict_confidence_tier",
    ]
    missing = [col for col in required_columns if col not in prediction_rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in ST0.5 predictions: {', '.join(missing)}")

    by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        by_bacteria[row["bacteria"]].append(row)

    recommendation_rows: List[Dict[str, object]] = []
    relaxed_strain_count = 0

    for bacteria in sorted(by_bacteria.keys()):
        rows = sorted(
            by_bacteria[bacteria],
            key=lambda r: (-float(r[args.score_column]), r["phage"]),
        )
        if args.max_per_family == 0:
            selected = rows[: args.top_k]
            relaxed_used = False
        else:
            family_counts = Counter()
            selected = []
            skipped_due_to_diversity: List[Dict[str, str]] = []
            for row in rows:
                if len(selected) >= args.top_k:
                    break
                family = row["phage_family"] or "missing_family"
                if family_counts[family] < args.max_per_family:
                    selected.append(row)
                    family_counts[family] += 1
                else:
                    skipped_due_to_diversity.append(row)

            relaxed_used = False
            if len(selected) < args.top_k:
                relaxed_used = True
                for row in skipped_due_to_diversity:
                    if len(selected) >= args.top_k:
                        break
                    selected.append(row)

            if len(selected) < args.top_k:
                for row in rows:
                    if len(selected) >= args.top_k:
                        break
                    if row in selected:
                        continue
                    selected.append(row)

            if relaxed_used:
                relaxed_strain_count += 1

        for rank, row in enumerate(selected[: args.top_k], start=1):
            recommendation_rows.append(
                {
                    "bacteria": bacteria,
                    "recommendation_rank": rank,
                    "phage": row["phage"],
                    "phage_family": row["phage_family"],
                    "score_column": args.score_column,
                    "score_value": row[args.score_column],
                    "score_logreg_raw": row.get("pred_logreg_raw", ""),
                    "score_logreg_platt": row.get("pred_logreg_platt", ""),
                    "score_logreg_isotonic": row.get("pred_logreg_isotonic", ""),
                    "split_holdout": row["split_holdout"],
                    "label_hard_binary": row["label_hard_binary"],
                    "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                    "diversity_relaxed_for_strain": 1 if relaxed_used else 0,
                }
            )

    recommendation_rows.sort(key=lambda r: (str(r["bacteria"]), int(r["recommendation_rank"])))
    output_recs = args.output_dir / "st06_top3_recommendations.csv"
    write_csv(output_recs, fieldnames=list(recommendation_rows[0].keys()), rows=recommendation_rows)

    holdout_rows = [row for row in prediction_rows if row["split_holdout"] == "holdout_test"]
    holdout_by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in holdout_rows:
        holdout_by_bacteria[row["bacteria"]].append(row)

    recs_by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
        recs_by_bacteria[str(row["bacteria"])].append(row)

    holdout_topk_metrics_by_slice: Dict[str, Dict[str, float]] = {}
    bootstrap_ci_by_slice: Dict[str, Dict[str, float]] = {}
    for slice_name in ("full_label", "strict_confidence"):
        holdout_topk_metrics_by_slice[slice_name] = evaluate_holdout_slice(
            holdout_by_bacteria=holdout_by_bacteria,
            recs_by_bacteria=recs_by_bacteria,
            slice_name=slice_name,
        )
        bootstrap_ci_by_slice[slice_name] = bootstrap_topk_ci(
            holdout_by_bacteria=holdout_by_bacteria,
            recs_by_bacteria=recs_by_bacteria,
            slice_name=slice_name,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_random_state=args.bootstrap_random_state,
        )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st06_recommend_top3",
        "input_predictions_path": str(args.st05_predictions_path),
        "parameters": {
            "top_k": args.top_k,
            "score_column": args.score_column,
            "max_per_family": args.max_per_family,
            "diversity_mode": "none" if args.max_per_family == 0 else f"max_family_{args.max_per_family}",
        },
        "recommendation_summary": {
            "recommended_strain_count": len({row["bacteria"] for row in recommendation_rows}),
            "recommended_row_count": len(recommendation_rows),
            "diversity_relaxed_strain_count": relaxed_strain_count,
        },
        "holdout_topk_metrics": holdout_topk_metrics_by_slice,
        "holdout_topk_bootstrap_ci": bootstrap_ci_by_slice,
    }

    write_json(args.output_dir / "st06_recommendation_summary.json", summary)

    print("ST0.6 completed.")
    print(f"- Recommended strains: {summary['recommendation_summary']['recommended_strain_count']}")
    print(f"- Recommendation rows: {summary['recommendation_summary']['recommended_row_count']}")
    print(
        f"- Holdout top-{args.top_k} hit rate: {summary['holdout_topk_metrics']['full_label']['topk_hit_rate_all_strains']}"
    )
    print(f"- Output recommendations: {output_recs}")


if __name__ == "__main__":
    main()
