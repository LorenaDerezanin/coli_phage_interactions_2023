#!/usr/bin/env python3
"""ST0.6b: Compare top-k recommendation policies side-by-side."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round

SCORE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("logreg_raw", "pred_logreg_raw"),
    ("logreg_platt", "pred_logreg_platt"),
    ("logreg_isotonic", "pred_logreg_isotonic"),
)


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
        help="Output directory for ST0.6b artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of phages to recommend per strain.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=2,
        help="Maximum recommendations from one phage family in family-capped policies.",
    )
    return parser.parse_args(argv)


def select_topk_with_policy(
    rows: List[Dict[str, str]],
    score_column: str,
    top_k: int,
    max_per_family: Optional[int],
) -> Tuple[List[Dict[str, str]], bool]:
    ordered = sorted(
        rows,
        key=lambda r: (-float(r[score_column]), r["phage"]),
    )

    if max_per_family is None:
        return ordered[:top_k], False

    family_counts: Counter[str] = Counter()
    selected: List[Dict[str, str]] = []
    skipped_due_to_diversity: List[Dict[str, str]] = []

    for row in ordered:
        if len(selected) >= top_k:
            break
        family = row["phage_family"] or "missing_family"
        if family_counts[family] < max_per_family:
            selected.append(row)
            family_counts[family] += 1
        else:
            skipped_due_to_diversity.append(row)

    relaxed_used = False
    if len(selected) < top_k:
        relaxed_used = True
        for row in skipped_due_to_diversity:
            if len(selected) >= top_k:
                break
            selected.append(row)

    if len(selected) < top_k:
        for row in ordered:
            if len(selected) >= top_k:
                break
            if row in selected:
                continue
            selected.append(row)

    return selected[:top_k], relaxed_used


def build_recommendations(
    by_bacteria: Dict[str, List[Dict[str, str]]],
    score_variant: str,
    score_column: str,
    diversity_mode: str,
    top_k: int,
    max_per_family: Optional[int],
) -> Tuple[List[Dict[str, object]], int]:
    policy_id = f"{score_variant}__{diversity_mode}"
    recommendation_rows: List[Dict[str, object]] = []
    relaxed_strain_count = 0

    for bacteria in sorted(by_bacteria.keys()):
        selected, relaxed_used = select_topk_with_policy(
            rows=by_bacteria[bacteria],
            score_column=score_column,
            top_k=top_k,
            max_per_family=max_per_family,
        )
        if relaxed_used:
            relaxed_strain_count += 1

        for rank, row in enumerate(selected, start=1):
            recommendation_rows.append(
                {
                    "policy_id": policy_id,
                    "score_variant": score_variant,
                    "score_column": score_column,
                    "diversity_mode": diversity_mode,
                    "bacteria": bacteria,
                    "recommendation_rank": rank,
                    "phage": row["phage"],
                    "phage_family": row["phage_family"],
                    "score_value": row[score_column],
                    "split_holdout": row["split_holdout"],
                    "label_hard_binary": row["label_hard_binary"],
                    "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                    "diversity_relaxed_for_strain": 1 if relaxed_used else 0,
                }
            )

    recommendation_rows.sort(
        key=lambda r: (
            str(r["policy_id"]),
            str(r["bacteria"]),
            int(r["recommendation_rank"]),
        )
    )
    return recommendation_rows, relaxed_strain_count


def evaluate_holdout(
    recommendation_rows: List[Dict[str, object]],
    holdout_by_bacteria: Dict[str, List[Dict[str, str]]],
) -> Dict[str, object]:
    recs_by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
        if str(row["split_holdout"]) != "holdout_test":
            continue
        recs_by_bacteria[str(row["bacteria"])].append(row)

    holdout_total = 0
    holdout_hits = 0
    susceptible_total = 0
    susceptible_hits = 0

    for bacteria in sorted(holdout_by_bacteria.keys()):
        available = holdout_by_bacteria[bacteria]
        recs = recs_by_bacteria.get(bacteria, [])
        if not recs:
            continue

        holdout_total += 1
        rec_hit = any(str(row["label_hard_binary"]) == "1" for row in recs)
        holdout_hits += 1 if rec_hit else 0

        susceptible = any(row["label_hard_binary"] == "1" for row in available if row["label_hard_binary"] != "")
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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")
    if args.max_per_family < 1:
        raise ValueError("max-per-family must be >= 1")

    st05_rows = read_csv_rows(args.st05_predictions_path)
    if not st05_rows:
        raise ValueError("ST0.5 calibrated predictions input is empty.")

    for _, column in SCORE_COLUMNS:
        if column not in st05_rows[0]:
            raise ValueError(f"Missing required score column in ST0.5 predictions: {column}")

    by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in st05_rows:
        by_bacteria[row["bacteria"]].append(row)

    holdout_by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in st05_rows:
        if row["split_holdout"] == "holdout_test":
            holdout_by_bacteria[row["bacteria"]].append(row)

    policy_comparison_rows: List[Dict[str, object]] = []
    all_recommendation_rows: List[Dict[str, object]] = []

    for score_variant, score_column in SCORE_COLUMNS:
        for diversity_mode, max_per_family in [
            ("none", None),
            (f"max_family_{args.max_per_family}", args.max_per_family),
        ]:
            recommendation_rows, relaxed_strain_count = build_recommendations(
                by_bacteria=by_bacteria,
                score_variant=score_variant,
                score_column=score_column,
                diversity_mode=diversity_mode,
                top_k=args.top_k,
                max_per_family=max_per_family,
            )
            all_recommendation_rows.extend(recommendation_rows)

            metrics = evaluate_holdout(
                recommendation_rows=recommendation_rows,
                holdout_by_bacteria=holdout_by_bacteria,
            )
            policy_comparison_rows.append(
                {
                    "policy_id": f"{score_variant}__{diversity_mode}",
                    "score_variant": score_variant,
                    "score_column": score_column,
                    "diversity_mode": diversity_mode,
                    "diversity_relaxed_strain_count": relaxed_strain_count,
                    **metrics,
                }
            )

    policy_comparison_rows.sort(
        key=lambda r: (
            -float(r["topk_hit_rate_all_strains"]),
            -float(r["topk_hit_rate_susceptible_only"]),
            int(r["diversity_relaxed_strain_count"]),
            str(r["policy_id"]),
        )
    )
    all_recommendation_rows.sort(
        key=lambda r: (
            str(r["policy_id"]),
            str(r["bacteria"]),
            int(r["recommendation_rank"]),
        )
    )

    best_policy = policy_comparison_rows[0]
    best_policy_id = str(best_policy["policy_id"])
    best_recommendation_rows = [row for row in all_recommendation_rows if str(row["policy_id"]) == best_policy_id]

    write_csv(
        args.output_dir / "st06b_policy_comparison.csv",
        fieldnames=list(policy_comparison_rows[0].keys()),
        rows=policy_comparison_rows,
    )
    write_csv(
        args.output_dir / "st06b_recommendations_all_policies.csv",
        fieldnames=list(all_recommendation_rows[0].keys()),
        rows=all_recommendation_rows,
    )
    write_csv(
        args.output_dir / "st06b_top3_recommendations_best.csv",
        fieldnames=list(best_recommendation_rows[0].keys()),
        rows=best_recommendation_rows,
    )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st06b_compare_ranking_policies",
        "input_predictions_path": str(args.st05_predictions_path),
        "parameters": {
            "top_k": args.top_k,
            "max_per_family": args.max_per_family,
            "score_variants": [variant for variant, _ in SCORE_COLUMNS],
        },
        "compared_policy_count": len(policy_comparison_rows),
        "best_policy": best_policy,
        "outputs": {
            "policy_comparison_csv": str(args.output_dir / "st06b_policy_comparison.csv"),
            "all_policy_recommendations_csv": str(args.output_dir / "st06b_recommendations_all_policies.csv"),
            "best_policy_recommendations_csv": str(args.output_dir / "st06b_top3_recommendations_best.csv"),
            "summary_json": str(args.output_dir / "st06b_summary.json"),
        },
    }
    write_json(args.output_dir / "st06b_summary.json", summary)

    print("ST0.6b completed.")
    print(f"- Policies compared: {len(policy_comparison_rows)}")
    print(
        "- Best policy: "
        f"{best_policy_id} (all={best_policy['topk_hit_rate_all_strains']}, "
        f"susceptible={best_policy['topk_hit_rate_susceptible_only']})"
    )
    print(f"- Comparison output: {args.output_dir / 'st06b_policy_comparison.csv'}")


if __name__ == "__main__":
    main()
