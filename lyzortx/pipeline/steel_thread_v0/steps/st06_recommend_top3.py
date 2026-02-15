#!/usr/bin/env python3
"""ST0.6: Generate top-k phage recommendations with simple diversity constraints."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st05-ranked-predictions-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st05_ranked_predictions.csv"),
        help="Input ST0.5 ranked predictions CSV.",
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
        "--max-per-family",
        type=int,
        default=2,
        help="Maximum recommendations from the same phage family before relaxing diversity.",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return [
            {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            for row in reader
        ]


def safe_round(value: float) -> float:
    return round(float(value), 6)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")
    if args.max_per_family < 1:
        raise ValueError("max-per-family must be >= 1")

    ranked_rows = read_csv_rows(args.st05_ranked_predictions_path)
    if not ranked_rows:
        raise ValueError("ST0.5 ranked predictions input is empty.")

    by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in ranked_rows:
        by_bacteria[row["bacteria"]].append(row)

    recommendation_rows: List[Dict[str, object]] = []
    relaxed_strain_count = 0

    for bacteria in sorted(by_bacteria.keys()):
        rows = sorted(
            by_bacteria[bacteria],
            key=lambda r: (-float(r["score_logreg_isotonic"]), r["phage"]),
        )
        family_counts: Counter[str] = Counter()
        selected: List[Dict[str, str]] = []
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
                    "score_logreg_isotonic": row["score_logreg_isotonic"],
                    "split_holdout": row["split_holdout"],
                    "label_hard_binary": row["label_hard_binary"],
                    "label_strict_confidence_tier": row["label_strict_confidence_tier"],
                    "diversity_relaxed_for_strain": 1 if relaxed_used else 0,
                }
            )

    recommendation_rows.sort(key=lambda r: (str(r["bacteria"]), int(r["recommendation_rank"])))
    output_recs = args.output_dir / "st06_top3_recommendations.csv"
    write_csv(output_recs, fieldnames=list(recommendation_rows[0].keys()), rows=recommendation_rows)

    holdout_rows = [row for row in ranked_rows if row["split_holdout"] == "holdout_test"]
    holdout_by_bacteria: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in holdout_rows:
        holdout_by_bacteria[row["bacteria"]].append(row)

    recs_by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in recommendation_rows:
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
        rec_hit = any(row["label_hard_binary"] == "1" for row in recs)
        holdout_hits += 1 if rec_hit else 0

        susceptible = any(row["label_hard_binary"] == "1" for row in available if row["label_hard_binary"] != "")
        if susceptible:
            susceptible_total += 1
            susceptible_hits += 1 if rec_hit else 0

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "step_name": "st06_recommend_top3",
        "input_ranked_predictions_path": str(args.st05_ranked_predictions_path),
        "parameters": {
            "top_k": args.top_k,
            "max_per_family": args.max_per_family,
        },
        "recommendation_summary": {
            "recommended_strain_count": len({row["bacteria"] for row in recommendation_rows}),
            "recommended_row_count": len(recommendation_rows),
            "diversity_relaxed_strain_count": relaxed_strain_count,
        },
        "holdout_topk_metrics": {
            "holdout_strain_count": holdout_total,
            "holdout_hit_count": holdout_hits,
            "topk_hit_rate_all_strains": safe_round(holdout_hits / holdout_total if holdout_total else 0.0),
            "holdout_susceptible_strain_count": susceptible_total,
            "holdout_susceptible_hit_count": susceptible_hits,
            "topk_hit_rate_susceptible_only": safe_round(
                susceptible_hits / susceptible_total if susceptible_total else 0.0
            ),
        },
    }

    write_json(args.output_dir / "st06_recommendation_summary.json", summary)

    print("ST0.6 completed.")
    print(f"- Recommended strains: {summary['recommendation_summary']['recommended_strain_count']}")
    print(f"- Recommendation rows: {summary['recommendation_summary']['recommended_row_count']}")
    print(f"- Holdout top-{args.top_k} hit rate: {summary['holdout_topk_metrics']['topk_hit_rate_all_strains']}")
    print(f"- Output recommendations: {output_recs}")


if __name__ == "__main__":
    main()
