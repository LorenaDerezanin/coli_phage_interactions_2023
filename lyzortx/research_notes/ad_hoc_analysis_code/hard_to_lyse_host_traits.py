#!/usr/bin/env python3
"""TB03: characterize hard-to-lyse strains by known host traits."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from scipy.stats import fisher_exact, kruskal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

LOW_SUSCEPTIBILITY_THRESHOLD = 3
MIN_GROUP_SIZE_FOR_ENRICHMENT_TEST = 4

TRAIT_FIELDS: Sequence[tuple[str, str]] = (
    ("host_serotype", "serotype"),
    ("host_phylogroup", "phylogroup"),
    ("host_st", "ST"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interaction-matrix-path",
        type=Path,
        default=Path("data/interactions/interaction_matrix.csv"),
        help="Interaction matrix CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Host metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/hard_to_lyse_host_traits"),
        help="Output directory.",
    )
    parser.add_argument(
        "--low-susceptibility-threshold",
        type=int,
        default=LOW_SUSCEPTIBILITY_THRESHOLD,
        help="Maximum number of lytic phages for the low-susceptibility slice.",
    )
    parser.add_argument(
        "--min-group-size-for-enrichment-test",
        type=int,
        default=MIN_GROUP_SIZE_FOR_ENRICHMENT_TEST,
        help="Minimum subgroup size to run category-vs-rest enrichment tests.",
    )
    return parser.parse_args()


def clean_trait_value(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Missing"
    text = str(value).strip()
    return text if text else "Missing"


def derive_serotype(o_type: object, h_type: object) -> str:
    o_value = clean_trait_value(o_type)
    h_value = clean_trait_value(h_type)
    if o_value == "Missing" and h_value == "Missing":
        return "Missing"
    if o_value == "Missing":
        return h_value
    if h_value == "Missing":
        return o_value
    return f"{o_value}:{h_value}"


def classify_susceptibility_bucket(n_lytic_phages: int, low_threshold: int) -> str:
    if n_lytic_phages == 0:
        return "zero"
    if n_lytic_phages <= low_threshold:
        return f"low_{1}_{low_threshold}"
    return "broad"


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    total = len(indexed)
    adjusted = [0.0] * total
    running_min = 1.0

    for reverse_rank, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = total - reverse_rank + 1
        q_value = min(1.0, p_value * total / rank)
        running_min = min(running_min, q_value)
        adjusted[original_index] = running_min

    return adjusted


def build_per_strain_summary(
    interaction_matrix: pd.DataFrame,
    host_metadata: pd.DataFrame,
    low_threshold: int,
) -> pd.DataFrame:
    binary_matrix = (interaction_matrix > 0).astype(int)
    n_lytic_phages = binary_matrix.sum(axis=1).rename("n_lytic_phages")

    strain_summary = n_lytic_phages.reset_index().rename(columns={"index": "bacteria"})
    host_columns = [
        "bacteria",
        "ABC_serotype",
        "Clermont_Phylo",
        "ST_Warwick",
        "O-type",
        "H-type",
    ]
    strain_summary = strain_summary.merge(host_metadata[host_columns], on="bacteria", how="left")
    strain_summary["host_serotype"] = [
        derive_serotype(o_type, h_type)
        for o_type, h_type in zip(strain_summary["O-type"], strain_summary["H-type"], strict=False)
    ]
    strain_summary["host_abc_serotype"] = strain_summary["ABC_serotype"].map(clean_trait_value)
    strain_summary["host_phylogroup"] = strain_summary["Clermont_Phylo"].map(clean_trait_value)
    strain_summary["host_st"] = strain_summary["ST_Warwick"].map(clean_trait_value)
    strain_summary["is_zero_lysis"] = strain_summary["n_lytic_phages"] == 0
    strain_summary["is_low_susceptibility"] = strain_summary["n_lytic_phages"] <= low_threshold
    strain_summary["susceptibility_bucket"] = [
        classify_susceptibility_bucket(int(count), low_threshold)
        for count in strain_summary["n_lytic_phages"]
    ]
    return strain_summary.sort_values(["n_lytic_phages", "bacteria"]).reset_index(drop=True)


def summarize_trait_field(
    strain_summary: pd.DataFrame,
    field_name: str,
    field_label: str,
    low_threshold: int,
    min_group_size: int,
) -> list[dict[str, object]]:
    work = strain_summary[["bacteria", "n_lytic_phages", "is_zero_lysis", "is_low_susceptibility", field_name]].copy()
    work[field_name] = work[field_name].map(clean_trait_value)

    grouped = work.groupby(field_name, dropna=False)
    groups_for_kruskal = [
        group["n_lytic_phages"].to_numpy()
        for _, group in grouped
    ]
    kruskal_p_value = float(kruskal(*groups_for_kruskal).pvalue) if len(groups_for_kruskal) >= 2 else None

    rows: list[dict[str, object]] = []
    tested_indexes: list[int] = []
    tested_p_values: list[float] = []

    for trait_value, group in grouped:
        total_strains = int(len(group))
        low_count = int(group["is_low_susceptibility"].sum())
        zero_count = int(group["is_zero_lysis"].sum())
        non_low_count = total_strains - low_count
        low_rate = low_count / total_strains if total_strains else 0.0
        row: dict[str, object] = {
            "summary_level": "trait_value",
            "field_name": field_label,
            "trait_value": trait_value,
            "total_strains": total_strains,
            "zero_lysis_count": zero_count,
            "low_susceptibility_count": low_count,
            "low_susceptibility_rate": low_rate,
            "mean_lytic_phages": float(group["n_lytic_phages"].mean()),
            "median_lytic_phages": float(group["n_lytic_phages"].median()),
            "tested_for_enrichment": total_strains >= min_group_size,
            "odds_ratio_vs_rest": None,
            "fisher_p_value": None,
            "fisher_q_value": None,
            "kruskal_p_value": kruskal_p_value,
            "low_susceptibility_threshold": low_threshold,
            "min_group_size_for_enrichment_test": min_group_size,
        }

        if total_strains >= min_group_size:
            rest = work.loc[work[field_name] != trait_value]
            contingency = [
                [low_count, non_low_count],
                [int(rest["is_low_susceptibility"].sum()), int((~rest["is_low_susceptibility"]).sum())],
            ]
            odds_ratio, fisher_p_value = fisher_exact(contingency, alternative="two-sided")
            row["odds_ratio_vs_rest"] = float(odds_ratio)
            row["fisher_p_value"] = float(fisher_p_value)
            tested_indexes.append(len(rows))
            tested_p_values.append(float(fisher_p_value))

        rows.append(row)

    for row_index, q_value in zip(tested_indexes, benjamini_hochberg(tested_p_values), strict=False):
        rows[row_index]["fisher_q_value"] = q_value

    rows.sort(
        key=lambda row: (
            row["summary_level"] != "trait_value",
            row["fisher_q_value"] if row["fisher_q_value"] is not None else 1.0,
            -float(row["low_susceptibility_rate"]),
            row["trait_value"],
        )
    )

    field_row = {
        "summary_level": "field",
        "field_name": field_label,
        "trait_value": "__all__",
        "total_strains": int(len(work)),
        "zero_lysis_count": int(work["is_zero_lysis"].sum()),
        "low_susceptibility_count": int(work["is_low_susceptibility"].sum()),
        "low_susceptibility_rate": float(work["is_low_susceptibility"].mean()),
        "mean_lytic_phages": float(work["n_lytic_phages"].mean()),
        "median_lytic_phages": float(work["n_lytic_phages"].median()),
        "tested_for_enrichment": True,
        "odds_ratio_vs_rest": None,
        "fisher_p_value": None,
        "fisher_q_value": None,
        "kruskal_p_value": kruskal_p_value,
        "low_susceptibility_threshold": low_threshold,
        "min_group_size_for_enrichment_test": min_group_size,
    }
    return [field_row, *rows]


def top_enriched_rows(summary_rows: Iterable[dict[str, object]], q_threshold: float = 0.1) -> list[dict[str, object]]:
    enriched = [
        row
        for row in summary_rows
        if row["summary_level"] == "trait_value"
        and row["fisher_q_value"] is not None
        and float(row["fisher_q_value"]) <= q_threshold
        and float(row["odds_ratio_vs_rest"] or 0.0) > 1.0
    ]
    return sorted(
        enriched,
        key=lambda row: (float(row["fisher_q_value"]), -float(row["low_susceptibility_rate"]), row["field_name"]),
    )


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    interaction_matrix = pd.read_csv(args.interaction_matrix_path, sep=";").set_index("bacteria")
    host_metadata = pd.read_csv(args.host_metadata_path, sep=";")

    strain_summary = build_per_strain_summary(
        interaction_matrix=interaction_matrix,
        host_metadata=host_metadata,
        low_threshold=args.low_susceptibility_threshold,
    )

    summary_rows: list[dict[str, object]] = []
    for field_name, field_label in TRAIT_FIELDS:
        summary_rows.extend(
            summarize_trait_field(
                strain_summary=strain_summary,
                field_name=field_name,
                field_label=field_label,
                low_threshold=args.low_susceptibility_threshold,
                min_group_size=args.min_group_size_for_enrichment_test,
            )
        )

    strain_output = args.output_dir / "hard_to_lyse_strain_summary.csv"
    summary_output = args.output_dir / "host_trait_low_susceptibility_summary.csv"
    manifest_output = args.output_dir / "tb03_summary.json"

    write_csv(
        strain_output,
        fieldnames=[
            "bacteria",
            "n_lytic_phages",
            "is_zero_lysis",
            "is_low_susceptibility",
            "susceptibility_bucket",
            "host_serotype",
            "host_abc_serotype",
            "host_phylogroup",
            "host_st",
        ],
        rows=strain_summary[
            [
                "bacteria",
                "n_lytic_phages",
                "is_zero_lysis",
                "is_low_susceptibility",
                "susceptibility_bucket",
                "host_serotype",
                "host_abc_serotype",
                "host_phylogroup",
                "host_st",
            ]
        ].to_dict(orient="records"),
    )

    write_csv(
        summary_output,
        fieldnames=[
            "summary_level",
            "field_name",
            "trait_value",
            "total_strains",
            "zero_lysis_count",
            "low_susceptibility_count",
            "low_susceptibility_rate",
            "mean_lytic_phages",
            "median_lytic_phages",
            "tested_for_enrichment",
            "odds_ratio_vs_rest",
            "fisher_p_value",
            "fisher_q_value",
            "kruskal_p_value",
            "low_susceptibility_threshold",
            "min_group_size_for_enrichment_test",
        ],
        rows=summary_rows,
    )

    low_strains = strain_summary.loc[strain_summary["is_low_susceptibility"], "bacteria"].tolist()
    zero_strains = strain_summary.loc[strain_summary["is_zero_lysis"], "bacteria"].tolist()
    manifest = {
        "analysis_id": "TB03",
        "interaction_matrix_path": str(args.interaction_matrix_path),
        "host_metadata_path": str(args.host_metadata_path),
        "strain_output_path": str(strain_output),
        "trait_summary_output_path": str(summary_output),
        "low_susceptibility_threshold": args.low_susceptibility_threshold,
        "min_group_size_for_enrichment_test": args.min_group_size_for_enrichment_test,
        "n_total_strains": int(len(strain_summary)),
        "n_zero_lysis_strains": int(strain_summary["is_zero_lysis"].sum()),
        "n_low_susceptibility_strains": int(strain_summary["is_low_susceptibility"].sum()),
        "zero_lysis_strains": zero_strains,
        "low_susceptibility_strains": low_strains,
        "top_enriched_trait_values_q_le_0_1": top_enriched_rows(summary_rows),
        "notes": [
            "Serotype is represented as a derived O-type:H-type label to avoid the heavy missingness of ABC_serotype.",
            "Category-vs-rest enrichment tests are only reported for groups meeting the minimum size threshold.",
        ],
    }
    write_json(manifest_output, manifest)

    print(
        f"TB03 summary: total={manifest['n_total_strains']}, zero={manifest['n_zero_lysis_strains']}, "
        f"low<={args.low_susceptibility_threshold}={manifest['n_low_susceptibility_strains']}"
    )
    for row in top_enriched_rows(summary_rows)[:10]:
        print(
            f"{row['field_name']}={row['trait_value']}: low_rate={row['low_susceptibility_rate']:.3f}, "
            f"odds_ratio={row['odds_ratio_vs_rest']:.3f}, q={row['fisher_q_value']:.4f}"
        )


if __name__ == "__main__":
    main()
