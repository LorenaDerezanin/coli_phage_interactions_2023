#!/usr/bin/env python3
"""TB03: characterize hard-to-lyse strains by known host traits."""

from __future__ import annotations

import argparse
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
UNKNOWN_SUSCEPTIBILITY_BUCKET = "unknown_missing_assays"

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
    if value is None or pd.isna(value):
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


def classify_susceptibility_bucket(n_lytic_phages: int | None, low_threshold: int) -> str:
    if n_lytic_phages is None:
        return UNKNOWN_SUSCEPTIBILITY_BUCKET
    if n_lytic_phages == 0:
        return "zero"
    if n_lytic_phages <= low_threshold:
        return f"low_{1}_{low_threshold}"
    return "broad"


def count_lytic_phages(interaction_matrix: pd.DataFrame) -> pd.Series:
    binary_matrix = interaction_matrix.gt(0).astype("Int64").where(interaction_matrix.notna(), pd.NA)
    return binary_matrix.sum(axis=1, min_count=binary_matrix.shape[1]).astype("Int64").rename("n_lytic_phages")


def count_known_lytic_phages(interaction_matrix: pd.DataFrame) -> pd.Series:
    return interaction_matrix.gt(0).where(interaction_matrix.notna(), False).sum(axis=1).astype(int).rename(
        "known_lytic_phages"
    )


def count_missing_assays(interaction_matrix: pd.DataFrame) -> pd.Series:
    return interaction_matrix.isna().sum(axis=1).astype(int).rename("missing_assay_count")


def true_count(values: pd.Series) -> int:
    return int(values.eq(True).sum())


def false_count(values: pd.Series) -> int:
    return int(values.eq(False).sum())


def observed_rate(values: pd.Series) -> float | None:
    observed_total = int(values.notna().sum())
    return true_count(values) / observed_total if observed_total else None


def optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


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
    n_lytic_phages = count_lytic_phages(interaction_matrix)
    known_lytic_phages = count_known_lytic_phages(interaction_matrix)
    missing_assay_count = count_missing_assays(interaction_matrix)

    strain_summary = pd.concat([n_lytic_phages, known_lytic_phages, missing_assay_count], axis=1)
    strain_summary = strain_summary.reset_index().rename(columns={"index": "bacteria"})
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
    has_missing_assays = strain_summary["missing_assay_count"].gt(0)
    strain_summary["is_zero_lysis"] = pd.Series(pd.NA, index=strain_summary.index, dtype="boolean")
    strain_summary["is_low_susceptibility"] = pd.Series(pd.NA, index=strain_summary.index, dtype="boolean")
    strain_summary["is_zero_lysis"] = strain_summary["is_zero_lysis"].mask(strain_summary["known_lytic_phages"].gt(0), False)
    strain_summary["is_low_susceptibility"] = strain_summary["is_low_susceptibility"].mask(
        strain_summary["known_lytic_phages"].gt(low_threshold), False
    )
    no_missing_assays = ~has_missing_assays
    strain_summary["is_zero_lysis"] = strain_summary["is_zero_lysis"].mask(
        no_missing_assays, strain_summary["known_lytic_phages"].eq(0)
    )
    strain_summary["is_low_susceptibility"] = strain_summary["is_low_susceptibility"].mask(
        no_missing_assays, strain_summary["known_lytic_phages"].le(low_threshold)
    )

    strain_summary["susceptibility_bucket"] = UNKNOWN_SUSCEPTIBILITY_BUCKET
    strain_summary["susceptibility_bucket"] = strain_summary["susceptibility_bucket"].mask(
        strain_summary["known_lytic_phages"].gt(low_threshold), "broad"
    )
    exact_count_mask = no_missing_assays
    strain_summary.loc[exact_count_mask, "susceptibility_bucket"] = [
        classify_susceptibility_bucket(int(count), low_threshold)
        for count in strain_summary.loc[exact_count_mask, "known_lytic_phages"]
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
        group["n_lytic_phages"].dropna().to_numpy(dtype=float)
        for _, group in grouped
        if group["n_lytic_phages"].notna().any()
    ]
    kruskal_p_value = float(kruskal(*groups_for_kruskal).pvalue) if len(groups_for_kruskal) >= 2 else None

    rows: list[dict[str, object]] = []
    tested_indexes: list[int] = []
    tested_p_values: list[float] = []

    for trait_value, group in grouped:
        total_strains = int(len(group))
        low_count = true_count(group["is_low_susceptibility"])
        zero_count = true_count(group["is_zero_lysis"])
        non_low_count = false_count(group["is_low_susceptibility"])
        low_rate = observed_rate(group["is_low_susceptibility"])
        observed_low_status_count = int(group["is_low_susceptibility"].notna().sum())
        row: dict[str, object] = {
            "summary_level": "trait_value",
            "field_name": field_label,
            "trait_value": trait_value,
            "total_strains": total_strains,
            "zero_lysis_count": zero_count,
            "low_susceptibility_count": low_count,
            "low_susceptibility_rate": low_rate,
            "mean_lytic_phages": optional_float(group["n_lytic_phages"].mean()),
            "median_lytic_phages": optional_float(group["n_lytic_phages"].median()),
            "tested_for_enrichment": observed_low_status_count >= min_group_size,
            "odds_ratio_vs_rest": None,
            "fisher_p_value": None,
            "fisher_q_value": None,
            "kruskal_p_value": kruskal_p_value,
            "low_susceptibility_threshold": low_threshold,
            "min_group_size_for_enrichment_test": min_group_size,
        }

        if observed_low_status_count >= min_group_size:
            rest = work.loc[work[field_name] != trait_value]
            rest_low_count = true_count(rest["is_low_susceptibility"])
            rest_non_low_count = false_count(rest["is_low_susceptibility"])
            contingency = [
                [low_count, non_low_count],
                [rest_low_count, rest_non_low_count],
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
            -(float(row["low_susceptibility_rate"]) if row["low_susceptibility_rate"] is not None else -1.0),
            row["trait_value"],
        )
    )

    field_row = {
        "summary_level": "field",
        "field_name": field_label,
        "trait_value": "__all__",
        "total_strains": int(len(work)),
        "zero_lysis_count": true_count(work["is_zero_lysis"]),
        "low_susceptibility_count": true_count(work["is_low_susceptibility"]),
        "low_susceptibility_rate": observed_rate(work["is_low_susceptibility"]),
        "mean_lytic_phages": optional_float(work["n_lytic_phages"].mean()),
        "median_lytic_phages": optional_float(work["n_lytic_phages"].median()),
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

    low_strains = strain_summary.loc[strain_summary["is_low_susceptibility"].eq(True), "bacteria"].tolist()
    zero_strains = strain_summary.loc[strain_summary["is_zero_lysis"].eq(True), "bacteria"].tolist()
    manifest = {
        "analysis_id": "TB03",
        "interaction_matrix_path": str(args.interaction_matrix_path),
        "host_metadata_path": str(args.host_metadata_path),
        "strain_output_path": str(strain_output),
        "trait_summary_output_path": str(summary_output),
        "low_susceptibility_threshold": args.low_susceptibility_threshold,
        "min_group_size_for_enrichment_test": args.min_group_size_for_enrichment_test,
        "n_total_strains": int(len(strain_summary)),
        "n_zero_lysis_strains": true_count(strain_summary["is_zero_lysis"]),
        "n_low_susceptibility_strains": true_count(strain_summary["is_low_susceptibility"]),
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
