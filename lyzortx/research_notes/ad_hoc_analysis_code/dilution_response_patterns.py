#!/usr/bin/env python3
"""TB05: analyze dilution-response patterns per phage and per bacterial subgroup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.research_notes.ad_hoc_analysis_code.hard_to_lyse_host_traits import (
    benjamini_hochberg,
    clean_trait_value,
    derive_serotype,
)

HIGH_POTENCY_DILUTIONS = {-4, -2}
MIN_LYTIC_PAIRS_FOR_ENRICHMENT_TEST = 40
SUBGROUP_FIELDS: Sequence[tuple[str, str]] = (
    ("host_phylogroup", "phylogroup"),
    ("host_st", "ST"),
    ("host_serotype", "serotype"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-labels-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/labels/label_set_v1_pairs.csv"),
        help="Track A pair label set with dilution potency outputs.",
    )
    parser.add_argument(
        "--pair-dilution-summary-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/labels/track_a_pair_dilution_summary.csv"),
        help="Track A pair+dilution summary table.",
    )
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Host metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Phage metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/dilution_response_patterns"),
        help="Output directory.",
    )
    parser.add_argument(
        "--min-lytic-pairs-for-enrichment-test",
        type=int,
        default=MIN_LYTIC_PAIRS_FOR_ENRICHMENT_TEST,
        help="Minimum lytic-pair count required before comparing a phage/subgroup against the rest.",
    )
    return parser.parse_args()


def serialize_dilution(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))


def build_pair_response_features(pair_dilution_summary: pd.DataFrame) -> pd.DataFrame:
    positive_rows = pair_dilution_summary[pair_dilution_summary["score_1_count"].gt(0)].copy()
    if positive_rows.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "n_positive_dilutions",
                "max_positive_dilution",
                "best_positive_dilution",
                "has_multidilution_support",
                "positive_dilution_list",
            ]
        )

    features = (
        positive_rows.groupby("pair_id", dropna=False)
        .agg(
            n_positive_dilutions=("log_dilution", "nunique"),
            max_positive_dilution=("log_dilution", "max"),
            best_positive_dilution=("log_dilution", "min"),
            positive_dilution_list=("log_dilution", lambda s: ",".join(str(int(value)) for value in sorted(set(s)))),
        )
        .reset_index()
    )
    features["has_multidilution_support"] = features["n_positive_dilutions"].ge(2)
    return features


def prepare_host_metadata(host_metadata: pd.DataFrame) -> pd.DataFrame:
    prepared = host_metadata.copy()
    prepared["host_phylogroup"] = prepared["Clermont_Phylo"].map(clean_trait_value)
    prepared["host_st"] = prepared["ST_Warwick"].map(clean_trait_value)
    prepared["host_serotype"] = [
        derive_serotype(o_type, h_type)
        for o_type, h_type in zip(prepared["O-type"], prepared["H-type"], strict=False)
    ]
    return prepared[["bacteria", "host_phylogroup", "host_st", "host_serotype"]].drop_duplicates()


def add_bh_q_values(
    summary: pd.DataFrame,
    tested_mask: pd.Series,
    p_value_column: str,
    q_value_column: str,
) -> pd.DataFrame:
    summary = summary.copy()
    summary[q_value_column] = None
    tested_index = summary.index[tested_mask.fillna(False)]
    if len(tested_index) == 0:
        return summary

    q_values = benjamini_hochberg(summary.loc[tested_index, p_value_column].astype(float).tolist())
    for index, q_value in zip(tested_index, q_values, strict=False):
        summary.at[index, q_value_column] = float(q_value)
    return summary


def summarize_binary_enrichment(
    summary: pd.DataFrame,
    group_column: str,
    event_count_column: str,
    non_event_count_column: str,
    tested_column: str,
    odds_ratio_column: str,
    p_value_column: str,
    q_value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        tested = bool(getattr(row, tested_column))
        odds_ratio: float | None = None
        p_value: float | None = None
        if tested:
            group_event = int(getattr(row, event_count_column))
            group_non_event = int(getattr(row, non_event_count_column))
            rest_event = int(summary[event_count_column].sum()) - group_event
            rest_non_event = int(summary[non_event_count_column].sum()) - group_non_event
            odds_ratio, p_value = fisher_exact(
                [[group_event, group_non_event], [rest_event, rest_non_event]],
                alternative="two-sided",
            )

        rows.append(
            {
                group_column: getattr(row, group_column),
                odds_ratio_column: odds_ratio,
                p_value_column: p_value,
            }
        )

    enrichment = pd.DataFrame(rows)
    summary = summary.merge(enrichment, on=group_column, how="left")
    return add_bh_q_values(summary, summary[tested_column], p_value_column, q_value_column)


def build_phage_dilution_response_summary(
    pair_labels: pd.DataFrame,
    pair_response_features: pd.DataFrame,
    phage_metadata: pd.DataFrame,
    min_lytic_pairs_for_test: int,
) -> pd.DataFrame:
    resolved_pairs = pair_labels[pair_labels["any_lysis"].notna()].copy()
    lytic_pairs = pair_labels[pair_labels["any_lysis"].eq(1)].copy()
    lytic_pairs = lytic_pairs.merge(pair_response_features, on="pair_id", how="left")
    lytic_pairs["has_multidilution_support"] = lytic_pairs["has_multidilution_support"].fillna(False)
    lytic_pairs["is_high_potency"] = lytic_pairs["best_lysis_dilution"].isin(HIGH_POTENCY_DILUTIONS)

    resolved_counts = (
        resolved_pairs.groupby("phage", dropna=False)
        .agg(n_resolved_pairs=("pair_id", "size"))
        .reset_index()
    )
    phage_summary = (
        lytic_pairs.groupby("phage", dropna=False)
        .agg(
            n_lytic_pairs=("pair_id", "size"),
            n_high_potency_pairs=("is_high_potency", "sum"),
            n_low_potency_pairs=("is_high_potency", lambda s: int((~s).sum())),
            high_potency_rate=("is_high_potency", "mean"),
            n_multidilution_pairs=("has_multidilution_support", "sum"),
            n_single_dilution_pairs=("has_multidilution_support", lambda s: int((~s).sum())),
            multi_dilution_support_rate=("has_multidilution_support", "mean"),
            mean_potency_rank=("dilution_potency_rank", "mean"),
            median_potency_rank=("dilution_potency_rank", "median"),
            best_dilution_0_count=("best_lysis_dilution", lambda s: int((s == 0).sum())),
            best_dilution_minus1_count=("best_lysis_dilution", lambda s: int((s == -1).sum())),
            best_dilution_minus2_count=("best_lysis_dilution", lambda s: int((s == -2).sum())),
            best_dilution_minus4_count=("best_lysis_dilution", lambda s: int((s == -4).sum())),
        )
        .reset_index()
    )
    phage_summary = resolved_counts.merge(phage_summary, on="phage", how="left").fillna(
        {
            "n_lytic_pairs": 0,
            "n_high_potency_pairs": 0,
            "n_low_potency_pairs": 0,
            "high_potency_rate": 0.0,
            "n_multidilution_pairs": 0,
            "n_single_dilution_pairs": 0,
            "multi_dilution_support_rate": 0.0,
            "best_dilution_0_count": 0,
            "best_dilution_minus1_count": 0,
            "best_dilution_minus2_count": 0,
            "best_dilution_minus4_count": 0,
        }
    )
    phage_summary["lytic_pair_rate"] = phage_summary["n_lytic_pairs"] / phage_summary["n_resolved_pairs"]
    phage_summary["tested_for_high_potency_enrichment"] = phage_summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)
    phage_summary["tested_for_multidilution_enrichment"] = phage_summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)
    phage_summary = summarize_binary_enrichment(
        summary=phage_summary,
        group_column="phage",
        event_count_column="n_high_potency_pairs",
        non_event_count_column="n_low_potency_pairs",
        tested_column="tested_for_high_potency_enrichment",
        odds_ratio_column="high_potency_odds_ratio_vs_rest",
        p_value_column="high_potency_fisher_p_value",
        q_value_column="high_potency_fisher_q_value",
    )
    phage_summary = summarize_binary_enrichment(
        summary=phage_summary,
        group_column="phage",
        event_count_column="n_multidilution_pairs",
        non_event_count_column="n_single_dilution_pairs",
        tested_column="tested_for_multidilution_enrichment",
        odds_ratio_column="multidilution_odds_ratio_vs_rest",
        p_value_column="multidilution_fisher_p_value",
        q_value_column="multidilution_fisher_q_value",
    )
    phage_summary = phage_summary.merge(
        phage_metadata[["phage", "Morphotype", "Family"]],
        on="phage",
        how="left",
    ).rename(columns={"Morphotype": "morphotype", "Family": "family"})
    phage_summary["morphotype"] = phage_summary["morphotype"].map(clean_trait_value)
    phage_summary["family"] = phage_summary["family"].map(clean_trait_value)
    return phage_summary.sort_values(
        ["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs", "phage"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_bacterial_subgroup_dilution_response_summary(
    pair_labels: pd.DataFrame,
    pair_response_features: pd.DataFrame,
    host_metadata: pd.DataFrame,
    min_lytic_pairs_for_test: int,
) -> pd.DataFrame:
    resolved_pairs = pair_labels[pair_labels["any_lysis"].notna()].copy().merge(host_metadata, on="bacteria", how="left")
    lytic_pairs = pair_labels[pair_labels["any_lysis"].eq(1)].copy()
    lytic_pairs = lytic_pairs.merge(pair_response_features, on="pair_id", how="left").merge(
        host_metadata,
        on="bacteria",
        how="left",
    )
    lytic_pairs["has_multidilution_support"] = lytic_pairs["has_multidilution_support"].fillna(False)
    lytic_pairs["is_high_potency"] = lytic_pairs["best_lysis_dilution"].isin(HIGH_POTENCY_DILUTIONS)

    subgroup_rows: list[pd.DataFrame] = []
    for subgroup_column, subgroup_label in SUBGROUP_FIELDS:
        resolved_counts = (
            resolved_pairs.groupby(subgroup_column, dropna=False)
            .agg(n_resolved_pairs=("pair_id", "size"))
            .reset_index()
            .rename(columns={subgroup_column: "subgroup_value"})
        )
        summary = (
            lytic_pairs.groupby(subgroup_column, dropna=False)
            .agg(
                n_lytic_pairs=("pair_id", "size"),
                n_high_potency_pairs=("is_high_potency", "sum"),
                n_low_potency_pairs=("is_high_potency", lambda s: int((~s).sum())),
                high_potency_rate=("is_high_potency", "mean"),
                n_multidilution_pairs=("has_multidilution_support", "sum"),
                n_single_dilution_pairs=("has_multidilution_support", lambda s: int((~s).sum())),
                multi_dilution_support_rate=("has_multidilution_support", "mean"),
                mean_potency_rank=("dilution_potency_rank", "mean"),
                median_potency_rank=("dilution_potency_rank", "median"),
                best_dilution_0_count=("best_lysis_dilution", lambda s: int((s == 0).sum())),
                best_dilution_minus1_count=("best_lysis_dilution", lambda s: int((s == -1).sum())),
                best_dilution_minus2_count=("best_lysis_dilution", lambda s: int((s == -2).sum())),
                best_dilution_minus4_count=("best_lysis_dilution", lambda s: int((s == -4).sum())),
            )
            .reset_index()
            .rename(columns={subgroup_column: "subgroup_value"})
        )
        summary = resolved_counts.merge(summary, on="subgroup_value", how="left").fillna(
            {
                "n_lytic_pairs": 0,
                "n_high_potency_pairs": 0,
                "n_low_potency_pairs": 0,
                "high_potency_rate": 0.0,
                "n_multidilution_pairs": 0,
                "n_single_dilution_pairs": 0,
                "multi_dilution_support_rate": 0.0,
                "best_dilution_0_count": 0,
                "best_dilution_minus1_count": 0,
                "best_dilution_minus2_count": 0,
                "best_dilution_minus4_count": 0,
            }
        )
        summary["field_name"] = subgroup_label
        summary["subgroup_value"] = summary["subgroup_value"].map(clean_trait_value)
        summary["lytic_pair_rate"] = summary["n_lytic_pairs"] / summary["n_resolved_pairs"]
        summary["tested_for_high_potency_enrichment"] = summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)
        summary["tested_for_multidilution_enrichment"] = summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)
        summary = summarize_binary_enrichment(
            summary=summary,
            group_column="subgroup_value",
            event_count_column="n_high_potency_pairs",
            non_event_count_column="n_low_potency_pairs",
            tested_column="tested_for_high_potency_enrichment",
            odds_ratio_column="high_potency_odds_ratio_vs_rest",
            p_value_column="high_potency_fisher_p_value",
            q_value_column="high_potency_fisher_q_value",
        )
        summary = summarize_binary_enrichment(
            summary=summary,
            group_column="subgroup_value",
            event_count_column="n_multidilution_pairs",
            non_event_count_column="n_single_dilution_pairs",
            tested_column="tested_for_multidilution_enrichment",
            odds_ratio_column="multidilution_odds_ratio_vs_rest",
            p_value_column="multidilution_fisher_p_value",
            q_value_column="multidilution_fisher_q_value",
        )
        subgroup_rows.append(summary)

    return (
        pd.concat(subgroup_rows, ignore_index=True)
        .sort_values(["field_name", "high_potency_rate", "n_lytic_pairs", "subgroup_value"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )


def top_rows(
    summary: pd.DataFrame,
    limit: int,
    sort_columns: Sequence[str],
) -> list[dict[str, object]]:
    if sort_columns:
        summary = summary.sort_values(list(sort_columns), ascending=[False] * len(sort_columns))
    summary = summary.head(limit).where(pd.notna(summary.head(limit)), None)
    return json.loads(summary.to_json(orient="records"))


def build_manifest(
    pair_labels: pd.DataFrame,
    phage_summary: pd.DataFrame,
    subgroup_summary: pd.DataFrame,
    min_lytic_pairs_for_test: int,
) -> dict[str, object]:
    lytic_pairs = pair_labels[pair_labels["any_lysis"].eq(1)].copy()
    lytic_pairs["is_high_potency"] = lytic_pairs["best_lysis_dilution"].isin(HIGH_POTENCY_DILUTIONS)
    phylogroup_rows = subgroup_summary[subgroup_summary["field_name"] == "phylogroup"].copy()
    st_rows = subgroup_summary[subgroup_summary["field_name"] == "ST"].copy()
    return {
        "analysis_id": "TB05",
        "output_dir": "lyzortx/generated_outputs/dilution_response_patterns",
        "min_lytic_pairs_for_enrichment_test": min_lytic_pairs_for_test,
        "n_resolved_pairs": int(pair_labels["any_lysis"].notna().sum()),
        "n_lytic_pairs": int(len(lytic_pairs)),
        "overall_high_potency_rate": float(lytic_pairs["is_high_potency"].mean()),
        "overall_multi_dilution_support_rate": float(phage_summary["n_multidilution_pairs"].sum() / len(lytic_pairs)),
        "best_dilution_distribution": {
            serialize_dilution(key): int(value)
            for key, value in lytic_pairs["best_lysis_dilution"].value_counts().sort_index().items()
        },
        "top_high_potency_phages": top_rows(
            phage_summary[phage_summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)],
            limit=5,
            sort_columns=["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs"],
        ),
        "top_low_potency_phages": top_rows(
            phage_summary[phage_summary["n_lytic_pairs"].ge(min_lytic_pairs_for_test)].sort_values(
                ["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs", "phage"],
                ascending=[True, True, False, True],
            ),
            limit=5,
            sort_columns=[],
        ),
        "top_phylogroups_by_high_potency": top_rows(
            phylogroup_rows[phylogroup_rows["n_lytic_pairs"].ge(min_lytic_pairs_for_test)],
            limit=5,
            sort_columns=["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs"],
        ),
        "bottom_phylogroups_by_high_potency": top_rows(
            phylogroup_rows[phylogroup_rows["n_lytic_pairs"].ge(min_lytic_pairs_for_test)].sort_values(
                ["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs", "subgroup_value"],
                ascending=[True, True, False, True],
            ),
            limit=5,
            sort_columns=[],
        ),
        "top_STs_by_high_potency": top_rows(
            st_rows[st_rows["n_lytic_pairs"].ge(min_lytic_pairs_for_test)],
            limit=5,
            sort_columns=["high_potency_rate", "multi_dilution_support_rate", "n_lytic_pairs"],
        ),
        "notes": [
            "High-potency lysis is defined as a best lysis dilution of -2 or -4.",
            "Multi-dilution support is defined as at least two dilution levels with any interpretable positive observation.",
            "Bacterial subgroup summaries are reported for phylogroup, ST, and derived O:H serotype.",
        ],
    }


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    pair_labels = pd.read_csv(args.pair_labels_path)
    pair_dilution_summary = pd.read_csv(args.pair_dilution_summary_path)
    host_metadata = prepare_host_metadata(pd.read_csv(args.host_metadata_path, sep=";"))
    phage_metadata = pd.read_csv(args.phage_metadata_path, sep=";")
    pair_response_features = build_pair_response_features(pair_dilution_summary)

    phage_summary = build_phage_dilution_response_summary(
        pair_labels=pair_labels,
        pair_response_features=pair_response_features,
        phage_metadata=phage_metadata,
        min_lytic_pairs_for_test=args.min_lytic_pairs_for_enrichment_test,
    )
    subgroup_summary = build_bacterial_subgroup_dilution_response_summary(
        pair_labels=pair_labels,
        pair_response_features=pair_response_features,
        host_metadata=host_metadata,
        min_lytic_pairs_for_test=args.min_lytic_pairs_for_enrichment_test,
    )
    manifest = build_manifest(
        pair_labels=pair_labels,
        phage_summary=phage_summary,
        subgroup_summary=subgroup_summary,
        min_lytic_pairs_for_test=args.min_lytic_pairs_for_enrichment_test,
    )

    write_csv(
        args.output_dir / "pair_dilution_response_features.csv",
        fieldnames=[
            "pair_id",
            "n_positive_dilutions",
            "max_positive_dilution",
            "best_positive_dilution",
            "has_multidilution_support",
            "positive_dilution_list",
        ],
        rows=pair_response_features.to_dict(orient="records"),
    )
    write_csv(
        args.output_dir / "per_phage_dilution_response_summary.csv",
        fieldnames=[
            "phage",
            "morphotype",
            "family",
            "n_resolved_pairs",
            "n_lytic_pairs",
            "lytic_pair_rate",
            "n_high_potency_pairs",
            "n_low_potency_pairs",
            "high_potency_rate",
            "n_multidilution_pairs",
            "n_single_dilution_pairs",
            "multi_dilution_support_rate",
            "mean_potency_rank",
            "median_potency_rank",
            "best_dilution_0_count",
            "best_dilution_minus1_count",
            "best_dilution_minus2_count",
            "best_dilution_minus4_count",
            "tested_for_high_potency_enrichment",
            "high_potency_odds_ratio_vs_rest",
            "high_potency_fisher_p_value",
            "high_potency_fisher_q_value",
            "tested_for_multidilution_enrichment",
            "multidilution_odds_ratio_vs_rest",
            "multidilution_fisher_p_value",
            "multidilution_fisher_q_value",
        ],
        rows=phage_summary.to_dict(orient="records"),
    )
    write_csv(
        args.output_dir / "bacterial_subgroup_dilution_response_summary.csv",
        fieldnames=[
            "field_name",
            "subgroup_value",
            "n_resolved_pairs",
            "n_lytic_pairs",
            "lytic_pair_rate",
            "n_high_potency_pairs",
            "n_low_potency_pairs",
            "high_potency_rate",
            "n_multidilution_pairs",
            "n_single_dilution_pairs",
            "multi_dilution_support_rate",
            "mean_potency_rank",
            "median_potency_rank",
            "best_dilution_0_count",
            "best_dilution_minus1_count",
            "best_dilution_minus2_count",
            "best_dilution_minus4_count",
            "tested_for_high_potency_enrichment",
            "high_potency_odds_ratio_vs_rest",
            "high_potency_fisher_p_value",
            "high_potency_fisher_q_value",
            "tested_for_multidilution_enrichment",
            "multidilution_odds_ratio_vs_rest",
            "multidilution_fisher_p_value",
            "multidilution_fisher_q_value",
        ],
        rows=subgroup_summary.to_dict(orient="records"),
    )
    write_json(args.output_dir / "tb05_summary.json", manifest)

    print(
        f"TB05 summary: lytic_pairs={manifest['n_lytic_pairs']}, "
        f"high_potency_rate={manifest['overall_high_potency_rate']:.3f}, "
        f"multi_dilution_support_rate={manifest['overall_multi_dilution_support_rate']:.3f}"
    )
    for row in manifest["top_high_potency_phages"][:5]:
        print(
            f"  phage={row['phage']} high_potency_rate={row['high_potency_rate']:.3f} "
            f"multi_rate={row['multi_dilution_support_rate']:.3f} n={int(row['n_lytic_pairs'])}"
        )


if __name__ == "__main__":
    main()
