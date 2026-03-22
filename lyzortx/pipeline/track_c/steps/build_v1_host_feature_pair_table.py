#!/usr/bin/env python3
"""TC04: Merge Track C host feature blocks and integrate them into a v1 pair table."""

from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import parse_float, read_csv_rows, safe_round
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    CATEGORICAL_FEATURE_COLUMNS as V0_CATEGORICAL_FEATURE_COLUMNS,
)
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    NUMERIC_FEATURE_COLUMNS as V0_NUMERIC_FEATURE_COLUMNS,
)
from lyzortx.pipeline.track_c.steps.build_extended_host_surface_feature_block import (
    build_capsule_index,
    build_feature_rows as build_extended_surface_feature_rows,
    build_lps_core_index,
    read_delimited_rows as read_extended_rows,
)
from lyzortx.pipeline.track_c.steps.build_omp_receptor_variant_feature_block import (
    RECEPTOR_COLUMNS,
    build_feature_rows as build_omp_feature_rows,
    read_delimited_rows as read_omp_rows,
    select_feature_categories,
    summarize_receptor_categories,
)

DEFENSE_DERIVED_COLUMNS: Tuple[str, ...] = (
    "host_defense_diversity",
    "host_defense_has_crispr",
    "host_defense_abi_burden",
)
EXTENDED_CATEGORICAL_COLUMNS: Tuple[str, ...] = (
    "host_surface_klebsiella_capsule_type",
    "host_surface_lps_core_type",
)
MISSING_VALUE_SENTINELS = {"", None}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table used as the v0 base table.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments for the LightGBM sanity check.",
    )
    parser.add_argument(
        "--defense-subtypes-path",
        type=Path,
        default=Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"),
        help="Semicolon-delimited defense-system subtype table.",
    )
    parser.add_argument(
        "--receptor-clusters-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"),
        help="Tab-delimited OMP receptor cluster table.",
    )
    parser.add_argument(
        "--umap-path",
        type=Path,
        default=Path("data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv"),
        help="Tab-delimited 8D host UMAP embedding table.",
    )
    parser.add_argument(
        "--capsule-path",
        type=Path,
        default=Path("data/genomics/bacteria/capsules/klebsiella_capsules/kaptive_results_high_hits_cured.txt"),
        help="Tab-delimited capsule typing table.",
    )
    parser.add_argument(
        "--lps-primary-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt"),
        help="Primary tab-delimited LPS core table.",
    )
    parser.add_argument(
        "--lps-supplemental-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_host.txt"),
        help="Supplemental tab-delimited LPS core table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table"),
        help="Directory for generated TC04 artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and manifests.",
    )
    parser.add_argument(
        "--defense-min-present-count",
        type=int,
        default=5,
        help="Minimum strain support required to retain a defense subtype column.",
    )
    parser.add_argument(
        "--defense-max-present-count",
        type=int,
        default=395,
        help="Maximum strain support allowed to retain a defense subtype column.",
    )
    parser.add_argument(
        "--omp-min-cluster-count",
        type=int,
        default=5,
        help="Rare-cluster threshold reused from TC02.",
    )
    parser.add_argument(
        "--omp-max-feature-count",
        type=int,
        default=22,
        help="Feature budget reused from TC02.",
    )
    parser.add_argument(
        "--lightgbm-random-state",
        type=int,
        default=42,
        help="Random seed for the TC04 LightGBM sanity check.",
    )
    parser.add_argument(
        "--allow-no-lift",
        action="store_true",
        help="Do not fail when the v1 LightGBM sanity check does not beat the v0 feature set.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify_token(value: str) -> str:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value.strip())
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"Cannot slugify empty feature token from {value!r}")
    return normalized


def _parse_binary_flag(value: str) -> int:
    normalized = value.strip()
    if normalized in {"", "0", "0.0"}:
        return 0
    parsed = float(normalized)
    if parsed < 0:
        raise ValueError(f"Expected non-negative defense subtype count, found {value!r}")
    return 1 if parsed > 0 else 0


def _is_missing_value(value: object) -> bool:
    return value in MISSING_VALUE_SENTINELS


def build_defense_feature_rows(
    defense_rows: Sequence[Mapping[str, str]],
    *,
    min_present_count: int = 5,
    max_present_count: int = 395,
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, object]]:
    if not defense_rows:
        raise ValueError("Defense subtype rows are empty.")
    if min_present_count < 1:
        raise ValueError("min_present_count must be >= 1")
    if max_present_count < min_present_count:
        raise ValueError("max_present_count must be >= min_present_count")

    subtype_columns = [column for column in defense_rows[0].keys() if column != "bacteria"]
    retained_subtypes: List[str] = []
    support_counts: Dict[str, int] = {}
    for column in subtype_columns:
        present_count = sum(_parse_binary_flag(str(row.get(column, ""))) for row in defense_rows)
        support_counts[column] = present_count
        if min_present_count <= present_count <= max_present_count:
            retained_subtypes.append(column)

    kept_column_names = [f"host_defense_subtype_{slugify_token(column)}" for column in retained_subtypes]
    output_rows: List[Dict[str, object]] = []
    for row in sorted(defense_rows, key=lambda item: item["bacteria"]):
        bacteria = row.get("bacteria", "").strip()
        if not bacteria:
            continue

        encoded_row: Dict[str, object] = {"bacteria": bacteria}
        defense_diversity = 0
        abi_burden = 0
        has_crispr = 0
        for subtype in retained_subtypes:
            value = _parse_binary_flag(str(row.get(subtype, "")))
            encoded_row[f"host_defense_subtype_{slugify_token(subtype)}"] = value
            defense_diversity += value
            if subtype.startswith("Abi"):
                abi_burden += value
            if subtype.startswith("CAS_"):
                has_crispr = max(has_crispr, value)

        encoded_row["host_defense_diversity"] = defense_diversity
        encoded_row["host_defense_has_crispr"] = has_crispr
        encoded_row["host_defense_abi_burden"] = abi_burden
        output_rows.append(encoded_row)

    manifest = {
        "host_count": len(output_rows),
        "source_subtype_count": len(subtype_columns),
        "retained_subtype_count": len(retained_subtypes),
        "retained_subtypes": retained_subtypes,
        "dropped_low_support_subtypes": [
            column for column in subtype_columns if support_counts[column] < min_present_count
        ],
        "dropped_high_support_subtypes": [
            column for column in subtype_columns if support_counts[column] > max_present_count
        ],
        "support_counts": dict(sorted(support_counts.items())),
    }
    return output_rows, kept_column_names + list(DEFENSE_DERIVED_COLUMNS), manifest


def _index_rows(rows: Sequence[Mapping[str, object]], *, key: str = "bacteria") -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if value in index:
            raise ValueError(f"Duplicate {key} value {value!r}")
        index[value] = dict(row)
    return index


def merge_host_feature_blocks(
    target_bacteria: Sequence[str],
    *,
    blocks: Mapping[str, Sequence[Mapping[str, object]]],
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[str]]:
    all_columns: List[str] = []
    seen_columns: set[str] = set()
    block_indexes = {name: _index_rows(rows) for name, rows in blocks.items()}

    for block_name, rows in blocks.items():
        if not rows:
            raise ValueError(f"Feature block {block_name!r} is empty.")
        block_columns = [column for column in rows[0].keys() if column != "bacteria"]
        duplicate_columns = sorted(column for column in block_columns if column in seen_columns)
        if duplicate_columns:
            raise ValueError(
                f"Duplicate feature columns detected while merging host blocks: {', '.join(duplicate_columns)}"
            )
        seen_columns.update(block_columns)
        all_columns.extend(block_columns)

    merged_rows: List[Dict[str, object]] = []
    join_audit: Dict[str, object] = {
        "target_host_count": len(target_bacteria),
        "block_summaries": {},
    }

    for bacteria in sorted(target_bacteria):
        merged_row: Dict[str, object] = {"bacteria": bacteria}
        for block_name, block_index in block_indexes.items():
            block_row = block_index.get(bacteria, {})
            for column in [name for name in blocks[block_name][0].keys() if name != "bacteria"]:
                merged_row[column] = block_row.get(column, "")
        merged_rows.append(merged_row)

    for block_name, rows in blocks.items():
        block_index = block_indexes[block_name]
        block_columns = [column for column in rows[0].keys() if column != "bacteria"]
        missing_summary: Dict[str, Dict[str, int]] = {}
        hosts_missing_from_block = sum(1 for bacteria in target_bacteria if bacteria not in block_index)
        for column in block_columns:
            source_missing_on_target = sum(
                1
                for bacteria in target_bacteria
                if bacteria in block_index and _is_missing_value(block_index[bacteria].get(column))
            )
            joined_missing = sum(
                1 for row in merged_rows if _is_missing_value(row[column]) and row["bacteria"] in target_bacteria
            )
            expected_missing = source_missing_on_target + hosts_missing_from_block
            unexpected_missing = joined_missing - expected_missing
            if unexpected_missing != 0:
                raise ValueError(
                    f"Unexpected missingness increase for {block_name}.{column}: "
                    f"joined={joined_missing} expected={expected_missing}"
                )
            missing_summary[column] = {
                "source_missing_on_target_hosts": source_missing_on_target,
                "hosts_absent_from_block": hosts_missing_from_block,
                "joined_missing_after_merge": joined_missing,
                "unexpected_missing_increase": unexpected_missing,
            }

        join_audit["block_summaries"][block_name] = {
            "source_host_count": len(block_index),
            "target_host_overlap_count": sum(1 for bacteria in target_bacteria if bacteria in block_index),
            "target_hosts_missing_from_block": hosts_missing_from_block,
            "column_missingness": missing_summary,
        }

    return merged_rows, join_audit, all_columns


def build_pair_table_rows(
    st02_rows: Sequence[Mapping[str, str]],
    host_matrix_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    host_index = _index_rows(host_matrix_rows)
    output_rows: List[Dict[str, object]] = []
    host_columns = [column for column in host_matrix_rows[0].keys() if column != "bacteria"]

    for row in st02_rows:
        bacteria = row["bacteria"]
        if bacteria not in host_index:
            raise KeyError(f"Missing merged host features for {bacteria}")
        pair_row: Dict[str, object] = dict(row)
        for column in host_columns:
            pair_row[column] = host_index[bacteria][column]
        output_rows.append(pair_row)

    output_rows.sort(key=lambda item: (str(item["bacteria"]), str(item["phage"])))
    return output_rows


def _build_feature_dict(
    row: Mapping[str, object],
    *,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> Dict[str, object]:
    features: Dict[str, object] = {}
    for column in categorical_columns:
        value = row.get(column, "")
        if value not in MISSING_VALUE_SENTINELS:
            features[column] = str(value)
    for column in numeric_columns:
        raw = row.get(column, "")
        if raw in MISSING_VALUE_SENTINELS:
            continue
        if isinstance(raw, (int, float)):
            features[column] = float(raw)
            continue
        parsed = parse_float(str(raw))
        if parsed is not None:
            features[column] = parsed
    return features


def _safe_binary_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "n": float(len(y_true)),
        "positive_rate": safe_round(sum(y_true) / len(y_true)) if y_true else None,
        "average_precision": safe_round(average_precision_score(y_true, y_prob)) if y_true else None,
        "roc_auc": None,
        "brier_score": safe_round(brier_score_loss(y_true, y_prob)) if y_true else None,
        "log_loss": safe_round(log_loss(y_true, y_prob, labels=[0, 1])) if y_true else None,
    }
    if len(set(y_true)) >= 2:
        metrics["roc_auc"] = safe_round(roc_auc_score(y_true, y_prob))
    return metrics


def run_lightgbm_sanity_check(
    pair_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, str]],
    *,
    v1_categorical_columns: Sequence[str],
    v1_numeric_columns: Sequence[str],
    random_state: int = 42,
    allow_no_lift: bool = False,
) -> Dict[str, object]:
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is required for the TC04 sanity check. Add it to requirements.txt and install dependencies."
        ) from exc

    split_by_pair = {row["pair_id"]: row for row in split_rows}
    joined_rows: List[Dict[str, object]] = []
    for row in pair_rows:
        pair_id = str(row["pair_id"])
        if pair_id not in split_by_pair:
            raise KeyError(f"Pair {pair_id} is missing from ST0.3 split assignments")
        merged = dict(row)
        merged.update(split_by_pair[pair_id])
        joined_rows.append(merged)

    train_rows = [
        row
        for row in joined_rows
        if row["split_holdout"] == "train_non_holdout" and str(row["is_hard_trainable"]) == "1"
    ]
    if not train_rows:
        raise ValueError("No non-holdout hard-trainable rows available for the TC04 sanity check.")

    fold_ids = sorted({int(row["split_cv5_fold"]) for row in train_rows if int(row["split_cv5_fold"]) >= 0})
    if len(fold_ids) < 2:
        raise ValueError("The TC04 sanity check requires at least two non-holdout CV folds.")

    v0_categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS)
    v0_numeric_columns = list(V0_NUMERIC_FEATURE_COLUMNS)

    fold_metrics: List[Dict[str, object]] = []
    v0_ap_values: List[float] = []
    v1_ap_values: List[float] = []

    for fold_id in fold_ids:
        train_fold_rows = [row for row in train_rows if int(row["split_cv5_fold"]) != fold_id]
        valid_fold_rows = [row for row in train_rows if int(row["split_cv5_fold"]) == fold_id]
        if not train_fold_rows or not valid_fold_rows:
            raise ValueError(f"Fold {fold_id} does not have both train and validation rows.")

        y_train = [int(str(row["label_hard_any_lysis"])) for row in train_fold_rows]
        y_valid = [int(str(row["label_hard_any_lysis"])) for row in valid_fold_rows]

        v0_vectorizer = DictVectorizer(sparse=True, sort=True)
        X_v0_train = v0_vectorizer.fit_transform(
            [
                _build_feature_dict(
                    row,
                    categorical_columns=v0_categorical_columns,
                    numeric_columns=v0_numeric_columns,
                )
                for row in train_fold_rows
            ]
        )
        X_v0_valid = v0_vectorizer.transform(
            [
                _build_feature_dict(
                    row,
                    categorical_columns=v0_categorical_columns,
                    numeric_columns=v0_numeric_columns,
                )
                for row in valid_fold_rows
            ]
        )

        v1_vectorizer = DictVectorizer(sparse=True, sort=True)
        X_v1_train = v1_vectorizer.fit_transform(
            [
                _build_feature_dict(
                    row,
                    categorical_columns=v1_categorical_columns,
                    numeric_columns=v1_numeric_columns,
                )
                for row in train_fold_rows
            ]
        )
        X_v1_valid = v1_vectorizer.transform(
            [
                _build_feature_dict(
                    row,
                    categorical_columns=v1_categorical_columns,
                    numeric_columns=v1_numeric_columns,
                )
                for row in valid_fold_rows
            ]
        )

        v1_model_kwargs = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 1,
            "min_data_in_bin": 1,
            "class_weight": "balanced",
            "random_state": random_state + fold_id,
            "n_jobs": 1,
            "verbosity": -1,
            "force_col_wise": True,
        }

        v0_model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state + fold_id,
            solver="liblinear",
        )
        v0_model.fit(X_v0_train, y_train)
        v0_prob = [float(value) for value in v0_model.predict_proba(X_v0_valid)[:, 1]]

        v1_model = LGBMClassifier(**v1_model_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            v1_model.fit(X_v1_train, y_train)
            v1_prob = [float(value) for value in v1_model.predict_proba(X_v1_valid)[:, 1]]

        v0_metrics = _safe_binary_metrics(y_valid, v0_prob)
        v1_metrics = _safe_binary_metrics(y_valid, v1_prob)
        v0_ap = float(v0_metrics["average_precision"] or 0.0)
        v1_ap = float(v1_metrics["average_precision"] or 0.0)
        v0_ap_values.append(v0_ap)
        v1_ap_values.append(v1_ap)

        fold_metrics.append(
            {
                "fold_id": fold_id,
                "train_rows": len(train_fold_rows),
                "validation_rows": len(valid_fold_rows),
                "validation_positive_count": int(sum(y_valid)),
                "v0": {
                    "model_type": "logistic_regression",
                    "n_vectorized_features": len(v0_vectorizer.get_feature_names_out()),
                    "metrics": v0_metrics,
                },
                "v1": {
                    "model_type": "lightgbm",
                    "n_vectorized_features": len(v1_vectorizer.get_feature_names_out()),
                    "metrics": v1_metrics,
                },
                "delta_average_precision": safe_round(v1_ap - v0_ap),
            }
        )

    v0_mean_ap = safe_round(sum(v0_ap_values) / len(v0_ap_values))
    v1_mean_ap = safe_round(sum(v1_ap_values) / len(v1_ap_values))
    ap_lift = safe_round(v1_mean_ap - v0_mean_ap)
    if ap_lift <= 0 and not allow_no_lift:
        raise ValueError(
            f"TC04 sanity check failed: v1 mean average precision {v1_mean_ap} did not beat v0 {v0_mean_ap}."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_non_holdout_row_count": len(train_rows),
        "fold_ids": fold_ids,
        "primary_metric": "average_precision",
        "comparison": "v0 logistic regression baseline vs v1 LightGBM on non-holdout CV folds",
        "v0_feature_space": {
            "categorical_columns": v0_categorical_columns,
            "numeric_columns": v0_numeric_columns,
        },
        "v1_feature_space": {
            "categorical_columns": list(v1_categorical_columns),
            "numeric_columns": list(v1_numeric_columns),
        },
        "fold_metrics": fold_metrics,
        "summary": {
            "v0_mean_average_precision": v0_mean_ap,
            "v1_mean_average_precision": v1_mean_ap,
            "average_precision_lift": ap_lift,
            "lift_confirmed": ap_lift > 0,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    st03_rows = read_csv_rows(args.st03_split_assignments_path)
    if not st02_rows:
        raise ValueError(f"No rows found in {args.st02_pair_table_path}")
    if not st03_rows:
        raise ValueError(f"No rows found in {args.st03_split_assignments_path}")

    target_bacteria = sorted({row["bacteria"] for row in st02_rows})

    defense_rows_raw = read_extended_rows(args.defense_subtypes_path, ";")
    defense_rows, defense_numeric_columns, defense_manifest = build_defense_feature_rows(
        defense_rows_raw,
        min_present_count=args.defense_min_present_count,
        max_present_count=args.defense_max_present_count,
    )

    omp_rows_raw = read_omp_rows(args.receptor_clusters_path, "\t")
    omp_summaries = summarize_receptor_categories(
        omp_rows_raw,
        receptor_columns=RECEPTOR_COLUMNS,
        min_cluster_count=args.omp_min_cluster_count,
    )
    omp_selected_categories = select_feature_categories(
        omp_summaries,
        receptor_columns=RECEPTOR_COLUMNS,
        max_feature_count=args.omp_max_feature_count,
    )
    omp_rows = build_omp_feature_rows(omp_rows_raw, omp_summaries, omp_selected_categories)
    omp_numeric_columns = [column for column in omp_rows[0].keys() if column != "bacteria"]

    umap_rows = read_extended_rows(args.umap_path, "\t")
    capsule_rows = read_extended_rows(args.capsule_path, "\t")
    lps_primary_rows = read_extended_rows(args.lps_primary_path, "\t")
    lps_supplemental_rows = read_extended_rows(args.lps_supplemental_path, "\t")
    capsule_index = build_capsule_index(capsule_rows)
    lps_index = build_lps_core_index(lps_primary_rows, lps_supplemental_rows)
    extended_rows = build_extended_surface_feature_rows(
        umap_rows=umap_rows,
        capsule_index=capsule_index,
        lps_index=lps_index,
    )
    extended_numeric_columns = [
        column
        for column in extended_rows[0].keys()
        if column != "bacteria" and column not in EXTENDED_CATEGORICAL_COLUMNS
    ]

    host_matrix_rows, join_audit, merged_host_columns = merge_host_feature_blocks(
        target_bacteria,
        blocks={
            "defense_subtypes": defense_rows,
            "omp_receptor_variants": omp_rows,
            "extended_surface": extended_rows,
        },
    )
    pair_table_rows = build_pair_table_rows(st02_rows, host_matrix_rows)

    v1_categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS) + list(EXTENDED_CATEGORICAL_COLUMNS)
    v1_numeric_columns = (
        list(V0_NUMERIC_FEATURE_COLUMNS)
        + list(defense_numeric_columns)
        + list(omp_numeric_columns)
        + list(extended_numeric_columns)
    )
    sanity = run_lightgbm_sanity_check(
        pair_table_rows,
        st03_rows,
        v1_categorical_columns=v1_categorical_columns,
        v1_numeric_columns=v1_numeric_columns,
        random_state=args.lightgbm_random_state,
        allow_no_lift=args.allow_no_lift,
    )

    host_matrix_output_path = args.output_dir / f"host_feature_matrix_{args.version}.csv"
    pair_table_output_path = args.output_dir / f"pair_table_{args.version}.csv"
    join_audit_output_path = args.output_dir / f"host_feature_join_audit_{args.version}.json"
    manifest_output_path = args.output_dir / f"pair_table_manifest_{args.version}.json"
    sanity_output_path = args.output_dir / f"lightgbm_sanity_check_{args.version}.json"

    write_csv(
        host_matrix_output_path,
        ["bacteria"] + merged_host_columns,
        host_matrix_rows,
    )
    write_csv(
        pair_table_output_path,
        list(pair_table_rows[0].keys()),
        pair_table_rows,
    )
    write_json(join_audit_output_path, join_audit)
    write_json(
        manifest_output_path,
        {
            "version": args.version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(pair_table_rows),
            "host_count": len(host_matrix_rows),
            "distinct_bacteria_count": len(target_bacteria),
            "distinct_phage_count": len({row["phage"] for row in pair_table_rows}),
            "host_feature_count": len(merged_host_columns),
            "column_groups": {
                "v0_base_columns": list(st02_rows[0].keys()),
                "track_c_host_columns": merged_host_columns,
                "categorical_columns_for_v1_models": v1_categorical_columns,
                "numeric_columns_for_v1_models": v1_numeric_columns,
            },
            "inputs": {
                "st02_pair_table": {
                    "path": str(args.st02_pair_table_path),
                    "sha256": _sha256(args.st02_pair_table_path),
                },
                "st03_split_assignments": {
                    "path": str(args.st03_split_assignments_path),
                    "sha256": _sha256(args.st03_split_assignments_path),
                },
                "defense_subtypes": {
                    "path": str(args.defense_subtypes_path),
                    "sha256": _sha256(args.defense_subtypes_path),
                    "summary": defense_manifest,
                },
                "receptor_clusters": {
                    "path": str(args.receptor_clusters_path),
                    "sha256": _sha256(args.receptor_clusters_path),
                },
                "umap_path": {"path": str(args.umap_path), "sha256": _sha256(args.umap_path)},
                "capsule_path": {"path": str(args.capsule_path), "sha256": _sha256(args.capsule_path)},
                "lps_primary_path": {"path": str(args.lps_primary_path), "sha256": _sha256(args.lps_primary_path)},
                "lps_supplemental_path": {
                    "path": str(args.lps_supplemental_path),
                    "sha256": _sha256(args.lps_supplemental_path),
                },
            },
            "outputs": {
                "host_feature_matrix_csv": str(host_matrix_output_path),
                "pair_table_csv": str(pair_table_output_path),
                "join_audit_json": str(join_audit_output_path),
                "lightgbm_sanity_json": str(sanity_output_path),
            },
            "notes": {
                "host_contract": "Rows are anchored to the interaction/ST0.2 host panel.",
                "scope_choice": (
                    "The older receptor_surface_feature_block is not merged here because it duplicates TC02/TC03 "
                    "signal and reintroduces a narrower 369-host contract."
                ),
            },
        },
    )
    write_json(sanity_output_path, sanity)

    print("TC04 completed.")
    print(f"- Host matrix rows: {len(host_matrix_rows)}")
    print(f"- Host feature count: {len(merged_host_columns)}")
    print(f"- Pair table rows: {len(pair_table_rows)}")
    print(f"- Mean AP lift over v0: {sanity['summary']['average_precision_lift']}")
    print(f"- Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
