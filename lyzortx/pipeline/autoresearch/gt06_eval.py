#!/usr/bin/env python3
"""GT06: GenoPHI k-mer receptor prediction to strengthen Gate 2.

Compares three arms on ST03 holdout:
  1. gt03_baseline — all_gates with genus-level Gate 2 (8/96 OMP phages)
  2. kmer_gate2   — all_gates with k-mer-based Gate 2 (39/96 OMP phages)
  3. kmer_gate2_only — k-mer Gate 2 replacing genus Gate 2 entirely

Usage:
    python -m lyzortx.pipeline.autoresearch.gt06_eval --device-type cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch.candidate_replay import (
    bootstrap_holdout_metric_cis,
    build_st03_training_frame,
    load_module_from_path,
    load_st03_holdout_frame,
    summarize_seed_metrics,
)
from lyzortx.pipeline.autoresearch.derive_pairwise_depo_capsule_features import (
    compute_pairwise_depo_capsule_features,
)
from lyzortx.pipeline.autoresearch.derive_pairwise_receptor_omp_features import (
    compute_pairwise_receptor_omp_features,
)
from lyzortx.pipeline.autoresearch.derive_pairwise_receptor_omp_kmer_features import (
    compute_pairwise_receptor_omp_kmer_features,
)
from lyzortx.pipeline.autoresearch.gt03_eval import (
    LGBM_PARAMS,
    apply_rfe,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")
DEFAULT_CANDIDATE_DIR = Path("lyzortx/autoresearch")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/gt06_eval")

SEEDS = [7, 42, 123]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42


def build_design_with_gate2_variant(
    *,
    candidate_module: ModuleType,
    context: Any,
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    gate2_variant: str,  # "genus", "kmer", or "both"
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build all-gates design with a specific Gate 2 variant."""
    host_slots = ["host_surface", "host_typing", "host_stats", "host_defense"]
    phage_slots = ["phage_projection", "phage_stats"]

    host_table = candidate_module.build_entity_feature_table(
        context.slot_artifacts, slot_names=host_slots, entity_key="bacteria"
    )
    phage_table = candidate_module.build_entity_feature_table(
        context.slot_artifacts, slot_names=phage_slots, entity_key="phage"
    )

    host_typed, _, host_categorical = candidate_module.type_entity_features(host_table, "bacteria")
    phage_typed, _, phage_categorical = candidate_module.type_entity_features(phage_table, "phage")

    train_design = candidate_module.build_raw_pair_design_matrix(
        training_frame, host_features=host_typed, phage_features=phage_typed
    )
    holdout_design = candidate_module.build_raw_pair_design_matrix(
        holdout_frame, host_features=host_typed, phage_features=phage_typed
    )

    # Gate 1: depolymerase × capsule (always included).
    compute_pairwise_depo_capsule_features(train_design)
    compute_pairwise_depo_capsule_features(holdout_design)

    # Gate 2: receptor × OMP — variant determines which method.
    prefixes = list(f"{s}__" for s in host_slots + phage_slots) + ["pair_depo_capsule__"]

    if gate2_variant in ("genus", "both"):
        compute_pairwise_receptor_omp_features(train_design)
        compute_pairwise_receptor_omp_features(holdout_design)
        prefixes.append("pair_receptor_omp__")

    if gate2_variant in ("kmer", "both"):
        compute_pairwise_receptor_omp_kmer_features(train_design)
        compute_pairwise_receptor_omp_kmer_features(holdout_design)
        prefixes.append("pair_receptor_omp_kmer__")

    feature_columns = [col for col in train_design.columns if col.startswith(tuple(prefixes))]
    categorical_columns = [col for col in (host_categorical + phage_categorical) if col in feature_columns]

    return train_design, holdout_design, feature_columns, categorical_columns


def evaluate_arm(
    *,
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    arm_id: str,
    device_type: str,
) -> list[dict[str, object]]:
    """Train LightGBM with RFE, evaluate on holdout for 3 seeds."""
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    rfe_features = apply_rfe(train_design, feature_columns, categorical_columns, y_train, seed=42)
    rfe_categorical = [c for c in categorical_columns if c in rfe_features]
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    LOGGER.info("Arm %s: %d features after RFE (from %d)", arm_id, len(rfe_features), len(feature_columns))

    all_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        LOGGER.info("Arm %s seed %d", arm_id, seed)
        estimator = LGBMClassifier(
            **LGBM_PARAMS,
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            device_type=device_type,
            **({"deterministic": True, "force_col_wise": True} if device_type == "cpu" else {}),
        )
        estimator.fit(
            train_design[rfe_features],
            y_train,
            sample_weight=sample_weight,
            categorical_feature=rfe_categorical,
        )
        predictions = estimator.predict_proba(holdout_design[rfe_features])[:, 1]

        # Feature importance by slot.
        imp = estimator.feature_importances_
        slot_imp: dict[str, float] = {}
        for col, val in zip(rfe_features, imp):
            slot = col.split("__")[0]
            slot_imp[slot] = slot_imp.get(slot, 0) + val
        total_imp = sum(slot_imp.values()) or 1
        parts = [f"{s}={v / total_imp * 100:.1f}%" for s, v in sorted(slot_imp.items(), key=lambda x: -x[1])]
        LOGGER.info("Feature importance: %s", ", ".join(parts))

        for row, prob in zip(
            holdout_design.loc[:, ["pair_id", "bacteria", "phage", "label_any_lysis"]].to_dict(orient="records"),
            predictions,
        ):
            all_rows.append(
                {
                    "arm_id": arm_id,
                    "seed": seed,
                    "pair_id": str(row["pair_id"]),
                    "bacteria": str(row["bacteria"]),
                    "phage": str(row["phage"]),
                    "label_hard_any_lysis": int(row["label_any_lysis"]),
                    "predicted_probability": round(float(prob), 6),
                }
            )

        metrics = summarize_seed_metrics(all_rows[-len(holdout_design) :])
        LOGGER.info(
            "Arm %s seed %d: AUC=%.4f, top-3=%.1f%%, Brier=%.4f",
            arm_id,
            seed,
            metrics.get("holdout_roc_auc", 0),
            metrics.get("holdout_top3_hit_rate_all_strains", 0) * 100,
            metrics.get("holdout_brier_score", 0),
        )
    return all_rows


def run_gt06_eval(
    *,
    candidate_module: ModuleType,
    context: Any,
    device_type: str,
    output_dir: Path,
) -> None:
    holdout_frame = load_st03_holdout_frame()
    training_frame = build_st03_training_frame()
    LOGGER.info("ST03 split: %d training, %d holdout rows", len(training_frame), len(holdout_frame))

    arms = [
        ("gt03_genus_gate2", "genus"),
        ("kmer_gate2_only", "kmer"),
        ("both_gate2", "both"),
    ]

    all_rows: list[dict[str, object]] = []
    for arm_id, gate2_variant in arms:
        LOGGER.info("=== Arm: %s (gate2=%s) ===", arm_id, gate2_variant)
        train_design, holdout_design, features, categoricals = build_design_with_gate2_variant(
            candidate_module=candidate_module,
            context=context,
            training_frame=training_frame,
            holdout_frame=holdout_frame,
            gate2_variant=gate2_variant,
        )
        LOGGER.info("Arm %s: %d total features", arm_id, len(features))
        all_rows.extend(
            evaluate_arm(
                train_design=train_design,
                holdout_design=holdout_design,
                feature_columns=features,
                categorical_columns=categoricals,
                arm_id=arm_id,
                device_type=device_type,
            )
        )

    # Aggregate and bootstrap.
    df = pd.DataFrame(all_rows)
    aggregated = (
        df.groupby(["arm_id", "pair_id", "bacteria", "phage", "label_hard_any_lysis"], as_index=False)[
            "predicted_probability"
        ]
        .mean()
        .sort_values(["arm_id", "bacteria", "phage"])
    )

    holdout_rows_by_arm: dict[str, list[dict[str, object]]] = {}
    for _, row in aggregated.iterrows():
        holdout_rows_by_arm.setdefault(str(row["arm_id"]), []).append(dict(row))

    bootstrap_results = bootstrap_holdout_metric_cis(
        holdout_rows_by_arm,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_random_state=BOOTSTRAP_RANDOM_STATE,
        baseline_arm_id="gt03_genus_gate2",
    )

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_predictions.csv", index=False)

    bootstrap_json = {}
    for arm_id, ci_dict in bootstrap_results.items():
        bootstrap_json[arm_id] = {
            metric: {"point_estimate": ci.point_estimate, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
            for metric, ci in ci_dict.items()
        }
    with open(output_dir / "bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_json, f, indent=2)

    # Summary.
    LOGGER.info("=" * 60)
    LOGGER.info("GT06 Results")
    LOGGER.info("=" * 60)
    for arm_id, ci_dict in bootstrap_results.items():
        if "__delta_vs_" in arm_id:
            continue
        auc = ci_dict.get("holdout_roc_auc")
        top3 = ci_dict.get("holdout_top3_hit_rate_all_strains")
        brier = ci_dict.get("holdout_brier_score")
        if auc and auc.point_estimate is not None:
            LOGGER.info(
                "  %s: AUC=%.4f [%.3f, %.3f], top-3=%.1f%%, Brier=%.4f",
                arm_id,
                auc.point_estimate,
                auc.ci_low or 0,
                auc.ci_high or 0,
                (top3.point_estimate or 0) * 100,
                brier.point_estimate or 0,
            )
    for arm_id, ci_dict in bootstrap_results.items():
        if "__delta_vs_gt03_genus_gate2" not in arm_id:
            continue
        auc = ci_dict.get("holdout_roc_auc")
        if auc and auc.ci_low is not None:
            clean = arm_id.replace("__delta_vs_gt03_genus_gate2", "")
            LOGGER.info("  Delta %s vs genus: [%+.4f, %+.4f]", clean, auc.ci_low, auc.ci_high)

    LOGGER.info("Results saved to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-type", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    LOGGER.info("GT06 k-mer receptor eval starting at %s", datetime.now(timezone.utc).isoformat())

    candidate_module = load_module_from_path("gt06_candidate", args.candidate_dir / "train.py")
    context = candidate_module.load_and_validate_cache(cache_dir=args.cache_dir, include_host_defense=True)

    run_gt06_eval(
        candidate_module=candidate_module,
        context=context,
        device_type=args.device_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
