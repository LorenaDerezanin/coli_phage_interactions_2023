#!/usr/bin/env python3
"""GT04: HPO with Optuna on the three-layer feature set.

Optimizes LightGBM hyperparameters via Optuna (~50 trials) using 5-fold
stratified CV on training data, then evaluates the best params on the
ST03 holdout with 3 seeds and 1000 bootstrap resamples.

Compares to the GT03 default-param baseline (0.823 AUC all_gates_rfe).

Usage:
    python -m lyzortx.pipeline.autoresearch.gt04_hpo --device-type cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
from lyzortx.pipeline.autoresearch.gt03_eval import (
    LGBM_PARAMS as GT03_DEFAULT_PARAMS,
    apply_rfe,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")
DEFAULT_CANDIDATE_DIR = Path("lyzortx/autoresearch")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/gt04_hpo")

SEEDS = [7, 42, 123]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
CV_FOLDS = 5


def build_all_gates_design(
    *,
    candidate_module: ModuleType,
    context: Any,
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build all-gates design matrices (baseline + defense + depo×capsule + receptor×OMP)."""
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

    compute_pairwise_depo_capsule_features(train_design)
    compute_pairwise_depo_capsule_features(holdout_design)
    compute_pairwise_receptor_omp_features(train_design)
    compute_pairwise_receptor_omp_features(holdout_design)

    all_slot_names = host_slots + phage_slots
    prefixes = tuple(f"{s}__" for s in all_slot_names) + ("pair_depo_capsule__", "pair_receptor_omp__")
    feature_columns = [col for col in train_design.columns if col.startswith(prefixes)]
    categorical_columns = [col for col in (host_categorical + phage_categorical) if col in feature_columns]

    return train_design, holdout_design, feature_columns, categorical_columns


def run_optuna_hpo(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    n_trials: int,
    device_type: str,
) -> dict[str, Any]:
    """Run Optuna HPO over LightGBM hyperparameters. Returns best params."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        estimator = LGBMClassifier(
            **params,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
            device_type=device_type,
            **({"deterministic": True, "force_col_wise": True} if device_type == "cpu" else {}),
        )
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv, scoring="roc_auc", params={"sample_weight": sample_weight}
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="gt04_lgbm_hpo")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    LOGGER.info(
        "Optuna best trial: AUC=%.4f, params=%s",
        study.best_value,
        json.dumps(study.best_params, indent=2),
    )
    return study.best_params


def evaluate_on_holdout(
    *,
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    arm_id: str,
    device_type: str,
) -> list[dict[str, object]]:
    """Train with given params on full training set, predict on holdout. Returns all seed rows."""
    all_rows: list[dict[str, object]] = []
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    for seed in SEEDS:
        LOGGER.info("Arm %s seed %d", arm_id, seed)
        estimator = LGBMClassifier(
            **params,
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            device_type=device_type,
            **({"deterministic": True, "force_col_wise": True} if device_type == "cpu" else {}),
        )
        estimator.fit(
            train_design[feature_columns],
            y_train,
            sample_weight=sample_weight,
            categorical_feature=categorical_columns,
        )
        predictions = estimator.predict_proba(holdout_design[feature_columns])[:, 1]

        # Log feature importance.
        imp = estimator.feature_importances_
        slot_imp: dict[str, float] = {}
        for col, val in zip(feature_columns, imp):
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


def run_hpo_eval(
    *,
    candidate_module: ModuleType,
    context: Any,
    device_type: str,
    output_dir: Path,
    n_trials: int,
) -> None:
    """Run HPO, then evaluate default vs tuned params on holdout."""
    holdout_frame = load_st03_holdout_frame()
    training_frame = build_st03_training_frame()
    LOGGER.info("ST03 split: %d training, %d holdout rows", len(training_frame), len(holdout_frame))

    # Build all-gates design matrices.
    train_design, holdout_design, feature_columns, categorical_columns = build_all_gates_design(
        candidate_module=candidate_module,
        context=context,
        training_frame=training_frame,
        holdout_frame=holdout_frame,
    )
    LOGGER.info("All-gates design: %d features, %d training pairs", len(feature_columns), len(train_design))

    # Apply RFE (same as GT03 all_gates_rfe).
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    rfe_features = apply_rfe(train_design, feature_columns, categorical_columns, y_train, seed=42)
    LOGGER.info("RFE selected %d features", len(rfe_features))

    rfe_categorical = [c for c in categorical_columns if c in rfe_features]
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    # Run Optuna HPO on RFE-selected features.
    LOGGER.info("Starting Optuna HPO with %d trials...", n_trials)
    best_params = run_optuna_hpo(
        X_train=train_design[rfe_features],
        y_train=y_train,
        sample_weight=sample_weight,
        n_trials=n_trials,
        device_type=device_type,
    )

    # Evaluate: GT03 default params vs tuned params.
    all_rows: list[dict[str, object]] = []

    LOGGER.info("=== Arm: gt03_default ===")
    all_rows.extend(
        evaluate_on_holdout(
            train_design=train_design,
            holdout_design=holdout_design,
            feature_columns=rfe_features,
            categorical_columns=rfe_categorical,
            params=GT03_DEFAULT_PARAMS,
            arm_id="gt03_default",
            device_type=device_type,
        )
    )

    LOGGER.info("=== Arm: optuna_tuned ===")
    all_rows.extend(
        evaluate_on_holdout(
            train_design=train_design,
            holdout_design=holdout_design,
            feature_columns=rfe_features,
            categorical_columns=rfe_categorical,
            params=best_params,
            arm_id="optuna_tuned",
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
        arm_id = str(row["arm_id"])
        holdout_rows_by_arm.setdefault(arm_id, []).append(dict(row))

    bootstrap_results = bootstrap_holdout_metric_cis(
        holdout_rows_by_arm,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_random_state=BOOTSTRAP_RANDOM_STATE,
        baseline_arm_id="gt03_default",
    )

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_predictions.csv", index=False)
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    bootstrap_json = {}
    for arm_id, ci_dict in bootstrap_results.items():
        bootstrap_json[arm_id] = {
            metric: {"point_estimate": ci.point_estimate, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
            for metric, ci in ci_dict.items()
        }
    with open(output_dir / "bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_json, f, indent=2)

    # Print summary.
    LOGGER.info("=" * 60)
    LOGGER.info("GT04 HPO Results")
    LOGGER.info("=" * 60)
    LOGGER.info("Best Optuna params: %s", json.dumps(best_params, indent=2))
    LOGGER.info("GT03 default params: %s", json.dumps(GT03_DEFAULT_PARAMS, indent=2))
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
        if "__delta_vs_gt03_default" not in arm_id:
            continue
        auc = ci_dict.get("holdout_roc_auc")
        if auc and auc.ci_low is not None:
            LOGGER.info("  Delta (tuned vs default): [%+.4f, %+.4f]", auc.ci_low, auc.ci_high)

    LOGGER.info("Results saved to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-type", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    LOGGER.info("GT04 HPO starting at %s", datetime.now(timezone.utc).isoformat())

    candidate_module = load_module_from_path("gt04_candidate", args.candidate_dir / "train.py")
    context = candidate_module.load_and_validate_cache(cache_dir=args.cache_dir, include_host_defense=True)

    run_hpo_eval(
        candidate_module=candidate_module,
        context=context,
        device_type=args.device_type,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
