#!/usr/bin/env python3
"""GT05: CatBoost comparison on the three-layer feature set.

Replaces LightGBM with CatBoost (GenoPHI-optimal algorithm) using the GT03
RFE-selected features. CatBoost handles categoricals natively — no one-hot
encoding needed for phylogroup, serotype, ST.

Runs Optuna HPO (~50 trials) over CatBoost-specific params, then evaluates
best params vs GT03 LightGBM default and GT04 tuned LightGBM on ST03 holdout.

Usage:
    python -m lyzortx.pipeline.autoresearch.gt05_catboost --device-type cpu
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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch.candidate_replay import (
    bootstrap_holdout_metric_cis,
    build_st03_training_frame,
    load_module_from_path,
    load_st03_holdout_frame,
    summarize_seed_metrics,
)
from lyzortx.pipeline.autoresearch.gt03_eval import (
    LGBM_PARAMS as GT03_DEFAULT_PARAMS,
    apply_rfe,
)
from lyzortx.pipeline.autoresearch.gt04_hpo import build_all_gates_design

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")
DEFAULT_CANDIDATE_DIR = Path("lyzortx/autoresearch")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/gt05_catboost")

SEEDS = [7, 42, 123]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
CV_FOLDS = 5


def label_encode_categoricals(
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Label-encode string categoricals for CatBoost and return cat_feature indices.

    CatBoost accepts string categoricals directly, but they must be str type
    (not mixed). Returns copies with categoricals cast to str and the column
    indices for cat_features parameter.
    """
    train_copy = train_design.copy()
    holdout_copy = holdout_design.copy()
    for col in categorical_columns:
        train_copy[col] = train_copy[col].astype(str).fillna("__missing__")
        holdout_copy[col] = holdout_copy[col].astype(str).fillna("__missing__")
    return train_copy, holdout_copy, categorical_columns


def run_catboost_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    cat_features: list[str],
    n_trials: int,
) -> dict[str, Any]:
    """Run Optuna HPO over CatBoost hyperparameters."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
                cat_features=cat_features,
            )
            train_pool = Pool(
                X_train.iloc[train_idx],
                y_train[train_idx],
                weight=sample_weight[train_idx],
                cat_features=cat_features,
            )
            val_pool = Pool(
                X_train.iloc[val_idx],
                y_train[val_idx],
                cat_features=cat_features,
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=0)
            val_pred = model.predict_proba(val_pool)[:, 1]

            from sklearn.metrics import roc_auc_score

            scores.append(roc_auc_score(y_train[val_idx], val_pred))

        return np.mean(scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="gt05_catboost_hpo")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    LOGGER.info(
        "CatBoost Optuna best trial: AUC=%.4f, params=%s",
        study.best_value,
        json.dumps(study.best_params, indent=2),
    )
    return study.best_params


def evaluate_catboost_on_holdout(
    *,
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    feature_columns: list[str],
    cat_features: list[str],
    params: dict[str, Any],
    arm_id: str,
) -> list[dict[str, object]]:
    """Train CatBoost with given params, predict on holdout."""
    all_rows: list[dict[str, object]] = []
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    for seed in SEEDS:
        LOGGER.info("Arm %s seed %d", arm_id, seed)
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
            cat_features=cat_features,
        )
        train_pool = Pool(
            train_design[feature_columns],
            y_train,
            weight=sample_weight,
            cat_features=cat_features,
        )
        model.fit(train_pool, verbose=0)
        predictions = model.predict_proba(Pool(holdout_design[feature_columns], cat_features=cat_features))[:, 1]

        # Log feature importance.
        imp = model.get_feature_importance()
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


def evaluate_lgbm_on_holdout(
    *,
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    arm_id: str,
    device_type: str,
) -> list[dict[str, object]]:
    """Train LightGBM baseline for comparison."""
    from lightgbm import LGBMClassifier

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


def run_catboost_eval(
    *,
    candidate_module: ModuleType,
    context: Any,
    device_type: str,
    output_dir: Path,
    n_trials: int,
) -> None:
    """Run CatBoost HPO, then compare to LightGBM on holdout."""
    holdout_frame = load_st03_holdout_frame()
    training_frame = build_st03_training_frame()
    LOGGER.info("ST03 split: %d training, %d holdout rows", len(training_frame), len(holdout_frame))

    train_design, holdout_design, feature_columns, categorical_columns = build_all_gates_design(
        candidate_module=candidate_module,
        context=context,
        training_frame=training_frame,
        holdout_frame=holdout_frame,
    )
    LOGGER.info("All-gates design: %d features, %d training pairs", len(feature_columns), len(train_design))

    # Apply RFE (same as GT03).
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)
    rfe_features = apply_rfe(train_design, feature_columns, categorical_columns, y_train, seed=42)
    rfe_categorical = [c for c in categorical_columns if c in rfe_features]
    LOGGER.info("RFE selected %d features (%d categorical)", len(rfe_features), len(rfe_categorical))

    # Prepare CatBoost-compatible data (string categoricals).
    train_cb, holdout_cb, _ = label_encode_categoricals(train_design, holdout_design, rfe_categorical)
    sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    # CatBoost HPO.
    LOGGER.info("Starting CatBoost Optuna HPO with %d trials...", n_trials)
    best_cb_params = run_catboost_optuna(
        X_train=train_cb[rfe_features],
        y_train=y_train,
        sample_weight=sample_weight,
        cat_features=rfe_categorical,
        n_trials=n_trials,
    )

    # Evaluate: LightGBM default vs CatBoost tuned.
    all_rows: list[dict[str, object]] = []

    LOGGER.info("=== Arm: lgbm_gt03_default ===")
    all_rows.extend(
        evaluate_lgbm_on_holdout(
            train_design=train_design,
            holdout_design=holdout_design,
            feature_columns=rfe_features,
            categorical_columns=rfe_categorical,
            params=GT03_DEFAULT_PARAMS,
            arm_id="lgbm_gt03_default",
            device_type=device_type,
        )
    )

    LOGGER.info("=== Arm: catboost_tuned ===")
    all_rows.extend(
        evaluate_catboost_on_holdout(
            train_design=train_cb,
            holdout_design=holdout_cb,
            feature_columns=rfe_features,
            cat_features=rfe_categorical,
            params=best_cb_params,
            arm_id="catboost_tuned",
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
        baseline_arm_id="lgbm_gt03_default",
    )

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_predictions.csv", index=False)
    with open(output_dir / "best_catboost_params.json", "w", encoding="utf-8") as f:
        json.dump(best_cb_params, f, indent=2)

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
    LOGGER.info("GT05 CatBoost Results")
    LOGGER.info("=" * 60)
    LOGGER.info("Best CatBoost params: %s", json.dumps(best_cb_params, indent=2))
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
        if "__delta_vs_lgbm_gt03_default" not in arm_id:
            continue
        auc = ci_dict.get("holdout_roc_auc")
        if auc and auc.ci_low is not None:
            LOGGER.info("  Delta (CatBoost vs LightGBM): [%+.4f, %+.4f]", auc.ci_low, auc.ci_high)

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
    LOGGER.info("GT05 CatBoost comparison starting at %s", datetime.now(timezone.utc).isoformat())

    candidate_module = load_module_from_path("gt05_candidate", args.candidate_dir / "train.py")
    context = candidate_module.load_and_validate_cache(cache_dir=args.cache_dir, include_host_defense=True)

    run_catboost_eval(
        candidate_module=candidate_module,
        context=context,
        device_type=args.device_type,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
