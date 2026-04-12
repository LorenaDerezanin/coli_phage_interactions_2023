#!/usr/bin/env python3
"""GT03: Three-layer integration with RFE and inverse-frequency class weighting.

Runs a multi-arm ablation on the ST03 holdout comparing:
  1. baseline         — 5-slot AUTORESEARCH (reproduce 0.810 AUC)
  2. +gate1           — baseline + depolymerase × capsule cross-terms
  3. +gate2           — baseline + receptor × OMP cross-terms
  4. +gate3           — baseline + host defense (79 features)
  5. all_gates        — baseline + all three gate feature sets
  6. all_gates_rfe    — all_gates with RFE feature selection
  7. all_gates_rfe_ifw — all_gates with RFE + per-phage inverse-frequency weighting

Each arm runs 3 seeds with 1000-resample bootstrap CIs on the 65-bacteria holdout.

Usage:
    python -m lyzortx.pipeline.autoresearch.gt03_eval --device-type cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch.candidate_replay import (
    bootstrap_holdout_metric_cis,
    build_st03_training_frame,
    load_module_from_path,
    load_st03_holdout_frame,
    summarize_seed_metrics,
    temporary_module_attribute,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("lyzortx/generated_outputs/autoresearch/search_cache_v1")
DEFAULT_CANDIDATE_DIR = Path("lyzortx/autoresearch")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/gt03_eval")

SEEDS = [7, 42, 123]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@dataclass
class AblationArm:
    arm_id: str
    include_host_defense: bool = False
    include_pairwise_depo_capsule: bool = False
    include_pairwise_receptor_omp: bool = False
    use_rfe: bool = False
    use_inverse_freq_weighting: bool = False


ABLATION_ARMS = [
    AblationArm("baseline"),
    AblationArm("+gate1_depo_capsule", include_pairwise_depo_capsule=True),
    AblationArm("+gate2_receptor_omp", include_pairwise_receptor_omp=True),
    AblationArm("+gate3_defense", include_host_defense=True),
    AblationArm(
        "all_gates",
        include_host_defense=True,
        include_pairwise_depo_capsule=True,
        include_pairwise_receptor_omp=True,
    ),
    AblationArm(
        "all_gates_rfe",
        include_host_defense=True,
        include_pairwise_depo_capsule=True,
        include_pairwise_receptor_omp=True,
        use_rfe=True,
    ),
    AblationArm(
        "all_gates_rfe_ifw",
        include_host_defense=True,
        include_pairwise_depo_capsule=True,
        include_pairwise_receptor_omp=True,
        use_rfe=True,
        use_inverse_freq_weighting=True,
    ),
]


def build_design_matrices(
    *,
    candidate_module: ModuleType,
    context: Any,
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    arm: AblationArm,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build train and holdout design matrices for an ablation arm.

    Returns (train_design, holdout_design, feature_columns, categorical_columns).
    """
    host_slots = ["host_surface", "host_typing", "host_stats"]
    if arm.include_host_defense:
        host_slots.append("host_defense")
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

    # Add pairwise cross-terms.
    if arm.include_pairwise_depo_capsule:
        from lyzortx.pipeline.autoresearch.derive_pairwise_depo_capsule_features import (
            compute_pairwise_depo_capsule_features,
        )

        compute_pairwise_depo_capsule_features(train_design)
        compute_pairwise_depo_capsule_features(holdout_design)

    if arm.include_pairwise_receptor_omp:
        from lyzortx.pipeline.autoresearch.derive_pairwise_receptor_omp_features import (
            compute_pairwise_receptor_omp_features,
        )

        compute_pairwise_receptor_omp_features(train_design)
        compute_pairwise_receptor_omp_features(holdout_design)

    # Build feature column list from prefixes.
    all_slot_names = host_slots + phage_slots
    prefixes = tuple(f"{s}__" for s in all_slot_names)
    if arm.include_pairwise_depo_capsule:
        prefixes = (*prefixes, "pair_depo_capsule__")
    if arm.include_pairwise_receptor_omp:
        prefixes = (*prefixes, "pair_receptor_omp__")

    feature_columns = [col for col in train_design.columns if col.startswith(prefixes)]
    categorical_columns = [col for col in (host_categorical + phage_categorical) if col in feature_columns]

    return train_design, holdout_design, feature_columns, categorical_columns


def compute_inverse_freq_weights(train_design: pd.DataFrame) -> np.ndarray:
    """Compute per-phage inverse-frequency sample weights for positive samples.

    Narrow-host phages (few positives) get higher weight on their positive samples.
    Weights are multiplied with the existing training_weight_v3.
    """
    base_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)
    labels = train_design["label_any_lysis"].astype(int).to_numpy()
    phages = train_design["phage"].astype(str).to_numpy()

    # Compute per-phage positive rate.
    unique_phages = np.unique(phages)
    phage_pos_rate = {}
    for phage in unique_phages:
        mask = phages == phage
        phage_pos_rate[phage] = labels[mask].mean()

    # Inverse-frequency: higher weight for rare positives.
    inv_freq = {p: 1.0 / max(rate, 0.01) for p, rate in phage_pos_rate.items()}
    mean_inv = np.mean(list(inv_freq.values()))
    inv_freq = {p: v / mean_inv for p, v in inv_freq.items()}  # normalize to mean=1

    # Apply to positive samples only.
    ifw = np.ones(len(base_weight))
    for i in range(len(base_weight)):
        if labels[i] == 1:
            ifw[i] = inv_freq[phages[i]]

    LOGGER.info(
        "Inverse-frequency weighting: min=%.2f, max=%.2f, mean=%.2f (positive samples only)",
        ifw[labels == 1].min(),
        ifw[labels == 1].max(),
        ifw[labels == 1].mean(),
    )
    return base_weight * ifw


def apply_rfe(
    train_design: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    y_train: np.ndarray,
    seed: int,
) -> list[str]:
    """Run RFECV on numeric features and return selected columns (categoricals always kept)."""
    # RFECV can't handle string categoricals — exclude them from RFE.
    # Categoricals (phylogroup, serotype, ST) are always useful; RFE focuses
    # on pruning the numeric features (HMM scores, cross-terms, etc.).
    numeric_features = [c for c in feature_columns if c not in categorical_columns]
    if not numeric_features:
        return feature_columns

    rfe_estimator = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=10,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    selector = RFECV(
        rfe_estimator,
        step=0.1,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        min_features_to_select=10,
    )
    LOGGER.info(
        "Running RFECV on %d numeric features (keeping %d categoricals)...",
        len(numeric_features),
        len(categorical_columns),
    )
    selector.fit(train_design[numeric_features], y_train)
    selected_numeric = [col for col, kept in zip(numeric_features, selector.support_) if kept]
    selected = categorical_columns + selected_numeric
    LOGGER.info(
        "RFECV selected %d/%d numeric features (%d total with categoricals)",
        len(selected_numeric),
        len(numeric_features),
        len(selected),
    )
    return selected


def run_arm_seed(
    *,
    candidate_module: ModuleType,
    train_design: pd.DataFrame,
    holdout_design: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    arm: AblationArm,
    seed: int,
    device_type: str,
) -> list[dict[str, object]]:
    """Train and evaluate one arm for one seed. Returns holdout prediction rows."""
    y_train = train_design["label_any_lysis"].astype(int).to_numpy(dtype=int)

    # Sample weights.
    if arm.use_inverse_freq_weighting:
        sample_weight = compute_inverse_freq_weights(train_design)
    else:
        sample_weight = train_design["training_weight_v3"].astype(float).to_numpy(dtype=float)

    # RFE feature selection.
    active_features = feature_columns
    if arm.use_rfe:
        active_features = apply_rfe(train_design, feature_columns, categorical_columns, y_train, seed)

    active_categorical = [c for c in categorical_columns if c in active_features]

    # Build and train estimator.
    with temporary_module_attribute(candidate_module, "PAIR_SCORER_RANDOM_STATE", seed):
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
        train_design[active_features],
        y_train,
        sample_weight=sample_weight,
        categorical_feature=active_categorical,
    )
    predictions = estimator.predict_proba(holdout_design[active_features])[:, 1]

    # Log feature importance by slot prefix.
    imp = estimator.feature_importances_
    slot_imp: dict[str, float] = {}
    for col, val in zip(active_features, imp):
        slot = col.split("__")[0]
        slot_imp[slot] = slot_imp.get(slot, 0) + val
    total_imp = sum(slot_imp.values()) or 1
    parts = [f"{s}={v / total_imp * 100:.1f}%" for s, v in sorted(slot_imp.items(), key=lambda x: -x[1])]
    LOGGER.info("Feature importance: %s (%d features)", ", ".join(parts), len(active_features))

    # Build output rows.
    rows = []
    for row, prob in zip(
        holdout_design.loc[:, ["pair_id", "bacteria", "phage", "label_any_lysis"]].to_dict(orient="records"),
        predictions,
    ):
        rows.append(
            {
                "arm_id": arm.arm_id,
                "seed": seed,
                "pair_id": str(row["pair_id"]),
                "bacteria": str(row["bacteria"]),
                "phage": str(row["phage"]),
                "label_hard_any_lysis": int(row["label_any_lysis"]),
                "predicted_probability": round(float(prob), 6),
            }
        )
    return rows


def run_ablation(
    *,
    candidate_module: ModuleType,
    context: Any,
    device_type: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all ablation arms and return results summary."""
    holdout_frame = load_st03_holdout_frame()
    training_frame = build_st03_training_frame()
    LOGGER.info("ST03 split: %d training, %d holdout rows", len(training_frame), len(holdout_frame))

    all_rows: list[dict[str, object]] = []
    seed_metric_rows: list[dict[str, object]] = []

    for arm in ABLATION_ARMS:
        LOGGER.info("=== Arm: %s ===", arm.arm_id)

        # Build design matrices once per arm (shared across seeds).
        train_design, holdout_design, feature_columns, categorical_columns = build_design_matrices(
            candidate_module=candidate_module,
            context=context,
            training_frame=training_frame,
            holdout_frame=holdout_frame,
            arm=arm,
        )
        LOGGER.info(
            "Arm %s: %d feature columns, %d training pairs", arm.arm_id, len(feature_columns), len(train_design)
        )

        for seed in SEEDS:
            LOGGER.info("Arm %s seed %d", arm.arm_id, seed)
            arm_seed_rows = run_arm_seed(
                candidate_module=candidate_module,
                train_design=train_design,
                holdout_design=holdout_design,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                arm=arm,
                seed=seed,
                device_type=device_type,
            )
            all_rows.extend(arm_seed_rows)
            metrics = summarize_seed_metrics(arm_seed_rows)
            seed_metric_rows.append({"arm_id": arm.arm_id, "seed": seed, **metrics})
            LOGGER.info(
                "Arm %s seed %d: AUC=%.4f, top-3=%.1f%%, Brier=%.4f",
                arm.arm_id,
                seed,
                metrics.get("holdout_roc_auc", 0),
                metrics.get("holdout_top3_hit_rate_all_strains", 0) * 100,
                metrics.get("holdout_brier_score", 0),
            )

    # Aggregate predictions across seeds (mean probability).
    df = pd.DataFrame(all_rows)
    aggregated = (
        df.groupby(["arm_id", "pair_id", "bacteria", "phage", "label_hard_any_lysis"], as_index=False)[
            "predicted_probability"
        ]
        .mean()
        .sort_values(["arm_id", "bacteria", "phage"])
    )
    aggregated_rows = aggregated.to_dict(orient="records")

    # Group by arm for bootstrap CIs.
    holdout_rows_by_arm: dict[str, list[dict[str, object]]] = {}
    for row in aggregated_rows:
        arm_id = str(row["arm_id"])
        holdout_rows_by_arm.setdefault(arm_id, []).append(row)

    bootstrap_results = bootstrap_holdout_metric_cis(
        holdout_rows_by_arm,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_random_state=BOOTSTRAP_RANDOM_STATE,
        baseline_arm_id="baseline",
    )

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    pd.DataFrame(seed_metric_rows).to_csv(output_dir / "seed_metrics.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_predictions.csv", index=False)

    # Format results summary.
    summary_lines = ["GT03 Ablation Results", "=" * 60]
    summary_lines.append(f"{'Arm':30s} {'AUC':>8s} {'Top-3':>8s} {'Brier':>8s}")
    summary_lines.append("-" * 60)
    for arm_id, ci_dict in bootstrap_results.items():
        if "__delta_vs_" in arm_id:
            continue
        auc_ci = ci_dict.get("holdout_roc_auc")
        top3_ci = ci_dict.get("holdout_top3_hit_rate_all_strains")
        brier_ci = ci_dict.get("holdout_brier_score")
        if auc_ci and top3_ci and brier_ci:
            summary_lines.append(
                f"{arm_id:30s} "
                f"{auc_ci.point_estimate:.3f}  "
                f"{top3_ci.point_estimate * 100:.1f}%  "
                f"{brier_ci.point_estimate:.3f}"
            )
    summary_lines.append("")
    summary_lines.append("Delta vs baseline (AUC):")
    for arm_id, ci_dict in bootstrap_results.items():
        if "__delta_vs_baseline" not in arm_id:
            continue
        auc_delta = ci_dict.get("holdout_roc_auc")
        if auc_delta:
            clean_id = arm_id.replace("__delta_vs_baseline", "")
            summary_lines.append(
                f"  {clean_id:28s} {auc_delta.point_estimate:+.4f} "
                f"[{auc_delta.ci_lower:+.4f}, {auc_delta.ci_upper:+.4f}]"
            )

    summary_text = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    LOGGER.info("\n%s", summary_text)

    # Save bootstrap results as JSON.
    bootstrap_json = {}
    for arm_id, ci_dict in bootstrap_results.items():
        bootstrap_json[arm_id] = {
            metric: {
                "point_estimate": ci.point_estimate,
                "ci_lower": ci.ci_lower,
                "ci_upper": ci.ci_upper,
            }
            for metric, ci in ci_dict.items()
        }
    with open(output_dir / "bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_json, f, indent=2)

    return {"bootstrap_results": bootstrap_json, "summary": summary_text}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-type",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Device for LightGBM.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Path to AUTORESEARCH search cache.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="Path to candidate train.py directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)

    LOGGER.info("GT03 three-layer ablation starting at %s", datetime.now(timezone.utc).isoformat())

    candidate_train_path = args.candidate_dir / "train.py"
    candidate_module = load_module_from_path("gt03_candidate", candidate_train_path)
    context = candidate_module.load_and_validate_cache(
        cache_dir=args.cache_dir,
        include_host_defense=True,
    )

    run_ablation(
        candidate_module=candidate_module,
        context=context,
        device_type=args.device_type,
        output_dir=args.output_dir,
    )
    LOGGER.info("GT03 ablation complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
