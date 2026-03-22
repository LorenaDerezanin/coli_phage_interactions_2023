#!/usr/bin/env python3
"""TP01: Build an interactive digital phagogram for per-strain phage ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.isotonic import IsotonicRegression

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    CATEGORICAL_FEATURE_COLUMNS as V0_CATEGORICAL_FEATURE_COLUMNS,
)
from lyzortx.pipeline.steel_thread_v0.steps.st04_train_baselines import (
    NUMERIC_FEATURE_COLUMNS as V0_NUMERIC_FEATURE_COLUMNS,
)
from lyzortx.pipeline.track_g.steps import run_feature_subset_sweep, train_v1_binary_classifier
from lyzortx.pipeline.track_g.steps.compute_shap_explanations import (
    _dense_row,
    format_contribution_summary,
    top_feature_contributions,
)

DEFAULT_LOCKED_V1_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
DEFAULT_TG05_SUMMARY_PATH = Path(
    "lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/tg05_feature_subset_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_p/digital_phagogram")
DEFAULT_DISPLAY_LIMIT = 12
DEFAULT_BOOTSTRAP_SAMPLES = 128
DEFAULT_CALIBRATION_FOLD = 0

TG02_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "bacteria",
    "phage",
    "split_holdout",
    "split_cv5_fold",
    "label_hard_any_lysis",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locked-v1-config-path",
        type=Path,
        default=DEFAULT_LOCKED_V1_CONFIG_PATH,
        help="Locked v1 feature configuration JSON from Track G TG05.",
    )
    parser.add_argument(
        "--tg05-summary-path",
        type=Path,
        default=DEFAULT_TG05_SUMMARY_PATH,
        help="TG05 summary JSON containing the locked LightGBM hyperparameters.",
    )
    parser.add_argument(
        "--st02-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv"),
        help="Input ST0.2 pair table path.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv"),
        help="Input ST0.3 split assignments path.",
    )
    parser.add_argument(
        "--track-c-pair-table-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv"),
        help="Input Track C v1 pair table path.",
    )
    parser.add_argument(
        "--track-d-genome-kmer-path",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv"),
        help="Input Track D genome k-mer feature CSV.",
    )
    parser.add_argument(
        "--track-d-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_d/phage_distance_embedding/phage_distance_embedding_features.csv"
        ),
        help="Input Track D phage-distance feature CSV.",
    )
    parser.add_argument(
        "--track-e-rbp-compatibility-path",
        type=Path,
        default=train_v1_binary_classifier.TRACK_E_REQUIRED_BLOCKS[0][1],
        help="Input Track E RBP-receptor compatibility feature CSV.",
    )
    parser.add_argument(
        "--track-e-defense-evasion-path",
        type=Path,
        default=train_v1_binary_classifier.TRACK_E_REQUIRED_BLOCKS[1][1],
        help="Input Track E defense-evasion proxy feature CSV.",
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=train_v1_binary_classifier.TRACK_E_REQUIRED_BLOCKS[2][1],
        help="Input Track E isolation-host distance feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Track P artifacts.",
    )
    parser.add_argument(
        "--initial-bacteria",
        type=str,
        default="",
        help="Initial strain to show when the HTML demo loads.",
    )
    parser.add_argument(
        "--display-limit",
        type=int,
        default=DEFAULT_DISPLAY_LIMIT,
        help="Initial number of ranked phages to render per arm.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples for confidence bands.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=DEFAULT_CALIBRATION_FOLD,
        help="ST0.3 calibration fold used to fit bootstrap calibrators.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for confidence-band bootstrapping.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume TG05 and upstream Track G prerequisites already exist.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(values: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def ensure_prerequisite_outputs(args: argparse.Namespace) -> None:
    if args.skip_prerequisites:
        return
    if not args.tg05_summary_path.exists():
        run_feature_subset_sweep.main([])


def load_locked_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tg05_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_feature_space_from_columns(
    base_feature_space: train_v1_binary_classifier.FeatureSpace,
    *,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> train_v1_binary_classifier.FeatureSpace:
    return train_v1_binary_classifier.FeatureSpace(
        categorical_columns=_deduplicate(categorical_columns),
        numeric_columns=_deduplicate(numeric_columns),
        track_c_additional_columns=base_feature_space.track_c_additional_columns,
        track_d_columns=base_feature_space.track_d_columns,
        track_e_columns=base_feature_space.track_e_columns,
    )


def build_locked_arm_feature_spaces(
    full_feature_space: train_v1_binary_classifier.FeatureSpace,
    *,
    winner_subset_blocks: Sequence[str],
    excluded_columns: Sequence[str],
) -> Dict[str, train_v1_binary_classifier.FeatureSpace]:
    blocks = run_feature_subset_sweep.build_feature_blocks(full_feature_space)
    categorical_columns = list(V0_CATEGORICAL_FEATURE_COLUMNS)
    numeric_columns = list(V0_NUMERIC_FEATURE_COLUMNS)
    for block_id in winner_subset_blocks:
        block = blocks[block_id]
        categorical_columns.extend(block.categorical_columns)
        numeric_columns.extend(block.numeric_columns)

    panel_space = _build_feature_space_from_columns(
        full_feature_space,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    deployment_space = _build_feature_space_from_columns(
        full_feature_space,
        categorical_columns=[
            column for column in panel_space.categorical_columns if column not in set(excluded_columns)
        ],
        numeric_columns=[column for column in panel_space.numeric_columns if column not in set(excluded_columns)],
    )
    return {"panel": panel_space, "deployment": deployment_space}


def _dense_matrix(matrix: Any, indices: Sequence[int]) -> np.ndarray:
    subset = matrix[indices]
    if sparse.issparse(subset):
        return subset.toarray()
    return np.asarray(subset)


def score_rows_for_arm(
    rows: Sequence[Mapping[str, object]],
    feature_space: train_v1_binary_classifier.FeatureSpace,
    *,
    params: Mapping[str, object],
    random_state: int,
) -> Tuple[Any, Any, List[Dict[str, object]], List[Dict[str, object]], np.ndarray, List[float]]:
    lightgbm_factory = lambda arm_params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        arm_params,
        seed_offset,
        base_random_state=random_state,
    )
    estimator, vectorizer, train_rows, eval_rows, probabilities = train_v1_binary_classifier.fit_final_estimator(
        rows,
        feature_space,
        estimator_factory=lightgbm_factory,
        params=params,
    )
    feature_matrix = vectorizer.transform(
        [
            train_v1_binary_classifier._build_feature_dict(
                row,
                categorical_columns=feature_space.categorical_columns,
                numeric_columns=feature_space.numeric_columns,
            )
            for row in rows
        ]
    )
    all_probabilities = train_v1_binary_classifier._predict_probabilities(estimator, feature_matrix)
    return estimator, vectorizer, train_rows, eval_rows, feature_matrix, all_probabilities


def bootstrap_probability_intervals(
    calibration_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    score_key: str = "predicted_probability",
    label_key: str = "label_hard_any_lysis",
    bootstrap_samples: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    if not candidate_rows:
        return {}
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    x_calib = np.asarray([float(row[score_key]) for row in calibration_rows], dtype=float)
    y_calib = np.asarray([int(row[label_key]) for row in calibration_rows], dtype=int)
    if len(x_calib) == 0:
        raise ValueError("No calibration rows available.")
    if len(np.unique(y_calib)) < 2:
        raise ValueError("Calibration rows must contain both classes for bootstrap CI estimation.")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(x_calib, y_calib)

    candidate_raw = np.asarray([float(row[score_key]) for row in candidate_rows], dtype=float)
    point_estimates = np.asarray(calibrator.predict(candidate_raw), dtype=float)

    rng = np.random.default_rng(random_state)
    bootstrap_predictions: List[np.ndarray] = []
    for _ in range(bootstrap_samples):
        sample_indices = rng.integers(0, len(calibration_rows), size=len(calibration_rows))
        y_sample = y_calib[sample_indices]
        if len(np.unique(y_sample)) < 2:
            continue
        sample_calibrator = IsotonicRegression(out_of_bounds="clip")
        sample_calibrator.fit(x_calib[sample_indices], y_sample)
        bootstrap_predictions.append(np.asarray(sample_calibrator.predict(candidate_raw), dtype=float))

    if not bootstrap_predictions:
        raise ValueError("Bootstrap CI estimation failed because no valid resamples contained both classes.")

    samples = np.vstack(bootstrap_predictions)
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    intervals: Dict[str, Dict[str, float]] = {}
    for index, row in enumerate(candidate_rows):
        intervals[str(row["pair_id"])] = {
            "calibrated_p_lysis": safe_round(float(point_estimates[index])),
            "ci_low": safe_round(float(low[index])),
            "ci_high": safe_round(float(high[index])),
            "bootstrap_samples_used": float(samples.shape[0]),
        }
    return intervals


def build_shap_driver_rows(
    feature_matrix: Any,
    estimator: Any,
    feature_names: Sequence[str],
    scored_rows: Sequence[Mapping[str, object]],
    *,
    top_k: int = 3,
) -> Dict[str, Dict[str, object]]:
    import shap

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(feature_matrix)
    shap_values = explanation.values

    rows_by_pair: Dict[str, Dict[str, object]] = {}
    for index, row in enumerate(scored_rows):
        row_values = _dense_row(shap_values, index)
        feature_values = _dense_row(feature_matrix, index)
        contributions = top_feature_contributions(row_values, feature_values, feature_names, top_k=top_k)
        rows_by_pair[str(row["pair_id"])] = {
            "top_positive": contributions["positive"],
            "top_negative": contributions["negative"],
            "top_shap_summary": (
                f"+ {format_contribution_summary(contributions['positive']) or 'no positive drivers'}; "
                f"- {format_contribution_summary(contributions['negative']) or 'no negative drivers'}"
            ),
        }
    return rows_by_pair


def build_arm_display_rows(
    scored_rows: Sequence[Mapping[str, object]],
    confidence_intervals: Mapping[str, Mapping[str, float]],
    shap_rows_by_pair_id: Mapping[str, Mapping[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    rows_by_bacteria: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in scored_rows:
        rows_by_bacteria[str(row["bacteria"])].append(row)

    display_rows: Dict[str, List[Dict[str, object]]] = {}
    for bacteria, bacteria_rows in rows_by_bacteria.items():
        ranked = sorted(
            bacteria_rows,
            key=lambda row: (-float(row["predicted_probability"]), str(row["phage"])),
        )
        rendered_rows: List[Dict[str, object]] = []
        for rank, row in enumerate(ranked, start=1):
            pair_id = str(row["pair_id"])
            interval = confidence_intervals[pair_id]
            shap_payload = shap_rows_by_pair_id[pair_id]
            rendered_rows.append(
                {
                    "rank": rank,
                    "pair_id": pair_id,
                    "phage": row["phage"],
                    "phage_family": row.get("phage_family", ""),
                    "prediction_context": row.get("prediction_context", ""),
                    "label_hard_any_lysis": row.get("label_hard_any_lysis", ""),
                    "p_lysis": safe_round(float(interval["calibrated_p_lysis"])),
                    "ci_low": safe_round(float(interval["ci_low"])),
                    "ci_high": safe_round(float(interval["ci_high"])),
                    "top_shap_summary": shap_payload["top_shap_summary"],
                    "top_positive": shap_payload["top_positive"],
                    "top_negative": shap_payload["top_negative"],
                    "raw_probability": safe_round(float(row["predicted_probability"])),
                }
            )
        display_rows[bacteria] = rendered_rows
    return dict(sorted(display_rows.items(), key=lambda item: item[0]))


def build_phagogram_bundle(
    *,
    config: Mapping[str, object],
    tg05_summary: Mapping[str, object],
    initial_bacteria: str,
    display_limit: int,
    panel_rows_by_strain: Mapping[str, Sequence[Mapping[str, object]]],
    deployment_rows_by_strain: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    bacteria = initial_bacteria if initial_bacteria in panel_rows_by_strain else sorted(panel_rows_by_strain)[0]
    strains = []
    for strain in sorted(panel_rows_by_strain):
        strains.append(
            {
                "bacteria": strain,
                "panel_rows": list(panel_rows_by_strain[strain]),
                "deployment_rows": list(deployment_rows_by_strain[strain]),
            }
        )
    locked = config["locked_v1_feature_configuration"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TP01",
        "initial_bacteria": bacteria,
        "display_limit": display_limit,
        "locked_v1_feature_configuration": locked,
        "panel_label": locked["winner_label"],
        "deployment_label": f"{locked['winner_label']} (deployment-realistic)",
        "panel_metrics": locked["panel_default"],
        "deployment_metrics": locked["deployment_realistic"],
        "tg05_locked_lightgbm_hyperparameters": tg05_summary["locked_lightgbm_hyperparameters"],
        "strain_count": len(strains),
        "strains": strains,
    }


def render_digital_phagogram_html(bundle: Mapping[str, object]) -> str:
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    panel_metrics = bundle["panel_metrics"]
    deployment_metrics = bundle["deployment_metrics"]
    winner_blocks = ", ".join(bundle["locked_v1_feature_configuration"]["winner_subset_blocks"])
    excluded = ", ".join(
        bundle["locked_v1_feature_configuration"]["deployment_realistic"]["excluded_label_derived_columns"]
    )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track P Digital Phagogram</title>
  <style>
    :root {{
      --bg: #07111f;
      --bg-soft: #0d1a2f;
      --panel: rgba(12, 21, 37, 0.88);
      --panel-strong: rgba(16, 28, 48, 0.96);
      --text: #edf6ff;
      --muted: #9fb3c8;
      --line: rgba(180, 206, 230, 0.16);
      --accent: #6ee7d8;
      --accent-2: #fbbf24;
      --danger: #fb7185;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(110, 231, 216, 0.17), transparent 32%),
        radial-gradient(circle at top right, rgba(251, 191, 36, 0.14), transparent 30%),
        linear-gradient(180deg, #04101d 0%, #07111f 100%);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    .wrap {{
      width: min(1600px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.7fr 1fr;
      gap: 20px;
      align-items: start;
      margin-bottom: 20px;
    }}
    .title {{
      background: linear-gradient(135deg, rgba(13, 26, 47, 0.84), rgba(8, 18, 33, 0.78));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 24px 26px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(32px, 4vw, 58px);
      line-height: 0.98;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .lede {{
      margin: 0;
      color: var(--muted);
      max-width: 76ch;
      line-height: 1.55;
    }}
    .controls {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 12px;
    }}
    .controls label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    .controls-row {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      align-items: end;
    }}
    select, input[type="text"], button {{
      background: rgba(6, 12, 23, 0.9);
      color: var(--text);
      border: 1px solid rgba(180, 206, 230, 0.18);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: linear-gradient(135deg, rgba(110, 231, 216, 0.18), rgba(251, 191, 36, 0.18));
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .stat {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .config {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 18px;
    }}
    .config-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .config-card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .config-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .arms {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .arm-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      min-height: 720px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 12px;
    }}
    .arm-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .arm-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .arm-head .sub {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(180, 206, 230, 0.16);
      background: rgba(6, 12, 23, 0.72);
      color: var(--muted);
    }}
    .table-head, .row {{
      display: grid;
      grid-template-columns: 48px 1fr 190px 1.5fr;
      gap: 12px;
      align-items: start;
    }}
    .table-head {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      padding: 0 4px;
    }}
    .scroll {{
      overflow: auto;
      padding-right: 4px;
    }}
    .row {{
      padding: 14px 10px;
      border-top: 1px solid rgba(180, 206, 230, 0.10);
      transition: background 120ms ease, border-color 120ms ease;
    }}
    .row:hover {{
      background: rgba(110, 231, 216, 0.05);
      border-color: rgba(110, 231, 216, 0.18);
    }}
    .rank {{
      font-size: 20px;
      font-weight: 700;
      color: var(--accent-2);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .phage {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .prob {{
      display: grid;
      gap: 8px;
    }}
    .probbar {{
      position: relative;
      height: 14px;
      border-radius: 999px;
      background: rgba(180, 206, 230, 0.10);
      overflow: hidden;
    }}
    .probbar .fill {{
      position: absolute;
      inset: 0 auto 0 0;
      width: var(--pct);
      border-radius: inherit;
      background: linear-gradient(90deg, rgba(110, 231, 216, 0.92), rgba(251, 191, 36, 0.92));
    }}
    .probbar .band {{
      position: absolute;
      top: -3px;
      bottom: -3px;
      left: var(--lo);
      width: calc(var(--hi) - var(--lo));
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.15);
      border: 1px solid rgba(255, 255, 255, 0.18);
    }}
    .probtext {{
      font-size: 13px;
      color: var(--text);
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 6px;
    }}
    .chip {{
      padding: 6px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(180, 206, 230, 0.12);
      background: rgba(6, 12, 23, 0.72);
      color: var(--text);
    }}
    .chip.positive {{
      color: #6ee7d8;
    }}
    .chip.negative {{
      color: #fb7185;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .footer-note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 1200px) {{
      .hero, .config, .arms {{
        grid-template-columns: 1fr;
      }}
      .arm-card {{
        min-height: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <section class="title">
        <div class="eyebrow">Track P presentation artifact</div>
        <h1>Digital phagogram</h1>
        <p class="lede">
          Ranked phage recommendations for any strain, rendered side by side for the locked panel model and the
          deployment-realistic lock. Each row surfaces calibrated P(lysis), a confidence band, and the SHAP drivers
          behind the ranking.
        </p>
    <div class="stats" style="margin-top:18px;">
          <div class="stat">
            <div class="label">Initial strain</div>
            <div class="value" id="initial-strain"></div>
          </div>
          <div class="stat">
            <div class="label">Panel model</div>
            <div class="value">__PANEL_TOP3__ top-3</div>
          </div>
          <div class="stat">
            <div class="label">Deployment-realistic</div>
            <div class="value">__DEPLOYMENT_TOP3__ top-3</div>
          </div>
        </div>
      </section>
      <section class="controls">
        <div class="controls-row">
          <label>
            Choose strain
            <select id="strain-select"></select>
          </label>
          <label>
            Initial display rows
            <input id="limit-input" type="text" value="__DISPLAY_LIMIT__">
          </label>
          <button id="show-all-button" type="button">Toggle full ranking</button>
        </div>
        <div class="small" id="strain-note"></div>
      </section>
    </div>

    <div class="config">
      <div class="config-card">
        <h2>Locked v1 feature config</h2>
        <p>
          Winner blocks: __WINNER_BLOCKS__. Deployment-realistic mode excludes __EXCLUDED__. The phagogram uses the locked
          TG05 panel configuration rather than the broader TG01 all-features reference.
        </p>
      </div>
      <div class="config-card">
        <h2>Demo contract</h2>
        <p>
          The left panel shows the panel-default lock with <code>host_n_infections</code> retained. The right panel
          removes label-derived columns so operators can compare live-panel and deployment-realistic ranking stability.
        </p>
      </div>
    </div>

    <div class="arms">
      <article class="arm-card">
        <div class="arm-head">
          <div>
            <h2 id="panel-title"></h2>
            <div class="sub" id="panel-subtitle"></div>
          </div>
          <div class="pill" id="panel-pill"></div>
        </div>
        <div class="table-head">
          <div>Rank</div>
          <div>Phage + SHAP</div>
          <div>P(lysis)</div>
          <div>Top SHAP features</div>
        </div>
        <div class="scroll" id="panel-table"></div>
      </article>
      <article class="arm-card">
        <div class="arm-head">
          <div>
            <h2 id="deployment-title"></h2>
            <div class="sub" id="deployment-subtitle"></div>
          </div>
          <div class="pill" id="deployment-pill"></div>
        </div>
        <div class="table-head">
          <div>Rank</div>
          <div>Phage + SHAP</div>
          <div>P(lysis)</div>
          <div>Top SHAP features</div>
        </div>
        <div class="scroll" id="deployment-table"></div>
      </article>
    </div>
      <div class="footer-note">
      Saved data bundle: <span id="bundle-path">tp01_digital_phagogram_bundle.json</span>
    </div>
  </div>

  <script id="phagogram-data" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("phagogram-data").textContent);
    const strains = data.strains;
    const strainSelect = document.getElementById("strain-select");
    const limitInput = document.getElementById("limit-input");
    const strainNote = document.getElementById("strain-note");
    const panelTable = document.getElementById("panel-table");
    const deploymentTable = document.getElementById("deployment-table");
    const panelTitle = document.getElementById("panel-title");
    const panelSubtitle = document.getElementById("panel-subtitle");
    const panelPill = document.getElementById("panel-pill");
    const deploymentTitle = document.getElementById("deployment-title");
    const deploymentSubtitle = document.getElementById("deployment-subtitle");
    const deploymentPill = document.getElementById("deployment-pill");
    const initialStrain = document.getElementById("initial-strain");
    const bundlePath = document.getElementById("bundle-path");
    const showAllButton = document.getElementById("show-all-button");

    let showAll = false;

    function chipClass(value) {{
      if (value.includes("+")) return "positive";
      if (value.includes("-")) return "negative";
      return "";
    }}

    function renderDriverChips(row) {{
      const chips = [];
      const combined = [...(row.top_positive || []), ...(row.top_negative || [])];
      for (const feature of combined.slice(0, 6)) {{
        const sign = feature.shap_value >= 0 ? "+" : "";
        chips.push(`<span class="chip ${feature.shap_value >= 0 ? "positive" : "negative"}">${feature.feature_name} ${sign}${Number(feature.shap_value).toFixed(3)}</span>`);
      }}
      if (!chips.length) {{
        return `<span class="chip">No SHAP drivers captured</span>`;
      }}
      return chips.join("");
    }}

    function renderRow(row, index) {{
      const lo = Math.max(0, Math.min(100, row.ci_low * 100));
      const hi = Math.max(0, Math.min(100, row.ci_high * 100));
      const p = Math.max(0, Math.min(100, row.p_lysis * 100));
      return `
        <div class="row">
          <div class="rank">${row.rank}</div>
          <div>
            <div class="phage">${row.phage}</div>
            <div class="meta">${row.phage_family || ""}${row.prediction_context ? " · " + row.prediction_context : ""}</div>
            <div class="meta">${row.top_shap_summary}</div>
          </div>
          <div class="prob">
            <div class="probbar" style="--lo:${lo}%; --hi:${hi}%; --pct:${p}%;">
              <div class="band"></div>
              <div class="fill"></div>
            </div>
            <div class="probtext">${row.p_lysis.toFixed(3)} [${row.ci_low.toFixed(3)}, ${row.ci_high.toFixed(3)}]</div>
          </div>
          <div class="chips">${renderDriverChips(row)}</div>
        </div>`;
    }}

    function renderArmRows(rows, container) {{
      const rawLimit = Number(limitInput.value || data.display_limit);
      const limit = Number.isFinite(rawLimit) ? rawLimit : data.display_limit;
      const slice = showAll ? rows : rows.slice(0, Math.max(1, limit));
      container.innerHTML = slice.map((row, index) => renderRow(row, index)).join("") +
        (rows.length > slice.length ? `<div class="small" style="padding:10px 10px 2px;">Showing ${slice.length} of ${rows.length} ranked phages.</div>` : "");
    }}

    function currentStrain() {{
      return strainSelect.value;
    }}

    function renderSelectedStrain() {{
      const strain = strains.find((item) => item.bacteria === currentStrain()) || strains[0];
      const panelRows = strain.panel_rows;
      const deploymentRows = strain.deployment_rows;
      initialStrain.textContent = strain.bacteria;
      panelTitle.textContent = data.panel_label;
      panelSubtitle.textContent = `${strain.bacteria} · locked panel-default ranking`;
      panelPill.textContent = `${panelRows.length} phages`;
      deploymentTitle.textContent = data.deployment_label;
      deploymentSubtitle.textContent = `${strain.bacteria} · deployment-realistic ranking`;
      deploymentPill.textContent = `${deploymentRows.length} phages`;
      strainNote.textContent = `The selected strain contains ${panelRows.length} scored panel rows and ${deploymentRows.length} scored deployment rows.`;
      renderArmRows(panelRows, panelTable);
      renderArmRows(deploymentRows, deploymentTable);
      bundlePath.textContent = data.generated_at_utc;
    }}

    function syncStrainOptions() {{
      strainSelect.innerHTML = strains.map((item) => `<option value="${item.bacteria}">${item.bacteria}</option>`).join("");
      strainSelect.value = data.initial_bacteria;
      initialStrain.textContent = data.initial_bacteria;
      panelTitle.textContent = data.panel_label;
      deploymentTitle.textContent = data.deployment_label;
      panelPill.textContent = `${strains[0].panel_rows.length} phages`;
      deploymentPill.textContent = `${strains[0].deployment_rows.length} phages`;
    }}

    strainSelect.addEventListener("change", renderSelectedStrain);
    limitInput.addEventListener("change", renderSelectedStrain);
    showAllButton.addEventListener("click", () => {{
      showAll = !showAll;
      showAllButton.textContent = showAll ? "Show initial ranking" : "Toggle full ranking";
      renderSelectedStrain();
    }});

    syncStrainOptions();
    renderSelectedStrain();
  </script>
</body>
</html>
"""
    while "{{" in html or "}}" in html:
        html = html.replace("{{", "{").replace("}}", "}")
    return (
        html.replace("__PAYLOAD__", payload)
        .replace("__PANEL_TOP3__", f"{float(panel_metrics['holdout_top3_hit_rate_all_strains']):.3f}")
        .replace("__DEPLOYMENT_TOP3__", f"{float(deployment_metrics['holdout_top3_hit_rate_all_strains']):.3f}")
        .replace("__DISPLAY_LIMIT__", str(bundle["display_limit"]))
        .replace("__WINNER_BLOCKS__", winner_blocks)
        .replace("__EXCLUDED__", excluded)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)

    locked_config = load_locked_config(args.locked_v1_config_path)
    tg05_summary = load_tg05_summary(args.tg05_summary_path)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_defense_rows = read_csv_rows(args.track_e_defense_evasion_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)

    track_d_feature_columns = train_v1_binary_classifier._deduplicate_preserving_order(
        [column for column in track_d_genome_rows[0].keys() if column != "phage"]
        + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
    )
    track_e_feature_columns = train_v1_binary_classifier._deduplicate_preserving_order(
        [column for column in track_e_rbp_rows[0].keys() if column not in train_v1_binary_classifier.IDENTIFIER_COLUMNS]
        + [
            column
            for column in track_e_defense_rows[0].keys()
            if column not in train_v1_binary_classifier.IDENTIFIER_COLUMNS
        ]
        + [
            column
            for column in track_e_isolation_rows[0].keys()
            if column not in train_v1_binary_classifier.IDENTIFIER_COLUMNS
        ]
    )
    all_feature_space = train_v1_binary_classifier.build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )
    merged_rows = train_v1_binary_classifier.merge_expanded_feature_rows(
        track_c_pair_rows,
        split_rows,
        phage_feature_blocks=(track_d_genome_rows, track_d_distance_rows),
        pair_feature_blocks=(track_e_rbp_rows, track_e_defense_rows, track_e_isolation_rows),
    )

    lock = locked_config["locked_v1_feature_configuration"]
    arm_spaces = build_locked_arm_feature_spaces(
        all_feature_space,
        winner_subset_blocks=lock["winner_subset_blocks"],
        excluded_columns=lock["deployment_realistic"]["excluded_label_derived_columns"],
    )
    params = dict(tg05_summary["locked_lightgbm_hyperparameters"])
    arm_results: Dict[str, Dict[str, object]] = {}
    for arm_name, arm_space in arm_spaces.items():
        estimator, vectorizer, _, _, feature_matrix, all_probabilities = score_rows_for_arm(
            merged_rows,
            arm_space,
            params=params,
            random_state=args.random_state,
        )
        scored_rows = []
        for row, probability in zip(merged_rows, all_probabilities):
            scored_row = dict(row)
            scored_row["predicted_probability"] = safe_round(float(probability))
            scored_rows.append(scored_row)

        calibration_rows = [
            row
            for row in scored_rows
            if row["split_holdout"] == "train_non_holdout"
            and row["split_cv5_fold"] == str(args.calibration_fold)
            and row["label_hard_any_lysis"] != ""
        ]
        confidence_intervals = bootstrap_probability_intervals(
            calibration_rows,
            scored_rows,
            bootstrap_samples=args.bootstrap_samples,
            random_state=args.random_state,
        )
        shap_rows_by_pair = build_shap_driver_rows(
            feature_matrix,
            estimator,
            list(vectorizer.get_feature_names_out()),
            scored_rows,
        )
        arm_results[arm_name] = {
            "feature_space": arm_space,
            "scored_rows": scored_rows,
            "confidence_intervals": confidence_intervals,
            "shap_rows_by_pair": shap_rows_by_pair,
            "display_rows": build_arm_display_rows(scored_rows, confidence_intervals, shap_rows_by_pair),
        }

    panel_rows_by_strain = arm_results["panel"]["display_rows"]
    deployment_rows_by_strain = arm_results["deployment"]["display_rows"]
    initial_bacteria = args.initial_bacteria or sorted(panel_rows_by_strain)[0]
    bundle = build_phagogram_bundle(
        config=locked_config,
        tg05_summary=tg05_summary,
        initial_bacteria=initial_bacteria,
        display_limit=args.display_limit,
        panel_rows_by_strain=panel_rows_by_strain,
        deployment_rows_by_strain=deployment_rows_by_strain,
    )

    output_bundle = args.output_dir / "tp01_digital_phagogram_bundle.json"
    output_html = args.output_dir / "tp01_digital_phagogram.html"
    output_summary = args.output_dir / "tp01_digital_phagogram_summary.json"

    output_bundle.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    output_html.write_text(render_digital_phagogram_html(bundle), encoding="utf-8")
    write_json(
        output_summary,
        {
            "generated_at_utc": bundle["generated_at_utc"],
            "task_id": "TP01",
            "initial_bacteria": bundle["initial_bacteria"],
            "strain_count": bundle["strain_count"],
            "display_limit": bundle["display_limit"],
            "outputs": {
                "bundle_json": str(output_bundle),
                "html": str(output_html),
            },
            "inputs": {
                "locked_v1_config": {
                    "path": str(args.locked_v1_config_path),
                    "sha256": sha256(args.locked_v1_config_path),
                },
                "tg05_summary": {
                    "path": str(args.tg05_summary_path),
                    "sha256": sha256(args.tg05_summary_path),
                },
            },
        },
    )

    print("TP01 completed.")
    print(f"- Strains covered: {bundle['strain_count']}")
    print(f"- Initial strain: {bundle['initial_bacteria']}")
    print(f"- Output HTML: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
