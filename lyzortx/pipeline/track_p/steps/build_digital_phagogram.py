#!/usr/bin/env python3
"""TP01: Build an interactive digital phagogram for per-strain phage ranking."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps import run_feature_subset_sweep
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier
from lyzortx.pipeline.track_g.steps.compute_shap_explanations import (
    format_contribution_summary,
    top_feature_contributions,
)
from lyzortx.pipeline.track_g.steps.run_feature_subset_sweep import (
    SweepArm,
    build_deployment_realistic_arm,
    build_subset_sweep_arms,
)

DEFAULT_LOCKED_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
DEFAULT_TG05_SUMMARY_PATH = Path(
    "lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/tg05_feature_subset_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_p/digital_phagogram")
CALIBRATION_FOLD = 0
TOP_K_RECOMMENDATIONS = 3
TOP_FEATURES_PER_PAIR = 3
PLOTLY_SCRIPT_PATTERN = re.compile(r'<script src="([^"]*plotly[^"]*)"></script>')


@dataclass(frozen=True)
class ModelArtifact:
    arm: SweepArm
    rows: List[Dict[str, object]]
    ranked_rows: List[Dict[str, object]]
    recommendation_rows: List[Dict[str, object]]
    holdout_binary_metrics: Dict[str, Optional[float]]
    holdout_top3_metrics: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locked-config-path",
        type=Path,
        default=DEFAULT_LOCKED_CONFIG_PATH,
        help="Locked v1 feature configuration JSON from TG05.",
    )
    parser.add_argument(
        "--tg05-summary-path",
        type=Path,
        default=DEFAULT_TG05_SUMMARY_PATH,
        help="TG05 summary JSON with the locked LightGBM hyperparameters.",
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
        default=Path(
            "lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/"
            "rbp_receptor_compatibility_features_v1.csv"
        ),
        help="Input Track E RBP-receptor compatibility feature CSV.",
    )
    parser.add_argument(
        "--track-e-defense-evasion-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/defense_evasion_proxy_feature_block/"
            "defense_evasion_proxy_features_v1.csv"
        ),
        help="Input Track E defense-evasion proxy feature CSV.",
    )
    parser.add_argument(
        "--track-e-isolation-distance-path",
        type=Path,
        default=Path(
            "lyzortx/generated_outputs/track_e/isolation_host_distance_feature_block/"
            "isolation_host_distance_features_v1.csv"
        ),
        help="Input Track E isolation-host distance feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated TP01 artifacts.",
    )
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=CALIBRATION_FOLD,
        help="Non-holdout CV fold used to fit the calibrators.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RECOMMENDATIONS,
        help="Number of phages to surface per strain in the dashboard.",
    )
    parser.add_argument(
        "--top-features-per-pair",
        type=int,
        default=TOP_FEATURES_PER_PAIR,
        help="Number of SHAP features to surface per recommended pair.",
    )
    parser.add_argument(
        "--initial-strain",
        type=str,
        default="",
        help="Strain to preselect in the dashboard. Defaults to the first available strain.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Fallback random state for the calibration models.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Assume prerequisite Track G outputs already exist instead of generating them when missing.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(values: Sequence[str]) -> Tuple[str, ...]:
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
    if not args.locked_config_path.exists() or not args.tg05_summary_path.exists():
        run_feature_subset_sweep.main(["--output-dir", str(args.tg05_summary_path.parent)])


def load_locked_v1_feature_configuration(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tg05_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_plotly_script_tag() -> str:
    import plotly.graph_objects as go
    import plotly.io as pio

    probe_html = pio.to_html(go.Figure(), include_plotlyjs="cdn", full_html=False)
    match = PLOTLY_SCRIPT_PATTERN.search(probe_html)
    if match is not None:
        return match.group(0)
    return '<script src="https://cdn.plot.ly/plotly-6.6.0.min.js"></script>'


def build_feature_space_from_inputs(
    st02_rows: Sequence[Mapping[str, str]],
    track_c_pair_rows: Sequence[Mapping[str, str]],
    track_d_feature_columns: Sequence[str],
    track_e_feature_columns: Sequence[str],
) -> train_v1_binary_classifier.FeatureSpace:
    return train_v1_binary_classifier.build_feature_space(
        st02_rows,
        track_c_pair_rows,
        track_d_feature_columns,
        track_e_feature_columns,
    )


def build_model_arms(
    feature_space: train_v1_binary_classifier.FeatureSpace,
    locked_configuration: Mapping[str, object],
) -> Tuple[SweepArm, SweepArm]:
    winner_blocks = tuple(str(block) for block in locked_configuration["winner_subset_blocks"])
    available_arms = build_subset_sweep_arms(feature_space)
    winning_arm = next(arm for arm in available_arms if arm.subset_blocks == winner_blocks)
    deployment_arm = build_deployment_realistic_arm(winning_arm)
    return winning_arm, deployment_arm


def _calibrate_probabilities(
    oof_rows: Sequence[Mapping[str, object]],
    *,
    calibration_fold: int,
    random_state: int,
) -> Tuple[IsotonicRegression, LogisticRegression]:
    calibration_rows = [
        row
        for row in oof_rows
        if str(row["split_cv5_fold"]) == str(calibration_fold) and row["label_hard_any_lysis"] != ""
    ]
    if not calibration_rows:
        raise ValueError("No calibration rows available for TP01.")

    x_calib = np.asarray([float(row["raw_probability"]) for row in calibration_rows], dtype=float)
    y_calib = np.asarray([int(str(row["label_hard_any_lysis"])) for row in calibration_rows], dtype=int)
    if len(np.unique(y_calib)) < 2:
        raise ValueError("Calibration fold has only one class for TP01.")

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(x_calib, y_calib)
    platt = LogisticRegression(solver="lbfgs", random_state=random_state, max_iter=1000)
    platt.fit(x_calib.reshape(-1, 1), y_calib)
    return isotonic, platt


def _dense_row(matrix: Any, row_index: int) -> np.ndarray:
    row = matrix[row_index]
    if hasattr(row, "toarray"):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


def _row_summary_from_contributions(contributions: Mapping[str, Sequence[Mapping[str, object]]]) -> str:
    positive = format_contribution_summary(contributions["positive"])
    negative = format_contribution_summary(contributions["negative"])
    if positive and negative:
        return f"+ {positive}; - {negative}"
    if positive:
        return f"+ {positive}"
    if negative:
        return f"- {negative}"
    return ""


def _select_top_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    probability_key: str,
    top_k: int,
) -> List[Dict[str, object]]:
    rows_by_bacteria: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_bacteria[str(row["bacteria"])].append(dict(row))

    selected: List[Dict[str, object]] = []
    for bacteria in sorted(rows_by_bacteria):
        ranked = sorted(rows_by_bacteria[bacteria], key=lambda row: (-float(row[probability_key]), str(row["phage"])))
        for rank, row in enumerate(ranked[:top_k], start=1):
            enriched = dict(row)
            enriched["recommendation_rank"] = rank
            selected.append(enriched)
    return selected


def _rank_holdout_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    probability_key: str,
) -> List[Dict[str, object]]:
    ranked_rows: List[Dict[str, object]] = []
    for row in rows:
        ranked_rows.append(dict(row))
    ranked_rows.sort(key=lambda row: (str(row["bacteria"]), -float(row[probability_key]), str(row["phage"])))
    return ranked_rows


def _fit_and_score_arm(
    merged_rows: Sequence[Mapping[str, object]],
    feature_space: train_v1_binary_classifier.FeatureSpace,
    arm: SweepArm,
    *,
    estimator_factory,
    best_params: Mapping[str, object],
    calibration_fold: int,
    random_state: int,
    top_k: int,
    top_features_per_pair: int,
) -> ModelArtifact:
    arm_feature_space = train_v1_binary_classifier.FeatureSpace(
        categorical_columns=arm.categorical_columns,
        numeric_columns=arm.numeric_columns,
        track_c_additional_columns=feature_space.track_c_additional_columns,
        track_d_columns=feature_space.track_d_columns,
        track_e_columns=feature_space.track_e_columns,
    )
    fold_datasets = train_v1_binary_classifier.prepare_fold_datasets(merged_rows, arm_feature_space)
    oof_rows = train_v1_binary_classifier.score_rows_with_cv_predictions(
        fold_datasets,
        estimator_factory=estimator_factory,
        best_params=best_params,
        probability_column="raw_probability",
    )
    isotonic, platt = _calibrate_probabilities(oof_rows, calibration_fold=calibration_fold, random_state=random_state)

    estimator, vectorizer, _, holdout_rows, raw_probabilities = train_v1_binary_classifier.fit_final_estimator(
        merged_rows,
        arm_feature_space,
        estimator_factory=estimator_factory,
        params=best_params,
    )
    scored_holdout_rows: List[Dict[str, object]] = []
    for row, raw_probability in zip(holdout_rows, raw_probabilities):
        calibrated_isotonic = float(isotonic.predict([raw_probability])[0])
        calibrated_platt = float(platt.predict_proba(np.asarray([[raw_probability]], dtype=float))[:, 1][0])
        confidence_low = min(calibrated_isotonic, calibrated_platt)
        confidence_high = max(calibrated_isotonic, calibrated_platt)
        scored_holdout_rows.append(
            {
                **row,
                "raw_probability": safe_round(float(raw_probability)),
                "predicted_probability": safe_round(calibrated_isotonic),
                "calibrated_probability_platt": safe_round(calibrated_platt),
                "confidence_band_low": safe_round(confidence_low),
                "confidence_band_high": safe_round(confidence_high),
                "confidence_band_width": safe_round(confidence_high - confidence_low),
            }
        )

    holdout_binary_metrics = train_v1_binary_classifier.compute_binary_metrics(
        [int(str(row["label_hard_any_lysis"])) for row in scored_holdout_rows],
        [float(row["predicted_probability"]) for row in scored_holdout_rows],
    )
    holdout_top3_metrics = train_v1_binary_classifier.compute_top3_hit_rate(
        scored_holdout_rows,
        probability_key="predicted_probability",
    )
    ranked_rows = _rank_holdout_rows(scored_holdout_rows, probability_key="predicted_probability")
    selected_rows = _select_top_rows(scored_holdout_rows, probability_key="predicted_probability", top_k=top_k)

    selected_pair_ids = {row["pair_id"] for row in selected_rows}
    explain_rows = [row for row in scored_holdout_rows if row["pair_id"] in selected_pair_ids]
    feature_matrix = vectorizer.transform(
        [
            train_v1_binary_classifier._build_feature_dict(
                row,
                categorical_columns=arm_feature_space.categorical_columns,
                numeric_columns=arm_feature_space.numeric_columns,
            )
            for row in explain_rows
        ]
    )

    import shap

    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(feature_matrix)
    shap_matrix = explanation.values
    base_values = np.asarray(explanation.base_values).ravel()
    feature_names = list(vectorizer.get_feature_names_out())
    explain_index_by_pair_id = {row["pair_id"]: index for index, row in enumerate(explain_rows)}

    shap_rows: List[Dict[str, object]] = []
    for row in selected_rows:
        explain_index = explain_index_by_pair_id.get(row["pair_id"])
        if explain_index is None:
            continue
        shap_row = _dense_row(shap_matrix, explain_index)
        feature_row = _dense_row(feature_matrix, explain_index)
        contributions = top_feature_contributions(
            shap_row,
            feature_row,
            feature_names,
            top_k=top_features_per_pair,
        )
        shap_rows.append(
            {
                **row,
                "shap_base_value": safe_round(float(base_values[explain_index])),
                "total_abs_shap": safe_round(float(np.abs(shap_row).sum())),
                "top_positive_feature_summary": format_contribution_summary(contributions["positive"]),
                "top_negative_feature_summary": format_contribution_summary(contributions["negative"]),
                "shap_summary": _row_summary_from_contributions(contributions),
                **{
                    f"top_positive_feature_{position}": item["feature_name"]
                    for position, item in enumerate(contributions["positive"], start=1)
                },
                **{
                    f"top_positive_shap_{position}": item["shap_value"]
                    for position, item in enumerate(contributions["positive"], start=1)
                },
                **{
                    f"top_negative_feature_{position}": item["feature_name"]
                    for position, item in enumerate(contributions["negative"], start=1)
                },
                **{
                    f"top_negative_shap_{position}": item["shap_value"]
                    for position, item in enumerate(contributions["negative"], start=1)
                },
            }
        )

    shap_rows.sort(key=lambda row: (str(row["bacteria"]), int(row["recommendation_rank"])))
    selected_rows.sort(key=lambda row: (str(row["bacteria"]), int(row["recommendation_rank"])))

    return ModelArtifact(
        arm=arm,
        rows=scored_holdout_rows,
        ranked_rows=ranked_rows,
        recommendation_rows=shap_rows,
        holdout_binary_metrics=holdout_binary_metrics,
        holdout_top3_metrics=holdout_top3_metrics,
    )


def build_dashboard_payload(
    panel_artifact: ModelArtifact,
    deployment_artifact: ModelArtifact,
    *,
    locked_configuration: Mapping[str, object],
    tg05_summary: Mapping[str, object],
    initial_strain: str,
) -> Dict[str, object]:
    strains = sorted({str(row["bacteria"]) for row in panel_artifact.rows})
    if initial_strain not in strains:
        initial_strain = strains[0] if strains else ""
    deployment_realistic = locked_configuration.get("deployment_realistic") or locked_configuration.get(
        "deployment_realistic_sensitivity"
    )

    def _rows_by_strain(rows: Sequence[Mapping[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["bacteria"])].append(dict(row))
        for strain_rows in grouped.values():
            strain_rows.sort(key=lambda row: int(row["recommendation_rank"]))
        return dict(grouped)

    def _summary_cards(artifact: ModelArtifact, model_label: str) -> Dict[str, object]:
        return {
            "model_label": model_label,
            "holdout_roc_auc": artifact.holdout_binary_metrics["roc_auc"],
            "holdout_brier_score": artifact.holdout_binary_metrics["brier_score"],
            "holdout_top3_all_strains": artifact.holdout_top3_metrics["top3_hit_rate_all_strains"],
            "holdout_top3_susceptible_only": artifact.holdout_top3_metrics["top3_hit_rate_susceptible_only"],
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TP01",
        "initial_strain": initial_strain,
        "strains": strains,
        "locked_configuration": {
            "winner_label": locked_configuration["winner_label"],
            "winner_subset_blocks": locked_configuration["winner_subset_blocks"],
            "panel_default": locked_configuration["panel_default"],
            "deployment_realistic": deployment_realistic,
            "selection_policy": locked_configuration["selection_policy"],
            "label_derived_columns_reviewed": locked_configuration["label_derived_columns_reviewed"],
        },
        "panel_summary": _summary_cards(panel_artifact, f"{panel_artifact.arm.display_name} (panel model)"),
        "deployment_summary": _summary_cards(
            deployment_artifact, f"{deployment_artifact.arm.display_name} (deployment-realistic)"
        ),
        "models": {
            "panel": {
                "arm_id": panel_artifact.arm.arm_id,
                "label": f"{panel_artifact.arm.display_name} (with host_n_infections)",
                "rows_by_strain": _rows_by_strain(panel_artifact.recommendation_rows),
            },
            "deployment": {
                "arm_id": deployment_artifact.arm.arm_id,
                "label": f"{deployment_artifact.arm.display_name} (without host_n_infections)",
                "rows_by_strain": _rows_by_strain(deployment_artifact.recommendation_rows),
            },
        },
        "inputs": {
            "tg05_summary": {
                "path": str(tg05_summary.get("source_path", "")),
                "sha256": tg05_summary.get("sha256", ""),
            },
        },
    }


def _render_metric_card(title: str, value: object, detail: str) -> str:
    return (
        f'<article class="metric-card"><span class="metric-title">{html.escape(title)}</span>'
        f'<span class="metric-value">{html.escape(str(value))}</span>'
        f'<span class="metric-detail">{html.escape(detail)}</span></article>'
    )


def render_dashboard_html(payload: Mapping[str, object]) -> str:
    plotly_script = _build_plotly_script_tag()
    strains = list(payload["strains"])
    initial_strain = str(payload["initial_strain"])
    panel_summary = payload["panel_summary"]
    deployment_summary = payload["deployment_summary"]
    locked_configuration = payload["locked_configuration"]
    payload_json = json.dumps(payload, sort_keys=True)

    control_options = "\n".join(
        f'<option value="{html.escape(str(strain))}">{html.escape(str(strain))}</option>' for strain in strains
    )

    summary_cards = (
        _render_metric_card("Panel top-3", panel_summary["holdout_top3_all_strains"], "with host_n_infections")
        + _render_metric_card(
            "Deployment top-3", deployment_summary["holdout_top3_all_strains"], "without host_n_infections"
        )
        + _render_metric_card("Panel AUC", panel_summary["holdout_roc_auc"], "TG05 locked panel model")
        + _render_metric_card(
            "Deployment AUC", deployment_summary["holdout_roc_auc"], "deployment-realistic sensitivity"
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track P digital phagogram</title>
  {plotly_script}
  <style>
    :root {{
      color-scheme: dark;
      --bg: #08111d;
      --bg-alt: #0f1b2f;
      --card: rgba(12, 23, 41, 0.84);
      --card-border: rgba(153, 198, 255, 0.16);
      --text: #eff5ff;
      --muted: #aab8d4;
      --accent: #75c4ff;
      --accent-2: #91f2c4;
      --warn: #ffb86b;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(117, 196, 255, 0.18), transparent 36%),
        radial-gradient(circle at top right, rgba(145, 242, 196, 0.12), transparent 28%),
        linear-gradient(180deg, #050b14 0%, var(--bg) 44%, #070f1a 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 24px 40px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.4fr 0.9fr;
      align-items: end;
      margin-bottom: 22px;
    }}
    .eyebrow {{
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      font-weight: 700;
      margin: 0 0 8px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.2rem, 4vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }}
    .lede {{
      margin: 14px 0 0;
      color: var(--muted);
      max-width: 74ch;
      font-size: 1rem;
      line-height: 1.6;
    }}
    .hero-side {{
      border: 1px solid var(--card-border);
      background: linear-gradient(180deg, rgba(15, 27, 47, 0.92), rgba(8, 17, 29, 0.92));
      border-radius: var(--radius);
      padding: 18px 18px 16px;
      box-shadow: var(--shadow);
    }}
    .hero-side .label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}
    .hero-side .value {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-top: 6px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      margin: 18px 0 24px;
    }}
    .control-card, .metric-card, .panel-card {{
      border: 1px solid var(--card-border);
      background: var(--card);
      backdrop-filter: blur(10px);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .control-card {{
      padding: 16px 18px;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    label {{
      display: block;
      font-size: 0.84rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    input[type="search"] {{
      width: 100%;
      padding: 14px 16px;
      font: inherit;
      color: var(--text);
      border: 1px solid rgba(153, 198, 255, 0.22);
      background: rgba(6, 13, 24, 0.78);
      border-radius: 14px;
    }}
    button {{
      padding: 13px 18px;
      border: 0;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent), #5e86ff);
      color: #08111d;
      font-weight: 800;
      cursor: pointer;
    }}
    button:hover {{ filter: brightness(1.04); }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      padding: 16px;
      display: grid;
      gap: 6px;
      min-height: 106px;
    }}
    .metric-title {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 0.76rem;
    }}
    .metric-value {{
      font-size: 1.7rem;
      line-height: 1;
      font-weight: 800;
    }}
    .metric-detail {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .board {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .panel-card {{
      padding: 18px;
      overflow: hidden;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .panel-title {{
      margin: 0;
      font-size: 1.24rem;
      letter-spacing: -0.02em;
    }}
    .panel-subtitle {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(117, 196, 255, 0.12);
      color: var(--accent);
      font-size: 0.82rem;
      white-space: nowrap;
    }}
    .plot {{
      width: 100%;
      height: 340px;
    }}
    .rank-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      overflow: hidden;
      border-radius: 16px;
    }}
    .rank-table th, .rank-table td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(153, 198, 255, 0.11);
      vertical-align: top;
      text-align: left;
      font-size: 0.88rem;
    }}
    .rank-table th {{
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }}
    .rank-table tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .confidence {{
      color: var(--accent-2);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .shap-block {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
    }}
    @media (max-width: 1100px) {{
      .hero, .controls, .board, .summary-grid {{ grid-template-columns: 1fr; }}
      .control-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Track P digital phagogram</p>
        <h1>Per-strain phage ranking, calibrated and explained.</h1>
        <p class="lede">
          Side-by-side ranking for the locked panel model and the deployment-realistic model. Both use the TG05
          winner configuration; the panel view keeps <code>host_n_infections</code>, while the deployment view removes
          that label-derived shortcut and exposes a more honest live-demo ranking.
        </p>
      </div>
      <div class="hero-side">
        <div class="label">Locked configuration</div>
        <div class="value">{html.escape(str(locked_configuration["winner_label"]))}</div>
        <div class="lede" style="margin-top:10px;">
          Winner blocks: {html.escape(", ".join(str(block) for block in locked_configuration["winner_subset_blocks"]))}<br>
          Deployment exclusion: {html.escape(", ".join(str(item) for item in locked_configuration["deployment_realistic"]["excluded_label_derived_columns"]))}
        </div>
      </div>
    </section>

    <section class="controls">
      <div class="control-card">
        <label for="strain-input">Strain</label>
        <div class="control-row">
          <input id="strain-input" type="search" list="strain-options" value="{html.escape(initial_strain)}" placeholder="Type a strain ID">
          <button id="update-button" type="button">Update phagogram</button>
        </div>
        <datalist id="strain-options">{control_options}</datalist>
      </div>
      <div class="summary-grid">
        {summary_cards}
      </div>
    </section>

    <main class="board">
      <article class="panel-card">
        <div class="panel-header">
          <div>
            <h2 class="panel-title" id="panel-title">Panel model</h2>
            <p class="panel-subtitle" id="panel-subtitle">With host_n_infections, TG05 winner arm.</p>
          </div>
          <div class="pill" id="panel-pill">P(lysis) with confidence band</div>
        </div>
        <div id="panel-plot" class="plot"></div>
        <table class="rank-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Phage</th>
              <th>P(lysis)</th>
              <th>Confidence band</th>
              <th>Top SHAP features</th>
            </tr>
          </thead>
          <tbody id="panel-table-body"></tbody>
        </table>
      </article>
      <article class="panel-card">
        <div class="panel-header">
          <div>
            <h2 class="panel-title" id="deployment-title">Deployment-realistic model</h2>
            <p class="panel-subtitle" id="deployment-subtitle">Without host_n_infections, same locked feature family.</p>
          </div>
          <div class="pill" id="deployment-pill">P(lysis) with confidence band</div>
        </div>
        <div id="deployment-plot" class="plot"></div>
        <table class="rank-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Phage</th>
              <th>P(lysis)</th>
              <th>Confidence band</th>
              <th>Top SHAP features</th>
            </tr>
          </thead>
          <tbody id="deployment-table-body"></tbody>
        </table>
      </article>
    </main>
    <p class="footer-note">
      Confidence bands are the isotonic-versus-Platt calibrated interval for each pair. Hover a bar to see the phage
      family, calibration range, and SHAP drivers. Both charts are ordered by calibrated P(lysis).
    </p>
  </div>
  <script>
    const PHAGOGRAM = {payload_json};

    function escapeHtml(value) {{
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function rowsFor(modelKey, strain) {{
      const model = PHAGOGRAM.models[modelKey];
      return model.rows_by_strain[strain] || [];
    }}

    function renderRows(modelKey, strain, plotDivId, tableBodyId, titleId, subtitleId, pillId, modelLabel) {{
      const rows = rowsFor(modelKey, strain);
      const title = document.getElementById(titleId);
      const subtitle = document.getElementById(subtitleId);
      const pill = document.getElementById(pillId);
      title.textContent = `${{modelLabel}} - ${{strain}}`;
      subtitle.textContent = rows.length ? `Top ${{rows.length}} phages for ${{strain}}.` : 'No ranked phages available.';
      pill.textContent = rows.length ? `Top-ranked phages: ${{rows.length}}` : 'No ranked phages';

      const x = rows.map(row => Number(row.predicted_probability));
      const y = rows.map(row => row.phage);
      const upper = rows.map(row => Number(row.confidence_band_high) - Number(row.predicted_probability));
      const lower = rows.map(row => Number(row.predicted_probability) - Number(row.confidence_band_low));
      const hoverText = rows.map(row => {{
        return `Rank ${{row.recommendation_rank}}<br>` +
          `${{escapeHtml(row.phage)}}<br>` +
          `P(lysis): ${{Number(row.predicted_probability).toFixed(3)}}<br>` +
          `Band: ${{Number(row.confidence_band_low).toFixed(3)}} - ${{Number(row.confidence_band_high).toFixed(3)}}<br>` +
          `${{escapeHtml(row.shap_summary)}}`;
      }});
      const trace = {{
        type: 'bar',
        orientation: 'h',
        x: x,
        y: y,
        text: rows.map(row => Number(row.predicted_probability).toFixed(3)),
        textposition: 'outside',
        marker: {{
          color: modelKey === 'panel' ? '#75c4ff' : '#91f2c4',
          line: {{ color: 'rgba(255,255,255,0.25)', width: 1 }},
        }},
        hovertemplate: '%{{customdata}}<extra></extra>',
        customdata: hoverText,
        error_x: {{
          type: 'data',
          array: upper,
          arrayminus: lower,
          color: 'rgba(255,255,255,0.4)',
          thickness: 1.5,
          width: 0,
        }},
      }};
      const layout = {{
        margin: {{ l: 18, r: 24, t: 8, b: 28 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {{
          range: [0, 1],
          title: 'P(lysis)',
          gridcolor: 'rgba(153, 198, 255, 0.14)',
          zeroline: false,
          tickfont: {{ color: '#aab8d4' }},
          titlefont: {{ color: '#aab8d4' }},
        }},
        yaxis: {{
          autorange: 'reversed',
          tickfont: {{ color: '#eff5ff' }},
          gridcolor: 'rgba(153, 198, 255, 0.08)',
        }},
        font: {{ family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#eff5ff' }},
        bargap: 0.38,
        showlegend: false,
        height: 340,
      }};
      Plotly.newPlot(plotDivId, [trace], layout, {{displayModeBar: false, responsive: true}});

      const tbody = document.getElementById(tableBodyId);
      tbody.innerHTML = rows.map(row => {{
        const band = `${{Number(row.confidence_band_low).toFixed(3)}} - ${{Number(row.confidence_band_high).toFixed(3)}}`;
        return `<tr>
          <td>${{row.recommendation_rank}}</td>
          <td>${{escapeHtml(row.phage)}}</td>
          <td>${{Number(row.predicted_probability).toFixed(3)}}</td>
          <td class="confidence">${{band}}</td>
          <td class="shap-block">${{escapeHtml(row.shap_summary || '')}}</td>
        </tr>`;
      }}).join('');
    }}

    function renderStrain(strain) {{
      const panelLabel = PHAGOGRAM.models.panel.label;
      const deploymentLabel = PHAGOGRAM.models.deployment.label;
      renderRows('panel', strain, 'panel-plot', 'panel-table-body', 'panel-title', 'panel-subtitle', 'panel-pill', panelLabel);
      renderRows(
        'deployment',
        strain,
        'deployment-plot',
        'deployment-table-body',
        'deployment-title',
        'deployment-subtitle',
        'deployment-pill',
        deploymentLabel,
      );
    }}

    document.getElementById('update-button').addEventListener('click', () => {{
      const strain = document.getElementById('strain-input').value.trim();
      if (strain && PHAGOGRAM.strains.includes(strain)) {{
        renderStrain(strain);
      }}
    }});

    document.getElementById('strain-input').addEventListener('change', event => {{
      const strain = event.target.value.trim();
      if (strain && PHAGOGRAM.strains.includes(strain)) {{
        renderStrain(strain);
      }}
    }});

    if (PHAGOGRAM.initial_strain) {{
      renderStrain(PHAGOGRAM.initial_strain);
    }}
  </script>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    ensure_prerequisite_outputs(args)

    locked_configuration = load_locked_v1_feature_configuration(args.locked_config_path)[
        "locked_v1_feature_configuration"
    ]
    tg05_summary = load_tg05_summary(args.tg05_summary_path)

    st02_rows = read_csv_rows(args.st02_pair_table_path)
    split_rows = read_csv_rows(args.st03_split_assignments_path)
    track_c_pair_rows = read_csv_rows(args.track_c_pair_table_path)
    track_d_genome_rows = read_csv_rows(args.track_d_genome_kmer_path)
    track_d_distance_rows = read_csv_rows(args.track_d_distance_path)
    track_e_rbp_rows = read_csv_rows(args.track_e_rbp_compatibility_path)
    track_e_defense_rows = read_csv_rows(args.track_e_defense_evasion_path)
    track_e_isolation_rows = read_csv_rows(args.track_e_isolation_distance_path)

    track_d_feature_columns = _deduplicate(
        [column for column in track_d_genome_rows[0].keys() if column != "phage"]
        + [column for column in track_d_distance_rows[0].keys() if column != "phage"]
    )
    track_e_feature_columns = _deduplicate(
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
    feature_space = build_feature_space_from_inputs(
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

    tg05_best_params = tg05_summary["locked_lightgbm_hyperparameters"]
    lightgbm_factory = lambda params, seed_offset: train_v1_binary_classifier.make_lightgbm_estimator(  # noqa: E731
        params,
        seed_offset,
        base_random_state=args.random_state,
    )
    panel_arm, deployment_arm = build_model_arms(feature_space, locked_configuration)

    panel_artifact = _fit_and_score_arm(
        merged_rows,
        feature_space,
        panel_arm,
        estimator_factory=lightgbm_factory,
        best_params=tg05_best_params,
        calibration_fold=args.calibration_fold,
        random_state=args.random_state,
        top_k=args.top_k,
        top_features_per_pair=args.top_features_per_pair,
    )
    deployment_artifact = _fit_and_score_arm(
        merged_rows,
        feature_space,
        deployment_arm,
        estimator_factory=lightgbm_factory,
        best_params=tg05_best_params,
        calibration_fold=args.calibration_fold,
        random_state=args.random_state,
        top_k=args.top_k,
        top_features_per_pair=args.top_features_per_pair,
    )

    payload = build_dashboard_payload(
        panel_artifact,
        deployment_artifact,
        locked_configuration=locked_configuration,
        tg05_summary={
            "source_path": str(args.tg05_summary_path),
            "sha256": _sha256(args.tg05_summary_path),
        },
        initial_strain=args.initial_strain,
    )

    ranked_rows: List[Dict[str, object]] = []
    for model_key, artifact in (("panel", panel_artifact), ("deployment", deployment_artifact)):
        for row in artifact.recommendation_rows:
            ranked_rows.append(
                {
                    "model_key": model_key,
                    "model_label": artifact.arm.display_name,
                    "bacteria": row["bacteria"],
                    "phage": row["phage"],
                    "phage_family": row["phage_family"],
                    "recommendation_rank": row["recommendation_rank"],
                    "predicted_probability": row["predicted_probability"],
                    "confidence_band_low": row["confidence_band_low"],
                    "confidence_band_high": row["confidence_band_high"],
                    "confidence_band_width": row["confidence_band_width"],
                    "top_positive_feature_summary": row["top_positive_feature_summary"],
                    "top_negative_feature_summary": row["top_negative_feature_summary"],
                    "shap_summary": row["shap_summary"],
                }
            )
    ranked_rows.sort(key=lambda row: (str(row["model_key"]), str(row["bacteria"]), int(row["recommendation_rank"])))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TP01",
        "locked_configuration": locked_configuration,
        "panel_metrics": {
            "holdout_roc_auc": panel_artifact.holdout_binary_metrics["roc_auc"],
            "holdout_brier_score": panel_artifact.holdout_binary_metrics["brier_score"],
            "holdout_top3_hit_rate_all_strains": panel_artifact.holdout_top3_metrics["top3_hit_rate_all_strains"],
            "holdout_top3_hit_rate_susceptible_only": panel_artifact.holdout_top3_metrics[
                "top3_hit_rate_susceptible_only"
            ],
        },
        "deployment_metrics": {
            "holdout_roc_auc": deployment_artifact.holdout_binary_metrics["roc_auc"],
            "holdout_brier_score": deployment_artifact.holdout_binary_metrics["brier_score"],
            "holdout_top3_hit_rate_all_strains": deployment_artifact.holdout_top3_metrics["top3_hit_rate_all_strains"],
            "holdout_top3_hit_rate_susceptible_only": deployment_artifact.holdout_top3_metrics[
                "top3_hit_rate_susceptible_only"
            ],
        },
        "recommendation_count": args.top_k,
        "top_features_per_pair": args.top_features_per_pair,
        "strain_count": len(payload["strains"]),
        "panel_recommendation_rows": len(panel_artifact.recommendation_rows),
        "deployment_recommendation_rows": len(deployment_artifact.recommendation_rows),
        "inputs": {
            "locked_config": {"path": str(args.locked_config_path), "sha256": _sha256(args.locked_config_path)},
            "tg05_summary": {"path": str(args.tg05_summary_path), "sha256": _sha256(args.tg05_summary_path)},
            "st02_pair_table": {"path": str(args.st02_pair_table_path), "sha256": _sha256(args.st02_pair_table_path)},
            "st03_split_assignments": {
                "path": str(args.st03_split_assignments_path),
                "sha256": _sha256(args.st03_split_assignments_path),
            },
            "track_c_pair_table": {
                "path": str(args.track_c_pair_table_path),
                "sha256": _sha256(args.track_c_pair_table_path),
            },
            "track_d_genome_kmers": {
                "path": str(args.track_d_genome_kmer_path),
                "sha256": _sha256(args.track_d_genome_kmer_path),
            },
            "track_d_distance": {
                "path": str(args.track_d_distance_path),
                "sha256": _sha256(args.track_d_distance_path),
            },
            "track_e_rbp_receptor_compatibility": {
                "path": str(args.track_e_rbp_compatibility_path),
                "sha256": _sha256(args.track_e_rbp_compatibility_path),
            },
            "track_e_defense_evasion": {
                "path": str(args.track_e_defense_evasion_path),
                "sha256": _sha256(args.track_e_defense_evasion_path),
            },
            "track_e_isolation_host_distance": {
                "path": str(args.track_e_isolation_distance_path),
                "sha256": _sha256(args.track_e_isolation_distance_path),
            },
        },
    }

    html_output = render_dashboard_html(payload)
    args.output_dir.joinpath("tp01_digital_phagogram.html").write_text(html_output, encoding="utf-8")
    write_csv(
        args.output_dir / "tp01_ranked_phagogram_rows.csv",
        [
            "model_key",
            "model_label",
            "bacteria",
            "phage",
            "phage_family",
            "recommendation_rank",
            "predicted_probability",
            "confidence_band_low",
            "confidence_band_high",
            "confidence_band_width",
            "top_positive_feature_summary",
            "top_negative_feature_summary",
            "shap_summary",
        ],
        ranked_rows,
    )
    write_json(args.output_dir / "tp01_phagogram_summary.json", summary)

    print("TP01 completed.")
    print(f"- Panel model top-3 hit rate: {panel_artifact.holdout_top3_metrics['top3_hit_rate_all_strains']}")
    print(f"- Deployment model top-3 hit rate: {deployment_artifact.holdout_top3_metrics['top3_hit_rate_all_strains']}")
    print(f"- Output HTML: {args.output_dir / 'tp01_digital_phagogram.html'}")
    print(f"- Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
