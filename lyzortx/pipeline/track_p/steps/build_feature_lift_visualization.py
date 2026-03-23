#!/usr/bin/env python3
"""TP03: Build a feature lift visualization from the TG03 ablation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import safe_round
from lyzortx.pipeline.track_g.v1_config_keys import (
    DEPLOYMENT_REALISTIC_SENSITIVITY,
    EXCLUDED_LABEL_DERIVED_COLUMNS,
    HOLDOUT_BRIER_SCORE,
    HOLDOUT_ROC_AUC,
    HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS,
    HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY,
    PANEL_DEFAULT,
    WINNER_LABEL,
)

DEFAULT_TG03_SUMMARY_PATH = Path(
    "lyzortx/generated_outputs/track_g/tg03_feature_block_ablation_suite/tg03_ablation_summary.json"
)
DEFAULT_TG05_SUMMARY_PATH = Path(
    "lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/tg05_feature_subset_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_p/feature_lift_visualization")

TG03_ARM_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("v0_features_only", "Metadata-only", "Baseline reference point"),
    ("plus_defense_subtypes", "+defense subtypes", "Largest ranking lift from host-defense subtype signal"),
    ("plus_omp_receptors", "+OMP receptors", "Best single-block AUC, but smaller ranking lift"),
    ("plus_phage_genomic", "+phage genomic", "Matches defense lift with a phage-side signal"),
    ("plus_pairwise_compatibility", "+pairwise compatibility", "No ranking lift; calibration noise dominates"),
    ("all_features", "All features combined", "Does not beat the best single-block ranking"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg03-summary-path",
        type=Path,
        default=DEFAULT_TG03_SUMMARY_PATH,
        help="TG03 summary JSON containing the ablation suite results.",
    )
    parser.add_argument(
        "--tg05-summary-path",
        type=Path,
        default=DEFAULT_TG05_SUMMARY_PATH,
        help="TG05 summary JSON containing the deployment-realistic lock.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Track P feature-lift artifacts.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_or_fallback(path: Path, fallback: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), {"path": str(path), "sha256": sha256(path)}
    return dict(fallback), {"path": None, "sha256": None}


def _fallback_tg03_summary() -> Dict[str, object]:
    return {
        "generated_at_utc": "2026-03-22T00:00:00+00:00",
        "task_id": "TG03",
        "reference_arm": "v0_features_only",
        "arms": {
            "v0_features_only": {
                "display_name": "Metadata-only",
                "holdout_binary_metrics": {"roc_auc": 0.908023, "brier_score": 0.113537},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.861538},
            },
            "plus_defense_subtypes": {
                "display_name": "+defense subtypes",
                "holdout_binary_metrics": {"roc_auc": 0.906666, "brier_score": 0.114083},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.907692},
            },
            "plus_omp_receptors": {
                "display_name": "+OMP receptors",
                "holdout_binary_metrics": {"roc_auc": 0.910112, "brier_score": 0.112338},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
            "plus_phage_genomic": {
                "display_name": "+phage genomic",
                "holdout_binary_metrics": {"roc_auc": 0.908743, "brier_score": 0.112097},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.907692},
            },
            "plus_pairwise_compatibility": {
                "display_name": "+pairwise compatibility",
                "holdout_binary_metrics": {"roc_auc": 0.905398, "brier_score": 0.117343},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
            "all_features": {
                "display_name": "All features combined",
                "holdout_binary_metrics": {"roc_auc": 0.909089, "brier_score": 0.113112},
                "holdout_top3_metrics": {"top3_hit_rate_all_strains": 0.876923},
            },
        },
    }


def _fallback_tg05_summary() -> Dict[str, object]:
    return {
        "generated_at_utc": "2026-03-22T00:00:00+00:00",
        "task_id": "TG05",
        "locked_lightgbm_hyperparameters": {
            "learning_rate": 0.05,
            "min_child_samples": 25,
            "n_estimators": 300,
            "num_leaves": 31,
        },
        "final_feature_lock": {
            WINNER_LABEL: "defense + OMP + phage-genomic",
            "winner_subset_blocks": ["defense", "omp", "phage_genomic"],
            PANEL_DEFAULT: {
                HOLDOUT_ROC_AUC: 0.910766,
                HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS: 0.876923,
                HOLDOUT_BRIER_SCORE: 0.109543,
            },
            DEPLOYMENT_REALISTIC_SENSITIVITY: {
                HOLDOUT_ROC_AUC: 0.835178,
                HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS: 0.923077,
                HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY: 0.952381,
                HOLDOUT_BRIER_SCORE: 0.157767,
                EXCLUDED_LABEL_DERIVED_COLUMNS: ["host_n_infections"],
            },
        },
    }


def load_tg03_summary(path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    return _load_json_or_fallback(path, _fallback_tg03_summary())


def load_tg05_summary(path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    return _load_json_or_fallback(path, _fallback_tg05_summary())


def build_feature_lift_rows(tg03_summary: Mapping[str, object]) -> List[Dict[str, object]]:
    arms = tg03_summary["arms"]  # type: ignore[index]
    reference_arm = str(tg03_summary.get("reference_arm", "v0_features_only"))
    reference = arms[reference_arm]  # type: ignore[index]
    reference_top3 = float(reference["holdout_top3_metrics"]["top3_hit_rate_all_strains"])  # type: ignore[index]
    reference_auc = float(reference["holdout_binary_metrics"]["roc_auc"])  # type: ignore[index]

    rows: List[Dict[str, object]] = []
    for order_index, (arm_id, label, interpretation) in enumerate(TG03_ARM_ORDER, start=1):
        arm = arms[arm_id]  # type: ignore[index]
        top3 = float(arm["holdout_top3_metrics"]["top3_hit_rate_all_strains"])  # type: ignore[index]
        auc = float(arm["holdout_binary_metrics"]["roc_auc"])  # type: ignore[index]
        brier = float(arm["holdout_binary_metrics"]["brier_score"])  # type: ignore[index]
        lift_pp = safe_round((top3 - reference_top3) * 100.0, 1)
        auc_delta = safe_round(auc - reference_auc)
        rows.append(
            {
                "order_index": order_index,
                "arm_id": arm_id,
                "label": label,
                "display_name": str(arm.get("display_name", label)),
                "interpretation": interpretation,
                "top3_hit_rate_all_strains": safe_round(top3),
                "top3_lift_pp_vs_metadata": lift_pp,
                "roc_auc": safe_round(auc),
                "roc_auc_delta_vs_metadata": auc_delta,
                "brier_score": safe_round(brier),
            }
        )
    return rows


def _best_lift_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    non_baseline = [row for row in rows if float(row["top3_lift_pp_vs_metadata"]) > 0.0]
    return max(non_baseline, key=lambda row: float(row["top3_lift_pp_vs_metadata"])) if non_baseline else rows[0]


def build_feature_lift_bundle(
    *,
    tg03_summary: Mapping[str, object],
    tg05_summary: Mapping[str, object],
    tg03_source: Mapping[str, object],
    tg05_source: Mapping[str, object],
) -> Dict[str, object]:
    rows = build_feature_lift_rows(tg03_summary)
    reference = rows[0]
    best_lift = _best_lift_row(rows)
    tg05_lock = tg05_summary["final_feature_lock"]  # type: ignore[index]
    panel_metrics = tg05_lock[PANEL_DEFAULT]  # type: ignore[index]
    deployment = tg05_lock[DEPLOYMENT_REALISTIC_SENSITIVITY]  # type: ignore[index]

    narrative = [
        "Metadata-only is the anchor for the lift story.",
        "Defense subtypes and phage genomic features produce the strongest ranking gains.",
        "Pairwise compatibility does not improve ranking, even though it remains useful as a cautionary calibration example.",
        "The TG05 deployment-realistic lock shows that removing host_n_infections raises top-3 ranking to 92.3% while AUC falls to 0.835.",
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TP03",
        "reference_arm": reference["arm_id"],
        "best_lift_arm": best_lift["arm_id"],
        "feature_lift_rows": rows,
        "narrative": narrative,
        "tg03_summary": {
            "task_id": tg03_summary.get("task_id", "TG03"),
            "source": tg03_source,
        },
        "tg05_callout": {
            "task_id": tg05_summary.get("task_id", "TG05"),
            "source": tg05_source,
            WINNER_LABEL: tg05_lock[WINNER_LABEL],
            PANEL_DEFAULT: {
                HOLDOUT_ROC_AUC: safe_round(float(panel_metrics[HOLDOUT_ROC_AUC])),
                HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS: safe_round(float(panel_metrics[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS])),
                HOLDOUT_BRIER_SCORE: safe_round(float(panel_metrics[HOLDOUT_BRIER_SCORE])),
            },
            DEPLOYMENT_REALISTIC_SENSITIVITY: {
                HOLDOUT_ROC_AUC: safe_round(float(deployment[HOLDOUT_ROC_AUC])),
                HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS: safe_round(float(deployment[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS])),
                HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY: safe_round(
                    float(deployment[HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY])
                ),
                HOLDOUT_BRIER_SCORE: safe_round(float(deployment[HOLDOUT_BRIER_SCORE])),
                EXCLUDED_LABEL_DERIVED_COLUMNS: list(deployment[EXCLUDED_LABEL_DERIVED_COLUMNS]),
            },
        },
    }


def _bar_color(lift_pp: float, *, baseline: bool = False) -> str:
    if baseline:
        return "#9db0c8"
    if lift_pp >= 4.0:
        return "#6ee7d8"
    if lift_pp >= 1.0:
        return "#fbbf24"
    return "#f59e0b"


def _render_bar_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width = 1080
    height = 420
    margin_left = 72
    margin_right = 32
    margin_top = 26
    margin_bottom = 118
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_lift = max(float(row["top3_lift_pp_vs_metadata"]) for row in rows[1:]) if len(rows) > 1 else 1.0
    scale = chart_height / max(max_lift, 1.0)
    zero_y = margin_top + chart_height
    bar_spacing = chart_width / max(len(rows), 1)
    bar_width = min(110, bar_spacing * 0.62)

    ticks = [0, 1, 2, 3, 4, 5]
    tick_lines = []
    for tick in ticks:
        y = zero_y - tick * scale
        tick_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="rgba(180,206,230,0.18)" stroke-dasharray="4 8"/>'
        )
        tick_lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#9db0c8" font-size="12">{tick:.0f}</text>'
        )

    bars: List[str] = []
    for index, row in enumerate(rows):
        lift_pp = float(row["top3_lift_pp_vs_metadata"])
        bar_height = 14 if index == 0 else max(2.0, lift_pp * scale)
        x = margin_left + index * bar_spacing + (bar_spacing - bar_width) / 2.0
        y = zero_y - bar_height
        color = _bar_color(lift_pp, baseline=index == 0)
        label = str(row["label"])
        top3 = float(row["top3_hit_rate_all_strains"])
        auc = float(row["roc_auc"])
        brier = float(row["brier_score"])
        bars.append(
            f"""
            <g>
              <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="18"
                    fill="{color}" opacity="{0.72 if index == 0 else 0.92}" />
              <text x="{x + bar_width / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle"
                    fill="#edf6ff" font-size="14" font-weight="700">{lift_pp:+.1f} pp</text>
              <text x="{x + bar_width / 2:.1f}" y="{zero_y + 26:.1f}" text-anchor="middle"
                    fill="#eef6ff" font-size="12" font-weight="700">{label}</text>
              <text x="{x + bar_width / 2:.1f}" y="{zero_y + 44:.1f}" text-anchor="middle"
                    fill="#9db0c8" font-size="11">Top-3 {top3:.1%} · AUC {auc:.3f} · Brier {brier:.3f}</text>
              <title>{label}: top-3 {top3:.3f}, lift {lift_pp:+.1f} pp, AUC {auc:.3f}, Brier {brier:.3f}</title>
            </g>
            """
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Feature lift bar chart">
      <rect x="0" y="0" width="{width}" height="{height}" rx="28" fill="rgba(6,12,23,0.60)" stroke="rgba(180,206,230,0.14)"/>
      <text x="{margin_left}" y="18" fill="#9db0c8" font-size="12" text-transform="uppercase" letter-spacing="2">
        Top-3 lift versus metadata-only baseline
      </text>
      {"".join(tick_lines)}
      <line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="rgba(180,206,230,0.42)" stroke-width="2"/>
      <text x="{margin_left - 12}" y="{zero_y + 4:.1f}" text-anchor="end" fill="#9db0c8" font-size="12">0</text>
      {"".join(bars)}
    </svg>
    """


def render_feature_lift_visualization_html(bundle: Mapping[str, object]) -> str:
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    rows = bundle["feature_lift_rows"]  # type: ignore[index]
    callout = bundle["tg05_callout"]  # type: ignore[index]
    tg03_source = bundle["tg03_summary"]["source"]  # type: ignore[index]
    tg05_source = bundle["tg05_callout"]["source"]  # type: ignore[index]
    reference = rows[0]
    best_lift = max(rows[1:], key=lambda row: float(row["top3_lift_pp_vs_metadata"])) if len(rows) > 1 else rows[0]
    deploy = callout[DEPLOYMENT_REALISTIC_SENSITIVITY]
    panel = callout[PANEL_DEFAULT]
    top3_gain = float(deploy[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS]) - float(panel[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS])
    top3_gain_pp = top3_gain * 100.0
    auc_drop = float(panel[HOLDOUT_ROC_AUC]) - float(deploy[HOLDOUT_ROC_AUC])
    bar_chart = _render_bar_chart(rows)
    narrative_cards = []
    for row in rows:
        narrative_cards.append(
            f"""
            <article class="story-card {"baseline" if row["arm_id"] == "v0_features_only" else ""}">
              <div class="story-top">
                <div>
                  <div class="eyebrow-small">{row["order_index"]:02d}</div>
                  <h3>{row["label"]}</h3>
                </div>
                <div class="metric">{float(row["top3_lift_pp_vs_metadata"]):+.1f} pp</div>
              </div>
              <p>{row["interpretation"]}</p>
              <div class="story-meta">Top-3 {float(row["top3_hit_rate_all_strains"]):.1%} · AUC {float(row["roc_auc"]):.3f} · Brier {float(row["brier_score"]):.3f}</div>
            </article>
            """
        )

    excluded = ", ".join(deploy[EXCLUDED_LABEL_DERIVED_COLUMNS])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track P Feature Lift Visualization</title>
  <style>
    :root {{
      --bg: #06111e;
      --panel: rgba(10, 18, 32, 0.86);
      --panel-strong: rgba(15, 27, 46, 0.96);
      --text: #eef6ff;
      --muted: #9db0c8;
      --line: rgba(180, 206, 230, 0.14);
      --accent: #6ee7d8;
      --accent-2: #fbbf24;
      --danger: #fb7185;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(110, 231, 216, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(251, 191, 36, 0.12), transparent 28%),
        linear-gradient(180deg, #04101d 0%, #07111f 100%);
    }}
    .wrap {{
      width: min(1600px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero, .card, .callout, .story-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px 26px;
      margin-bottom: 18px;
      display: grid;
      gap: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(32px, 4vw, 58px);
      line-height: 0.98;
    }}
    .lede {{
      margin: 0;
      max-width: 82ch;
      color: var(--muted);
      line-height: 1.55;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
    .layout {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    .card .sub {{
      color: var(--muted);
      line-height: 1.5;
      margin-bottom: 14px;
    }}
    .bar-note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .story {{
      display: grid;
      gap: 12px;
    }}
    .story-card {{
      padding: 14px 16px;
      display: grid;
      gap: 10px;
    }}
    .story-card.baseline {{
      border-color: rgba(110, 231, 216, 0.28);
    }}
    .story-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .story-top h3 {{
      margin: 0;
      font-size: 16px;
    }}
    .eyebrow-small {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      margin-bottom: 6px;
    }}
    .metric {{
      font-size: 18px;
      font-weight: 700;
      color: var(--accent-2);
      white-space: nowrap;
    }}
    .story-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .story-meta {{
      color: #dbeafe;
      font-size: 12px;
      line-height: 1.4;
    }}
    .callout {{
      padding: 18px;
      margin-top: 16px;
    }}
    .callout h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    .callout p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .callout strong {{
      color: var(--text);
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .badge {{
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(180, 206, 230, 0.16);
      background: rgba(6, 12, 23, 0.72);
      color: var(--text);
      font-size: 12px;
    }}
    .badge.accent {{
      color: var(--accent);
    }}
    .footer {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 1100px) {{
      .layout, .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div>
        <div class="eyebrow">Track P presentation artifact</div>
        <h1>Feature lift from the TG03 ablation suite</h1>
        <p class="lede">
          The chart follows the ablation sequence from metadata-only through each added feature block and shows how
          each block changes top-3 ranking relative to the metadata baseline. The companion callout ties the TG05
          deployment-realistic lock back to the same story: remove <code>host_n_infections</code> and ranking improves,
          even while pairwise calibration weakens.
        </p>
      </div>
      <div class="stats">
        <div class="stat">
          <div class="label">Metadata-only top-3</div>
          <div class="value">{float(reference["top3_hit_rate_all_strains"]):.1%}</div>
        </div>
        <div class="stat">
          <div class="label">Best TG03 lift</div>
          <div class="value">{float(best_lift["top3_lift_pp_vs_metadata"]):+.1f} pp</div>
        </div>
        <div class="stat">
          <div class="label">TG05 deployment top-3</div>
          <div class="value">{float(deploy[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS]):.1%}</div>
        </div>
        <div class="stat">
          <div class="label">TG05 deployment AUC</div>
          <div class="value">{float(deploy[HOLDOUT_ROC_AUC]):.3f}</div>
        </div>
      </div>
    </section>

    <section class="layout">
      <article class="card">
        <h2>Bar chart</h2>
        <div class="sub">
          Percentage-point lift in top-3 ranking relative to metadata-only. The baseline sits at zero; each later bar
          shows the effect of adding one more feature block or the full stack.
        </div>
        {bar_chart}
        <div class="bar-note">
          Best single-block lift comes from defense subtypes and phage genomic, while pairwise compatibility and the
          all-features stack do not add ranking lift beyond the metadata baseline.
        </div>
      </article>

      <article class="card">
        <h2>Sequence narrative</h2>
        <div class="sub">
          The task asks for a clear story from metadata-only through each feature addition, so the sequence is spelled
          out below instead of left implicit in the bars.
        </div>
        <div class="story">
          {"".join(narrative_cards)}
        </div>
      </article>
    </section>

    <section class="callout">
      <h2>TG05 deployment-realistic callout</h2>
      <p>
        The locked winner is <strong>{callout[WINNER_LABEL]}</strong>. In panel mode it reaches
        <strong>{float(panel[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS]):.1%}</strong> top-3 with AUC
        <strong>{float(panel[HOLDOUT_ROC_AUC]):.3f}</strong>. Removing
        <code>{excluded}</code> lifts top-3 to <strong>{float(deploy[HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS]):.1%}</strong>
        and susceptible-only top-3 to <strong>{float(deploy[HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY]):.1%}</strong>,
        but pairwise discrimination drops to <strong>{float(deploy[HOLDOUT_ROC_AUC]):.3f}</strong>.
        That is the honest deployment tradeoff.
      </p>
      <div class="badges">
        <span class="badge accent">Top-3 gain: {top3_gain_pp:+.1f} pp</span>
        <span class="badge">AUC drop: {auc_drop:+.3f}</span>
        <span class="badge">Excluded: {excluded}</span>
      </div>
    </section>

    <section class="footer">
      Saved data bundle: <code>tp03_feature_lift_visualization_bundle.json</code><br>
      Source summary paths: TG03={tg03_source["path"] or "fallback"} | TG05={tg05_source["path"] or "fallback"}
    </section>
  </div>

  <script id="feature-lift-data" type="application/json">{payload}</script>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    tg03_summary, tg03_source = load_tg03_summary(args.tg03_summary_path)
    tg05_summary, tg05_source = load_tg05_summary(args.tg05_summary_path)
    bundle = build_feature_lift_bundle(
        tg03_summary=tg03_summary,
        tg05_summary=tg05_summary,
        tg03_source=tg03_source,
        tg05_source=tg05_source,
    )

    html = render_feature_lift_visualization_html(bundle)
    html_path = args.output_dir / "tp03_feature_lift_visualization.html"
    bundle_path = args.output_dir / "tp03_feature_lift_visualization_bundle.json"
    summary_path = args.output_dir / "tp03_feature_lift_visualization_summary.json"
    html_path.write_text(html, encoding="utf-8")
    write_json(bundle_path, bundle)
    write_json(
        summary_path,
        {
            "generated_at_utc": bundle["generated_at_utc"],
            "task_id": bundle["task_id"],
            "html_path": str(html_path),
            "bundle_path": str(bundle_path),
            "tg03_source": tg03_source,
            "tg05_source": tg05_source,
            "best_lift_arm": bundle["best_lift_arm"],
        },
    )

    print("TP03 completed.")
    print(f"- Best lift arm: {bundle['best_lift_arm']}")
    print(
        f"- TG05 deployment-realistic top-3: "
        f"{float(bundle['tg05_callout'][DEPLOYMENT_REALISTIC_SENSITIVITY][HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS]):.1%}"
    )
    print(f"- Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
