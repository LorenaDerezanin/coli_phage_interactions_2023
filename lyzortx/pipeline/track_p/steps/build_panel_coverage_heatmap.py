#!/usr/bin/env python3
"""TP02: Build a panel coverage heatmap across host phylogroup and phage family."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows, safe_round
from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier
from lyzortx.pipeline.track_p.steps.build_digital_phagogram import (
    build_locked_arm_feature_spaces,
    ensure_prerequisite_outputs,
    load_locked_config,
    load_tg05_summary,
    score_rows_for_arm,
)

DEFAULT_LOCKED_V1_CONFIG_PATH = Path("lyzortx/pipeline/track_g/v1_feature_configuration.json")
DEFAULT_TG05_SUMMARY_PATH = Path(
    "lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/tg05_feature_subset_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_p/panel_coverage_heatmap")


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
        help="Directory for generated Track P heatmap artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for the locked LightGBM scorer.",
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


def _normalize_axis_value(value: object, *, fallback: str) -> str:
    if value in {"", None}:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _unique_count(values: Iterable[str]) -> int:
    return len({value for value in values if value})


def aggregate_heatmap_layer(
    scored_rows: Sequence[Mapping[str, object]],
    *,
    row_key: str,
    col_key: str,
    value_key: str = "predicted_probability",
) -> Dict[str, object]:
    if not scored_rows:
        raise ValueError("No scored rows available for heatmap aggregation.")

    cell_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    row_bacteria: Dict[str, set[str]] = defaultdict(set)
    row_phages: Dict[str, set[str]] = defaultdict(set)
    col_bacteria: Dict[str, set[str]] = defaultdict(set)
    col_phages: Dict[str, set[str]] = defaultdict(set)
    row_values: Dict[str, List[float]] = defaultdict(list)
    col_values: Dict[str, List[float]] = defaultdict(list)

    for row in scored_rows:
        row_label = _normalize_axis_value(row.get(row_key), fallback="unknown")
        col_label = _normalize_axis_value(row.get(col_key), fallback="unknown")
        value = float(row[value_key])
        pair_bacteria = _normalize_axis_value(row.get("bacteria"), fallback="")
        pair_phage = _normalize_axis_value(row.get("phage"), fallback="")
        cell_values[(row_label, col_label)].append(value)
        row_bacteria[row_label].add(pair_bacteria)
        row_phages[row_label].add(pair_phage)
        col_bacteria[col_label].add(pair_bacteria)
        col_phages[col_label].add(pair_phage)
        row_values[row_label].append(value)
        col_values[col_label].append(value)

    return {
        "cell_values": cell_values,
        "row_bacteria": row_bacteria,
        "row_phages": row_phages,
        "col_bacteria": col_bacteria,
        "col_phages": col_phages,
        "row_values": row_values,
        "col_values": col_values,
    }


def _fallback_strain(panel_rows_by_strain: Mapping[str, Sequence[Mapping[str, object]]]) -> str:
    return next(iter(sorted(panel_rows_by_strain)), "")


def _sorted_axis_labels(
    labels: Iterable[str],
    *,
    panel_stats: Mapping[str, Mapping[str, float]],
    deployment_stats: Mapping[str, Mapping[str, float]],
) -> List[str]:
    def sort_key(label: str) -> Tuple[float, float, str]:
        stats = panel_stats.get(label) or deployment_stats.get(label) or {}
        mean_probability = stats.get("mean_probability")
        support = stats.get("n_pairs", 0.0)
        if mean_probability is None:
            return (1e9, -support, label)
        return (float(mean_probability), -float(support), label)

    return sorted(dict.fromkeys(labels), key=sort_key)


def _summarize_axis(
    values_by_label: Mapping[str, Sequence[float]],
    bacteria_by_label: Mapping[str, set[str]],
    phages_by_label: Mapping[str, set[str]],
) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for label, values in values_by_label.items():
        summary[label] = {
            "n_pairs": len(values),
            "mean_probability": safe_round(_mean(values)),
            "min_probability": safe_round(float(np.min(values))),
            "max_probability": safe_round(float(np.max(values))),
            "unique_bacteria_count": _unique_count(bacteria_by_label[label]),
            "unique_phage_count": _unique_count(phages_by_label[label]),
        }
    return summary


def _build_layer_rows(
    layer: Mapping[str, object],
    *,
    row_order: Sequence[str],
    col_order: Sequence[str],
) -> List[Dict[str, object]]:
    cell_values: Mapping[Tuple[str, str], Sequence[float]] = layer["cell_values"]  # type: ignore[assignment]
    rows: List[Dict[str, object]] = []
    for row_label in row_order:
        for col_label in col_order:
            values = list(cell_values.get((row_label, col_label), []))
            if values:
                rows.append(
                    {
                        "row_label": row_label,
                        "col_label": col_label,
                        "mean_probability": safe_round(_mean(values)),
                        "min_probability": safe_round(float(np.min(values))),
                        "max_probability": safe_round(float(np.max(values))),
                        "n_pairs": len(values),
                    }
                )
            else:
                rows.append(
                    {
                        "row_label": row_label,
                        "col_label": col_label,
                        "mean_probability": None,
                        "min_probability": None,
                        "max_probability": None,
                        "n_pairs": 0,
                    }
                )
    return rows


def build_panel_coverage_bundle(
    *,
    config: Mapping[str, object],
    tg05_summary: Mapping[str, object],
    initial_bacteria: str,
    panel_rows_by_strain: Mapping[str, Sequence[Mapping[str, object]]],
    deployment_rows_by_strain: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    panel_rows: List[Mapping[str, object]] = [row for rows in panel_rows_by_strain.values() for row in rows]
    deployment_rows: List[Mapping[str, object]] = [row for rows in deployment_rows_by_strain.values() for row in rows]

    panel_layer = aggregate_heatmap_layer(panel_rows, row_key="host_phylogroup", col_key="phage_family")
    deployment_layer = aggregate_heatmap_layer(deployment_rows, row_key="host_phylogroup", col_key="phage_family")

    row_order = _sorted_axis_labels(
        set(panel_layer["row_values"]) | set(deployment_layer["row_values"]),
        panel_stats=_summarize_axis(panel_layer["row_values"], panel_layer["row_bacteria"], panel_layer["row_phages"]),
        deployment_stats=_summarize_axis(
            deployment_layer["row_values"], deployment_layer["row_bacteria"], deployment_layer["row_phages"]
        ),
    )
    column_order = _sorted_axis_labels(
        set(panel_layer["col_values"]) | set(deployment_layer["col_values"]),
        panel_stats=_summarize_axis(panel_layer["col_values"], panel_layer["col_bacteria"], panel_layer["col_phages"]),
        deployment_stats=_summarize_axis(
            deployment_layer["col_values"], deployment_layer["col_bacteria"], deployment_layer["col_phages"]
        ),
    )

    panel_rows_rendered = _build_layer_rows(panel_layer, row_order=row_order, col_order=column_order)
    deployment_rows_rendered = _build_layer_rows(deployment_layer, row_order=row_order, col_order=column_order)
    panel_map = {(row["row_label"], row["col_label"]): row for row in panel_rows_rendered}
    deployment_map = {(row["row_label"], row["col_label"]): row for row in deployment_rows_rendered}
    delta_rows: List[Dict[str, object]] = []
    for row_label in row_order:
        for col_label in column_order:
            panel_cell = panel_map[(row_label, col_label)]
            deployment_cell = deployment_map[(row_label, col_label)]
            panel_value = panel_cell["mean_probability"]
            deployment_value = deployment_cell["mean_probability"]
            delta_rows.append(
                {
                    "row_label": row_label,
                    "col_label": col_label,
                    "panel_probability": panel_value,
                    "deployment_probability": deployment_value,
                    "delta_probability": (
                        None
                        if panel_value is None or deployment_value is None
                        else safe_round(float(panel_value) - float(deployment_value))
                    ),
                    "n_pairs": max(int(panel_cell["n_pairs"]), int(deployment_cell["n_pairs"])),
                }
            )

    panel_row_summary = _summarize_axis(
        panel_layer["row_values"], panel_layer["row_bacteria"], panel_layer["row_phages"]
    )
    deployment_row_summary = _summarize_axis(
        deployment_layer["row_values"], deployment_layer["row_bacteria"], deployment_layer["row_phages"]
    )
    panel_col_summary = _summarize_axis(
        panel_layer["col_values"], panel_layer["col_bacteria"], panel_layer["col_phages"]
    )
    deployment_col_summary = _summarize_axis(
        deployment_layer["col_values"], deployment_layer["col_bacteria"], deployment_layer["col_phages"]
    )

    hardest_phylogroups = sorted(
        row_order,
        key=lambda label: (
            panel_row_summary.get(label, deployment_row_summary.get(label, {})).get("mean_probability", 1e9),
            label,
        ),
    )[:5]
    largest_gaps = sorted(
        [row for row in delta_rows if row["delta_probability"] is not None],
        key=lambda row: abs(float(row["delta_probability"])),
        reverse=True,
    )[:8]

    locked = config["locked_v1_feature_configuration"]
    fallback_initial = _fallback_strain(panel_rows_by_strain)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "TP02",
        "initial_bacteria": initial_bacteria if initial_bacteria in panel_rows_by_strain else fallback_initial,
        "locked_v1_feature_configuration": locked,
        "panel_label": f"{locked['winner_label']} (panel-default)",
        "deployment_label": f"{locked['winner_label']} (deployment-realistic)",
        "row_axis": "host_phylogroup",
        "col_axis": "phage_family",
        "row_order": row_order,
        "column_order": column_order,
        "panel_heatmap": {
            "rows": panel_rows_rendered,
            "row_summary": panel_row_summary,
            "column_summary": panel_col_summary,
        },
        "deployment_heatmap": {
            "rows": deployment_rows_rendered,
            "row_summary": deployment_row_summary,
            "column_summary": deployment_col_summary,
        },
        "delta_heatmap": {
            "rows": delta_rows,
            "row_summary": panel_row_summary,
            "column_summary": panel_col_summary,
        },
        "summaries": {
            "hardest_panel_phylogroups": hardest_phylogroups,
            "largest_panel_deployment_gaps": largest_gaps,
            "panel_missing_cell_count": sum(1 for row in panel_rows_rendered if row["mean_probability"] is None),
            "deployment_missing_cell_count": sum(
                1 for row in deployment_rows_rendered if row["mean_probability"] is None
            ),
        },
        "tg05_locked_lightgbm_hyperparameters": tg05_summary["locked_lightgbm_hyperparameters"],
    }


def render_panel_coverage_heatmap_html(bundle: Mapping[str, object]) -> str:
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    row_label = bundle["row_axis"]
    col_label = bundle["col_axis"]
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track P Panel Coverage Heatmap</title>
  <style>
    :root {{
      --bg: #05111d;
      --panel: rgba(11, 19, 33, 0.88);
      --panel-strong: rgba(16, 28, 48, 0.96);
      --text: #eef6ff;
      --muted: #9db0c8;
      --line: rgba(180, 206, 230, 0.16);
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(110, 231, 216, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(251, 191, 36, 0.14), transparent 28%),
        linear-gradient(180deg, #04101d 0%, #07111f 100%);
    }}
    .wrap {{
      width: min(1640px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero, .card, .legend, .note {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px 26px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: #6ee7d8;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 0.98;
    }}
    .lede {{
      margin: 0;
      max-width: 82ch;
      color: var(--muted);
      line-height: 1.55;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .stat {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .stat .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .legend {{
      padding: 16px 18px;
      margin-bottom: 18px;
    }}
    .legend p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .sections {{
      display: grid;
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .card-head .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid rgba(180, 206, 230, 0.16);
      background: rgba(6, 12, 23, 0.72);
      color: var(--muted);
      font-size: 12px;
    }}
    .heatmap-wrap {{
      overflow: auto;
      padding-bottom: 6px;
    }}
    .heatmap {{
      display: grid;
      gap: 6px;
      align-items: stretch;
      min-width: max-content;
    }}
    .axis {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .axis-row, .heatmap-row {{
      display: grid;
      grid-template-columns: 170px repeat(var(--cols), minmax(108px, 1fr));
      gap: 6px;
    }}
    .axis-row {{
      margin-bottom: 2px;
    }}
    .row-label, .col-label, .cell {{
      border-radius: 16px;
      border: 1px solid rgba(180, 206, 230, 0.12);
      background: rgba(6, 12, 23, 0.68);
      padding: 12px 12px;
    }}
    .row-label {{
      display: grid;
      gap: 5px;
      align-content: center;
      min-height: 84px;
    }}
    .row-label .name {{
      font-weight: 700;
      font-size: 15px;
    }}
    .row-label .meta, .cell .meta {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .col-label {{
      min-height: 82px;
      display: grid;
      gap: 5px;
      align-content: center;
      text-align: center;
    }}
    .cell {{
      min-height: 84px;
      display: grid;
      gap: 8px;
      align-content: center;
      text-align: center;
      color: var(--text);
    }}
    .cell.missing {{
      background:
        linear-gradient(135deg, rgba(180, 206, 230, 0.08), rgba(180, 206, 230, 0.02)),
        repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.04) 0 10px, rgba(255, 255, 255, 0.01) 10px 20px);
      color: var(--muted);
    }}
    .cell .value {{
      font-size: 22px;
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 700;
    }}
    .cell .support {{
      font-size: 12px;
      color: rgba(255, 255, 255, 0.84);
    }}
    .delta-note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
      line-height: 1.5;
    }}
    .note {{
      padding: 16px 18px;
    }}
    .note p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    @media (max-width: 1000px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Track P presentation artifact</div>
      <h1>Panel coverage heatmap</h1>
      <p class="lede">
        Predicted lysis probability aggregated across host phylogroup × phage family, with panel-default and
        deployment-realistic views shown side by side. Rows are ordered to surface hard-to-lyse host groups first, and
        the delta layer highlights where removing <code>host_n_infections</code> changes coverage confidence.
      </p>
      <div class="grid">
        <div class="stat">
          <div class="label">Hardest panel phylogroups</div>
          <div class="value" id="hardest-phylogroups"></div>
        </div>
        <div class="stat">
          <div class="label">Panel missing cells</div>
          <div class="value" id="panel-missing"></div>
        </div>
        <div class="stat">
          <div class="label">Deployment missing cells</div>
          <div class="value" id="deployment-missing"></div>
        </div>
      </div>
    </section>

    <section class="legend">
      <p>
        Heatmap cells show mean predicted lysis probability for the scored pairs in each bin. Missing cells are left
        blank so phylogroup/family gaps stay visible. The delta layer is computed as panel-default minus
        deployment-realistic probability.
      </p>
    </section>

    <section class="sections">
      <article class="card">
        <div class="card-head">
          <div>
            <h2 id="panel-title"></h2>
            <div class="sub" id="panel-subtitle"></div>
          </div>
          <div class="pill" id="panel-pill"></div>
        </div>
        <div class="heatmap-wrap">
          <div class="heatmap" id="panel-heatmap"></div>
        </div>
      </article>

      <article class="card">
        <div class="card-head">
          <div>
            <h2 id="deployment-title"></h2>
            <div class="sub" id="deployment-subtitle"></div>
          </div>
          <div class="pill" id="deployment-pill"></div>
        </div>
        <div class="heatmap-wrap">
          <div class="heatmap" id="deployment-heatmap"></div>
        </div>
      </article>

      <article class="card">
        <div class="card-head">
          <div>
            <h2>Panel minus deployment gap</h2>
            <div class="sub">Positive values mean the panel-default lock is more optimistic than the deployment-realistic lock.</div>
          </div>
          <div class="pill">Delta heatmap</div>
        </div>
        <div class="heatmap-wrap">
          <div class="heatmap" id="delta-heatmap"></div>
        </div>
        <div class="delta-note">
          The gap layer makes novel-strain confidence differences explicit instead of hiding them in the phagogram.
        </div>
      </article>
    </section>

    <section class="note" style="margin-top:16px;">
      <p>Saved data bundle: <span id="bundle-path">tp02_panel_coverage_heatmap_bundle.json</span></p>
    </section>
  </div>

  <script id="coverage-data" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("coverage-data").textContent);
    const rowLabel = __ROW_LABEL__;
    const colLabel = __COL_LABEL__;
    const panelTitle = document.getElementById("panel-title");
    const panelSubtitle = document.getElementById("panel-subtitle");
    const panelPill = document.getElementById("panel-pill");
    const deploymentTitle = document.getElementById("deployment-title");
    const deploymentSubtitle = document.getElementById("deployment-subtitle");
    const deploymentPill = document.getElementById("deployment-pill");
    const hardest = document.getElementById("hardest-phylogroups");
    const panelMissing = document.getElementById("panel-missing");
    const deploymentMissing = document.getElementById("deployment-missing");
    const panelHeatmap = document.getElementById("panel-heatmap");
    const deploymentHeatmap = document.getElementById("deployment-heatmap");
    const deltaHeatmap = document.getElementById("delta-heatmap");
    const bundlePath = document.getElementById("bundle-path");

    function colorForProbability(value) {{
      const clamped = Math.max(0, Math.min(1, value));
      if (clamped <= 0.5) {{
        const ratio = clamped / 0.5;
        const r = Math.round(10 + (47 - 10) * ratio);
        const g = Math.round(20 + (140 - 20) * ratio);
        const b = Math.round(37 + (220 - 37) * ratio);
        return `rgb(${r}, ${g}, ${b})`;
      }}
      const ratio = (clamped - 0.5) / 0.5;
      const r = Math.round(47 + (251 - 47) * ratio);
      const g = Math.round(140 + (191 - 140) * ratio);
      const b = Math.round(220 + (36 - 220) * ratio);
      return `rgb(${r}, ${g}, ${b})`;
    }}

    function colorForDelta(value) {{
      const clamped = Math.max(-0.6, Math.min(0.6, value)) / 0.6;
      if (clamped >= 0) {{
        const ratio = clamped;
        const r = Math.round(17 + (110 - 17) * ratio);
        const g = Math.round(24 + (231 - 24) * ratio);
        const b = Math.round(39 + (216 - 39) * ratio);
        return `rgb(${r}, ${g}, ${b})`;
      }}
      const ratio = Math.abs(clamped);
      const r = Math.round(17 + (251 - 17) * ratio);
      const g = Math.round(24 + (113 - 24) * ratio);
      const b = Math.round(39 + (133 - 39) * ratio);
      return `rgb(${r}, ${g}, ${b})`;
    }}

    function renderHeatmap(container, layer, mode) {{
      const rows = data.row_order;
      const cols = data.column_order;
      container.style.setProperty("--cols", cols.length);
      container.innerHTML = `
        <div class="axis-row">
          <div class="axis" style="padding:12px 12px;">${rowLabel} × ${colLabel}</div>
          ${cols.map((label) => `<div class="col-label"><div class="name">${label}</div><div class="meta">${layer.column_summary && layer.column_summary[label] ? layer.column_summary[label].n_pairs + " pairs" : "no pairs"}</div></div>`).join("")}
        </div>
        ${rows.map((rowName) => {{
          const rowMeta = (layer.row_summary && layer.row_summary[rowName]) || {{}};
          const rowCells = cols.map((colName) => {{
            const cell = layer.rows.find((item) => item.row_label === rowName && item.col_label === colName) || {{}};
            const value = mode === "delta" ? cell.delta_probability : cell.mean_probability;
            if (value === null || value === undefined) {{
              return `<div class="cell missing"><div class="value">—</div><div class="support">0 pairs</div></div>`;
            }}
            const background = mode === "delta" ? colorForDelta(value) : colorForProbability(value);
            const display = Number(value).toFixed(3);
            const support = `${cell.n_pairs} pair${cell.n_pairs === 1 ? "" : "s"}`;
            return `<div class="cell" style="background:${background}; color:${mode === "delta" ? "#edf6ff" : "#08111d"}" title="${rowName} × ${colName}">
              <div class="value">${display}</div>
              <div class="support">${support}</div>
            </div>`;
          }}).join("");
          return `
            <div class="heatmap-row">
              <div class="row-label">
                <div class="name">${rowName}</div>
                <div class="meta">${rowMeta.n_pairs || 0} pairs · ${rowMeta.unique_bacteria_count || 0} strains</div>
                <div class="meta">mean ${rowMeta.mean_probability !== undefined ? Number(rowMeta.mean_probability).toFixed(3) : "—"}</div>
              </div>
              ${rowCells}
            </div>`;
        }}).join("")}
      `;
    }}

    panelTitle.textContent = data.panel_label;
    panelSubtitle.textContent = `${data.row_axis} × ${data.col_axis} | panel-default`;
    panelPill.textContent = `${data.panel_heatmap.rows.length} cells`;
    deploymentTitle.textContent = data.deployment_label;
    deploymentSubtitle.textContent = `${data.row_axis} × ${data.col_axis} | deployment-realistic`;
    deploymentPill.textContent = `${data.deployment_heatmap.rows.length} cells`;
    hardest.textContent = (data.summaries.hardest_panel_phylogroups || []).join(", ");
    panelMissing.textContent = String(data.summaries.panel_missing_cell_count || 0);
    deploymentMissing.textContent = String(data.summaries.deployment_missing_cell_count || 0);
    bundlePath.textContent = data.generated_at_utc;

    renderHeatmap(panelHeatmap, data.panel_heatmap, "probability");
    renderHeatmap(deploymentHeatmap, data.deployment_heatmap, "probability");
    renderHeatmap(deltaHeatmap, data.delta_heatmap, "delta");
  </script>
</body>
</html>
"""
    while "{{" in html or "}}" in html:
        html = html.replace("{{", "{").replace("}}", "}")
    return (
        html.replace("__PAYLOAD__", payload)
        .replace("__ROW_LABEL__", json.dumps(row_label))
        .replace("__COL_LABEL__", json.dumps(col_label))
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
        _, _, _, _, _, probabilities = score_rows_for_arm(
            merged_rows,
            arm_space,
            params=params,
            random_state=args.random_state,
        )
        scored_rows = []
        for row, probability in zip(merged_rows, probabilities):
            scored_row = dict(row)
            scored_row["predicted_probability"] = safe_round(float(probability))
            scored_rows.append(scored_row)
        arm_results[arm_name] = scored_rows

    initial_bacteria = next(iter(sorted({str(row["bacteria"]) for row in merged_rows})), "")
    panel_rows_by_strain = defaultdict(list)
    deployment_rows_by_strain = defaultdict(list)
    for row in arm_results["panel"]:
        panel_rows_by_strain[str(row["bacteria"])].append(row)
    for row in arm_results["deployment"]:
        deployment_rows_by_strain[str(row["bacteria"])].append(row)

    bundle = build_panel_coverage_bundle(
        config=locked_config,
        tg05_summary=tg05_summary,
        initial_bacteria=initial_bacteria,
        panel_rows_by_strain=panel_rows_by_strain,
        deployment_rows_by_strain=deployment_rows_by_strain,
    )

    output_bundle = args.output_dir / "tp02_panel_coverage_heatmap_bundle.json"
    output_html = args.output_dir / "tp02_panel_coverage_heatmap.html"
    output_summary = args.output_dir / "tp02_panel_coverage_heatmap_summary.json"

    output_bundle.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    output_html.write_text(render_panel_coverage_heatmap_html(bundle), encoding="utf-8")
    write_json(
        output_summary,
        {
            "generated_at_utc": bundle["generated_at_utc"],
            "task_id": "TP02",
            "initial_bacteria": bundle["initial_bacteria"],
            "row_order": bundle["row_order"],
            "column_order": bundle["column_order"],
            "outputs": {
                "bundle_json": str(output_bundle),
                "html": str(output_html),
            },
            "summaries": bundle["summaries"],
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

    print("TP02 completed.")
    print(f"- Heatmap rows: {len(bundle['row_order'])}")
    print(f"- Heatmap columns: {len(bundle['column_order'])}")
    print(f"- Output HTML: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
