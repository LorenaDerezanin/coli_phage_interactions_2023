### 2026-03-22: TP01 implemented (digital phagogram visualization for per-strain ranking)

#### Executive summary

Built an interactive Track P phagogram demo that renders per-strain phage rankings side by side for the locked panel
model and the deployment-realistic lock. The visualization shows calibrated `P(lysis)`, bootstrap confidence bands,
and SHAP drivers for each ranked phage, and it is wired to the locked v1 feature configuration from TG05.

#### Implementation

- Added `lyzortx/pipeline/track_p/run_track_p.py` and
  `lyzortx/pipeline/track_p/steps/build_digital_phagogram.py`.
- The demo loader now:
  - reads the locked v1 feature config from `lyzortx/pipeline/track_g/v1_feature_configuration.json`
  - uses the TG05 locked LightGBM parameters as the scoring model family
  - rebuilds the panel-default feature lock and the deployment-realistic lock from the same audited feature space
  - scores the full ranked phage list for each strain, then attaches bootstrap confidence bands and SHAP summaries
- The HTML output is self-contained so it can be opened directly in a browser during partner demos.

#### Output summary

- Presentation bundle:
  - `lyzortx/generated_outputs/track_p/digital_phagogram/tp01_digital_phagogram.html`
  - `lyzortx/generated_outputs/track_p/digital_phagogram/tp01_digital_phagogram_bundle.json`
  - `lyzortx/generated_outputs/track_p/digital_phagogram/tp01_digital_phagogram_summary.json`

#### Interpretation

1. The right abstraction boundary is a presentation layer, not another predictor. The phagogram reuses the locked v1
   feature configuration and existing Track G scoring logic, then packages the results into an interactive demo view.
2. Showing panel and deployment-realistic rankings together makes the `host_n_infections` tradeoff explicit instead of
   burying it in a metrics table.
3. Confidence bands and SHAP summaries are kept on every ranked row so operators can inspect both uncertainty and local
   drivers without leaving the demo.

#### Next steps

1. Use the same rendering pattern for the planned Track P coverage heatmap if the partner demo needs a broader panel
   overview.
2. If the demo needs slimmer initial load times, trim the initial rank limit in the HTML controls rather than changing
   the underlying data bundle.

### 2026-03-22: TP02 implemented (panel coverage heatmap across strain diversity)

#### Executive summary

Built a Track P coverage heatmap that aggregates predicted lysis probability across host phylogroup and phage family for
the locked 96-phage panel. The artifact renders panel-default and deployment-realistic heatmaps side by side, plus a
delta layer that makes `host_n_infections`-driven confidence shifts visible where novel-strain confidence diverges from
panel confidence.

#### Implementation

- Added `lyzortx/pipeline/track_p/steps/build_panel_coverage_heatmap.py` and wired it into
  `lyzortx/pipeline/track_p/run_track_p.py`.
- The step reuses the locked Track G v1 feature lock and TG05 scorer, then aggregates pair-level predictions into
  phylogroup-by-family bins.
- Rows are ordered by panel mean predicted lysis so the hardest-to-lyse host groups surface first, and missing bins are
  left blank so panel gaps stay visually obvious.

#### Output summary

- Heatmap bundle:
  - `lyzortx/generated_outputs/track_p/panel_coverage_heatmap/tp02_panel_coverage_heatmap.html`
  - `lyzortx/generated_outputs/track_p/panel_coverage_heatmap/tp02_panel_coverage_heatmap_bundle.json`
  - `lyzortx/generated_outputs/track_p/panel_coverage_heatmap/tp02_panel_coverage_heatmap_summary.json`

#### Interpretation

1. The panel heatmap highlights where susceptibility is concentrated by phylogroup/family rather than by individual
   strain, which makes hard-to-lyse regions much easier to explain in a partner demo.
2. The deployment-realistic layer shows the cost of removing label-derived host features, so the demo can distinguish
   panel confidence from novel-strain confidence without switching tools.
3. The delta layer is the useful part for planning: large positive panel-minus-deployment cells are the bins where the
   current panel is leaning hardest on `host_n_infections`.

#### Next steps

1. If the partner deck needs a compact static figure, export the same bundle to a publication-style PNG/SVG without
   changing the aggregation logic.
2. If the panel expands, keep the same row/column ordering logic so future coverage maps remain comparable.
