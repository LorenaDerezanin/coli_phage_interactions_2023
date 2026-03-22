### 2026-03-22: TP01 implemented (digital phagogram dashboard for per-strain phage ranking)

#### Executive summary

Built a standalone interactive digital phagogram at `lyzortx/generated_outputs/track_p/digital_phagogram/tp01_digital_phagogram.html`
that shows, for any panel strain, the ranked phage list, calibrated `P(lysis)`, confidence bands, and top SHAP
drivers. The dashboard uses the locked TG05 configuration from `lyzortx/pipeline/track_g/v1_feature_configuration.json`
and presents the panel model and deployment-realistic model side by side. The deployment-realistic view improves
top-3 ranking slightly, while the panel view retains the stronger discrimination and calibration.

#### What was implemented

- Added a new Track P entrypoint at `lyzortx/pipeline/track_p/run_track_p.py`.
- Added the TP01 builder at `lyzortx/pipeline/track_p/steps/build_digital_phagogram.py`.
- The TP01 builder now:
  - reads the locked TG05 feature configuration and TG05 hyperparameter summary
  - reconstructs the winning panel arm and the deployment-realistic arm
  - fits calibrated LightGBM rankings for both views
  - computes SHAP summaries for the displayed top phages
  - renders a standalone Plotly dashboard with a strain selector and side-by-side panels
- Added tests in `lyzortx/tests/test_track_p_digital_phagogram.py` covering:
  - locked-feature-arm reconstruction
  - payload assembly for both model views
  - HTML rendering / interactive control wiring

#### Output summary

- TP01 artifact directory:
  - `tp01_digital_phagogram.html`
  - `tp01_ranked_phagogram_rows.csv`
  - `tp01_phagogram_summary.json`
- Strain coverage:
  - `65` strains
  - `195` ranked recommendation rows per model view
- Holdout metrics from the generated TP01 artifact:
  - panel model: ROC-AUC `0.909961`, Brier `0.100321`, top-3 hit rate `0.892308`
  - deployment-realistic model: ROC-AUC `0.834627`, Brier `0.139331`, top-3 hit rate `0.907692`

#### Interpretation

1. The dashboard meets the demo requirement. It is interactive, strain-selectable, and shows the two required
   prediction views side by side with the same locked feature family.
2. The confidence band is intentionally conservative: it uses the interval between isotonic and Platt-calibrated
   probabilities for each ranked pair, which gives a stable visual uncertainty cue without inventing a new estimator.
3. The panel model remains the stronger pairwise discriminator, but the deployment-realistic view is slightly better at
   top-3 ranking. That matches the TG05 pattern: removing `host_n_infections` hurts AUC but can help ranking quality.
4. The top SHAP features make the demo explainable enough for live use. The phage list is not just a score dump; each
   recommendation carries a short driver summary so the ranking can be defended in real time.

#### Next steps

1. Reuse the TP01 dashboard as the live demo artifact for Track P.
2. If we need a faster cold-start demo path, consider caching the TG05/TP01 artifacts rather than recomputing the model
   fits on each run.
