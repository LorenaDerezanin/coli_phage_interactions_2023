### 2026-03-22: TH02 implemented (explained recommendations with calibrated P(lysis), CI, and SHAP features)

#### What was implemented

- Added `lyzortx/pipeline/track_h/run_track_h.py` and
  `lyzortx/pipeline/track_h/steps/build_explained_recommendations.py`.
- The Track H step now joins TG02 calibrated holdout predictions with TG04 SHAP explanations and emits:
  - a top-3 recommendation CSV for every holdout strain
  - a per-strain summary CSV
  - a clinician-facing markdown report
  - a JSON manifest with input hashes and row counts
- Each recommended phage now carries:
  - calibrated `P(lysis)`
  - a bootstrap 95% CI from the TG02 calibration fold
  - the top-3 SHAP features surfaced from TG04 explanations

#### Output summary

- Report scope:
  - all holdout strains in the TG02 prediction file
  - top-3 recommendations per strain
- Report format:
  - compact overview table for scan review
  - per-strain detail blocks for deeper operator review

#### Interpretation

1. TH02 is now a presentation layer over the existing Track G model outputs rather than a new ranking model. That is the
   right boundary for this task because the acceptance criteria ask for explainable recommendations, not a new learner.
2. The bootstrap interval makes the recommendation confidence explicit instead of implying that the calibrated
   probability is exact. That is the main operational value for clinician or CDMO review.
3. Surfacing the top-3 SHAP features per recommendation gives a short, auditable rationale for each phage call while
   keeping the report compact enough for downstream review.

#### Next steps

1. If the upstream TG02/TG04 artifacts change, regenerate the TH02 report from the new calibrated predictions and SHAP
   rows.
2. If operators want a narrower display, the markdown report can be trimmed to the overview table without changing the
   CSV artifact.
