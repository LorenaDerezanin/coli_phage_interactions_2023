### 2026-03-24: TK01 VHRdb lift measurement

#### Executive summary

Added the TK01 Track K runner to measure VHRdb lift against the locked v1 baseline and record the result in a
manifest plus summary tables under `lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement/`. The first local
run found no joinable TI08 VHRdb rows, so the lift deltas were all `0.0` and the decision remained
`pending_external_artifact`. The step now requires the locked TG01 summary artifact rather than silently substituting
defaults, so the baseline comparison stays comparable to the Track G lock.

#### What was implemented

- Added a new Track K runner at `lyzortx/pipeline/track_k/run_track_k.py` and a TK01 lift-measurement step at
  `lyzortx/pipeline/track_k/steps/build_vhrdb_lift_report.py`.
- The step reuses the locked `defense + phage-genomic` v1 feature contract, trains the baseline internal-only model,
  then attempts to add TI08 VHRdb rows only when they join safely into the ST03 train split.
- Emitted a TK01 summary CSV, top-3 ranking CSV, and manifest under
  `lyzortx/generated_outputs/track_k/tk01_vhrdb_lift_measurement/`.

#### Findings

- In this checkout, the run completed with `0` joinable VHRdb rows from the TI08 cohort artifact.
- Baseline internal-only holdout metrics were:
  - ROC-AUC `0.831075`
  - top-3 hit rate `0.876923`
  - Brier score `0.166377`
- The VHRdb-augmented arm was identical because no cohort rows joined into the training split, so all deltas were `0.0`.
- The TK01 manifest marked the decision as `pending_external_artifact`.

#### Interpretation

- The Track K seam is now wired correctly: if TI08 VHRdb rows appear later, they will enter training only through the
  locked ST03-safe path.
- For this repository state, VHRdb should not be added to later arms yet. The code path is ready, but the local
  artifact set does not contain joinable VHRdb training rows, so there is no empirical lift to carry forward.
