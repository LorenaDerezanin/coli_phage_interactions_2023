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

### 2026-03-24: TK02 BASEL cumulative lift measurement

#### Executive summary

Added the TK02 Track K runner to measure BASEL lift on top of the best-so-far TK01 cohort. The real Track G/I
generated artifacts are not present in this checkout, so the new path was validated on a minimal fixture instead of a
production rerun. On that fixture, BASEL was neutral: ROC-AUC, top-3, and Brier deltas vs the previous best were all
`0.0`.

#### What was implemented

- Added shared Track K lift helpers in `lyzortx/pipeline/track_k/steps/build_source_lift_helpers.py`.
- Added the TK02 runner at `lyzortx/pipeline/track_k/steps/build_basel_lift_report.py`.
- Updated `lyzortx/pipeline/track_k/run_track_k.py` so `--step all` runs TK01 followed by TK02.
- The TK02 manifest now records the previous-best source systems, cumulative source set, metric deltas, and the lift
  assessment.

#### Findings

- On the validation fixture, TK01 kept `internal_plus_vhrdb` as the best-so-far cohort.
- TK02 evaluated `internal_plus_vhrdb_plus_basel`.
- Metric deltas vs the previous best were all `0.0`:
  - ROC-AUC `0.0`
  - top-3 `0.0`
  - Brier `0.0`
- The TK02 lift assessment was `neutral`.

#### Interpretation

- BASEL is now wired as a cumulative add-on after TK01, not as a replacement path.
- On the available fixture, BASEL is neutral rather than additive or harmful.
- A production rerun can replace the fixture note once the Track G/I generated artifacts are available locally.

### 2026-03-24: TK03 KlebPhaCol cumulative lift measurement

#### Executive summary

Added the TK03 Track K runner to measure KlebPhaCol lift on top of the current best-so-far cohort. The new step
reads the TK02 manifest to recover the prior external source chain, then retrains the locked v1 model with
`+KlebPhaCol` appended. On the validation fixture, KlebPhaCol was neutral: ROC-AUC, top-3, and Brier deltas vs the
previous best were all `0.0`.

#### What was implemented

- Added `lyzortx/pipeline/track_k/steps/build_klebphacol_lift_report.py` and wired it into
  `lyzortx/pipeline/track_k/run_track_k.py`.
- Extended the shared Track K manifest loader so TK03 can recover the previous best external cohort from TK02
  manifests without duplicating `internal`.
- Added regression coverage for the TK03 runner and the TK02-to-TK03 cohort handoff.

#### Findings

- On the validation fixture, TK03 carried forward `internal_plus_vhrdb` as the best-so-far cohort.
- TK03 evaluated `internal_plus_vhrdb_plus_klebphacol`.
- On the validation fixture, both arms scored ROC-AUC `0.5`, top-3 hit rate `1.0`, and Brier score `0.25`.
- The measured deltas vs the previous best were all `0.0`:
  - ROC-AUC `0.0`
  - top-3 `0.0`
  - Brier `0.0`
- The TK03 lift assessment was `neutral`.

#### Interpretation

- KlebPhaCol now slots cleanly into the cumulative Track K sequence after BASEL.
- On the available fixture, KlebPhaCol neither helps nor hurts the current best-so-far cohort, so it should be
  treated as neutral until a production rerun with the real Track I artifacts is available.
