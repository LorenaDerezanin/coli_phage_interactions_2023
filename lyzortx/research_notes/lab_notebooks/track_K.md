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

### 2026-03-24: TK04 GPB cumulative lift measurement

#### Executive summary

Added the TK04 Track K runner to measure GPB lift on top of the current best-so-far cohort. The local Track I cohort
artifact is still absent in this checkout, so the new path was validated on the same minimal fixture pattern used for
TK02/TK03. On that fixture, GPB was neutral: ROC-AUC, top-3, and Brier deltas vs the previous best were all `0.0`.

#### What was implemented

- Added `lyzortx/pipeline/track_k/steps/build_gpb_lift_report.py` and wired it into
  `lyzortx/pipeline/track_k/run_track_k.py`.
- Extended the Track K runner so `--step all` now runs TK01 through TK04 in order.
- Added regression coverage for the TK04 runner and the TK03-to-TK04 cohort handoff.

#### Findings

- On the validation fixture, TK04 carried forward `internal_plus_vhrdb_plus_basel_plus_klebphacol` as the
  best-so-far cohort.
- TK04 evaluated `internal_plus_vhrdb_plus_basel_plus_klebphacol_plus_gpb`.
- On the fixture, both arms scored ROC-AUC `0.5`, top-3 hit rate `1.0`, and Brier score `0.25`.
- The measured deltas vs the previous best were all `0.0`:
  - ROC-AUC `0.0`
  - top-3 `0.0`
  - Brier `0.0`
- The TK04 lift assessment was `neutral`.

#### Interpretation

- GPB now fits cleanly after KlebPhaCol in the cumulative Track K sequence.
- On the available fixture, GPB neither adds nor hurts the current best-so-far cohort, so there is no basis yet for
  changing the locked external-data chain.

### 2026-03-24: TK05 Tier B cumulative lift measurement

#### Executive summary

Added the TK05 Track K runner to measure the confidence-weighted Tier B add-on on top of the current best-so-far
cohort. A scratch rerun confirmed that both TI07-derived Tier B sources join cleanly and remain auditable through the
TI08 training cohort path, but the lift was neutral on the validation fixture: ROC-AUC, top-3, and Brier deltas vs
the previous best were all `0.0`. Because the measured source combination did not improve any metric, the baseline
should stay internal-only for v1.

#### What was implemented

- Added `lyzortx/pipeline/track_k/steps/build_tier_b_lift_report.py` and wired it into
  `lyzortx/pipeline/track_k/run_track_k.py`.
- Extended the shared Track K lift helper so multi-source additions can carry both Tier B source systems forward with
  per-source row counts.
- Added regression coverage for the TK05 runner, the combined Tier B cohort path, and the Track K dispatch sequence.

#### Findings

- The scratch validation run emitted `lyzortx/generated_outputs/track_k/tk05_tier_b_lift_measurement/`-style outputs
  under `.scratch/tk05_demo/out/` for the local fixture.
- The previous best cohort remained `internal_plus_vhrdb_plus_basel_plus_klebphacol_plus_gpb`.
- The augmented cohort was `internal_plus_vhrdb_plus_basel_plus_klebphacol_plus_gpb_plus_virus_host_db_plus_ncbi_virus_biosample`.
- The Tier B rows joined cleanly:
  - `virus_host_db`: `1` joined row
  - `ncbi_virus_biosample`: `1` joined row
- Metric deltas vs the previous best were all `0.0`:
  - ROC-AUC `0.0`
  - top-3 `0.0`
  - Brier `0.0`
- The TK05 lift assessment was `neutral`.

#### Interpretation

- Tier B is wired correctly as a cumulative add-on, but on this fixture it does not improve the model.
- Since no source combination improved metrics here, there is no basis to promote a new locked config yet; internal-only
  remains the safest v1 baseline until a real production rerun shows different behavior.

### 2026-03-24: TK06 synthesize per-source lift results and lock the external-data decision

#### Executive summary

Added a dedicated TK06 synthesis step at `lyzortx/pipeline/track_k/steps/build_external_data_decision_report.py` so
Track K can normalize TK01-TK05 manifest metrics back to the same internal-only baseline and make one explicit lock
decision. In this checkout, that decision remains `internal_only`: TK01 found no joinable VHRdb training rows, and the
fixture-based TK02-TK05 follow-up arms were all neutral. Because no external source combination earned inclusion, I
updated `lyzortx/pipeline/track_g/v1_feature_configuration.json` to record that the v1 training-data lock stays
internal-only and did not retrain the final model.

#### What was implemented

- Added `lyzortx/pipeline/track_k/steps/build_external_data_decision_report.py`, which reads the five Track K
  manifests, recomputes every arm's deltas vs the TK01 internal-only baseline, emits a single comparison CSV, and
  writes a lock manifest under `lyzortx/generated_outputs/track_k/tk06_external_data_decision/`.
- Extended `lyzortx/pipeline/track_k/run_track_k.py` so `--step external-data-decision` runs TK06 and `--step all`
  now executes TK01 through TK06 in order.
- Added regression coverage for the TK06 normalization and decision policy plus the new runner dispatch path.
- Recorded the locked external-data outcome in `lyzortx/pipeline/track_g/v1_feature_configuration.json` via
  `external_data_lock_task_id`, `locked_training_data_arm`, and `locked_external_source_systems`.

#### Findings

The cumulative comparison vs the locked internal-only baseline is:

| Source / evaluated arm | Delta AUC | Delta top-3 | Delta Brier |
| --- | ---: | ---: | ---: |
| `+VHRdb` | `0.000000` | `0.000000` | `0.000000` |
| `+BASEL` | `0.000000` | `0.000000` | `0.000000` |
| `+KlebPhaCol` | `0.000000` | `0.000000` | `0.000000` |
| `+GPB` | `0.000000` | `0.000000` | `0.000000` |
| `+Tier B (Virus-Host DB + NCBI Virus/BioSample)` | `0.000000` | `0.000000` | `0.000000` |

- Best-performing source combination for this repo state: `internal_only`.
- TK01 remains the only non-fixture measurement here, and it found `0` joinable VHRdb rows from the available TI08
  artifact.
- TK02-TK05 proved the cumulative Track K wiring on bounded fixtures, but none of those follow-up arms beat the
  internal-only baseline on ROC-AUC, top-3 hit rate, or Brier score.
- No external arm improved at least one tracked metric without harming another metric, so TK06 locked
  `locked_external_source_systems: []`.

#### Interpretation

- The correct decision here is to lock the absence of evidence, not to fabricate a promotion path. The repo policy is
  explicit that missing production artifacts do not justify empty or placeholder wins.
- TK06 now gives Track K a reproducible decision boundary for future reruns: if real manifests later show a
  strictly non-harmful gain, the synthesis step will surface that arm cleanly; until then, internal-only remains the
  honest v1 baseline.
- No final-model retrain was run for TK06 because the promotion condition was not met.
