### 2026-03-22: TG01 implemented (LightGBM binary classifier on v1 expanded feature set)

#### What was implemented

- Added a new Track G entrypoint at `lyzortx/pipeline/track_g/run_track_g.py` and the main TG01 trainer at
  `lyzortx/pipeline/track_g/steps/train_v1_binary_classifier.py`.
- The TG01 trainer now:
  - bootstraps missing prerequisite outputs from ST0.1 through ST0.3 plus Tracks C, D, and E
  - merges the canonical Track C `pair_table_v1.csv` with Track D phage-genomic features and Track E pairwise
    compatibility features
  - tunes LightGBM on the existing leakage-safe ST0.3 5-fold grouped CV contract (`split_cv5_fold` derived from
    `cv_group`)
  - keeps logistic regression as an interpretable comparator on the same expanded feature space
  - writes reusable artifacts under `lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/`
- Added tests in `lyzortx/tests/test_track_g_v1_binary_classifier.py` covering:
  - expanded-feature row merging across Track C/D/E
  - feature-space construction
  - top-3 hit-rate calculation
  - CV candidate selection rules
  - `run_track_g.py` dispatch

#### Output summary

- TG01 output directory:
  - `tg01_model_summary.json`
  - `tg01_cv_candidate_results.csv`
  - `tg01_pair_predictions.csv`
  - `tg01_holdout_top3_rankings.csv`
- Final modeled feature space:
  - `21` categorical columns
  - `170` numeric columns
  - composition: `115` Track C host-genomic columns, `34` Track D phage-genomic columns, `14` Track E pairwise
    columns, plus the audited ST0.4 v0 baseline feature columns
- Scored-pair coverage:
  - `35,266` rows in `tg01_pair_predictions.csv`
  - this is the union of non-holdout out-of-fold predictions and final-model holdout predictions for hard-trainable
    pairs
- Best hyperparameters:
  - LightGBM: `learning_rate=0.05`, `min_child_samples=10`, `n_estimators=300`, `num_leaves=31`
  - Logistic regression: `C=3.0`
- Best LightGBM metrics:
  - mean CV ROC-AUC: `0.908344`
  - mean CV top-3 hit rate (all strains): `0.933153`
  - holdout ROC-AUC: `0.910007`
  - holdout top-3 hit rate (all strains): `0.892308`
  - holdout top-3 hit rate (susceptible strains only): `0.920635`
- Logistic regression comparator metrics:
  - mean CV ROC-AUC: `0.867378`
  - mean CV top-3 hit rate (all strains): `0.825475`
  - holdout ROC-AUC: `0.856172`
  - holdout top-3 hit rate (all strains): `0.723077`

#### Interpretation

1. The core Track G modeling upgrade is validated. The tuned LightGBM model materially outperforms the logistic
   comparator on both CV and holdout, so the nonlinear expanded-feature stack is carrying real signal rather than just
   overfitting the grouped folds.
2. The AUC target was met and slightly exceeded. Holdout ROC-AUC reached `0.910007`, which is above the stated
   `0.87-0.90` target band.
3. The top-3 target was almost met but not fully met on the strictest denominator. Holdout top-3 on all holdout strains
   reached `0.892308`, just below the `90%+` target, while susceptible-only holdout top-3 reached `0.920635`.
4. That denominator split matters. The all-strain miss is small in absolute terms (`58 / 65` strains hit), but the two
   non-susceptible or effectively no-hit holdout strains keep the all-strain metric below the acceptance threshold even
   though ranking quality on susceptible strains is already above target.
5. TG01 should therefore be considered a successful model-training milestone with one honest caveat: the LightGBM model
   is strong enough to move forward to calibration and interpretation work, but the repo should not claim that the
   all-strain top-3 target is fully achieved yet.

#### Next steps

1. Use `tg01_pair_predictions.csv` as the raw-score input for TG02 calibration work so isotonic and Platt scaling are
   evaluated on the exact tuned LightGBM configuration rather than a proxy model.
2. Use the same TG01 output directory as the reference point for TG03 ablations, with the logistic comparator retained
   as the linear baseline.
3. Inspect the `7` holdout miss strains in `tg01_holdout_top3_rankings.csv` before claiming the remaining gap is purely
   calibration-related; some of that gap may still be a ranking or abstention-policy problem rather than a probability
   problem.

### 2026-03-22: TG02 implemented (GBM calibration with isotonic and Platt scaling)

#### What was implemented

- Added the TG02 calibration step at `lyzortx/pipeline/track_g/steps/calibrate_gbm_outputs.py`.
- Updated `lyzortx/pipeline/track_g/run_track_g.py` and `lyzortx/pipeline/track_g/README.md` so Track G now exposes a
  dedicated `calibrate-gbm` step in addition to TG01 training.
- TG02 now:
  - bootstraps TG01 automatically when raw GBM predictions are missing
  - fits isotonic regression and Platt scaling on TG01 LightGBM out-of-fold predictions from one fixed ST0.3
    non-holdout calibration fold
  - evaluates raw, isotonic, and Platt probabilities on both the calibration fold and the fixed ST0.3 holdout
  - reports metrics separately for the `full_label` and `strict_confidence` slices
  - writes reusable artifacts under `lyzortx/generated_outputs/track_g/tg02_gbm_calibration/`
- Added test coverage in `lyzortx/tests/test_track_g_v1_binary_classifier.py` for:
  - Track G CLI dispatch of the new calibration step
  - TG02 end-to-end artifact generation on a synthetic calibration fixture

#### Output summary

- TG02 output directory:
  - `tg02_calibration_summary.csv`
  - `tg02_pair_predictions_calibrated.csv`
  - `tg02_ranked_predictions.csv`
  - `tg02_calibration_artifacts.json`
- Calibration/evaluation row counts:
  - calibration fold rows: `5,755`
  - holdout rows: `6,235`
  - strict-confidence calibration rows: `4,556`
  - strict-confidence holdout rows: `5,130`
- Holdout `full_label` metrics:
  - raw: ECE `0.083442`, Brier `0.113112`, log-loss `0.360181`
  - isotonic: ECE `0.020480`, Brier `0.103067`, log-loss `0.344266`
  - Platt: ECE `0.027842`, Brier `0.103604`, log-loss `0.333431`
- Holdout `strict_confidence` metrics:
  - raw: ECE `0.150698`, Brier `0.094783`, log-loss `0.307661`
  - isotonic: ECE `0.094470`, Brier `0.069391`, log-loss `0.250986`
  - Platt: ECE `0.097377`, Brier `0.070347`, log-loss `0.238036`

#### Interpretation

1. TG02 met the stated acceptance target on the required denominator. On the `full_label` holdout slice, isotonic
   achieved ECE `0.020480` and Platt achieved ECE `0.027842`, so both calibration methods landed below the
   `0.03` target without changing the fixed ST0.3 holdout contract.
2. Isotonic produced the best holdout calibration error and the best holdout Brier score. Relative to raw LightGBM, it
   cut holdout `full_label` ECE from `0.083442` to `0.020480` and improved Brier from `0.113112` to `0.103067`.
3. Platt produced the best holdout log-loss. Its `full_label` holdout log-loss of `0.333431` beat both isotonic
   (`0.344266`) and raw (`0.360181`), which suggests the sigmoid fit is slightly better behaved in the tails even
   though isotonic is better calibrated on average.
4. The strict-confidence slice remains materially harder to calibrate than the full-label slice. Both methods improve
   strict-confidence Brier and log-loss, but holdout ECE remains around `0.095-0.097`, so that slice still has a
   noticeable reliability gap.
5. Calibration improved probability quality, but it did not by itself solve the ranking shortfall seen in TG01. The
   honest claim is therefore narrower: Track G now has well-calibrated full-label probabilities suitable for downstream
   recommendation confidence reporting, while strict-confidence reliability still needs follow-up work.

#### Next steps

1. Use isotonic-scaled probabilities as the default calibrated `P(lysis)` for TG04 recommendation outputs because it
   gives the best holdout ECE and Brier on the required full-label slice.
2. Keep Platt-scaled probabilities available in the artifacts for sensitivity checks and downstream uncertainty
   comparisons because it has the best holdout log-loss.
3. In TG03, test whether the remaining strict-confidence calibration gap is driven by feature ablations, class-balance
   shifts, or group-specific error concentration rather than the choice of calibrator alone.
