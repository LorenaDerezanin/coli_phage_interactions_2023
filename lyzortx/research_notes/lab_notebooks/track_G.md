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
