# Track A: Data Integrity and Labeling — Lab Notebook

## 2026-03-28: TA11 — Fix label policy for borderline matrix_score=0 pairs

### Executive summary

Added a `training_weight_v3` column to the pair table that downweights borderline noise positives (pairs where
`label_hard_any_lysis=1` but `aux_matrix_score_0_to_4=0`) to 0.1. These are single-replicate lysis events where the
matrix aggregated to 0, identified as noise by VHRdb cross-validation. Sample weights are now threaded through both
5-fold CV evaluation and holdout training in the Track G pipeline.

#### Problem

The v1 label policy assigns `any_lysis=1` to any pair with at least one replicate showing lysis. This captures ~2,557
pairs where the interaction matrix score is 0 — meaning the aggregated signal across all replicates and dilutions rounds
to no infection. VHRdb independently confirms these as "No infection" (project notebook 2026-03-24). These noise
positives add label noise to the training set.

#### Approach: downweighting over label flipping

The acceptance criteria allowed either a label_v3 policy (flip labels to 0) or training weights (downweight). We chose
downweighting because:

1. **Preserves information.** Flipping labels discards the fact that some lysis was observed; downweighting reduces
   influence while keeping the signal available.
2. **Existing infrastructure.** `fit_final_estimator` already accepted a `sample_weight_key` parameter (unused). Only
   the CV path needed weight threading.
3. **VHRdb finding used downweighting.** The +3.1pp top-3 improvement in the VHRdb analysis came from downweighting,
   not label flipping.

#### Implementation details

1. **Pair table** (`st02_build_pair_table.py`): Added `BORDERLINE_NOISE_WEIGHT = 0.1` constant. The `training_weight_v3`
   column is computed as: `0.1` if `label_hard_any_lysis == "1"` and `aux_matrix_score_0_to_4 == "0"`, else `1.0`.
2. **CV training** (`train_v1_binary_classifier.py`): Added `sample_weights` field to `FoldDataset`. Weights are
   extracted in `prepare_fold_datasets` (defaulting to 1.0 if missing), and passed to `estimator.fit()` in both
   `evaluate_candidate_grid` and `score_rows_with_cv_predictions`.
3. **Holdout training**: Both LightGBM and logistic regression holdout fits now pass
   `sample_weight_key="training_weight_v3"` to `fit_final_estimator`.

#### Metric deltas

Metric deltas require running the full pipeline locally (Track A through Track G). The code changes are complete; deltas
will be reported after the next local pipeline run.

#### Scripts and outputs

- Weight column: `lyzortx/pipeline/steel_thread_v0/steps/st02_build_pair_table.py`
- Training pipeline: `lyzortx/pipeline/track_g/steps/train_v1_binary_classifier.py`
- Tests: `lyzortx/tests/test_ta11_borderline_weight.py`
- Generated output (after pipeline run): `lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/`
