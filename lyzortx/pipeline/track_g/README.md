# Track G Modeling Pipeline

`python lyzortx/pipeline/track_g/run_track_g.py`

This command runs the implemented Track G modeling step:

1. `train-v1-binary`: build any missing Track C/D/E prerequisites, train tuned LightGBM and logistic-regression
   binary classifiers on the v1 expanded feature space, and write outputs under
   `lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/`

The TG01 trainer reuses the canonical ST0.2 / ST0.3 leakage-safe contract:

1. ST0.3's `split_holdout` assignment remains the fixed holdout boundary.
2. ST0.3's `split_cv5_fold` assignments provide the grouped 5-fold non-holdout CV used for tuning.
3. The feature space combines:
   - ST0.4's audited v0 metadata feature columns
   - Track C's additional host-genomic columns from `pair_table_v1.csv`
   - Track D's phage-genomic feature CSVs
   - Track E's pairwise compatibility feature CSVs

The output directory includes:

1. `tg01_model_summary.json`: best hyperparameters plus CV and holdout metrics for both models
2. `tg01_cv_candidate_results.csv`: flattened candidate-level tuning summaries
3. `tg01_pair_predictions.csv`: non-holdout out-of-fold predictions and final holdout predictions
4. `tg01_holdout_top3_rankings.csv`: top-3 holdout rankings for both model families
