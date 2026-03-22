# Track G Modeling Pipeline

`python lyzortx/pipeline/track_g/run_track_g.py`

This command runs the implemented Track G modeling step:

1. `train-v1-binary`: build any missing Track C/D/E prerequisites, train tuned LightGBM and logistic-regression
   binary classifiers on the v1 expanded feature space, and write outputs under
   `lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/`
2. `calibrate-gbm`: fit isotonic and Platt calibrators on TG01 LightGBM outputs using the ST0.3 fold contract and
   write outputs under `lyzortx/generated_outputs/track_g/tg02_gbm_calibration/`
3. `feature-block-ablation`: run TG03 LightGBM ablations on the fixed ST0.3 holdout split and write outputs under
   `lyzortx/generated_outputs/track_g/tg03_feature_block_ablation_suite/`

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

The TG02 calibration directory includes:

1. `tg02_calibration_summary.csv`: calibration and holdout metrics for raw, isotonic, and Platt-scaled LightGBM
   probabilities across `full_label` and `strict_confidence` slices
2. `tg02_pair_predictions_calibrated.csv`: pair-level raw and calibrated LightGBM probabilities
3. `tg02_ranked_predictions.csv`: isotonic-ranked per-strain predictions with raw and Platt scores for comparison
4. `tg02_calibration_artifacts.json`: fitted isotonic thresholds, Platt coefficients, and input hashes

The TG03 ablation directory includes:

1. `tg03_ablation_summary.json`: per-arm metrics, best hyperparameters, feature-block membership, and lift vs the
   `v0_features_only` reference arm
2. `tg03_ablation_metrics.csv`: flat arm-level table with holdout AUC, top-3 hit rate, Brier, CV summaries, and deltas
   vs v0
3. `tg03_ablation_cv_candidate_results.csv`: candidate-level CV summaries for each ablation arm
4. `tg03_ablation_pair_predictions.csv`: non-holdout out-of-fold and final holdout probabilities for each arm
5. `tg03_ablation_holdout_top3_rankings.csv`: holdout top-3 rankings for each arm
