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
4. `compute-shap`: compute TG04 TreeExplainer SHAP explanations for the tuned LightGBM and write outputs under
   `lyzortx/generated_outputs/track_g/tg04_shap_explanations/`
5. `feature-subset-sweep`: run TG05 2-block and 3-block subset sweeps with TG01-locked LightGBM hyperparameters and
   write outputs under `lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/`

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
5. `tg02_benchmark_summary.json`: v1 benchmark protocol summary with dual-slice point estimates and bootstrap CIs for
   ROC-AUC, Brier score, ECE, and top-3 hit rate

The benchmark summary is locked to the ST0.3 split protocol (`steel_thread_v0_st03_split_v1`) so downstream Track F
and Track G analyses stay on the same canonical evaluation contract.

The TG03 ablation directory includes:

1. `tg03_ablation_summary.json`: per-arm metrics, best hyperparameters, feature-block membership, and lift vs the
   `v0_features_only` reference arm
2. `tg03_ablation_metrics.csv`: flat arm-level table with holdout AUC, top-3 hit rate, Brier, CV summaries, and deltas
   vs v0
3. `tg03_ablation_cv_candidate_results.csv`: candidate-level CV summaries for each ablation arm
4. `tg03_ablation_pair_predictions.csv`: non-holdout out-of-fold and final holdout probabilities for each arm
5. `tg03_ablation_holdout_top3_rankings.csv`: holdout top-3 rankings for each arm

The TG04 SHAP explanation directory includes:

1. `tg04_recommendation_pair_explanations.csv`: top-3 per-strain phage recommendations with top positive and negative
   SHAP drivers for each recommended pair
2. `tg04_global_feature_importance.csv`: global mean-absolute SHAP ranking across the full hard-trainable panel
3. `tg04_per_strain_difficulty_summary.csv`: per-strain easy/moderate/hard summary with top drivers and confidence
   separation metrics
4. `tg04_shap_summary.json`: top global drivers, difficulty counts, and input hashes for notebook/report reuse

The TG05 feature-subset sweep directory includes:

1. `tg05_feature_subset_summary.json`: per-arm metrics, TG01 locked hyperparameters, and winner selection metadata
2. `tg05_feature_subset_metrics.csv`: holdout ROC-AUC, top-3 hit rate, and Brier for every 2-block and 3-block arm
3. `tg05_feature_subset_pair_predictions.csv`: non-holdout OOF and final holdout probabilities for every evaluated arm
4. `tg05_feature_subset_holdout_top3_rankings.csv`: holdout top-3 rankings for every evaluated arm
5. `tg05_locked_v1_feature_config.json`: flat locked v1 feature-block decision for downstream Track F/H/P work
6. `tg05_locked_v1_feature_columns.csv`: concrete categorical/numeric columns for the locked subset
