# Steel Thread v0

This directory contains a minimal end-to-end pipeline to validate feasibility before full plan execution.

## Purpose

- Prove we can run a reproducible path from raw interactions to recommendation-ready artifacts.
- Surface label quality and leakage risks early.
- Establish regression checks so refactors do not silently change behavior.

## Current Status

- Implemented:
  - `ST0.1` label policy and uncertainty flags: `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`.
  - `ST0.1b` strict confidence tiers: `lyzortx/pipeline/steel_thread_v0/steps/st01b_confidence_tiers.py`.
  - `ST0.2` canonical pair table builder: `lyzortx/pipeline/steel_thread_v0/steps/st02_build_pair_table.py`.
  - `ST0.3` split builder: `lyzortx/pipeline/steel_thread_v0/steps/st03_build_splits.py`.
  - `ST0.3b` split-suite builder: `lyzortx/pipeline/steel_thread_v0/steps/st03b_build_split_suite.py`.
  - `ST0.4` baseline model trainer: `lyzortx/pipeline/steel_thread_v0/steps/st04_train_baselines.py`.
  - `ST0.5` calibrator and ranker: `lyzortx/pipeline/steel_thread_v0/steps/st05_calibrate_rank.py`.
  - `ST0.6` recommender: `lyzortx/pipeline/steel_thread_v0/steps/st06_recommend_top3.py`.
  - `ST0.6b` ranking-policy comparator: `lyzortx/pipeline/steel_thread_v0/steps/st06b_compare_ranking_policies.py`.
  - `ST0.7` report artifact builder: `lyzortx/pipeline/steel_thread_v0/steps/st07_build_report.py`.
  - Regression gate for ST0.1: `lyzortx/pipeline/steel_thread_v0/checks/check_st01_regression.py`.
  - Regression gate for ST0.1b: `lyzortx/pipeline/steel_thread_v0/checks/check_st01b_regression.py`.
  - Regression gate for ST0.2: `lyzortx/pipeline/steel_thread_v0/checks/check_st02_regression.py`.
  - Regression gate for ST0.3: `lyzortx/pipeline/steel_thread_v0/checks/check_st03_regression.py`.
  - Regression gate for ST0.3b: `lyzortx/pipeline/steel_thread_v0/checks/check_st03b_regression.py`.
  - Regression gate for ST0.4: `lyzortx/pipeline/steel_thread_v0/checks/check_st04_regression.py`.
  - Regression gate for ST0.5: `lyzortx/pipeline/steel_thread_v0/checks/check_st05_regression.py`.
  - Regression gate for ST0.6: `lyzortx/pipeline/steel_thread_v0/checks/check_st06_regression.py`.
  - Regression gate for ST0.7: `lyzortx/pipeline/steel_thread_v0/checks/check_st07_regression.py`.

## Step Map

- `ST0.1`: Aggregate replicate/dilution observations into hard labels and uncertainty flags.
- `ST0.1b`: Add stricter confidence tiers as a parallel label view for dual-slice evaluation.
- `ST0.2`: Build canonical pair table with IDs, labels, uncertainty, and v0 features.
- `ST0.3`: Build fixed leakage-safe host-group splits.
- `ST0.3b`: Build split-suite artifacts for phage-family holdout and host+phage dual-axis stress tests.
- `ST0.4`: Train baseline models.
- `ST0.5`: Calibrate probabilities and generate rankings.
- `ST0.6`: Produce top-3 recommendations.
- `ST0.6b`: Compare recommendation ranking policies side-by-side (`raw`, `platt`, `isotonic`; with and without diversity
  cap).
- `ST0.7`: Emit reproducible report artifacts.

## How To Run

Run from repository root.

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st01
```

Run ST0.1b confidence tiering (consumes ST0.1 pair-level output):

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st01b
```

Run the ST0.1 regression gate (recomputes ST0.1 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st01_regression --run-st01
```

Run the ST0.1b regression gate (recomputes ST0.1 and ST0.1b then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st01b_regression --run-st01 --run-st01b
```

Run ST0.2 canonical pair table:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st02
```

Run the ST0.2 regression gate (recomputes ST0.1, ST0.1b, and ST0.2 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st02_regression --run-st01 --run-st01b --run-st02
```

Run ST0.3 split assignment builder:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st03
```

Run the ST0.3 regression gate (recomputes ST0.1, ST0.1b, ST0.2, and ST0.3 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st03_regression --run-st01 --run-st01b --run-st02 --run-st03
```

Run ST0.3b split-suite builder:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st03b
```

Run the ST0.3b regression gate (recomputes ST0.1, ST0.1b, ST0.2, ST0.3, and ST0.3b then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st03b_regression --run-st01 --run-st01b --run-st02 --run-st03 --run-st03b
```

Run ST0.4 baseline training:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st04
```

Run the ST0.4 regression gate (recomputes ST0.1 through ST0.4 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st04_regression \
  --run-st01 --run-st01b --run-st02 --run-st03 --run-st04
```

Run ST0.5 calibration and ranking:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st05
```

Run the ST0.5 regression gate (recomputes ST0.1 through ST0.5 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st05_regression \
  --run-st01 --run-st01b --run-st02 --run-st03 --run-st04 --run-st05
```

Run ST0.6 recommendation generation:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st06
```

ST0.6 defaults to `score_column=pred_logreg_platt` with `max_per_family=0` (no family-cap diversity constraint).

Run the ST0.6 regression gate (recomputes ST0.1 through ST0.6 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st06_regression \
  --run-st01 --run-st01b --run-st02 --run-st03 --run-st04 --run-st05 --run-st06
```

Run ST0.7 report build:

```bash
python -m lyzortx.pipeline.steel_thread_v0.steps.st07_build_report
```

Run ST0.6b ranking-policy comparison:

```bash
python -m lyzortx.pipeline.steel_thread_v0.steps.st06b_compare_ranking_policies
```

Run the ST0.7 regression gate (recomputes ST0.1 through ST0.7 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st07_regression \
  --run-st01 --run-st01b --run-st02 --run-st03 --run-st04 --run-st05 --run-st06 --run-st07
```

Alternative via orchestrator:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st01
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st01b
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st02
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st03
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st03b
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st04
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st05
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st06
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st07
```

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st06b
```

## Outputs

- Intermediate outputs: `lyzortx/generated_outputs/steel_thread_v0/intermediate/`.
- ST0.1 files:
  - `st01_label_policy_definition.json`
  - `st01_label_policy_audit.json`
  - `st01_pair_label_audit.csv`
- ST0.1b files:
  - `st01b_confidence_policy_definition.json`
  - `st01b_confidence_audit.json`
  - `st01b_pair_confidence_audit.csv`
- ST0.2 files:
  - `st02_pair_table.csv`
  - `st02_pair_table_audit.json`
  - `st02_feature_manifest.json`
- ST0.3 files:
  - `st03_split_assignments.csv`
  - `st03_split_protocol.json`
  - `st03_split_audit.json`
- ST0.3b files:
  - `st03b_split_suite_assignments.csv`
  - `st03b_split_suite_protocol.json`
  - `st03b_split_suite_audit.json`
- ST0.4 files:
  - `st04_pair_predictions_raw.csv`
  - `st04_model_metrics_raw.json`
  - `st04_model_artifacts.json`
- ST0.5 files:
  - `st05_calibration_summary.csv`
  - `st05_pair_predictions_calibrated.csv`
  - `st05_ranked_predictions.csv`
  - `st05_calibration_artifacts.json`
- ST0.6 files:
  - `st06_top3_recommendations.csv`
  - `st06_recommendation_summary.json`
- ST0.6b files:
  - `st06b_policy_comparison.csv`
  - `st06b_recommendations_all_policies.csv`
  - `st06b_top3_recommendations_best.csv`
  - `st06b_summary.json`
- ST0.7 files (`lyzortx/generated_outputs/steel_thread_v0/`):
  - `metrics_summary.csv`
  - `top3_recommendations.csv`
  - `calibration_summary.csv`
  - `error_analysis.csv`
  - `run_manifest.json`
- Baseline snapshots used by regression checks:
  - `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st05_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st06_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st07_expected_metrics.json`

## CI

GitHub Actions workflow: `.github/workflows/steel-thread-regression.yml`.
