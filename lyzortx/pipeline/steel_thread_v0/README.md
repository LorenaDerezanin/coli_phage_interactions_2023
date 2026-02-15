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
  - `ST0.4` baseline model trainer: `lyzortx/pipeline/steel_thread_v0/steps/st04_train_baselines.py`.
  - Regression gate for ST0.1: `lyzortx/pipeline/steel_thread_v0/checks/check_st01_regression.py`.
  - Regression gate for ST0.1b: `lyzortx/pipeline/steel_thread_v0/checks/check_st01b_regression.py`.
  - Regression gate for ST0.2: `lyzortx/pipeline/steel_thread_v0/checks/check_st02_regression.py`.
  - Regression gate for ST0.3: `lyzortx/pipeline/steel_thread_v0/checks/check_st03_regression.py`.
  - Regression gate for ST0.4: `lyzortx/pipeline/steel_thread_v0/checks/check_st04_regression.py`.
- Planned next:
  - `ST0.5` through `ST0.7` (currently placeholders).

## Step Map

- `ST0.1`: Aggregate replicate/dilution observations into hard labels and uncertainty flags.
- `ST0.1b`: Add stricter confidence tiers as a parallel label view for dual-slice evaluation.
- `ST0.2`: Build canonical pair table with IDs, labels, uncertainty, and v0 features.
- `ST0.3`: Build fixed leakage-safe splits.
- `ST0.4`: Train baseline models.
- `ST0.5`: Calibrate probabilities and generate rankings.
- `ST0.6`: Produce top-3 recommendations.
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

Run ST0.4 baseline training:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step st04
```

Run the ST0.4 regression gate (recomputes ST0.1 through ST0.4 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st04_regression --run-st01 --run-st01b --run-st02 --run-st03 --run-st04
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
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st04
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
- ST0.4 files:
  - `st04_pair_predictions_raw.csv`
  - `st04_model_metrics_raw.json`
  - `st04_model_artifacts.json`
- Baseline snapshots used by regression checks:
  - `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json`
  - `lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json`

## CI

GitHub Actions workflow: `.github/workflows/steel-thread-regression.yml`.
