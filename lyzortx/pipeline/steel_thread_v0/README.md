# Steel Thread v0

This directory contains a minimal end-to-end pipeline to validate feasibility before full plan execution.

## Purpose

- Prove we can run a reproducible path from raw interactions to recommendation-ready artifacts.
- Surface label quality and leakage risks early.
- Establish regression checks so refactors do not silently change behavior.

## Current Status

- Implemented:
  - `ST0.1` label policy and uncertainty flags: `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`.
  - Regression gate for ST0.1: `lyzortx/pipeline/steel_thread_v0/checks/check_st01_regression.py`.
- Planned next:
  - `ST0.1b` strict confidence tiering (`high_conf_pos`, `high_conf_neg`, `ambiguous`).
  - `ST0.2` through `ST0.7` (currently placeholders).

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

Run the ST0.1 regression gate (recomputes ST0.1 then compares against baseline):

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st01_regression --run-st01
```

Alternative via orchestrator:

```bash
python -m lyzortx.pipeline.steel_thread_v0.run_steel_thread_v0 --step check-st01
```

## Outputs

- Intermediate outputs: `lyzortx/generated_outputs/steel_thread_v0/intermediate/`.
- ST0.1 files:
  - `st01_label_policy_definition.json`
  - `st01_label_policy_audit.json`
  - `st01_pair_label_audit.csv`
- Baseline snapshot used by regression check: `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`.

## CI

GitHub Actions workflow: `.github/workflows/steel-thread-st01-regression.yml`.

It runs:

```bash
python -m lyzortx.pipeline.steel_thread_v0.checks.check_st01_regression --run-st01
```
