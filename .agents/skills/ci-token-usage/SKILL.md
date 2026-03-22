---
name: ci-token-usage
description: >-
  Analyze token usage and cost across Codex CI and Claude PR review workflow
  runs — overview, per-ticket breakdown, and waste detection.
user-invocable: true
allowed-tools: Bash, Read
argument-hint: "[--runs N] [--ticket <issue_number>] [--waste]"
---

# CI Token Usage Analysis

Analyze token consumption and cost across Codex CI workflow runs and Claude PR
review runs for the `LorenaDerezanin/coli_phage_interactions_2023` repository.

Codex runs (implement + lifecycle) report token counts. Claude review runs
report USD cost, number of turns, and duration.

## How to run

All commands go through the Python script at `lyzortx/orchestration/ci_token_usage.py`.
Run it from the repository root using the `phage_env` environment:

```bash
micromamba activate phage_env
python -m lyzortx.orchestration.ci_token_usage $ARGUMENTS
```

## Subcommands

Route on `$ARGUMENTS`:

### No arguments (or `--runs N`)

Show a recent overview of the last N workflow runs (default 10):
- Run ID, workflow name, date, status, usage (tokens or USD cost), associated PR/issue
- Summary stats split by Codex (tokens) and Claude review (USD cost)

```bash
python -m lyzortx.orchestration.ci_token_usage
python -m lyzortx.orchestration.ci_token_usage --runs 20
```

### `--ticket <issue_number>`

Track ALL spend for a given orchestrator ticket:
1. Implementation runs (Codex) — token counts
2. Lifecycle review runs (Codex) — token counts
3. Claude PR review runs — USD cost and turn counts
4. Breakdown: implementation vs review rounds

```bash
python -m lyzortx.orchestration.ci_token_usage --ticket 42
```

### `--waste`

For each run, identify and report token waste patterns:
- Environment discovery waste (micromamba/conda fumbling)
- Failed commands (exit codes 1, 127, 128)
- Git config failures
- Number of file reads (sed -n commands)
- Number of shell commands (/bin/bash -lc)

```bash
python -m lyzortx.orchestration.ci_token_usage --waste
python -m lyzortx.orchestration.ci_token_usage --waste --runs 20
```

## Presenting results

After running the command, present the stdout output directly to the user. The script formats
its own tables and summary statistics. Add brief interpretation if the numbers reveal notable
patterns (e.g., review rounds costing more than implementation, high failure rates).
