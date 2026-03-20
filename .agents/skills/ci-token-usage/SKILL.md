---
name: ci-token-usage
description: >-
  Analyze token usage and waste across Codex CI workflow runs — overview, per-ticket
  breakdown, and waste detection.
user-invocable: true
allowed-tools: Bash, Read
argument-hint: "[--runs N] [--ticket <issue_number>] [--waste]"
---

# CI Token Usage Analysis

Analyze token consumption across Codex CI workflow runs for the
`LorenaDerezanin/coli_phage_interactions_2023` repository.

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
- Run ID, workflow name, date, status, tokens used, associated PR/issue
- Total tokens across all runs
- Average tokens per successful run vs failed run

```bash
python -m lyzortx.orchestration.ci_token_usage
python -m lyzortx.orchestration.ci_token_usage --runs 20
```

### `--ticket <issue_number>`

Track ALL token spend for a given orchestrator ticket:
1. The implement run that created the PR
2. All lifecycle (review feedback) runs for that PR
3. Total tokens burned for the ticket
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
