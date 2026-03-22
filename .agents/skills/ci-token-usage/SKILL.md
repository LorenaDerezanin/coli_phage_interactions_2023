---
name: ci-token-usage
description: >-
  Analyze token usage and cost across Codex CI and Claude PR review workflow
  runs — overview, per-ticket breakdown, and waste detection.
user-invocable: true
allowed-tools: Bash, Read, WebFetch
argument-hint: "[--runs N] [--ticket <issue_number>] [--waste]"
---

# CI Token Usage Analysis

Analyze token consumption and estimated USD cost across Codex CI workflow runs
and Claude PR review runs for the `LorenaDerezanin/coli_phage_interactions_2023`
repository.

- **Claude review runs** report exact `total_cost_usd` from the action logs.
- **Codex runs** report only a total token count. Cost is estimated using
  cached per-model pricing with a 30% input / 70% output blended rate. The
  pricing table lives in `_OPENAI_RATES` in
  `lyzortx/orchestration/ci_token_usage.py` and includes an `as_of` date per
  entry so historical runs use the rate that was current when they ran.

## Pricing verification (once per session)

Before presenting results, **always verify that the cached OpenAI rates are
still current**. Check `_OPENAI_RATES` in
`lyzortx/orchestration/ci_token_usage.py`, then fetch
<https://developers.openai.com/api/docs/pricing> and compare. If rates have
changed, update the table (add a new entry with today's date; keep old entries
for historical accuracy). This check only needs to happen once per conversation.

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
- Run ID, workflow, date, status, model, estimated USD cost, associated PR/issue
- Totals and averages split by Codex vs Claude

```bash
python -m lyzortx.orchestration.ci_token_usage
python -m lyzortx.orchestration.ci_token_usage --runs 20
```

### `--ticket <issue_number>`

Track ALL spend for a given orchestrator ticket:
1. Implementation runs (Codex) — tokens, model, estimated cost
2. Lifecycle runs — usually no LLM invocation
3. Claude PR review runs — exact USD cost and turn counts
4. Total estimated cost across all run types

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
