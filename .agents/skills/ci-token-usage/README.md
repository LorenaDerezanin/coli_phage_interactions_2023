# ci-token-usage skill

Analyze token consumption and estimated USD cost across all CI workflows that invoke LLMs in this repository.

## Architecture

```
SKILL.md                            -- agent-facing prompt (subcommands, how to run, pricing check)
lyzortx/orchestration/ci_token_usage.py  -- CLI script (pure functions + gh CLI boundary)
lyzortx/tests/test_ci_token_usage.py     -- unit tests for pure functions
```

The skill is a thin SKILL.md prompt that delegates all real work to the Python CLI. The script follows the project's
standard pattern: pure functions for extraction/computation, side-effect boundary for `gh` CLI calls, and a
`cmd_overview` / `cmd_ticket` / `cmd_waste` command layer.

This skill intentionally does **not** own general GitHub Actions log browsing.
Use:
- `gh` to discover runs/jobs and download logs into `.scratch/gh-actions-logs/`
- `gh-actions-logs` to inspect those saved logs manually
- `ci-token-usage` only when the goal is spend, model, token, or waste analysis

## Covered workflows

| Workflow | File | What it does | LLM data available |
|---|---|---|---|
| Claude Implement | `claude-implement.yml` | Generates code from orchestrator issues | `total_cost_usd`, `num_turns`, `duration_ms` |
| Claude PR Lifecycle | `claude-pr-lifecycle.yml` | Addresses review feedback on PRs | `total_cost_usd`, `num_turns`, `duration_ms` |
| Claude PR Review | `claude-pr-review.yml` | Reviews PRs via `anthropics/claude-code-action` | `total_cost_usd`, `num_turns`, `duration_ms` |

## Design decisions

### Log-based LLM detection (not workflow-name based)

The script classifies runs by grepping log content, not by workflow filename or labels:

- **Claude invocation detected** = log contains `"total_cost_usd"` in a JSON result block
- **Codex invocation detected** (legacy) = log contains `tokens used` followed by a number
- **No LLM** = neither marker found

This is robust against workflow renames, label changes, and the fact that a single workflow can have both LLM-invoking
and non-LLM jobs. All implementation and lifecycle workflows now use Claude Code action.

### Two different cost sources

Codex and Claude log usage in fundamentally different formats:

- **Codex** (OpenAI) emits a single `tokens used` line with a raw count. No input/output breakdown, no cost.
- **Claude code action** (Anthropic) emits a JSON result block with `total_cost_usd` — an exact cost already computed
  by the action.

This means Claude costs are exact and Codex costs are estimates. The output marks this clearly with a warning line.

### Blended rate estimation for Codex

Since Codex logs only a total token count (no input/output split), we estimate cost using a blended rate:

```
blended = 30% * input_rate + 70% * output_rate
```

The 30/70 ratio reflects typical coding-agent behavior: agents read context (input) then generate code (output), so
output tokens dominate. This is an approximation — actual ratios may vary per run.

### Historical pricing with `as_of` dates

The `_OPENAI_RATES` table stores pricing as a list of `(input_per_1M, output_per_1M, as_of_date)` tuples per model,
ordered newest-first. When computing cost for a run, `_rate_for_model()` picks the newest entry whose `as_of` date is
<= the run date. This ensures:

- Current runs use current prices.
- Historical runs use the price that was in effect when they ran.
- When prices change, you add a new entry — old entries stay for historical accuracy.

### Runtime pricing verification

The SKILL.md instructs agents to check `_OPENAI_RATES` against the live OpenAI pricing page
(`https://developers.openai.com/api/docs/pricing`) once per session before presenting results. This avoids silently
using stale rates without requiring an automated scraping mechanism.

### Model extraction from logs

The script detects Claude model usage from the `total_cost_usd` JSON field in run logs. For legacy Codex runs, it falls
back to parsing `model: gpt-5.4` or `CODEX_MODEL: gpt-5.4` lines. `skipped` and `cancelled` runs show their status
directly.

### Unified cost column

The overview and waste tables use a single `Cost` column that shows `$X.XX` for any run with a computable cost,
regardless of whether it came from Codex token estimation or Claude's exact `total_cost_usd`. This makes cross-workflow
cost comparison straightforward.

## Lab notebook

Token usage snapshots and cost analysis are recorded in `lyzortx/research_notes/lab_notebooks/devops.md`, the lab
notebook for all coding infrastructure changes (CI/CD, tooling, automation, developer experience).

## Adding a new workflow

1. Add a `NEW_WORKFLOW = "filename.yml"` constant and append it to `ALL_WORKFLOWS`.
2. Determine what the workflow logs: tokens, cost JSON, or nothing.
3. If it has a new log format, add an extraction function (pure, unit-testable) and wire it into `_enrich_run()`.
4. If it uses a new pricing model, add entries to `_OPENAI_RATES` or create a parallel rate table.
5. Update the SKILL.md subcommand descriptions and this README.
