---
description: >
  Fetch and parse GitHub Actions workflow logs. Accepts any combination of filters:
  run ID, PR number, workflow name, job name, step name, text pattern, failed-only.
  Designed for diagnosing long Codex implement and feedback runs, but works with any workflow.
  Use this to find long-running steps, measure wall-clock duration of Codex tasks, identify
  where time is spent in CI, spot stuck or slow runs, and extract timestamps that the GitHub
  web UI hides (the web UI strips timestamps from log lines — raw logs accessed via gh CLI
  include them).
---

# Parse Workflow Logs

Fetch, filter, and display logs from GitHub Actions workflow runs. All filters are combinable.

**Why this exists:** The GitHub Actions web UI does not show timestamps on individual log lines —
it only shows line numbers. The raw logs accessed via `gh run view --log` include full ISO 8601
timestamps on every line. This command is the way to answer questions like "how long did the Codex
step take?", "where is time being spent?", "is this run stuck or just slow?", and "what happened
at 14:32 UTC?".

## Resolve the run

The user can provide **any combination** of these to narrow down which run(s) to inspect.
Layer them as needed — more filters = more precise.

| Filter | How to apply |
|--------|-------------|
| Run ID | Use directly: `RUN_ID=23507135839` |
| PR number | `BRANCH=$(gh pr view <N> --json headRefName --jq '.headRefName')` then add `--branch "$BRANCH"` to `gh run list` |
| Workflow name | Add `--workflow "<name>.yml"` to `gh run list` |
| Status | Add `--status completed` or `--status in_progress` to `gh run list` |
| Latest N | Add `--limit N` to `gh run list` (default: show latest) |

### Example: combining PR + workflow + status

```bash
BRANCH=$(gh pr view 42 --json headRefName --jq '.headRefName')
RUN_ID=$(gh run list \
  --branch "$BRANCH" \
  --workflow "claude-implement.yml" \
  --status completed \
  --limit 1 \
  --json databaseId --jq '.[0].databaseId')
```

If no run ID was given, always resolve one before proceeding.

## Get run overview

```bash
gh run view "$RUN_ID" --verbose
```

Shows all jobs and steps with status and duration. Use this to identify which job/step to drill into.

## Fetch and filter logs

All of the following filters are combinable. Apply whichever the user asked for.

### Base: full log with boilerplate stripped

```bash
gh run view "$RUN_ID" --log 2>&1 \
  | grep -v -E '(Set up job|Complete job|Post |actions/checkout|actions/setup-python|##\[group\]|##\[endgroup\]|Runner Image|GITHUB_TOKEN Perm|Secret source|Prepare workflow)' \
  | head -200
```

### Filter by job

```bash
# List jobs
gh run view "$RUN_ID" --json jobs --jq '.jobs[] | {name, databaseId, status, conclusion}'

# Fetch logs for a specific job
gh run view "$RUN_ID" --log --job <JOB_ID> 2>&1
```

### Filter by step name

The log format is `<job>\t<step>\t<timestamp> <message>`. Filter by step using `awk` (not `grep -P`,
which is unavailable on macOS):

```bash
gh run view "$RUN_ID" --log 2>&1 \
  | awk -F'\t' '$2 == "Implement task with Codex" || $2 == "Fix with Codex"'
```

### Filter by text pattern

```bash
gh run view "$RUN_ID" --log 2>&1 | grep -i -E '<pattern>'
```

Common patterns:
- Errors: `'(error|warning|failed|fatal|exception|::error::)'`
- Git ops: `'(git commit|git push|gh pr create|Created pull request)'`
- Tests: `'(PASSED|FAILED|ERROR|pytest|test)'`

### Failed steps only

```bash
gh run view "$RUN_ID" --log-failed 2>&1
```

### Combining filters

Chain them with pipes. Example — errors in the Codex step of a specific PR's latest lifecycle run:

```bash
BRANCH=$(gh pr view 42 --json headRefName --jq '.headRefName')
RUN_ID=$(gh run list --branch "$BRANCH" --workflow "claude-pr-lifecycle.yml" --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --log 2>&1 \
  | awk -F'\t' '$2 == "Fix with Claude"' \
  | grep -i -E '(error|warning|failed)'
```

## Timing and duration analysis

Use these to find where time is being spent, identify stuck steps, and measure wall-clock duration.
The GitHub web UI only shows "Started 11m 21s ago" at the job level — these commands give you
per-step timing.

```bash
# Wall-clock duration of a specific step (first and last timestamp)
gh run view "$RUN_ID" --log 2>&1 \
  | awk -F'\t' '$2 == "<step name>"' \
  | sed -n '1p; $p'

# Step transitions with timestamps — shows when each step started (timeline view)
gh run view "$RUN_ID" --log 2>&1 \
  | awk -F'\t' '{step=$2} step!=prev {print; prev=step}'
```

**Portability note:** Use `awk -F'\t'` for step filtering, not `grep -P`. macOS ships BSD grep
which does not support `-P` (Perl regex). Similarly, `date -d` is GNU-only — on macOS use
`date -jf` or Python for timestamp arithmetic.

### Common timing questions

| Question | Command |
|----------|---------|
| How long did the Codex step take? | Filter by step name, compare first/last timestamp |
| Is this run stuck or just slow? | Check step transitions — if the latest timestamp is recent, it's working |
| Which step took the longest? | Use step transitions to see the timeline |
| What was happening at time X? | `grep '<YYYY-MM-DD>T<HH:MM>'` in the log output |

## Gotchas

- **No `grep -P` on macOS.** BSD grep does not support Perl regex (`-P` flag). Use `awk -F'\t'` for
  tab-delimited field matching and `grep -E` for extended regex. All examples in this command use
  portable syntax.
- **No `date -d` on macOS.** GNU coreutils `date -d` is not available. For timestamp arithmetic,
  use `date -jf '%Y-%m-%dT%H:%M:%S' "$ts" +%s` or a quick Python one-liner.
- **BOM in first log line.** The very first line of `gh run view --log` output often has a UTF-8
  BOM (`\xEF\xBB\xBF`) before the timestamp. This is invisible but can cause regex matches or
  string comparisons to fail on that line. The step-transition `awk` command handles this fine
  since it only compares the step field.
- **`in_progress` runs have limited logs.** `--log` only returns output for completed steps. For
  a run still in progress, use `--verbose` to see which step is currently executing and its status.
- **Large log output.** Codex runs produce megabytes of logs. Always pipe through `head -N` or
  filter by step/pattern to avoid flooding the terminal. Use `--job <JOB_ID>` to scope when possible.
- **`--log-failed` is narrow.** It only shows steps that actually failed — not steps that were
  cancelled or skipped. If a run was cancelled mid-step, check `--log` instead.
- **Logs are retained for 90 days** by GitHub's default retention policy.
