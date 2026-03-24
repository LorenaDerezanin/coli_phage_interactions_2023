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
  --workflow "codex-implement.yml" \
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

The log format is `<job>\t<step>\t<timestamp> <message>`. Filter by step:

```bash
gh run view "$RUN_ID" --log 2>&1 | grep -P '^\S+\t(Implement task with Codex|Fix with Codex)\t'
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
RUN_ID=$(gh run list --branch "$BRANCH" --workflow "codex-pr-lifecycle.yml" --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --log 2>&1 \
  | grep -P '^\S+\tFix with Codex\t' \
  | grep -i -E '(error|warning|failed)'
```

## Timing and duration analysis

Use these to find where time is being spent, identify stuck steps, and measure wall-clock duration.
The GitHub web UI only shows "Started 11m 21s ago" at the job level — these commands give you
per-step timing.

```bash
# Wall-clock duration of a specific step (first and last timestamp)
gh run view "$RUN_ID" --log 2>&1 \
  | grep -P '^\S+\t<step name>\t' \
  | sed -n '1p; $p'

# Step transitions with timestamps — shows when each step started (timeline view)
gh run view "$RUN_ID" --log 2>&1 \
  | awk -F'\t' '{step=$2} step!=prev {print; prev=step}'

# Find the longest gap between consecutive log lines (detect stuck/slow points)
gh run view "$RUN_ID" --log 2>&1 \
  | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' \
  | awk 'NR>1 {cmd="date -d \""prev"\" +%s"; cmd | getline t1; close(cmd); cmd="date -d \""$0"\" +%s"; cmd | getline t2; close(cmd); if(t2-t1>30) print t2-t1"s gap before "$0} {prev=$0}'
```

### Common timing questions

| Question | Command |
|----------|---------|
| How long did the Codex step take? | Filter by step name, compare first/last timestamp |
| Is this run stuck or just slow? | Check step transitions — if the latest timestamp is recent, it's working |
| Which step took the longest? | Use step transitions to see the timeline |
| What was happening at time X? | `grep '2026-03-24T14:32'` in the log output |

## Notes

- `gh run view --log` can be slow for large runs (Codex runs produce megabytes). Use `--job` to scope when possible.
- `--log-failed` only shows steps that failed — skips successful steps entirely.
- For `in_progress` runs, `--log` may not be available yet. Use `--verbose` for step status.
- Logs are retained by GitHub for 90 days.
