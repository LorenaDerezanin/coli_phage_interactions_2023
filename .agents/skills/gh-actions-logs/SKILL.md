---
name: gh-actions-logs
description: >
  Read GitHub Actions logs that are already downloaded under `.scratch/gh-actions-logs/`, especially Codex `codex exec`
  jobs. Use this skill when the user asks why a workflow failed, whether a run really timed out, how close a job was
  to finishing, how to navigate a saved Actions log, or what happened inside a Codex implement/lifecycle run. Fetch
  the logs with the `gh` skill first, then use this skill to inspect them efficiently.
---

# GitHub Actions Log Reading

## Extract and clean Codex logs

Use `lyzortx.orchestration.github_logs` to extract the Codex session from a run
and strip noise (diffs, exec output, ANSI codes). This is the fastest way to
understand what a Codex run did.

```bash
# From a GitHub Actions URL — fetches, extracts, cleans, saves to .scratch/
python -m lyzortx.orchestration.github_logs extract-codex \
    https://github.com/LyzorTx/coli_phage_interactions_2023/actions/runs/<run>/job/<job>

# From an already-downloaded log file
python -m lyzortx.orchestration.github_logs extract-codex --local \
    .scratch/gh-actions-logs/<run-id>/<job-id>.log

# Only Codex reasoning (drops exec blocks entirely)
python -m lyzortx.orchestration.github_logs extract-codex --codex-only <URL-or-path>

# Keep full exec output (default strips it, keeping only command + status)
python -m lyzortx.orchestration.github_logs extract-codex --keep-exec-output <URL-or-path>
```

Output files are saved to `.scratch/gh-actions-logs/<run-id>/`:
- `<job-id>.cleaned.log` — reasoning + commands (no diffs, no exec output, no ANSI)
- `<job-id>.codex-only.log` — just the Codex commentary and patch summaries

**Always start with the cleaned or codex-only log** before falling back to raw
log inspection. These are typically 100-500 lines vs 100K+ raw.

## Raw log inspection

If the cleaned log is insufficient, fall back to raw logs under
`.scratch/gh-actions-logs/`. If they do not exist, use the `gh` skill first to
fetch them.

1. Identify the exact run/job log file you need.
2. Use `rg -n` to find failure markers before reading large chunks.
3. Read narrow windows with `sed -n '<start>,<end>p'`.
4. Separate the first real failure from wrapper noise that appears later.
5. Only after locating the real failure, decide whether timeout, quota, CI
   plumbing, or application logic was the actual blocker.

```bash
rg -n "##\\[error\\]|Traceback|ERROR:|FAILED|exited [1-9]|timed out|Quota exceeded|tokens used" \
  .scratch/gh-actions-logs/<run-id>/<job-id>.log
```

Prefer local targeted reads over dumping whole logs.

## How to interpret what you find

- Treat the earliest concrete traceback or failed command as the primary lead.
- Treat final wrapper lines like `Process completed with exit code 1` as summary
  signals, not root cause.
- If the user asks whether a job timed out, compare the timestamp span in the
  log against the workflow timeout and search for explicit timeout language
  before concluding it was a timeout.
- `gh run view --log` may label lines as `UNKNOWN STEP`; that is often a GitHub
  log-format limitation rather than evidence that the workflow itself lost step
  metadata.

## References

- General GitHub Actions log structure: see
  [references/log-layout.md](references/log-layout.md)
- Codex `codex exec` anatomy and failure reading: see
  [references/codex-exec.md](references/codex-exec.md)
