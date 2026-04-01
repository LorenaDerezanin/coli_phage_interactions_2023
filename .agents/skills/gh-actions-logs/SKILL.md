---
name: gh-actions-logs
description: >
  Read GitHub Actions logs that are already downloaded under `.scratch/gh-actions-logs/`, especially Codex `codex exec`
  jobs. Use this skill when the user asks why a workflow failed, whether a run really timed out, how close a job was
  to finishing, how to navigate a saved Actions log, or what happened inside a Codex implement/lifecycle run. Fetch
  the logs with the `gh` skill first, then use this skill to inspect them efficiently.
---

# GitHub Actions Log Reading

Assume the relevant logs already exist under `.scratch/gh-actions-logs/`. If
they do not, use the `gh` skill first to fetch them.

## First-pass workflow

1. Identify the exact run/job log file you need.
2. Use `rg -n` to find failure markers before reading large chunks.
3. Read narrow windows with `sed -n '<start>,<end>p'`.
4. Separate the first real failure from wrapper noise that appears later.
5. Only after locating the real failure, decide whether timeout, quota, CI
   plumbing, or application logic was the actual blocker.

Start with searches like:

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
