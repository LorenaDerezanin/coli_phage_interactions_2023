# GitHub Actions log layout

## Line shape

Logs fetched with `gh run view --log` typically look like:

```text
<job name>\t<step name>\t<timestamp>Z <content>
```

That means every line mixes:
- job context
- step context
- timestamp
- the actual message

When scanning, focus on the content after the timestamp. The job and step
prefixes are useful for orientation but are rarely the root cause.

## Common regions

Most job logs follow this order:

1. runner and container setup
2. workflow step execution
3. post-step cleanup
4. final GitHub wrapper status lines

Useful anchor strings:
- `##[group]` / `##[endgroup]` for grouped workflow sections
- `Run ...` for the command a workflow step executed
- `Post ...` for cleanup steps
- `##[error]Process completed with exit code 1.` for the final wrapper summary

## Discovery workflow

When you only have a run ID:

```bash
gh run view <run-id> --json jobs \
  --jq '.jobs[] | {databaseId, name, status, conclusion}'
```

That gives the job IDs you can map to saved files under
`.scratch/gh-actions-logs/<run-id>/`.

## Triage workflow

Start with pattern search:

```bash
rg -n "##\\[error\\]|Traceback|ERROR:|FAILED|exited [1-9]|timed out" <log-file>
```

Then inspect small windows around each hit:

```bash
sed -n '120,180p' <log-file>
```

Avoid reading the whole file unless the failure surface is still unclear.

## Job-step caveats

- `UNKNOWN STEP` is a known `gh`/GitHub limitation when logs cannot be matched
  back to exact step metadata.
- `--log-failed` is good for fast triage, but it may omit setup or earlier
  context that explains why the failed block happened.
- `gh run download` is for artifacts, not logs.
