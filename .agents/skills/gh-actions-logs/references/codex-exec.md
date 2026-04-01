# Reading `codex exec` logs

This repo's Codex workflow logs contain two layers:

1. the outer GitHub Actions workflow and `openai/codex-action`
2. the inner `codex exec` session itself

Read them in that order, but diagnose failures from the inner session first.

## Reliable anchors

Search for:

```bash
rg -n 'Running: .*codex "exec"|provider:|model:|sandbox:|session id:|apply patch|Traceback|tokens used|Quota exceeded' \
  <log-file>
```

The most informative anchors are:
- `Running: ... codex "exec"`: the handoff from the workflow to Codex
- `provider:`, `model:`, `sandbox:`, `session id:`: session metadata
- `codex`: natural-language progress updates from the agent
- `exec`: shell commands and outputs from the agent
- `apply patch`: file edits made through Codex's patch tool
- `tokens used`: Codex token footer

## Actor meanings

Inside the `codex exec` region:

- `user` is the injected task prompt from the workflow
- `codex` is the agent's commentary updates
- `exec` is a shell command plus its stdout/stderr
- `apply patch` means Codex edited files directly

If a patch succeeds, the log often prints a unified diff immediately after
`patch: completed`. In these runs, that diff is patch-tool output, not evidence
that the agent manually ran `git diff`.

## Typical `codex exec` sequence

1. workflow invokes `openai/codex-action`
2. action prints the exact `codex exec` command
3. session metadata appears
4. the full task prompt is logged under `user`
5. agent commentary and `exec` commands interleave
6. `apply patch` blocks show edits and diffs
7. targeted tests or reruns appear
8. the first real traceback or command failure occurs
9. outer wrapper errors appear after the inner failure
10. `tokens used` appears near the end if Codex finished enough to report usage

## Root-cause ordering

Prefer this order when interpreting failures:

1. first concrete application error or traceback
2. failed command wrapper around that error
3. Codex/action wrapper summary
4. final GitHub `##[error]` line

Example pattern from the 2026-04-01 TL18 implement run:
- real inner failure:
  `AttributeError: 'dict' object has no attribute 'panel_match'`
- later wrapper noise:
  `Quota exceeded. Check your plan and billing details.`
- final action summary:
  `Error: codex exited with code 1`

That means the run had both an application bug and a quota problem, but the
traceback still tells you what code path failed first.

## Timeout vs non-timeout

Do not infer timeout from "the job ended before finishing."

Check:
- the job's configured `timeout-minutes`
- the timestamp span between the first and last relevant log lines
- whether the log explicitly says `timed out`

If the run ends with a traceback or `Quota exceeded` before the timeout window
is exhausted, it was not a timeout failure.

## Practical search patterns

Use these to navigate large Codex logs quickly:

```bash
rg -n 'Running: .*codex "exec"|provider:|model:|sandbox:' <log-file>
rg -n 'codex$|exec$|apply patch' <log-file>
rg -n 'Traceback|AttributeError|RuntimeError|ValueError|FAILED|ERROR:' <log-file>
rg -n 'tokens used|Quota exceeded|Process completed with exit code' <log-file>
```

Then read windows around the hits with `sed -n`.
