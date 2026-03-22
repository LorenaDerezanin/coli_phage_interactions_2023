---
name: graphite
description: |
  Use for Graphite CLI (gt) stacked PRs workflow. This repo has Graphite initialized locally only.
  Triggers: graphite, stacked PRs, dependent PRs, chained PRs, PR stack, gt create,
  gt modify, gt sync, gt restack, gt log, gt checkout, gt up, gt down,
  rebase my stack, fix stack conflicts, split PR, land my stack, merge stack,
  sync with main/trunk, reorder branches, fold commits, amend stack, move branch
  to different parent, stack out of date, update my stack, create a stack of PRs.
---

# Graphite Stacked PRs — Project Customizations

This skill wraps the vendored Graphite skill with project-specific policies.

**Base reference:** follow the workflow in `vendor/claude-code-graphite/SKILL.md` with the overrides below.
For the command cheatsheet see `vendor/claude-code-graphite/references/cheatsheet.md`.
For conflict resolution patterns see `vendor/claude-code-graphite/references/conflict-resolution.md`.

## Overrides

### Repo not synced with Graphite (critical)

This repo is **not** synced with Graphite's remote service. Only local Graphite commands work. Remote-dependent
commands like `gt submit` will fail. Use `git push` (with `-u` for new branches) instead of `gt submit` for pushing
branches to the remote.

### Deprecated commands

The vendored skill references `gt commit` and `gt commit create` in some places. These are deprecated. Use `gt create`
instead (e.g., `gt create -am "msg"`). The vendored cheatsheet already reflects this.

### Non-interactive mode (required)

Always pass `--no-interactive` to `gt create` and other commands that support it. Agents cannot respond to
interactive prompts.

### Stack planning

Plan stacks before coding (as the base skill says), but do **not** use TodoWrite. Present the stack structure
in the conversation and ask for confirmation.

### PR conventions

For orchestrator tasks, follow the AGENTS.md PR creation policy:

- Title pattern: `[ORCH][TASK_ID] Brief description`
- Body must include `Closes #<issue_number>`
- Add `--label orchestrator-task`

For non-orchestrator stacked PRs, use descriptive titles without the `[ORCH]` prefix.

### PR URLs

Return GitHub PR URLs, not Graphite web URLs.

### Conflict resolution for requirements.txt

`requirements.txt` must stay alphabetically sorted with exact version pins. Do not accept-theirs blindly —
resolve manually by merging both sides and re-sorting.

### Sync and conflict workflow (overrides base skill)

Use `gt sync` instead of `git fetch && git rebase` to stay current with trunk (aligns with AGENTS.md Branch
Protection policy).

**Important correction to the base skill's conflict flow:** `gt sync` does **not** pause on conflicts. It
restacks only branches that can be restacked cleanly and reports any conflicted branches. To resolve:

```bash
gt sync                          # Pull trunk, restack clean branches, report conflicts
# gt sync reports: "Could not restack branch-X"
gt checkout branch-X             # Go to the conflicted branch
gt restack                       # This is where the actual conflict resolution happens
# ... resolve conflicts in editor ...
gt continue -a                   # Stage resolved files and continue the restack
git push                         # Push the fixed branch (gt submit will not work)
```

Do **not** run `gt continue -a` immediately after `gt sync` — there is no in-progress operation to continue
at that point.

### When to use Graphite vs plain git

Use Graphite (`gt`) **only** when creating stacked PRs. For single-branch workflows (one PR, no stack), use plain
`git` commands for branching, committing, and pushing. Graphite adds overhead and branch-naming surprises (e.g.,
`gt create` generates its own branch name from the commit message) that are unnecessary for non-stacked work.
