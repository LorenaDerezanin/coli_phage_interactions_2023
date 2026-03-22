---
name: graphite
description: |
  Use for Graphite CLI (gt) stacked PRs workflow. This repo has Graphite initialized.
  Triggers: graphite, stacked PRs, dependent PRs, chained PRs, PR stack, gt create,
  gt modify, gt submit, gt sync, gt restack, gt log, gt checkout, gt up, gt down,
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

### Non-interactive mode (required)

Always pass `--no-interactive` to `gt submit` and other commands that support it. Agents cannot respond to
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

### Sync command

Use `gt sync` instead of `git fetch && git rebase` to stay current with trunk (aligns with AGENTS.md Branch
Protection policy).
