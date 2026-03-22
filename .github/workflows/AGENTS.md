# CI Workflows Directory

- When modifying workflows in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README documents the trigger model, actors, and automation lifecycle for all orchestrator and Codex workflows.
- `ci-duplicate-check.yml` runs pylint `symilar` on PRs and pushes to main to detect duplicate code blocks in
  `lyzortx/`. This is informational only (`continue-on-error: true`) and does not block merges.
- `claude-pr-review.yml` uses `anthropics/claude-code-action@v1` to auto-review `orchestrator-task`-labeled PRs on
  open/push, and supports interactive `@claude` mentions on any PR. Requires the `ANTHROPIC_API_KEY` repository secret.
  Claude posts reviews as `claude[bot]` (the action's own OIDC app identity).
- `codex-pr-lifecycle.yml` triggers on COMMENTED reviews from `chatgpt-codex-connector[bot]` or `claude[bot]` only.
  Reviews from other users (including the ORCHESTRATOR_PAT holder) are excluded to prevent feedback loops. A concurrency
  group ensures only one lifecycle run per PR at a time.

# Workflow Logic Encapsulation

- Keep inline YAML expressions simple (single `contains()`, direct equality checks). If the expression is getting
  complex, that is a signal to move it into a Python helper under `lyzortx/` and call it from the workflow step.
- See root `AGENTS.md` → "External Service Integration Development" for the general policy on testing against live
  services before writing code and unit tests.
