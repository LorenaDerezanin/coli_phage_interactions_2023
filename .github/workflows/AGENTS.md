# CI Workflows Directory

- When modifying workflows in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README documents the trigger model, actors, and automation lifecycle for all orchestrator and Codex workflows.
- `ci-duplicate-check.yml` runs pylint `symilar` on PRs and pushes to main to detect duplicate code blocks in
  `lyzortx/`. This is informational only (`continue-on-error: true`) and does not block merges.
- `claude-pr-review.yml` uses `anthropics/claude-code-action@v1` to auto-review every PR on open/push, and supports
  interactive `@claude` mentions. Requires the `ANTHROPIC_API_KEY` repository secret. Claude posts reviews as
  `claude[bot]` (the action's own OIDC app identity).
- `codex-pr-lifecycle.yml` triggers on COMMENTED reviews from `chatgpt-codex-connector[bot]` or `claude[bot]` only.
  Reviews from other users (including the ORCHESTRATOR_PAT holder) are excluded to prevent feedback loops. A concurrency
  group ensures only one lifecycle run per PR at a time.
