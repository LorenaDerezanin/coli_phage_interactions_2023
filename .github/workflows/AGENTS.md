# CI Workflows Directory

- When modifying workflows in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README documents the trigger model, actors, and automation lifecycle for all orchestrator and Codex workflows.
- `ci-duplicate-check.yml` runs pylint `symilar` on PRs and pushes to main to detect duplicate code blocks in
  `lyzortx/`. This is informational only (`continue-on-error: true`) and does not block merges.
- `claude-pr-review.yml` uses `anthropics/claude-code-action@v1` to auto-review `orchestrator-task`-labeled PRs on
  open/push, and supports interactive `@claude` mentions on any PR. Requires the `ANTHROPIC_API_KEY` repository secret.
  Claude posts reviews as `claude[bot]` (the action's own OIDC app identity). After reviewing, it dispatches downstream
  actions: auto-merge on approval, or `codex-pr-lifecycle.yml` on commented reviews.
- `codex-pr-lifecycle.yml` is triggered exclusively via `workflow_dispatch` (from `claude-pr-review.yml` or manually).
  It runs the Codex fix loop for unresolved review threads. The 3-round cap (`codex-review-round-N` labels) prevents
  infinite loops. A concurrency group ensures only one lifecycle run per PR at a time. The `workflow_dispatch`-only
  trigger prevents a self-cancellation loop where Codex thread replies would fire `pull_request_review` events that
  cancel the in-progress run.
- `publish-codex-ci-image.yml` builds and publishes the prebaked GitHub Container Registry image used by the Codex
  workflows. The image is rebuilt from `.github/ci/Dockerfile` whenever its inputs change on `main`, and it can also be
  triggered manually.

# Concurrency and Thread Safety

- When adding or editing a workflow, consider whether parallel runs could cause problems (duplicate issues, race
  conditions on shared state, conflicting pushes to the same branch). If they can, add a `concurrency` group.
- Choose `cancel-in-progress: true` when only the latest run matters (e.g., PR reviews superseded by new pushes).
  Choose `cancel-in-progress: false` when every run carries unique side effects that must not be dropped (e.g.,
  orchestrator ticks that mark tasks done or create issues).
- Workflows triggered by multiple event types (e.g., `issues.closed` + `workflow_dispatch`) are especially prone to
  parallel runs — a single merge can fire both triggers simultaneously.
- For workflows that run inside Docker or other containers, write long-running command logs to a host/workspace-visible
  path while the job is still running (for example, a mounted workspace file under `.scratch/`). Do not put the only
  useful progress log in an ephemeral container-only path like `/tmp`.

# Workflow Logic Encapsulation

- Keep inline YAML expressions simple (single `contains()`, direct equality checks). If the expression is getting
  complex, that is a signal to move it into a Python helper under `lyzortx/` and call it from the workflow step.
- See root `AGENTS.md` for the general policy on testing against live services before writing code and unit tests.

# CI and Workflow Changes

- Manually test affected commands locally before committing CI changes. Workflow syntax errors and expensive to debug
  via push-and-wait.
- Run shell snippets (`gh pr view`, `printf`, variable substitutions) against real PRs to verify behavior.
- Document non-trivial devops changes in `lyzortx/research_notes/lab_notebooks/devops.md`.
