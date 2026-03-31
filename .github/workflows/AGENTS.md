# CI Workflows Directory

- When modifying workflows in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README documents the trigger model, actors, and automation lifecycle for all orchestrator and Claude workflows.
- `ci-duplicate-check.yml` runs pylint `symilar` on PRs and pushes to main to detect duplicate code blocks in
  `lyzortx/`. This is informational only (`continue-on-error: true`) and does not block merges.
- `claude-implement.yml` uses `anthropics/claude-code-action@v1` to implement orchestrator-dispatched tasks. Triggered
  on `orchestrator-task`-labeled issues (opened/reopened) and via `workflow_dispatch`. Resolves the CI image profile
  from the issue's `ci-image:*` label and runs Claude inside the matching prebaked GHCR container. Resolves the
  semantic model tier (`smart`/`simple`) from the issue's `<!-- model: ... -->` directive to a concrete Claude model ID
  via `parse_model_directive.py --provider claude`. Passes the Czarphage GitHub App token into the Claude action so git
  pushes and PR creation stay under `czarphage[bot]`. Requires `ANTHROPIC_API_KEY` and Czarphage app credentials.
- `claude-pr-review.yml` uses `anthropics/claude-code-action@v1` to auto-review `orchestrator-task`-labeled PRs on
  open/push, and supports interactive `@claude` mentions on any PR. Requires the `ANTHROPIC_API_KEY` repository secret.
  Claude posts reviews as `claude[bot]` (the action's own default app identity). Do not override the action's
  `github_token` here unless the downstream `claude[bot]` review-state checks are updated too. After reviewing, it
  dispatches downstream actions: auto-merge on approval, or `claude-pr-lifecycle.yml` on commented reviews.
- `claude-pr-lifecycle.yml` is triggered exclusively via `workflow_dispatch` (from `claude-pr-review.yml` or manually).
  It runs the Claude fix loop for unresolved review threads. Like `claude-implement.yml`, it passes the Czarphage token
  into the Claude action so branch updates stay under `czarphage[bot]`. The 3-round cap (`claude-review-round-N`
  labels) prevents infinite loops. A concurrency group ensures only one lifecycle run per PR at a time. The
  `workflow_dispatch`-only trigger prevents a self-cancellation loop where agent thread replies would fire
  `pull_request_review` events that cancel the in-progress run.
- `publish-codex-ci-image.yml` builds and publishes the prebaked GitHub Container Registry image used by the CI
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
- See root `AGENTS.md` → "External Service Integration Development" for the general policy on testing against live
  services before writing code and unit tests.
