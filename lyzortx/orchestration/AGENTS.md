# Orchestration Directory

- When modifying files in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README contains a Mermaid state diagram of the full automation lifecycle and descriptions of all components. Keep
  both in sync with the actual workflow logic.

# Implementation Workflow

- `codex-implement.yml` and `codex-pr-lifecycle.yml` are disabled in GitHub Actions. Task implementation and review
  feedback fixes happen locally (laptop-based Claude sessions), not in CI.
- The orchestrator (`orchestrator.yml`) still runs in CI: it ticks on issue close events and `workflow_dispatch`, marks
  tasks done, and creates new issues.
- `claude-pr-review.yml` still runs in CI: it auto-reviews orchestrator-task PRs and either approves+merges or posts
  review comments.
- Full lifecycle:
  1. Orchestrator tick creates a GitHub issue for the next pending task.
  2. Local Claude implements the task and pushes a PR (`Closes #N`).
  3. `claude-pr-review.yml` reviews the PR.
  4. If the review has comments, local Claude reads the feedback, pushes fixes, and requests re-review.
  5. On approval, `claude-pr-review.yml` enables auto-merge. If auto-merge fails, local Claude merges manually.
  6. The merged PR closes the linked issue, which triggers an orchestrator tick to dispatch the next task.

# Knowledge Model Rendering

- `knowledge.yml` is the source of truth for consolidated project knowledge. `render_knowledge.py` renders it to
  `lyzortx/KNOWLEDGE.md`.
- Every field in `knowledge.yml` must appear in the rendered output. If a field exists in the YAML, the renderer must
  surface it — statement, sources, status, confidence, context, and relates_to.
- When adding new fields to `knowledge.yml`, update `render_knowledge.py` to render them.
- Run `python -m lyzortx.orchestration.render_knowledge` after modifying `knowledge.yml` to regenerate `KNOWLEDGE.md`.
- The validator (`knowledge_parser.validate_knowledge()`) must pass before rendering. Invalid YAML is rejected.

# Plan Rendering Completeness

- Every field in `plan.yml` must appear in the rendered `PLAN.md`. If a field exists in the YAML, the renderer must
  surface it — acceptance criteria, model, implemented_in, baseline, description, and any future fields.
- When adding new fields to `plan.yml`, update `render_plan.py` to render them. Do not add YAML fields that are silently
  dropped during rendering.

# Pending Task Requirements

- Every pending task in `plan.yml` must have both a `model` field and non-empty `acceptance_criteria`.
- Done tasks may omit either — they are historical records, not dispatchable work.
- The orchestrator validates both fields at load time and raises `ValueError` if either is missing on a pending task.

# Acceptance Criteria for Artifact-Boundary Tasks

- If a task touches generated artifacts or evaluation semantics, the acceptance criteria must constrain boundary
  behavior explicitly. Include, when relevant:
  - a stale-artifact check: existing default outputs must be regenerated or rejected if their schema/provenance is old
  - a provenance-path check: validation must follow the actual CLI-provided artifact path, not a hardcoded default
  - a narrow-fallback check: permissive behavior must be limited to the intended context
- When adding a permissive fallback, require two tests:
  - one proving the fallback works in the intended narrow context
  - one proving strict failure still occurs outside that context
- Never accept criteria that say only "rerun downstream task on clean outputs" when the upstream change alters what rows
  or files exist. The task text must restate the changed contract.

# Task Status Management

- Never manually set a task's `status` to `done` in `plan.yml`. The orchestrator automatically marks tasks as done when
  their corresponding GitHub issue is closed as completed (via PR merge with `Closes #N`).
- Agents should only add new tasks or modify pending task fields (acceptance_criteria, model, title). Status transitions
  are the orchestrator's responsibility.

# Done Task Immutability

- Never modify the title, acceptance_criteria, or any other field on a done task. Done tasks are historical records —
  they document what was originally required, not the current state of the code.
- If a later task changes the codebase in ways that make a done task's criteria look stale (e.g., deleting features that
  a done task created), that is expected. The done task records what was true when it was completed; the later task
  records what changed. Git history tells the full story.

# Push Before Closing Issues

- Always push code changes to main before closing GitHub issues. Issue close events trigger orchestrator runs
  immediately. If the new code hasn't been pushed yet, the old code runs and can recreate the issue you just closed.
- This applies to both "completed" and "not planned" closures — any close event triggers a tick.

# Orchestration Robustness vs Fail-Fast

- The root AGENTS.md fail-fast rule applies to pipeline code. Orchestration code is different: the orchestrator should
  be robust against transient failures (GitHub API errors, network timeouts, race conditions) via retries.
- But robustness means retrying on transient errors, not tolerating missing data or silently skipping work. If the
  orchestrator cannot dispatch a task because its inputs are missing, that is an error to surface, not a condition to
  swallow.

# PR Lifecycle Feedback Contract

- `codex-pr-lifecycle.yml` is disabled. Review feedback is addressed by local Claude sessions, not CI.
- `claude-pr-review.yml` decides whether to auto-merge (on approval with zero unresolved threads) or post review
  comments for local Claude to address.
- Keep lifecycle feedback detection simple: read visible PR feedback surfaces (top-level PR comments, inline review
  comments, and non-empty review bodies) and do not reintroduce unresolved-thread counting as the criterion for
  whether there is work to do.
