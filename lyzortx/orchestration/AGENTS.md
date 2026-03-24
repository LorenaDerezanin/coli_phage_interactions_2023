# Orchestration Directory

- When modifying files in this directory, update `lyzortx/orchestration/README.md` to reflect the changes.
- The README contains a Mermaid state diagram of the full automation lifecycle and descriptions of all components. Keep
  both in sync with the actual workflow logic.

# Plan Rendering Completeness

- Every field in `plan.yml` must appear in the rendered `PLAN.md`. If a field exists in the YAML, the renderer must
  surface it — acceptance criteria, model, implemented_in, baseline, description, and any future fields.
- When adding new fields to `plan.yml`, update `render_plan.py` to render them. Do not add YAML fields that are silently
  dropped during rendering.

# Pending Task Requirements

- Every pending task in `plan.yml` must have both a `model` field and non-empty `acceptance_criteria`.
- Done tasks may omit either — they are historical records, not dispatchable work.
- The orchestrator validates both fields at load time and raises `ValueError` if either is missing on a pending task.

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

# Orchestration Robustness vs Fail-Fast

- The root AGENTS.md fail-fast rule applies to pipeline code. Orchestration code is different: the orchestrator should
  be robust against transient failures (GitHub API errors, network timeouts, race conditions) via retries.
- But robustness means retrying on transient errors, not tolerating missing data or silently skipping work. If the
  orchestrator cannot dispatch a task because its inputs are missing, that is an error to surface, not a condition to
  swallow.
