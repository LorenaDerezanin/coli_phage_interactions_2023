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
