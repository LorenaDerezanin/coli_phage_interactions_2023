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

# Model Tier Selection for Plan Tasks

- Do not choose `gpt-5.4-mini` by surface appearance alone. Some tasks look mechanical but are actually fragile
  artifact-boundary work.
- Escalate to `gpt-5.4` when a task does any of the following:
  - consumes outputs from an upstream task whose schema, provenance, or exclusion rules recently changed
  - changes evaluation, locking, sweep-selection, or model-comparison logic
  - depends on gitignored generated outputs that may exist locally in stale pre-replan form
  - introduces a permissive fallback such as zero-fill, cache reuse, or auto-regeneration
  - requires interpreting notebook or PR postmortem findings rather than just applying explicit edits
- Reserve `gpt-5.4-mini` for truly bounded mechanical work where the main risks are local code edits rather than
  cross-artifact contract mismatches.

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

# Skills and Model Fit

- Treat the `replan` skill as guidance that must work for the weakest model tier named in the resulting tasks.
- If a task would require extra judgment, tighter sequencing, or more explicit artifact-handling instructions for
  `gpt-5.4-mini` to succeed, either:
  - add that low-freedom guidance directly to the acceptance criteria, or
  - assign the task to `gpt-5.4` instead

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
