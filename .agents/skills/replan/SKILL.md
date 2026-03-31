---
name: replan
description: >
  Critically review and restructure the project plan (plan.yml). Use this skill when the user asks to review the plan,
  reassess priorities, question whether completed work is valid, investigate whether metrics are honest, restructure
  tracks, or respond to a discovery that invalidates prior assumptions (e.g., feature leakage, broken reproducibility,
  wasted work). Also use when the user says things like "replan", "rethink the plan", "what should we do next",
  "is this work still valid", "audit the pipeline", or "something seems wrong with the model".
---

# Replan: Critical Plan Review and Restructuring

This skill guides a systematic review of `lyzortx/orchestration/plan.yml` to identify invalid assumptions, wasted work,
integrity issues, and structural improvements. It was distilled from a real replanning session where label leakage
invalidated the v1 model and required deleting an entire track, restructuring the pipeline, and re-running downstream
work.

The principles below are ordered roughly by the sequence you'd follow in a review, but jump to whatever is relevant.

## Phase 1: Audit before trusting

Before accepting that completed work is valid, verify what the pipeline actually produced — not just that it ran
successfully.

### Check what the model learned, not just its metrics

Metrics can be inflated by leaked features. Before trusting AUC, top-3, or any evaluation number:

1. Look at SHAP / feature importance. If the top features are derived from training labels, the metrics are meaningless.
2. For every high-importance feature, trace it back to its source code. Read the function that constructs it. If it
   encodes training labels — directly (like counting positive pairs) or indirectly (like collaborative filtering on
   fold-excluded labels) — it's leakage regardless of fold-awareness.
3. Check whether any "deployment-realistic" or "novel-strain" arm exists. If the model performs better *without* certain
   features, those features are likely hurting generalization by compressing scores.

### Audit for adjacent leakage

When one leaked feature is found, audit sibling features from the same track or feature engineering step. Leakage
patterns cluster — if `host_n_infections` leaked, check every other feature derived from training labels in the same
block. Don't stop at the first finding.

## Phase 2: Decide what to keep, kill, or dead-end

### Kill invalid work

If completed work was built on invalid assumptions (e.g., a model that was memorizing labels, visualizations designed
around leaked metrics), delete it. Don't preserve it as "optional" or "alternative." The git history is the archive.
Keeping invalid work around as a variant signals that it might be acceptable to use, which it isn't.

### Dead-end vs delete

Tracks with completed code but no downstream consumers should be marked as dead ends, not deleted. Update the track
description to say it's a dead end, and remove it from other tracks' `depends_on` lists. The code stays in the repo and
may be useful later — but nothing should depend on it until it's validated.

### Assess downstream impact

For each piece of invalid work you're removing, trace what depends on it:
- Which tracks consume its outputs?
- Which tests validate against its metrics?
- Which visualizations or reports reference its numbers?
- Which config files encode its decisions?

All of these need updating. Make a list before you start changing things.

## Phase 3: Restructure the plan

### Decompose pipeline re-runs into sequential tickets

When a fundamental change requires re-running the pipeline, don't create one monolithic task. Decompose into:
1. **Make the change** — delete features, fix configs, update schemas
2. **Retrain / recompute** — run the modified pipeline
3. **Verify downstream** — confirm consumers still work with new outputs

Each step is a separate ticket. Each must pass CI independently.

### Size tickets to CI constraints

Each ticket must complete within the CI timeout (e.g., 45 minutes for Codex). Don't combine training + calibration +
SHAP + ablation + downstream verification into one task. If you're unsure whether a task fits, check the logs from
similar past runs:

```bash
gh run list --workflow=claude-implement.yml --limit=10 \
  --json displayTitle,startedAt,updatedAt \
  --jq '.[] | "\(.displayTitle) started=\(.startedAt) ended=\(.updatedAt)"'
```

### Match model tier to task complexity

Use the `simple` tier for mechanical tasks: deletions, re-running existing code, schema updates, downstream
verification. Reserve the `smart` tier for tasks requiring research judgment: investigating alternatives, proposing
new features, analyzing failure modes.

Ask: "Could a junior engineer do this by following explicit instructions?" If yes, it's usually a `simple` task.

### Artifact-boundary tasks are not mini by default

Some tasks look mechanical but are actually fragile because they sit at the boundary between old and new artifacts.
These should usually be `smart`, not `simple`.

Escalate to `smart` when a task does any of the following:
- consumes outputs from an upstream task whose schema, provenance, exclusion, or holdout rules recently changed
- re-runs a downstream evaluation after fixing leakage, reproducibility, or artifact validity upstream
- changes lock rules, sweep selection, or any "winner" decision logic
- depends on gitignored generated outputs that may exist locally in stale pre-replan form
- introduces a permissive fallback such as zero-fill, cache reuse, or auto-regeneration

Why: mini models often miss the boundary failures rather than the main logic:
- validating the wrong sibling manifest because a default path happened to exist
- widening a permissive fallback across training and CV paths when it was only meant for holdout evaluation
- trusting stale generated outputs because the filenames match
- failing one step later because the upstream artifact bootstrap chain was not restated in the task

If you still assign such a task to `simple`, the acceptance criteria must compensate by being much more specific.

### Set low freedom for fragile replan tasks

Anthropic's skill guidance says to reduce degrees of freedom when an operation is fragile and consistency matters.
Apply that here: artifact-boundary tasks need low-freedom acceptance criteria.

Do not write:
- "rerun mechanistic lift on clean outputs"

Write instead:
- what artifact contract changed
- what stale-artifact case must be handled
- what permissive behavior, if any, is allowed
- where strict failure must still happen

Example pattern:
- "Re-run TL05 on TL11-clean outputs, handling the fact that TL11 intentionally omits holdout pair rows."
- "Allow zero-fill only for holdout evaluation joins; non-holdout join misses remain a hard failure."
- "Validate provenance against the actual CLI-provided TL03/TL04 artifact paths, not default sibling manifests."
- "If default generated artifacts predate the new manifest schema, regenerate them or fail loudly."

### Lock decisions as human-approved

When model selection or configuration sweeps produce results within noise of each other, the winner should be a human
decision, not an auto-regenerated output. Separate the sweep (evidence gathering) from the lock (decision) in the
pipeline. Track J should train from the locked config — it should not re-run the sweep.

### Validate pending task completeness

Every pending task in `plan.yml` must have:
- `model` field — which Codex model runs it
- `acceptance_criteria` — what "done" means, specific enough to verify

The orchestrator validates both at load time and raises `ValueError` if either is missing. Check this before pushing
plan changes.

## Phase 4: Verify reproducibility

### Check for nondeterminism

When two runs of the same pipeline produce different results, investigate before assuming either is correct:

1. Check random seeds — are they set and consistent across all steps?
2. Check deterministic flags — for LightGBM, `deterministic=True` + `force_col_wise=True`
3. Check thread safety — `n_jobs` settings, parallel execution
4. Run the same step twice locally and compare outputs
5. Note: cross-machine differences are expected for many ML libraries. Lock decisions on one environment.

When documenting findings about library behavior (e.g., "LightGBM deterministic mode handles parallel threads"), always
include a URL to the official docs and a direct quote. Don't assert from memory.

## Phase 4.5: Test the task spec against the model tier

Anthropic's skill guidance also says to test a skill with all model tiers you plan to use. Apply the same idea to task
authoring: acceptance criteria must be written for the weakest model you intend to assign.

Before assigning `simple`, ask:
- Would a smaller model know which artifact boundary is dangerous here?
- Does the task text explicitly name the stale-artifact, path-provenance, and fallback-scope risks?
- If the model followed the acceptance criteria literally, could it still "complete" the task while being wrong?

If the answer to any of these is yes, either:
- tighten the acceptance criteria until the unsafe path is ruled out, or
- assign `smart` instead

## Phase 5: Update all artifacts

After restructuring the plan, make sure everything is consistent:

1. **Re-render PLAN.md** — `python -m lyzortx.orchestration.render_plan`
2. **Run tests** — `python -m pytest -q lyzortx/tests/`
3. **Write lab notebook entries** — document what was found, what was decided, and why. Include specific metrics,
   file paths, and source citations.
4. **Write project notebook entry** — summarize the strategic impact for readers who don't follow individual tracks.
5. **Update track descriptions** — if a track's status changed (dead-ended, invalidated, re-scoped), update its
   `description` field in `plan.yml`.
6. **Update `depends_on`** — if you removed a track or made it a dead end, remove it from other tracks' dependency
   lists.
7. **Verify orchestrator loads cleanly**:
   ```python
   from lyzortx.orchestration.orchestrator import load_pending_tasks
   from pathlib import Path
   tasks = load_pending_tasks(Path("lyzortx/orchestration/plan.yml"))
   for t in tasks:
       print(f"{t.task_id}: model={t.model}, criteria={len(t.acceptance_criteria)} items")
   ```

## Phase 6: Suggest next steps

If the plan has no pending tasks, or the remaining tasks don't address the most important open questions, propose next
steps. Don't just report "the plan is done" — the user invoked `/replan` because they want direction.

### Harvest future notes

Scan `lyzortx/research_notes/lab_notebooks/project.md` for entries marked "Future:" or containing phrases like
"consider", "revisit when", "follow-up", "defer". These are seeds planted by previous sessions that may now be
actionable. Evaluate each:
- Is the trigger condition met? (e.g., "revisit when external data is wired in")
- Is it still relevant given what's changed since it was written?
- Should it become a new plan task, or be deleted as obsolete?

### Propose new tasks

When proposing next steps:
1. Frame each as a concrete task with a clear acceptance criterion, not a vague direction
2. Justify why it's the highest-value next step given the current state
3. Size it (mini vs full model, estimated CI time)
4. Identify dependencies — can it start now, or does something need to happen first?

For artifact-boundary follow-up tasks, also state:
- which upstream artifact contract changed
- whether stale default artifacts must be regenerated or rejected
- which fallback behavior is permitted and where

Do not suggest presentation or visualization tasks (dashboards, demos, HTML reports). Agents produce bad visual
artifacts. Presentation work is human-driven.

## Checklist

Use this as a quick reference when starting a replan:

- [ ] Read the current `plan.yml` and `PLAN.md`
- [ ] For completed modeling tasks: check SHAP / feature importance — are top features honest?
- [ ] For each high-importance feature: trace to source code — does it encode labels?
- [ ] If leakage found: audit sibling features in the same block
- [ ] Identify downstream consumers of invalid work
- [ ] Delete invalid work (code, tests, notebooks) — don't preserve as "optional"
- [ ] Dead-end tracks with no consumers (update description + remove from depends_on)
- [ ] Decompose re-run into sequential tickets sized to CI timeout
- [ ] Assign model tiers (mini for mechanical, full for research)
- [ ] Lock human-decided configurations — separate sweep from lock
- [ ] Add model + acceptance_criteria to every pending task
- [ ] Re-render PLAN.md and run tests
- [ ] Write lab notebook + project notebook entries with citations
- [ ] Verify `load_pending_tasks` succeeds on the updated plan
- [ ] If no pending tasks remain: harvest "Future:" notes from project.md and propose next steps
