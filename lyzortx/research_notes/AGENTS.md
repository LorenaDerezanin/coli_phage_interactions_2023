# Research Notes Directory

## How lab notebooks are written

Lab notebooks live under `lab_notebooks/`. Each file is either track-specific (`track_ST.md`, `track_B.md`, etc.) or
cross-track (`project.md`).

Entries are written by agents dispatched through the orchestrator pipeline:

1. `lyzortx/orchestration/orchestrator.py` creates GitHub issues for pending plan tasks.
2. `.github/workflows/orchestrator.yml` triggers the orchestrator on issue close and workflow dispatch.
3. When an agent implements a task, the orchestrator issue body instructs it to write findings to
   `lyzortx/research_notes/lab_notebooks/track_{TRACK}.md` following the existing entry format.
4. `project.md` is used for cross-track strategic decisions and plan-level notes that are not specific to one track.
5. `devops.md` records coding infrastructure changes: CI/CD workflows, tooling, automation, token/cost analysis,
   developer experience improvements, and other non-science engineering decisions.

## When to write replanning entries

When the `/replan` skill is used and results in plan changes, write entries to both the affected track notebook and
`project.md`. The track entry should document the technical findings (e.g., specific leaked features, nondeterminism
evidence, metric comparisons). The project entry should document the strategic decision (e.g., which tracks were killed
or restructured, and why). Include source citations (URLs + quotes) for any claims about external library behavior.

## In-flight entry editing

When working on a feature branch or open PR, treat notebook entries touched in that branch as mutable working
documents. If subsequent work in the same branch invalidates statements in an in-flight entry (e.g., a metric was
wrong, a method was replaced, the split contract changed), delete the stale statements and replace them with the final
correct ones. Do not preserve churn with "initially we thought X" hedging — the git history already records the
evolution. The entry as merged should read as a clean, accurate record of the final state.

This applies only to entries being developed in the current branch/PR. Entries that were already merged to `main` are
historical records and must not be modified (see Done Task Immutability in the orchestration AGENTS.md).

## Entry format

- Each entry starts with `### YYYY-MM-DD HH:MM UTC: Title` (date heading level 3; timestamp helps resolve merge
  conflicts when two entries are added on the same day).
- Entries within a track file are ordered by task code, earliest first.
- Every entry must begin with an `#### Executive summary` section: 2-4 sentences covering what changed, why, and the
  key outcome or metric. A reader should be able to skip the rest of the entry and still understand the decision.
- Subsequent sections typically include: problem statement, design decisions, interpretation, and next steps.
- Entries should reference generated output paths and script paths so findings are traceable.
- Do not list files changed — that is what git is for.
- **"Future:" notes** — When an agent discovers something worth revisiting later (a deferred cleanup, a tool adoption
  trigger, a feature idea), add a section to `project.md` with a heading starting with `#### Future:`. Include the
  trigger condition ("revisit when...") so the `/replan` skill can evaluate whether the condition is now met. These notes
  are seeds for future plan tasks — they should be concrete enough to act on, not vague aspirations.

## Relationship to the knowledge model

Lab notebooks are the **episodic** record — specific experiments, dates, metrics, and intermediate findings. The
knowledge model (`lyzortx/orchestration/knowledge.yml` → `lyzortx/KNOWLEDGE.md`) is the **semantic** distillation —
validated facts, dead-end lessons, and active assumptions extracted from notebooks via the `/sleeponit` skill.

- Notebooks are the source of truth for *what happened*. The knowledge model is the source of truth for *what we know*.
- When writing a notebook entry that invalidates existing knowledge (e.g., a feature that was "active" is now proven
  leaky), note the invalidation explicitly. This helps the next `/sleeponit` run update the knowledge model.
- Source references in the knowledge model (e.g., `[TG02]`) point back to notebook entries. Keep notebook entry IDs
  (task codes) stable so these references remain valid.

## Other contents

- `PLAN.md` is auto-generated from `lyzortx/orchestration/plan.yml` by `lyzortx/orchestration/render_plan.py`. Do not
  edit it by hand.
- `KNOWLEDGE.md` (at `lyzortx/KNOWLEDGE.md`) is auto-generated from `lyzortx/orchestration/knowledge.yml` by
  `lyzortx/orchestration/render_knowledge.py`. Do not edit it by hand.
- `LITERATURE.md` is a curated reading list maintained manually.
- `external_data/` contains the source registry and external dataset metadata.
- `ad_hoc_analysis_code/` contains one-off analysis scripts referenced by lab notebook entries.
- `TIER_BENCHMARK_DENOMINATOR_POLICY.md` documents denominator rules for benchmark reporting.
