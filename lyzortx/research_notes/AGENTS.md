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

## Entry format

- Each entry starts with `### YYYY-MM-DD: Title` (date heading level 3).
- Entries within a track file are ordered by task code, earliest first.
- Every entry must begin with an `#### Executive summary` section: 2–4 sentences covering what changed, why, and the
  key outcome or metric. A reader should be able to skip the rest of the entry and still understand the decision.
- Subsequent sections typically include: problem statement, design decisions, interpretation, and next steps.
- Entries should reference generated output paths and script paths so findings are traceable.
- Do not list files changed — that is what git is for.

## Other contents

- `PLAN.md` is auto-generated from `lyzortx/orchestration/plan.yml` by `lyzortx/orchestration/render_plan.py`. Do not
  edit it by hand.
- `LITERATURE.md` is a curated reading list maintained manually.
- `external_data/` contains the source registry and external dataset metadata.
- `ad_hoc_analysis_code/` contains one-off analysis scripts referenced by lab notebook entries.
- `TIER_BENCHMARK_DENOMINATOR_POLICY.md` documents denominator rules for benchmark reporting.
