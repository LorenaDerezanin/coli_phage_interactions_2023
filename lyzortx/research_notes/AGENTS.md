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

## Entry format

- Each entry starts with `### YYYY-MM-DD: Title` (date heading level 3).
- Entries within a track file are ordered by task code, earliest first.
- Sections typically include: what was implemented, output summary, interpretation, and next steps.
- Entries should reference generated output paths and script paths so findings are traceable.

## Other contents

- `PLAN.md` is auto-generated from `lyzortx/orchestration/plan.yml` by `lyzortx/orchestration/render_plan.py`. Do not
  edit it by hand.
- `LITERATURE.md` is a curated reading list maintained manually.
- `external_data/` contains the source registry and external dataset metadata.
- `ad_hoc_analysis_code/` contains one-off analysis scripts referenced by lab notebook entries.
- `TIER_BENCHMARK_DENOMINATOR_POLICY.md` documents denominator rules for benchmark reporting.
