---
name: sleeponit
description: >
  Consolidates lab notebook knowledge into a unified, thematically organized knowledge model. Transforms episodic
  experimental records into semantic project knowledge, inspired by sleep-based memory consolidation. Triggers when the
  user says "sleep on it", "consolidate knowledge", "compact the notebooks", "update the knowledge model", "what do we
  know", or wants to distill accumulated findings into a reusable context artifact. Also triggers when starting a new
  project phase and wanting to capture what's been learned so far.
user-invocable: true
argument-hint: "[--incremental] — update existing knowledge model instead of rebuilding"
---

# Sleep On It: Knowledge Consolidation

This skill guides a structured consolidation of lab notebook knowledge into `lyzortx/orchestration/knowledge.yml`,
which renders to `lyzortx/KNOWLEDGE.md` and is loaded into Claude's context for all lyzortx work.

The metaphor is deliberate: like sleep-based memory consolidation, this process transforms episodic records (specific
experiments, dates, intermediate steps) into semantic knowledge (validated facts, dead ends, active assumptions).

## Phase 1: Extract & Organize

Read all notebooks in `lyzortx/research_notes/lab_notebooks/`. For each notebook entry, extract knowledge that is:

- **Worth remembering**: findings, validated methods, calibration results, performance bounds
- **Not recoverable from code**: the *why* behind decisions, dead-end lessons, caveats
- **Still relevant**: not superseded by later work

Discard:
- Implementation details recoverable from the code itself
- Git history (who changed what, when)
- Intermediate debugging steps
- Exact file paths (they change)

### Incremental mode

If `lyzortx/orchestration/knowledge.yml` already exists:

1. Load it — see API reference below for correct calling convention
2. Read notebooks looking for entries newer than `last_consolidated` date or entries not yet reflected in the model
3. Propose additions, updates (refined statements), and deletions (superseded findings — delete, don't mark)
4. Use `knowledge_parser.diff_knowledge()` to generate a clear change report

### Organizing into themes

Themes emerge from the content — do not force per-track structure. A finding from Track L and Track E that both
relate to features belong together under a features theme.

Suggested starting themes (adapt based on content):
- Data & Labels — labeling policy, data quality, confidence tiers
- Features — what works, what doesn't, leakage risks
- Model — architecture, calibration, performance bounds
- Evaluation — holdout protocol, benchmark methodology
- Infrastructure — operational knowledge not in code
- Dead Ends — compressed lessons from things that didn't work
- Open Questions — unresolved items that still matter

### Knowledge unit structure

Each unit in the YAML has:

```yaml
- id: short-kebab-case-id          # unique, stable across updates
  statement: >                      # one clear sentence — the knowledge itself
    Isotonic calibration outperforms Platt scaling for this dataset's
    non-Gaussian probability distribution.
  sources: [TG02, TG05]            # notebook entry references (enough to grep)
  status: active                    # active | dead-end (superseded units are deleted, not marked)
  confidence: validated             # validated | preliminary (optional)
  context: >                        # caveats, conditions (optional)
    Only tested with LightGBM; may not hold for other architectures.
  relates_to: [prob-distribution]   # cross-refs to other unit IDs (optional)
```

## Phase 2: Curate & Approve

1. Delete any existing `.scratch/sleeponit_draft.yml` first (avoids Write-tool Read-before-write errors on stale
   drafts): `rm -f .scratch/sleeponit_draft.yml`
2. Write draft YAML to `.scratch/sleeponit_draft.yml`
3. Validate and diff — see API reference below for correct calling convention
4. Present a **concise summary** to the user:
   - Unit count change (e.g., "43 -> 39 units")
   - Deleted units: list IDs with one-line reason each
   - Added units: list IDs with one-line description each
   - Refined units: list IDs with what changed
   - Flag any units you're uncertain about
5. Wait for user approval. If the summary is clear, a simple "y" is enough. Only do a multi-step
   interactive feedback session if changes are unwieldy for a summary or if important info would be
   lost in the summary.
6. On approval, proceed to Phase 3. On feedback, adjust the draft and re-summarize.

## Phase 3: Emit

1. Copy validated YAML to `lyzortx/orchestration/knowledge.yml`
2. Run `python -m lyzortx.orchestration.render_knowledge` to produce `lyzortx/KNOWLEDGE.md`
3. If `lyzortx/CLAUDE.md` does not already contain `@KNOWLEDGE.md`, add it as a second line

## What Makes a Good Knowledge Unit

**Good** (semantic, transferable):
> Isotonic calibration outperforms Platt scaling for our data's non-Gaussian distribution.

**Bad** (episodic, implementation detail):
> On 2026-03-15, we ran TG02 and found that isotonic calibration gave AUC 0.847 vs Platt's 0.831 using
> the script at lyzortx/pipeline/calibration.py with seed=42.

The good version captures the *knowledge*. The bad version captures the *event*. The event is in the notebook
and git history; the knowledge model stores only the distilled insight.

## Quality Checks

Before finalizing, verify:
- [ ] No per-track siloing — themes are conceptual, not organizational
- [ ] Each statement is one clear sentence (not a paragraph)
- [ ] Sources are traceable (grep for the ID in lab notebooks)
- [ ] `relates_to` cross-references point to valid IDs
- [ ] Dead ends include the *lesson*, not just "it didn't work"
- [ ] No implementation details that belong in code comments
- [ ] `validate_knowledge()` returns no errors

## knowledge_parser API reference

The parser uses frozen dataclasses, not dicts. Do not call `.get()` or use `['key']` on model objects.

```python
from pathlib import Path
from lyzortx.orchestration.knowledge_parser import load_knowledge, validate_knowledge, diff_knowledge

# load_knowledge takes a Path, not a string
km = load_knowledge(Path("lyzortx/orchestration/knowledge.yml"))

# KnowledgeModel attributes (NOT dict-style access):
km.last_consolidated   # str
km.themes              # list[KnowledgeTheme]
km.all_units()         # list[KnowledgeUnit]
km.all_unit_ids()      # set[str]

# KnowledgeTheme attributes:
theme.key              # str (e.g., "data-labels")
theme.title            # str (e.g., "Data & Labels")
theme.units            # list[KnowledgeUnit]

# KnowledgeUnit attributes:
unit.id                # str
unit.statement         # str
unit.sources           # list[str]
unit.status            # str ("active" or "dead-end")
unit.confidence        # str | None
unit.context           # str | None
unit.relates_to        # list[str]

# Validate returns a list of error strings (empty = valid)
errors = validate_knowledge(km)

# Diff compares two KnowledgeModel objects, returns a formatted string report
report = diff_knowledge(old_km, new_km)
```
