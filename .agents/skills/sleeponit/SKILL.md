---
name: sleeponit
description: >
  Consolidate lab notebook knowledge into a unified, thematically organized knowledge model. Inspired by sleep-based
  memory consolidation — transforms episodic experimental records into semantic project knowledge. Use this skill when
  the user says "sleep on it", "consolidate knowledge", "compact the notebooks", "update the knowledge model",
  "what do we know", or wants to distill accumulated findings into a reusable context artifact. Also use when starting
  a new project phase and wanting to capture what's been learned so far.
user-invocable: true
argument-hint: "[--incremental] — update existing knowledge model instead of rebuilding"
---

# Sleep On It: Knowledge Consolidation

This skill guides a structured consolidation of lab notebook knowledge into `lyzortx/orchestration/knowledge.yml`,
which renders to `lyzortx/KNOWLEDGE.md` and is loaded into Claude's context for all lyzortx work.

The metaphor is deliberate: like sleep-based memory consolidation, this process transforms episodic records (specific
experiments, dates, intermediate steps) into semantic knowledge (validated facts, dead ends, active assumptions).

## Architecture

```
lab notebooks (episodic)  →  /sleeponit (consolidation)  →  knowledge.yml (semantic)  →  KNOWLEDGE.md (context)
```

- **Source of truth**: `lyzortx/orchestration/knowledge.yml` (YAML, machine-manipulable)
- **Rendered output**: `lyzortx/KNOWLEDGE.md` (markdown, loaded into Claude context via `lyzortx/CLAUDE.md`)
- **Parser/validator**: `lyzortx/orchestration/knowledge_parser.py`
- **Renderer**: `lyzortx/orchestration/render_knowledge.py`

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

1. Load it with `knowledge_parser.load_knowledge()`
2. Read notebooks looking for entries newer than `last_consolidated` date or entries not yet reflected in the model
3. Propose additions, updates (changed status, refined statements), and removals (superseded findings)
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
  status: active                    # active | superseded | dead-end
  confidence: validated             # validated | preliminary (optional)
  context: >                        # caveats, conditions (optional)
    Only tested with LightGBM; may not hold for other architectures.
  relates_to: [prob-distribution]   # cross-refs to other unit IDs (optional)
```

## Phase 2: Curate

This is the critical human-in-the-loop step. The user must review and approve the knowledge model.

1. Write draft YAML to `.scratch/sleeponit_draft.yml`
2. Run validation: `knowledge_parser.validate_knowledge()` — fix any errors
3. Present a summary to the user:
   - Total units by theme and status
   - If incremental: the diff (what's new, changed, removed)
   - Any units you're uncertain about — flag them explicitly
4. Tell the user: "Review and edit `.scratch/sleeponit_draft.yml`, then tell me when ready."
5. After user confirms, re-validate the edited draft

## Phase 3: Emit

1. Copy validated YAML to `lyzortx/orchestration/knowledge.yml`
2. Run `python -m lyzortx.orchestration.render_knowledge` to produce `lyzortx/KNOWLEDGE.md`
3. If `lyzortx/CLAUDE.md` does not already contain `@KNOWLEDGE.md`, add it as a second line
4. Report:
   - Total knowledge units by status
   - If incremental: the diff summary
   - Remind user to commit both `knowledge.yml` and `KNOWLEDGE.md`

## Merge Conflict Guidance

When merging branches that both modified `knowledge.yml`:
- Resolve the YAML conflict (usually: keep both sets of additions)
- Update `last_consolidated` to the later date
- Re-run `python -m lyzortx.orchestration.render_knowledge` to regenerate `KNOWLEDGE.md`
- The rendered markdown is always regenerable — never hand-edit it

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
