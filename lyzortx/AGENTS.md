# Lyzor Tx Mission

- Build the best possible phage-lysis prediction pipeline for _E. coli_ using only in-silico methods.
- Primary local data source: `data/interactions/raw/raw_interactions.csv` and related repo metadata/features.
- No wet-lab access is assumed for this project.
- External data and literature can be added when they improve model quality or rigor.

# Operating Rules

- Keep methods reproducible and auditable (deterministic where possible).
- Use two KPI tiers:
  - **Tier 1 (Current Panel, Feasible):** evaluation with current 96-phage panel and current interaction matrix.
  - **Tier 2 (North-Star):** aspirational targets that may require panel expansion and external data.

# Steel Thread v0 Go / No-Go Gates

These are validation checkpoints, not dispatchable tasks. All must hold before expanding beyond v0:

- `run_steel_thread_v0.py` must complete without error on a fresh clone with only `phage_env` dependencies.
- No leakage violations detected by the v0 regression checks (ST0.1 through ST0.7).
- v0 model must materially outperform a naive baseline on the same split.

# Knowledge Model

- `lyzortx/KNOWLEDGE.md` is auto-generated from `lyzortx/orchestration/knowledge.yml` by
  `lyzortx/orchestration/render_knowledge.py`. Do not edit `KNOWLEDGE.md` by hand.
- The knowledge model contains consolidated project knowledge distilled from lab notebook entries. It is organized
  thematically (not per-track) and loaded into context for all lyzortx work.
- Consult the knowledge model before starting work on a track — it captures validated findings, dead ends, and active
  assumptions that may not be obvious from the code alone.
- When your work invalidates or extends existing knowledge (e.g., a finding changes status from "active" to
  "superseded"), note it in your lab notebook entry so the next `/sleeponit` run can update the model.
- After merging branches that both modified `knowledge.yml`, re-run `render_knowledge.py` to regenerate `KNOWLEDGE.md`.

# Paper Availability

- The research paper is in the local `paper/` directory (gitignored, paywalled — do not redistribute).
- A detailed gist is at `lyzortx/research_notes/GIST Prediction Ecoli nature paper.md`.

# Generated Outputs

- Store generated outputs under `lyzortx/generated_outputs/`, organized by analysis name.
- Do not write new generated artifacts to top-level directories like `figures/` for `lyzortx` analyses.
- This directory is gitignored. **Never commit generated outputs** — do not use `git add -f`.

# Fail-Fast on Missing Data

- Pipeline code must raise exceptions on missing inputs. Never silently fall back to empty DataFrames or zero rows.
- `FileNotFoundError` for missing files, `ValueError` for unexpected empty joins.
- A task producing zero deltas on empty inputs has **failed**, not succeeded.
- Tests that pass on empty fixtures test nothing.

# Coding Principles

- **Test data quality** — prefer real data (or programmatically generated realistic data) over hand-crafted dummy values.
- **No magic numbers or inline string literals** — define named constants for repeated or meaningful values.
- **Progress feedback** — long-running operations must log "starting" and "completed" messages with elapsed time.
  Loops over many items should log periodic progress (e.g., every N items) so silence never exceeds ~30 seconds.
- **Timestamped logging** — use Python's `logging` module with timestamps, not bare `print()`. Use
  `lyzortx.log_config.setup_logging()` in track runners.
- **Timezone-aware timestamps** — use `datetime.now(timezone.utc)`, never bare `datetime.now()` or `datetime.utcnow()`.
- **Top-level imports** — no lazy imports unless there is a documented circular-import reason.
- **Performance from the start** — vectorized numpy/pandas over Python loops, parallelize embarrassingly parallel work,
  batch operations over per-element calls.
- **Optimize the repeated hotspot first** — remove duplicated expensive work before adding clever machinery.
- **Keep performance fixes readable** — named helpers, clear constants, phase-level logging over opaque one-liners.

# One-Off Analyses

- Store one-off analysis scripts under `lyzortx/research_notes/ad_hoc_analysis_code/`.

# Function Design and Testing

- Prefer pure functions. Keep side effects at module boundaries.
- For new/changed logic, add concise unit tests covering core behavior and critical edge cases.
- When fixing a bug, write a failing test first (TDD-style).
- Place tests under `lyzortx/tests/`.
- Never write tests for functions defined only in the test file — tests must exercise production code.
- Keep CI unit-test workflows enabled and green.

# External Service Integration

- First explore APIs manually (e.g., `gh api`, `curl`) to understand real data shapes before writing code.
- Extract core logic into pure functions under `lyzortx/` and write unit tests.
- **Never read large API responses into the conversation** — write to file, check size, then inspect selectively.
- **Prove it works before updating tests.** When fixing an integration bug (e.g., API payload format), run the real
  call first to confirm the fix. Only update unit tests after the live call succeeds. Updating tests to match an
  untested hypothesis just moves the bug into the test suite.

# Claims About External Libraries

- Back any claim about library behavior with a link to official docs and a direct quote.
- Include URL and quote in lab notebook entries so future readers can verify.

# Scientific Review Standards

When reviewing analysis, modeling, or feature-engineering PRs, review the scientific substance — not just code
correctness.

## Substance over plumbing

Before approving or marking done: did this task produce real results, or just scaffolding?

- Zero rows, zero deltas, or "pending" placeholders means acceptance criteria are not met.
- If real results cannot be produced (e.g., data doesn't exist in CI), fail — no PR. Leave the issue open.

## Biology and domain sense

- Distinguish **plausible screening signals** from **confirmed mechanisms**. Do not turn correlations into mechanistic
  claims.
- RBP-receptor/LPS associations may be plausible screening candidates; anti-defense-defense subtype associations are
  vulnerable to lineage confounding.

## Statistical review

- Check that the statistical test matches the data-generating structure, not just the table shape.
- Verify comparison groups match the biological question. Check multiple-testing correction scope.
- Check p-value resolution and sample size — results pinned at numeric floors require caveats.
- Separate screening-valid statistics from publication-grade inference.

## Reality checks

- Verify headline counts against raw data (`data/interactions/raw/raw_interactions.csv`).
- Spot-check top hits. Confirm generated outputs were regenerated from current code.
- Do not treat "significant association" counts as independent discoveries — account for correlated features and
  duplicate profiles.

## Write-up standards

- Require notebook/PR text to state what the test controls for, what it doesn't, what was excluded, and whether results
  are exploratory or validated.
- If the implementation changes the analysis population or denominators, update the text in the same PR.

## Feature engineering and evaluation integrity

- **Train/test boundary applies to feature construction.** Any statistic derived from outcomes that becomes a feature
  must be computed on training data only.
- **Flawed pipeline results are not evidence.** Do not cite metrics from leaked/flawed pipelines.
- **Deployability means "derivable from inputs."** Separate architectural limitations from implementation gaps.
- **Small holdouts (<~100 units) require bootstrap confidence intervals.** Unquantified point estimates are noise.
- **Acceptance criteria define goals, not solutions.** Treat named tools as suggestions, not allowlists.
