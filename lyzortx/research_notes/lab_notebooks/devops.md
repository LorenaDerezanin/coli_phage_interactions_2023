### 2026-03-24: Pre-push hook to enforce rebase on origin/main

#### Executive summary

Added a `check-rebase-on-main` pre-push hook via pre-commit that blocks `git push` when the branch is not rebased on
`origin/main`. The hook enforces two things: (1) `origin/main`'s tip is an ancestor of HEAD, and (2) no merge commits
exist between `origin/main` and HEAD (enforcing linear history). Contributors activate it once per clone with
`pre-commit install --hook-type pre-push`.

#### Design decisions

**1. Pre-commit framework rather than a standalone `.githooks/` directory.**

The repo already uses pre-commit for pre-commit stage hooks (ruff, pymarkdown, gitignore enforcement). Adding a
pre-push stage hook to the same `.pre-commit-config.yaml` keeps everything in one system. The alternative — checking
in a `.githooks/` directory and setting `core.hooksPath` — would create a parallel hook management system and conflict
with pre-commit's own `core.hooksPath` usage.

**2. Requires a separate install command: `pre-commit install --hook-type pre-push`.**

`pre-commit install` (without flags) only installs the `pre-commit` hook type. There is no single-command way to
install all hook types — each needs a separate `-t` invocation, and multiple `-t` flags in one call are not supported.
This is a pre-commit framework limitation. The install command is documented in `INSTALL.md` and `AGENTS.md`.

**3. Hook logic extracted to `scripts/check-rebase-on-main.sh`.**

The initial implementation inlined all logic as a bash one-liner in `.pre-commit-config.yaml`. This was hard to read,
test, and edit. Extracting to a standalone script referenced via `language: script` in pre-commit makes it maintainable
and directly testable (`bash scripts/check-rebase-on-main.sh`).

**4. Rejects merge commits — enforces linear history.**

The merge-base check alone passes for both `git rebase origin/main` and `git merge origin/main`. Since the policy
requires rebase (linear history), the hook additionally checks `git log --merges origin/main..HEAD` and rejects any
merge commits between origin/main and HEAD.

**5. Skips check on main branch.**

Pushing main itself (e.g., after a merge) should not be blocked. The hook exits 0 immediately when
`git rev-parse --abbrev-ref HEAD` is `main`.

**6. Fetches origin/main before checking.**

The hook runs `git fetch origin main --quiet` to ensure it checks against the latest remote state, not a stale local
ref. This adds a small network call but prevents false passes when origin/main has advanced since the last fetch.

#### PRs

- PR #193: hook implementation, AGENTS.md/INSTALL.md docs, CI workflow updates.

### 2026-03-22: CI token usage baseline — 100-run snapshot

#### Summary

First comprehensive token/cost analysis across all LLM-invoking CI workflows using the `ci-token-usage` skill.
Covers 100 most recent workflow runs (2026-03-21 to 2026-03-22).

#### Report

```
Run ID       Workflow              Date        Status     Model    Cost       PR/Issue
-----------  --------------------  ----------  ---------  -------  ---------  ------------------
23393507574  Codex Implement Task  2026-03-22  failure    gpt-5.4  $1.23      TG04
23393126080  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.36      TG03
23392903619  Codex Implement Task  2026-03-22  success    gpt-5.4  $0.96      TG02
23392501930  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.46      TG01
23392240588  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.29      TE03
23392054053  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.17      TE02
23391859832  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.51      TE01
23391685029  Codex Implement Task  2026-03-22  success    gpt-5.4  $0.66      TD03
23401448088  Codex PR Lifecycle    2026-03-22  skipped             skipped    PR#110
23401446130  Codex PR Lifecycle    2026-03-22  failure             ?          PR#111
23401099955  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#110
23400156019  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#108
23399190036  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#107
23399162621  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#107
23393491419  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#105 / Issue#104
23393109495  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#103 / Issue#102
23392886948  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#101 / Issue#100
23392484693  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#99 / Issue#98
23392445365  Codex PR Lifecycle    2026-03-22  skipped             skipped    PR#99 / Issue#98
23392445346  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392445342  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392408551  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392222905  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#97 / Issue#96
23392037136  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#95 / Issue#94
23391842693  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#93 / Issue#92
23391664643  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#91 / Issue#90
23401661800  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401564745  Claude PR Review      2026-03-22  skipped             skipped    PR#109
23401448139  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401446395  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401446310  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401433093  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401425335  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401377585  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401364528  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401099963  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401063163  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23400672332  Claude PR Review      2026-03-22  skipped             skipped    PR#109
23400156013  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23400156012  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23400099350  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23399190076  Claude PR Review      2026-03-22  skipped             skipped    PR#107
23399190041  Claude PR Review      2026-03-22  skipped             skipped    PR#107
23399143141  Claude PR Review      2026-03-22  skipped             skipped
23399138290  Claude PR Review      2026-03-22  success             $0.38      PR#107
23393461248  Claude PR Review      2026-03-22  skipped             skipped    TG03
23393459493  Claude PR Review      2026-03-22  cancelled           cancelled  TG03
23393458339  Claude PR Review      2026-03-22  success             $0.84      PR#105 / Issue#104
23393109763  Claude PR Review      2026-03-22  skipped             skipped    PR#103 / Issue#102
23393109483  Claude PR Review      2026-03-22  cancelled           cancelled  PR#103 / Issue#102
23393079767  Claude PR Review      2026-03-22  cancelled           cancelled  TG02
23393076800  Claude PR Review      2026-03-22  cancelled           cancelled  TG02
23393075508  Claude PR Review      2026-03-22  success             $0.69      PR#103 / Issue#102
23392886997  Claude PR Review      2026-03-22  skipped             skipped    PR#101 / Issue#100
23392886969  Claude PR Review      2026-03-22  cancelled           cancelled  PR#101 / Issue#100
23392886955  Claude PR Review      2026-03-22  cancelled           cancelled  PR#101 / Issue#100
23392834873  Claude PR Review      2026-03-22  cancelled           cancelled  TG01
23392831636  Claude PR Review      2026-03-22  cancelled           cancelled  TG01
23392830381  Claude PR Review      2026-03-22  success             $1.47      PR#101 / Issue#100
23392445476  Claude PR Review      2026-03-22  skipped             skipped    PR#99 / Issue#98
23392445394  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392445372  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392443487  Claude PR Review      2026-03-22  success             $0.40      PR#99 / Issue#98
23392408605  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392408604  Claude PR Review      2026-03-22  skipped             skipped    PR#99 / Issue#98
23392408584  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392367636  Claude PR Review      2026-03-22  cancelled           cancelled  TE03
23392365847  Claude PR Review      2026-03-22  cancelled           cancelled  TE03
23392364225  Claude PR Review      2026-03-22  success             $1.01      PR#99 / Issue#98
23392222717  Claude PR Review      2026-03-22  skipped             skipped    PR#97 / Issue#96
23392188368  Claude PR Review      2026-03-22  cancelled           cancelled  TE02
23392186591  Claude PR Review      2026-03-22  cancelled           cancelled  TE02
23392184746  Claude PR Review      2026-03-22  success             $0.87      PR#97 / Issue#96
23392037153  Claude PR Review      2026-03-22  skipped             skipped    PR#95 / Issue#94
23392003520  Claude PR Review      2026-03-22  cancelled           cancelled  TE01
23392001064  Claude PR Review      2026-03-22  cancelled           cancelled  TE01
23391999833  Claude PR Review      2026-03-22  success             $0.95      PR#95 / Issue#94
23391842750  Claude PR Review      2026-03-22  skipped             skipped    PR#93 / Issue#92
23391842735  Claude PR Review      2026-03-22  cancelled           cancelled  PR#93 / Issue#92
23391789152  Claude PR Review      2026-03-22  cancelled           cancelled  TD03
23391785627  Claude PR Review      2026-03-22  success             $1.15      PR#93 / Issue#92
23391466934  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.85      TD02
23390906065  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.45
23390432373  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.85      TC04
23390282751  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.96      TC03
23390122019  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.74      TC02
23376412851  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.09
23368200679  Codex Implement Task  2026-03-21  failure             no LLM     TI05
23367963808  Codex Implement Task  2026-03-21  failure    gpt-5.4  $0.83
23391593132  Codex PR Lifecycle    2026-03-21  success    gpt-5.4  $0.34      PR#91 / Issue#90
23391450667  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#87 / Issue#21
23391423648  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#87 / Issue#21
23391423647  Codex PR Lifecycle    2026-03-21  cancelled           cancelled  PR#87 / Issue#21
23391345012  Codex PR Lifecycle    2026-03-21  cancelled  gpt-5.4  $0.39      PR#87 / Issue#21
23391247403  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#86 / Issue#83
23391242099  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#87 / Issue#21
23391221579  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#86 / Issue#83
23391208905  Codex PR Lifecycle    2026-03-21  cancelled           cancelled  PR#87 / Issue#21
23391182628  Codex PR Lifecycle    2026-03-21  cancelled  gpt-5.4  $0.28      PR#86 / Issue#83
23390764607  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#85

Total estimated cost: $26.18
  Codex:   $18.42  (18 runs, avg $1.02)
  Claude:  $7.76  (9 runs, avg $0.86)

  ⚠ Codex costs are estimates (blended 30% in / 70% out rate)
```

#### Interpretation

**Cost breakdown.** Total LLM spend across 100 runs: **$26.18** (Codex $18.42, Claude $7.76). Codex costs are
estimates using a 30/70 input/output blended rate against gpt-5.4 pricing; Claude costs are exact values reported by
`anthropics/claude-code-action`.

**Codex implementation runs** average $1.02 per run (range $0.66–$1.85). All use gpt-5.4. One failure (TI05) never
reached the LLM — correctly shows `no LLM`. One failure (run 23367963808) did consume tokens ($0.83) before failing.

**Claude review runs** average $0.86 per successful review (range $0.38–$1.47). Most expensive was PR#101/Issue#100
at $1.47. Many runs show `skipped` or `cancelled` — this is expected concurrency behavior when multiple pushes
trigger the review workflow in quick succession; only the latest run executes.

**Lifecycle runs** correctly split between `no LLM` (auto-merge jobs) and Codex-detected runs (address-feedback jobs
that invoke Codex, e.g. PR#91 at $0.34). Log-based detection handles this mixed-workflow case well.

**Concurrency waste.** PR#99/Issue#98 triggered 7 Claude review runs but only 2 completed ($0.40 + $1.01). The rest
were cancelled or skipped. Similarly, 4 lifecycle runs were cancelled for PR#87/Issue#21. This is not token waste
(cancelled runs do not consume LLM resources) but does consume GitHub Actions minutes.

#### Per-ticket drill-down

Two tickets verified with `--ticket`:

- **Issue #104 (TG03):** Codex implementation $1.36 (120,957 tok, gpt-5.4) + Claude review $0.84 (20 turns) =
  **$2.20 total**. Lifecycle run correctly shows `no LLM`.
- **Issue #98 (TE03):** Codex implementation $1.29 + 2 Claude reviews ($0.40 at 17 turns + $1.01 at 35 turns) =
  **$2.71 total**. Cancelled/skipped concurrency runs correctly show zero cost.

#### Tool used

Report generated with: `python -m lyzortx.orchestration.ci_token_usage --runs 100` and `--ticket 104` / `--ticket 98`
(see `.agents/skills/ci-token-usage/` for skill documentation and design decisions).

### 2026-03-22: Per-task model selection for Codex CI and orchestrator safety fix

#### Executive summary

Added per-task LLM model selection to the Codex orchestration pipeline. Each task in `plan.yml` now has a required
`model` field (`gpt-5.4` or `gpt-5.4-mini`) that flows through the orchestrator into CI workflows. 5 complex tasks
(SHAP, feature sweep, harmonization protocol, confidence tiers, external data integration) use `gpt-5.4`; 11
straightforward tasks (stats, visualization, parameterized loops, docs) use `gpt-5.4-mini` at ~70% lower cost. Projected
savings: ~48% on implementation runs (~$8.25 saved across the remaining 16 tasks). During rollout, discovered and fixed
a latent orchestrator bug where manually closing an issue would incorrectly mark its task as done — now gated on
GitHub's `state_reason` field.

#### Problem statement

All Codex implementation runs use `gpt-5.4` regardless of task complexity. The CI token usage baseline above shows 8
implementation runs consuming 856K tokens at an average of $1.07/run. Many tasks are straightforward (bootstrap CIs, bar
charts, parameterized training loops) and don't need the full model. `gpt-5.4-mini` costs ~70% less ($0.75/1M input,
$4.50/1M output vs $2.50/$15.00) and scores 54.4% on SWE-Bench Pro with a 400K context window — sufficient for this
repo's tasks.

#### Design decisions

**1. Model field lives in plan.yml, not in the workflow or issue template.**

The user runs a single planning conversation with subscription-model Claude Opus 4.6 to assign models to all pending
tasks at once. This avoids an additional LLM call at dispatch time and keeps model assignment as a reviewable data change
in version control. The `plan.yml` file already contains implementation-adjacent fields (`implemented_in`, `baseline`),
so `model` fits the existing schema pattern.

Alternative considered: parsing model from the issue body at dispatch time (set by the orchestrator based on heuristics).
Rejected because it requires either a second LLM call per dispatch or hand-coded heuristics that would be fragile.

**2. Model is a required field — no defaults, no fallbacks.**

`Task.model` is a required positional field on the dataclass (no default value). `load_pending_tasks()` and `run_once()`
both validate that every pending task has a non-empty model before proceeding, raising `ValueError` if any are missing.
The `codex-implement.yml` workflow also fails with `::error::` if the model directive is absent from the issue body.

This was a deliberate tightening during implementation. The initial version had `model: str = ""` with validation after
construction, but the user correctly identified that a default empty string is a silent fallback that defeats the purpose.
Making it required at the type level means you cannot construct a `Task` without specifying a model, which is enforced by
Python's `TypeError` on missing arguments to frozen dataclasses.

**3. Model travels as an HTML comment in the issue body: `<!-- model: gpt-5.4-mini -->`.**

The orchestrator emits this directive when creating issues. Both `codex-implement.yml` and `codex-pr-lifecycle.yml`
extract it using `parse_model_directive.py`, a small pure-function module that reads stdin and prints the model ID.

Why an HTML comment rather than a visible field: the directive is machine-readable metadata, not something the
implementing agent needs to see or act on. HTML comments don't render in GitHub's issue view, keeping the issue body
clean for human readers.

Why a dedicated Python module rather than inline `grep -oP`: PCRE lookbehind (`grep -P`) portability is uncertain across
CI runner images. A 12-line Python script is portable, testable, and reusable from both workflows.

**4. Lifecycle (review feedback) runs use the same model as the original implementation.**

`codex-pr-lifecycle.yml` extracts the linked issue number from the PR body (`Closes #N`), fetches that issue's body, and
extracts the model directive from it. This ensures consistency: if TG04 was implemented with `gpt-5.4`, all review
feedback rounds also use `gpt-5.4`.

Alternative considered: always use the cheaper model for feedback rounds since they're typically simpler fixes. Rejected
for now — consistency is more important until we have data on whether mini handles feedback adequately.

**5. Two-PR rollout: data first, then code.**

PR #109 (merged) added the `model` field to all 16 pending tasks in `plan.yml` as a pure data change. The existing
`load_plan()` function ignores unknown YAML fields, so this was a no-op. PR #112 wires the field through the system.
This separation allows model assignments to be reviewed and adjusted independently of code changes, and ensures a clean
rollback path if the wiring has issues.

#### Model assignments

5 tasks assigned `gpt-5.4` (complex reasoning, architectural design, domain interpretation):

| Task | Rationale |
|------|-----------|
| TG04 — SHAP explanations | TreeExplainer integration (new to codebase), cross-referencing ablation + model outputs, prescriptive recommendation of which feature blocks to keep |
| TG05 — Feature-subset sweep | 10 combinatorial model runs, comparison logic, locks final v1 feature config for all downstream tracks |
| TI05 — Harmonization protocol | Multi-source schema design, domain-critical decisions cascading to TI06–TI10 |
| TI07 — Confidence tiers | Subjective tier design + weighting strategy, cascades to TI08/TI09 |
| TI08 — External data integration | Conditional injection architecture, leakage prevention, fallback handling |

11 tasks assigned `gpt-5.4-mini` (stats, visualization, parameterized loops, docs):

| Task | Rationale |
|------|-----------|
| TF01 — Bootstrap CIs | Standard NumPy resampling, dual-slice filtering already exists in codebase |
| TF02 — v0 vs v1 comparison | Side-by-side metric table, algorithmic error bucket identification |
| TH02 — Explained recommendations | Data assembly from TG02+TG04 outputs, formatting |
| TI06 — Tier B ingestion | ID cross-referencing, lookup joins, follows TI04 patterns |
| TI09 — Sequential ablations | Parameterized loop over TG01 training, metric collection |
| TI10 — Lift tracking | GroupBy aggregation + failure mode detection |
| TJ01 — One-command regeneration | Orchestration script calling existing run_track_*.py |
| TJ02 — Environment freeze | Documentation + version pinning |
| TP01 — Digital phagogram | Plotly/Matplotlib visualization |
| TP02 — Panel coverage heatmap | Standard Seaborn heatmap |
| TP03 — Feature lift bar chart | Bar chart from existing TG03 CSV |

Each task was evaluated on four axes: novelty (new pattern vs reuse), domain criticality (does a mistake cascade?),
reasoning depth (multi-step logic vs straightforward assembly), and established patterns (can it largely copy TG01/TG03
structure?).

#### Projected cost impact

Assuming average ~107K tokens per run (observed today):

- **Status quo (all gpt-5.4):** 16 × $1.07 ≈ $17.12
- **With model selection:** (5 × $1.07) + (11 × $0.32) ≈ $5.35 + $3.52 ≈ $8.87
- **Estimated savings:** ~48% on implementation runs

This does not include lifecycle runs, which scale proportionally.

#### Orchestrator safety fix: state_reason gate for issue closure

During implementation, we closed 4 pre-existing orchestrator-task issues (#22, #25, #70, #106) that predated model
selection. This exposed a latent bug: the orchestrator's `sync_status_from_issues` treated any closed issue as
"completed," which would incorrectly mark unfinished tasks as done in `plan.yml`.

GitHub's REST API provides `state_reason` on issues: `"completed"` (closed by PR merge or explicitly marked done) vs
`"not_planned"` (manually closed without completion). The fix:

- Added `state_reason` field to the `IssueRef` dataclass.
- `list_task_issues()` now parses `state_reason` from the API response.
- `sync_status_from_issues()` only marks a task "completed" when `state_reason == "completed"`. Issues closed as
  `"not_planned"` revert to their previous status (pending or blocked), preserving the task's position in the dispatch
  queue.

This is a correctness fix independent of model selection — the bug existed before but was never triggered because issues
were only closed by PR merges (which set `state_reason: "completed"`). The model-selection migration was the first time
issues were manually closed for housekeeping.

Tests added: 5 test cases covering all `state_reason` paths (completed, not_planned, open, blocked preservation,
no-issue fallback).

#### PRs

- PR #109 (merged): data-only — added `model` field to all 16 pending tasks in `plan.yml`.
- PR #112: wired model through orchestrator, workflows, and added state_reason safety gate.

### 2026-03-24: Orchestrator task invalidation — reopening issues triggers spurious runs

#### Executive summary

Reopening 13 closed orchestrator issues (TI03-TI10, TK01-TK05) to change their close reason from "completed" to "not
planned" accidentally triggered 13 Codex implementation runs. The `issues.reopened` event fires the implementation
workflow before the issue can be re-closed. The safe workaround is to change `state_reason` via the GitHub API without
reopening: `gh api repos/OWNER/REPO/issues/NUMBER -X PATCH -f state=closed -f state_reason=not_planned`.

#### Design note

The root issue is that the orchestrator uses issue state changes as its trigger. Reopening an issue is indistinguishable
from a new dispatch signal. A more robust design would trigger on merged PRs rather than closed issues — PR merges are
unambiguous completion signals and cannot be accidentally triggered by state changes on issues.

### 2026-03-24: Claude PR review approvals are nondeterministic — formal reviews silently skipped

#### Executive summary

`claude-code-action@v1`'s built-in system prompt tells the agent "You CANNOT submit formal GitHub PR reviews" and "You
CANNOT approve pull requests (for security reasons)." This directly contradicts our custom prompt that instructs Claude
to use `mcp__github__submit_pending_pull_request_review`. The result is nondeterministic: sometimes the agent follows
our instructions and submits a formal review (PRs #217, #220, #223), sometimes it follows the built-in restriction and
writes its entire review into the sticky issue comment instead (PR #225). When this happens, the `claude-pr-review.yml`
dispatch step sees no formal review and takes no action — no auto-merge, no Codex lifecycle dispatch.

#### Root cause analysis

The conflicting instructions are in `anthropics/claude-code-action` source file `src/create-prompt/index.ts`
(the `generateDefaultPrompt()` function, "CAPABILITIES AND LIMITATIONS" section). They have been present since the
action's initial commit on 2025-05-19 and have never been modified. The action provides the MCP review tools in its
`allowedTools` list, but the prompt-level "CANNOT" instruction causes the model to sometimes refuse to use them.

Evidence from PR #225 logs:
- `permission_denials_count: 3` — the agent attempted to use the review tools and was denied.
- `No buffered inline comments` — the `classify_inline_comments` feature captured nothing.
- `Unexpected review state: . No action taken.` — the dispatch step found zero formal reviews.

The contradiction is a prompt-level issue, not a tool-permission issue. The MCP tools are available and allowed; the
model simply refuses to call them when it prioritizes the "CANNOT" instruction over the custom prompt.

#### Impact on downstream workflows

When Claude fails to submit a formal review:
1. `claude-pr-review.yml` dispatch step sees empty `LATEST_REVIEW` → takes the `else` branch → no auto-merge, no Codex
   dispatch.
2. `codex-pr-lifecycle.yml` (if manually triggered) queries `review_threads.py` which only reads formal review threads
   → finds zero unresolved threads → labels PR `ready-for-human-review` and posts "Review passed with no issues" even
   though actionable feedback exists in the issue comment.

#### Workarounds investigated

- `USE_SIMPLE_PROMPT=true` env var: switches to a simplified prompt that omits the CANNOT section. Stays in tag mode.
  Risk: the simplified prompt may lose other useful scaffolding.
- Stronger override language in custom prompt: fragile — sometimes works, sometimes doesn't, as PR #225 demonstrated.
- Post-condition guard in the workflow: check whether a formal review was submitted after the action completes; if not,
  fail loudly or submit a synthetic review via `gh api`. This is a band-aid.

#### Long-term fix: LangGraph review orchestrator (PR #48)

The correct fix is to wrap the review agent in a LangGraph orchestrator that can verify tool calls were actually made
and loop the agent back with feedback if the formal review submission is missing. This moves the reliability guarantee
from prompt-level persuasion (unreliable) to programmatic verification (deterministic). PR #48 contains the
implementation plan.

#### Immediate actions taken

- Updated `/gh` skill (section 10) to document that PR feedback can appear in either review threads or issue comments,
  and that both must be checked.
- PR #225 was merged manually after human review confirmed Claude's comment-based review was accurate.

#### Future: migrate orchestrator trigger from issues to PRs

Revisit when: the current trigger model causes more incidents, or the orchestrator is being refactored for other
reasons. The change would involve updating `codex-implement.yml` to trigger on `pull_request.closed` (with
`merged == true`) instead of `issues.reopened`/`issues.opened`, and updating `sync_status_from_issues` to read from PR
merge state instead of issue close state.

#### Future: ticket dependency support in plan.yml with GitHub issue relationships

Revisit when: the orchestrator is being refactored or task scheduling becomes more complex.

Currently `plan.yml` tasks have a flat `status` field (pending, blocked, done) with no structured dependency information.
Task ordering is implicit — humans and agents infer dependencies from track structure and naming conventions (e.g., TI05
depends on TI04 because it follows in the track). This is fragile and prevents the orchestrator from automatically
determining which tasks are unblocked when a predecessor completes.

The improvement would add a `depends_on` field to each task in `plan.yml` listing predecessor task IDs. The orchestrator
would use this to:
1. Automatically set tasks to `blocked` when predecessors are not yet `done`.
2. Unblock tasks when all predecessors complete.
3. Reflect dependencies as GitHub issue relationships (sub-issues or "blocked by" links) so the dependency graph is
   visible in the GitHub UI, not just in the YAML file.

This would eliminate manual `blocked`/`pending` status management and make the dispatch order deterministic based on the
dependency graph rather than YAML ordering.
