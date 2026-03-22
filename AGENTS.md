# Documentation Style Rules

- Markdown style relies on tool settings in `.pre-commit-config.yaml` and `.pymarkdown.yaml`.
- The line-length policy is 120 characters for prose, with configured exceptions for code blocks, tables, and headings.
- Use `pymarkdown --config .pymarkdown.yaml fix -r .` for optional Markdown auto-fixes.
- Use `pre-commit run pymarkdown --all-files` for optional manual lint checks.
- After auto-fixes, stage updated files explicitly before committing.

# Knowledge Persistence Policy

- Almost all learnings, rules, corrections, and project context must be written to `AGENTS.md` files (root or
  subfolder), **not** to user memory. `AGENTS.md` is shared, version-controlled, and benefits every agent and
  contributor. User memory is private and ephemeral.
- Reserve user memory exclusively for truly personal information that does not belong in a shared repo file (e.g., the
  user's role, private account details, or confidential partner names).
- When in doubt, default to `AGENTS.md`. The bar for writing to user memory instead should be very high — roughly 0.1%
  of cases.

# Environment Policy

- **Local development:** Use the `phage_env` micromamba environment. Activate it with `micromamba activate phage_env`.
- **CI / Codex sandbox:** Use plain `python` and `pip` directly — micromamba is not installed in CI. Dependencies are
  pre-installed via `pip install -r requirements.txt` before the agent runs.
- **How to detect CI:** Check for the `CI` environment variable (`[ -n "$CI" ]`). If set, skip micromamba activation.
- **Git identity in CI:** Git `user.name` and `user.email` are pre-configured before the agent runs. Do not attempt to
  set them yourself.

# Dependency Selection Policy

- Do not choose dependency-light implementations solely to keep CI minimal.
- When a library-based approach is expected to improve model quality, scientific validity, or maintainability, prefer
  the library-based approach.
- If adding dependencies is needed for the technically better implementation, update CI/environment setup accordingly
  rather than degrading the implementation.
- Keep `requirements.txt` alphabetically sorted by package name.
- Every new dependency must be pinned to an exact version in `requirements.txt` (e.g., `ruff==0.11.6`, not
  `ruff>=0.11.6`). Use the newest stable version available at the time of adding.
- Never install packages with bare `pip install <package>`. Always add the pinned version to `requirements.txt` first,
  then install via `pip install -r requirements.txt`.

# AGENTS.md and CLAUDE.md Pairing

- Whenever you create an `AGENTS.md` in a directory, also create an accompanying `CLAUDE.md` in the same directory that
  imports it with `@AGENTS.md`. This ensures cross-compatibility between Claude, Cursor, and Codex.

# Agent Scratch Space

- Use `.scratch/` for temporary agent-generated files (draft commit messages, notes, intermediate artifacts, and diff
  comparisons).
- Treat `.scratch/` as non-source workspace; it is ignored by git and should not contain canonical project content.
- Prefer `.scratch/` over `/tmp/` for temporary files so sandbox permissions are not needed.

# Paper Availability

- The research paper is expected in the local `paper/` directory.
- `paper/` is listed in `.gitignore` because the paper is paywalled and should not be redistributed in git history.
- Contributors should obtain the paper themselves for their local copy of the repository.
- A detailed gist of the paper is available at `lyzortx/research_notes/GIST Prediction Ecoli nature paper.md`.

# Code Placement Policy

- All new code should be added under `lyzortx/`.
- Why: this repository is a fork, and the new goal is an alternative pipeline based on the paper's raw inputs.
- Why: keeping new work in `lyzortx/` prevents mixing with original upstream sources and keeps provenance clear.
- Why: separation reduces accidental regressions in upstream code and makes reproduction-vs-new-pipeline comparisons
  easier to audit.
- Do not add new pipeline code at repository root or in original upstream directories by default.
- Modify code outside `lyzortx/` only when intentionally fixing a bug in the original repository code.

# Plan-Driven Execution

- The main driver for project execution is `lyzortx/orchestration/plan.yml`, rendered to
  `lyzortx/research_notes/PLAN.md`.
- Follow that plan for task sequencing; the orchestrator updates checklist states automatically.
- When scope decisions are ambiguous, prefer alignment with the plan unless the user overrides it.

# Branch Protection

- Never push directly to main. All changes go through pull requests.
- Create a feature branch, push it, and open a PR.
- Always rebase on main before starting work — whether implementing a new task or addressing review feedback.
- Always rebase on main before pushing. For plain git: `git fetch origin main && git rebase origin/main`. For Graphite
  stacks: `gt stack restack` to rebase the entire stack on its trunk.

# PR and Issue Linkage Policy

- Any PR that addresses a tracked GitHub issue must include `Closes #<issue_number>` in its description.
- Use one `Closes #...` line per issue when a PR intentionally resolves multiple issues.
- Keep closure references explicit so orchestration and audit flows can advance automatically on merge.

# Codex Review Connector

- The repo owner has a `chatgpt-codex-connector` GitHub App installed that automatically reviews every pull request.
- Codex reviews always use `state: "COMMENTED"` — never `APPROVED` or `CHANGES_REQUESTED"`.
- When Codex has suggestions, it posts inline comments on specific lines. When it finds no issues, the review body
  contains only boilerplate (e.g., "Didn't find any major issues") with no inline comments.
- Re-reviews can be requested by commenting `@codex review` on the PR.

# Review guidelines

- Do not approve a PR unless all CI checks pass. A branch with failing tests is not mergeable regardless of code
  quality.
- Review must verify: code quality, test coverage, alignment with acceptance criteria, and adherence to AGENTS.md
  policies.
- When addressing review feedback, apply the Requirement Challenge Policy: push back on comments that are wrong,
  overcomplicated, or low-value rather than blindly implementing every suggestion.
- Before raising an issue, check existing review threads and replies on the PR. Do not re-raise concerns that have
  already been addressed with a code fix or explicitly pushed back on with a reasoned explanation.

## Review focus areas

1. **Correctness** — bugs, logic errors, off-by-one, wrong variable usage.
2. **Test coverage** — are new/changed functions tested? Are critical edge cases covered?
3. **Security** — no secrets committed, no injection risks.
4. **AGENTS.md compliance** — verify the PR follows all policies in this file (code placement, dependency pinning,
   generated outputs, git staging, etc.).
5. **Clarity** — naming, structure, readability.

Do NOT nitpick style — ruff handles formatting. Focus on substantive issues only. Do not invent problems.

# PR Creation for Orchestrator Tasks

- When implementing an orchestrator task, create the PR using `gh pr create` from the CLI.
- PR title pattern: `[ORCH][TASK_ID] Brief description`.
- PR body MUST include `Closes #<issue_number>` (the orchestrator issue that dispatched the task).
- Add the `orchestrator-task` label: `gh pr create --label orchestrator-task`.

# Agent Transparency

- When an agent posts comments on GitHub (PR reviews, issue comments, etc.) using a human's credentials or a shared PAT,
  it must identify itself as an agent, including the model name and version (e.g., "Posted by Claude Opus 4.6" or
  "Posted by Codex gpt-5.4").

# Requirement Challenge Policy

- For any non-trivial user request, first question the requirement before implementing.
- If the request is unreasonable, overcomplicated, or lower-value than a simpler option, push back clearly.
- Suggest deletion or simplification when that is the better technical path.
- Only comply directly when the requirement is technically reasonable after this check.
- Apply this policy to all plan decisions, including whether to keep, change, or remove planned work.
- This policy applies equally to PR review feedback: do not blindly address every comment. Push back on feedback that is
  wrong, overcomplicated, or low-value. Explain why in the reply.

# One-Off Analyses

- If a one-off analysis is referenced in research notes, store the script under
  `lyzortx/research_notes/ad_hoc_analysis_code/`.

# Generated Outputs

- Store generated analysis outputs (CSVs, figures, reports) under `lyzortx/generated_outputs/`.
- Organize outputs by analysis name, for example `lyzortx/generated_outputs/raw_interactions_summary/`.
- Do not write new generated artifacts to top-level directories like `figures/` for `lyzortx` analyses.
- `lyzortx/generated_outputs/` is listed in `.gitignore`. **Never commit generated outputs to git.** They are
  reproducible from the analysis scripts and should not bloat the repository. If `git add` refuses to stage a file
  because it is gitignored, that refusal is correct — do not override it with `git add -f` or `git add --force`.

# Dead Code Policy

- Err heavily on the side of deleting unused code. All callers are within this repo — there are no external consumers.
- If a function, class, or import has no callers, delete it immediately. Do not keep it "just in case."

# Function Design and Testing Policy

- Prefer pure functions as the default design for new logic where practical.
- Keep side effects (I/O, network, subprocesses, global state mutation) at module boundaries.
- For new or changed pure logic, add concise unit tests that cover the most important functionality pragmatically.
- Do not over-cement with tests when there is a lot of code in flux — cover core behavior and critical edge cases, not
  every line.
- When fixing a bug, write a failing test first that proves the regression, then implement the fix to make it pass
  (TDD-style).
- Place tests under `lyzortx/tests/` unless the user explicitly requests a different location.
- **Never write tests for functions defined only in the test file itself.** Tests must exercise real production code. If
  there is no production function to test, either the logic belongs in a helper under `lyzortx/` or it does not need a
  test.
- Keep CI unit-test workflows enabled and green; do not merge changes that silently bypass tests.

# External Service Integration Development

- When writing code that interacts with an external service (GitHub API, GitHub Actions, CI systems, public databases
  like NCBI/UniProt/PDB, etc.), first explore and test the integration manually against the live service (e.g.,
  `gh api`, `curl`, CLI tools) to understand the real data shapes and edge cases.
- Based on that learning, extract core logic into pure, reusable functions under `lyzortx/` and write unit tests for
  those functions.
- Do not skip the manual exploration step and guess at API behavior. Do not write unit tests that mock everything
  without first verifying assumptions against the real service.

# Git Staging Policy

- Never use `git add -f` or `git add --force`. If git refuses to stage a file, it is gitignored for a reason.
- Stage files by explicit path (`git add <file> ...`). Do not use `git add .` or `git add -A` in CI, as these can
  accidentally stage untracked artifacts, build outputs, or temporary files present in the working tree.

# Commit Granularity

- One kind of change per commit. Do not mix unrelated changes (e.g., feature + policy update + dead code cleanup).
- If multiple changes are in flight, commit them separately with focused messages.

# Commit Shortcut

- If the user says exactly `commit staged`, do the following:
  1. Commit only currently staged files with `git commit`.
  2. Do not stage, unstage, or modify tracked state before committing.
  3. Generate a concise, descriptive commit message from staged diffs only.
  4. Do not mention what was excluded or not committed unless the user asks.
  5. If no files are staged, report that and stop.
