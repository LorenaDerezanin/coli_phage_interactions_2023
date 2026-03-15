# Documentation Style Rules

- Markdown style relies on tool settings in `.pre-commit-config.yaml` and `.markdownlint.yaml`.
- The line-length policy is 120 characters for prose, with configured exceptions for code blocks, tables, and headings.
- Use `pre-commit run prettier --all-files` for bulk auto-fixes.
- Use `pre-commit run markdownlint --all-files` for optional manual lint checks.
- After auto-fixes, stage updated files explicitly before committing.

# Environment Policy

- Use the `phage_env` micromamba environment for project commands by default.
- Before running project tooling, activate it with `micromamba activate phage_env`.
- Do not use system Python for repository tasks unless the user explicitly asks.

# Dependency Selection Policy

- Do not choose dependency-light implementations solely to keep CI minimal.
- When a library-based approach is expected to improve model quality, scientific validity, or maintainability, prefer
  the library-based approach.
- If adding dependencies is needed for the technically better implementation, update CI/environment setup accordingly
  rather than degrading the implementation.
- Keep `requirements.txt` alphabetically sorted by package name.

# AGENTS.md and CLAUDE.md Pairing

- Whenever you create an `AGENTS.md` in a directory, also create an accompanying `CLAUDE.md` in the same directory that
  imports it with `@AGENTS.md`.

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

# PR and Issue Linkage Policy

- Any PR that addresses a tracked GitHub issue must include `Closes #<issue_number>` in its description.
- Use one `Closes #...` line per issue when a PR intentionally resolves multiple issues.
- Keep closure references explicit so orchestration and audit flows can advance automatically on merge.

# PR Review Policy

- Do not approve a PR unless all CI checks pass. A branch with failing tests is not mergeable regardless of code quality.
- Review must verify: code quality, test coverage, alignment with acceptance criteria, and adherence to AGENTS.md policies.

# PR Creation for Orchestrator Tasks

- When implementing an orchestrator task, create the PR using `gh pr create` from the CLI.
- PR title pattern: `[ORCH][TASK_ID] Brief description`.
- PR body MUST include `Closes #<issue_number>` (the orchestrator issue that dispatched the task).
- Add the `orchestrator-task` label: `gh pr create --label orchestrator-task`.

# Requirement Challenge Policy

- For any non-trivial user request, first question the requirement before implementing.
- If the request is unreasonable, overcomplicated, or lower-value than a simpler option, push back clearly.
- Suggest deletion or simplification when that is the better technical path.
- Only comply directly when the requirement is technically reasonable after this check.
- Apply this policy to all plan decisions, including whether to keep, change, or remove planned work.

# One-Off Analyses

- If a one-off analysis is referenced in research notes, store the script under
  `lyzortx/research_notes/ad_hoc_analysis_code/`.

# Generated Outputs

- Store generated analysis outputs (CSVs, figures, reports) under `lyzortx/generated_outputs/`.
- Organize outputs by analysis name, for example `lyzortx/generated_outputs/raw_interactions_summary/`.
- Do not write new generated artifacts to top-level directories like `figures/` for `lyzortx` analyses.

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
- Keep CI unit-test workflows enabled and green; do not merge changes that silently bypass tests.

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
