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

# Agent Scratch Space

- Use `.scratch/` for temporary agent-generated files (draft commit messages, notes, and intermediate artifacts).
- Treat `.scratch/` as non-source workspace; it is ignored by git and should not contain canonical project content.

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

- The main driver for project execution is `lyzortx/research_notes/PLAN.md`.
- Follow that plan for task sequencing and update its checklist states as work progresses.
- When scope decisions are ambiguous, prefer alignment with the plan unless the user overrides it.

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

# Commit Shortcut

- If the user says exactly `commit staged`, do the following:
  1. Commit only currently staged files with `git commit`.
  2. Do not stage, unstage, or modify tracked state before committing.
  3. Generate a concise, descriptive commit message from staged diffs only.
  4. Do not mention what was excluded or not committed unless the user asks.
  5. If no files are staged, report that and stop.
