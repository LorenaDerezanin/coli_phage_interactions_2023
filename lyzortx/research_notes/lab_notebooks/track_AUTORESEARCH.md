### 2026-04-04 20:45 UTC: AUTORESEARCH track rationale — strict autoresearch readiness

#### Executive summary

Track AUTORESEARCH was added as a literal `autoresearch` preparation track, not as a vague "agentic research" bucket. The design
goal is to preserve the small-surface-area contract from `karpathy/autoresearch`: fixed `prepare.py`, single editable
`train.py`, human-owned `program.md`, fixed single-GPU wall-clock budget, and one scalar inner-loop validation metric.
The repo adaptation is that ST03 holdout labels must stay outside the search workspace entirely, so any cloud-search
winner still has to come back through a sealed replication harness before it can affect the main model line.

#### Design decisions

- **Keep the sandbox tiny.** The search surface lives under `lyzortx/autoresearch_strict/` and should behave like a
  miniature repo inside the repo.
- **Freeze data prep.** `prepare.py` exports a cached dataset from frozen deployment-paired artifacts and is not part of
  the search loop.
- **Mutate one file only.** `train.py` is the only file the search agent may edit.
- **Keep the real benchmark sealed.** The RunPod workspace gets `train` and `inner_val` only, never ST03 holdout
  labels.
- **Separate cloud spend from normal Codex CI.** RunPod provisioning belongs in a dedicated manual workflow and
  GitHub environment, not in `.github/workflows/codex-implement.yml`.

#### Immediate task sequence

1. `AR01`: build the sealed sandbox and export cache.
2. `AR02`: define the one-file baseline and guardrails.
3. `AR03`: add the dedicated RunPod workflow/environment contract.
4. `AR04`: add champion import plus sealed-holdout replication.

#### Interpretation

This is intentionally a readiness track, not yet an "overnight search succeeded" claim. The first paid RunPod search
should happen only after the dedicated workflow and environment-scoped secret path exist, because otherwise the repo
would be mixing ordinary implementation automation with spend-bearing GPU provisioning and a benchmark-visible search
loop.
