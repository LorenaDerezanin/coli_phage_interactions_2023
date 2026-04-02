# Deployment-Paired Feature Pipeline

This directory contains the DEPLOY track implementation — re-deriving all host features from raw genome assemblies
using the same pipeline that runs at inference time.

## Assembly Download Parallelism

The Picard collection assemblies (~1.9GB zip from figshare) take ~7 minutes to download. When starting a task that
needs assemblies:

1. **Kick off the download in the background first** — call the download function or run the download script
   asynchronously before doing anything else.
2. **Do prep work in parallel** — read source code, plan the implementation, set up output directories, load non-assembly
   dependencies, write tests — while the download runs.
3. **Only block on the download when you actually need a FASTA file** — typically right before the bioinformatics tool
   invocation.

Do not serialize: start download → wait 7 minutes → then start thinking about the task. The download is I/O-bound and
the planning is CPU-bound; they should overlap.

The download function should skip if the assemblies directory already contains 403 FASTA files. Check before
downloading. The assemblies live at `lyzortx/data/assemblies/picard/` (gitignored, under `lyzortx/` per the fork's
code placement policy).

## Feature Design Principles

- **Continuous scores over binary thresholds** where the gradient carries biological signal (receptor phmmer scores, RBP
  mmseqs identity, capsule HMM scores). See `track_DEPLOY.md` for the full rationale per feature block.
- **Integer counts over binary presence** for defense subtypes (2 copies of MazEF ≠ 1 copy).
- **Keep categoricals categorical** — phylogroup, serotype, ST, O-antigen type, LPS core type are genuinely categorical.
- **No derived summary features** — the model can learn `defense_diversity` from the constituent subtype counts if it
  matters. Pre-computing aggregates wastes feature budget.
- **No duplicate features** — each biological signal should appear exactly once in the feature vector.

## Output Directory

Store generated features in `lyzortx/generated_outputs/deployment_paired_features/` (gitignored), organized by
feature block: `host_defense/`, `host_surface/`, `host_typing/`, `phage_rbp/`.
