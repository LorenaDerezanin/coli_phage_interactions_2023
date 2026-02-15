# Lyzor Tx In-Silico Pipeline Plan

Last updated: 2026-02-15

## Mission

- Build the best possible phage-lysis prediction pipeline for _E. coli_ using only in-silico methods.
- Primary local data source: `data/interactions/raw/raw_interactions.csv` and related repo metadata/features.
- No wet-lab access is assumed for this project.
- External data and literature can be added when they improve model quality or rigor.

## Operating Rules

- Treat this file as the main execution driver for the repo.
- Update checklist status in this file as work progresses.
- Link every major code change or note update to one or more items here.
- Keep methods reproducible and auditable (deterministic where possible).
- Use two KPI tiers:
  - **Tier 1 (Current Panel, Feasible):** evaluation with current 96-phage panel and current interaction matrix.
  - **Tier 2 (North-Star):** aspirational targets that may require panel expansion and external data.

## Steel Thread v0 (Start Small)

- Goal: prove end-to-end viability quickly with a minimal but honest pipeline.
- Philosophy: internal paper data first; external data only after the end-to-end path is proven.
- Success condition: one command runs data prep -> training -> calibration -> recommendation -> report.

### Scope Guardrails

- Use only local internal inputs for v0: `data/interactions/raw/raw_interactions.csv`,
  `data/interactions/interaction_matrix.csv`, `data/genomics/bacteria/picard_collection.csv`,
  `data/genomics/phages/guelin_collection.csv`, `data/metadata/370+host_cross_validation_groups_1e-4.csv`.
- Exclude for v0: external datasets, mechanistic Stage A/B decomposition, and complex optimization recommender logic.
- Keep features simple: existing host metadata, phage metadata, and lightweight pairwise flags only.

### Execution Checklist

- [x] ST0.1 Define v0 label policy and uncertainty flags from raw interactions (`score='n'` included). Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`.
- [x] ST0.1b Add strict confidence tiering (`high_conf_pos`, `high_conf_neg`, `ambiguous`) as a parallel output from
      ST0.1 to support dual-slice evaluation. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st01b_confidence_tiers.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json`.
- [ ] ST0.2 Build one canonical pair table with IDs, labels, uncertainty, and v0 feature blocks.
- [ ] ST0.3 Lock one leakage-safe split protocol and one fixed holdout benchmark for v0.
- [ ] ST0.4 Train one strong tabular baseline and one simple comparator baseline.
- [ ] ST0.5 Calibrate probabilities and export ranked per-strain phage predictions.
- [ ] ST0.6 Generate minimal top-3 recommendations with simple diversity constraints.
- [ ] ST0.7 Emit one reproducible report to `lyzortx/generated_outputs/steel_thread_v0/`.

### Required Artifacts

- `lyzortx/generated_outputs/steel_thread_v0/metrics_summary.csv`
- `lyzortx/generated_outputs/steel_thread_v0/top3_recommendations.csv`
- `lyzortx/generated_outputs/steel_thread_v0/calibration_summary.csv`
- `lyzortx/generated_outputs/steel_thread_v0/error_analysis.csv`
- `lyzortx/generated_outputs/steel_thread_v0/run_manifest.json`

### Go / No-Go Gates

- [ ] End-to-end command completes on a clean environment without manual patching.
- [ ] No leakage violations detected by the v0 checks.
- [ ] Top-3 hit-rate and calibration metrics are reported for the locked protocol.
- [ ] Top-3 and calibration metrics are reported on both slices: full-label and high-confidence (ST0.1b).
- [ ] v0 model materially outperforms a naive baseline on the same split.
- [ ] Failure cases are documented with at least one concrete hypothesis per major error bucket.

### Expansion Rule After v0

- Only after v0 passes: add Track I Tier A datasets in strict order (`VHRdb -> BASEL -> KlebPhaCol -> GPB`) with
  one-source-at-a-time ablations.

## Parallel Execution View

- Use this view for planning workstreams.
- Tracks in the same stage box can run in parallel unless blocked by their own incoming dependencies.
- Keep the dependency DAG above as the source of truth for strict ordering.

```mermaid
graph LR
  subgraph s0["Stage 0 (Serial Foundation)"]
    ta["Track A: Data Integrity and Labeling"]
  end

  subgraph s1["Stage 1 (Parallelizable Build-Out)"]
    tb["Track B: Exploratory Analysis and Signal Discovery"]
    tc["Track C: Feature Engineering (Host)"]
    td["Track D: Feature Engineering (Phage)"]
    tf["Track F: Splits, Evaluation Protocol, and Leakage Control"]
    ti["Track I: External Data and Literature Integration"]
  end

  subgraph s2["Stage 2 (Parallelizable Integration)"]
    te["Track E: Pairwise Compatibility Features"]
    tg["Track G: Modeling Pipeline"]
    th["Track H: In-Silico Cocktail Recommendation"]
    tk["Track K: Sentinel Benchmarks"]
  end

  subgraph s3["Stage 3 (Release and Audit)"]
    tj["Track J: Reproducibility and Release Quality"]
  end

  ta --> tb
  ta --> tc
  ta --> td
  ta --> tf
  ta --> ti

  tb --> te
  tc --> te
  td --> te

  tc --> tg
  td --> tg
  te --> tg
  tf --> tg

  tf --> th
  tg --> th

  tf --> tk
  tg --> tk

  th --> tj
  tk --> tj
  tg --> tj
  tf --> tj
  ti --> tj
```

## Track A: Data Integrity and Labeling

- [ ] Build a canonical ID map for bacteria and phages across all tables.
- [ ] Resolve naming/alias mismatches (for example legacy phage names).
- [ ] Add automated data integrity checks for row/column consistency.
- [ ] Define and document handling policy for uninterpretable labels (`score='n'`).
- [ ] Define cohort contracts and denominator rules (`raw369`, `matrix402`, `features404`) for all reports.
- [ ] Preserve replicate and dilution structure in intermediate tables.
- [ ] Create label set v1: `any_lysis`, `lysis_strength`, `dilution_potency`, `uncertainty_flags`.
- [ ] Create label set v2 with alternative aggregation assumptions and compare impact.
- [ ] Add scripts that regenerate all derived labels from raw data in one command.

## Track B: Exploratory Analysis and Signal Discovery

- [x] Profile raw interaction matrix composition and replicate consistency.
- [x] Quantify morphotype breadth and narrow-susceptibility patterns.
- [ ] Characterize hard-to-lyse strains by known host traits.
- [ ] Characterize "rescuer phages" for narrow-susceptibility strains.
- [ ] Analyze dilution-response patterns per phage and per bacterial subgroup.
- [ ] Build uncertainty map: where annotation conflicts are concentrated.
- [ ] Prioritize candidate mechanistic feature hypotheses from EDA findings.

## Track C: Feature Engineering (Host)

- [ ] Build receptor/surface feature block: O/K/LPS-related loci and known receptor proxies.
- [ ] Add outer membrane receptor variant features.
- [ ] Encode phylogeny-aware host embeddings with leakage-safe generation.
- [ ] Build defense-system context block (presence, subtype, burden, co-occurrence).
- [ ] Add missingness indicators and confidence scores for host features.
- [ ] Version host feature matrix with schema and provenance manifest.

## Track D: Feature Engineering (Phage)

- [ ] Build phage sequence processing pipeline from genome/protein files.
- [ ] Extract RBP/depolymerase/domain features (HMM/domain and structure-aware proxies).
- [ ] Build phage protein family embeddings or pangenome cluster features.
- [ ] Add phage architecture/taxonomy/module features.
- [ ] Add isolation-host and lineage priors as weak features (not dominant).
- [ ] Version phage feature matrix with schema and provenance manifest.

## Track E: Pairwise Compatibility Features

- [ ] Design phage-host compatibility features (RBP family vs host receptor proxies).
- [ ] Add domain-level compatibility scores.
- [ ] Add feature interactions for adsorption-relevant host/phage pairs.
- [ ] Add uncertainty-aware pairwise features (confidence-weighted signals).

## Track F: Splits, Evaluation Protocol, and Leakage Control

- [ ] Define fixed split protocol before model iteration: leave-cluster-out host splits and phage-clade holdouts.
- [ ] Keep a strict untouched external test benchmark for final validation.
- [ ] Add leakage checks for all split strategies.
- [ ] Add source-aware evaluation for external integration: leave-one-datasource-out and cross-source transfer
      benchmarks.
- [ ] Add source-aware leakage checks: ensure no duplicated isolates/genomes leak across datasource boundaries.
- [ ] Define Tier 1 (current-panel feasible) benchmark suite:
  - **Top-3 Lytic Hit Rate (all strains, fixed panel) >= 95%**; stretch target >= 96.5%.
  - **Top-3 Lytic Hit Rate (susceptible strains only) >= 98%**.
  - **Precision at high confidence >= 99%** with minimum support threshold (report both precision and support).
  - **Calibration quality gates:** Brier score and ECE tracked for each model version.
- [ ] Define Tier 2 (north-star) benchmark suite:
  - **Top-3 Lytic Hit Rate (all strains) > 98%** after justified panel expansion and external integration.
  - **Simulated 3-phage cocktail coverage > 98%** in expanded-panel evaluation.
- [ ] Add benchmark report template for fair model-to-model comparison.

## Track G: Modeling Pipeline

- **Guiding Principle:** A "meaningful model" for this project is one that produces a **calibrated probability of
  lysis** for any given phage-bacterium pair, enabling nuanced downstream cocktail recommendations.
- [ ] Baseline 1: strong tabular binary model on existing host-only features.
- [ ] Baseline 2: joint host+phage feature model without pairwise interactions.
- [ ] Milestone G0: ship a calibrated Baseline 2 with leakage-safe protocol before mechanistic branching.
- [ ] External-data training order: internal-only baseline -> +Tier A supervised sources -> +Tier B weak-label sources.
- [ ] Stretch branch: Stage A model `P(adsorption)` from host-surface + phage-RBP + compatibility features.
- [ ] Stretch branch: Stage B model `P(productive_lysis | adsorption)` from post-entry features.
- [ ] Stretch branch: compose final probability `P(lysis) = P(adsorption) * P(productive_lysis | adsorption)`.
- [ ] Add multi-task formulation for binary + strength + potency targets.
- [ ] Add calibrated outputs (isotonic/Platt) and uncertainty intervals.
- [ ] Add robust handling of class imbalance and label uncertainty.
- [ ] Add model interpretation outputs (global and per-sample).

## Track H: In-Silico Cocktail Recommendation

- [ ] Replace heuristic-only recommender with optimization-based recommender.
- [ ] Define objective: maximize expected coverage and potency under uncertainty.
- [ ] Add constraints: diversity, redundancy penalties, and risk-aware terms.
- [ ] Compare against baseline and generic recipes on held-out evaluation sets.
- [ ] Evaluate robustness under perturbations of uncertain interactions.
- [ ] Add recommendation explanations at per-strain and per-cocktail levels.

## Track I: External Data and Literature Integration

- [x] Create a curated reading list of closely related phage-host prediction papers. Reference:
      `lyzortx/research_notes/LITERATURE.md`.
- [ ] Build `source_registry.csv` for all external sources: source type, label kind, host resolution, assay type,
      license, access path, last checked.
- [ ] Tier A supervised ingestion priority:
  1. VHRdb, 2) BASEL, 3) KlebPhaCol, 4) GPB.
- [ ] Define harmonization protocol for Tier A datasets: taxonomy normalization, ID mapping, assay-scale mapping,
      uncertainty flags.
- [ ] Tier B weak-label ingestion: Virus-Host DB and NCBI Virus/BioSample metadata with confidence tiering.
- [ ] Define confidence tiers for external labels (for example assay-backed, metadata-only, inferred).
- [ ] Integrate external data as a non-blocking enhancer: internal-only baseline must remain runnable and reportable.
- [ ] Run strict ablations in sequence: internal-only -> +VHRdb -> +BASEL -> +KlebPhaCol -> +GPB -> +Tier B weak labels.
- [ ] Track incremental lift and failure modes by datasource and confidence tier.

## Track J: Reproducibility and Release Quality

- [ ] One command to regenerate core figures/tables from raw and versioned inputs.
- [ ] Freeze environment specs and seeds for each benchmark run.
- [ ] Publish data/feature/model manifests with checksums.
- [ ] Add CI checks for schema drift, reproducibility scripts, and key metrics.
- [ ] Track external data-use restrictions and license terms in manifests and release notes.
- [ ] Keep generated outputs under `lyzortx/generated_outputs/` only.
- [ ] Keep one-off scripts that feed notes under `lyzortx/research_notes/ad_hoc_analysis_code/`.

## Track K: Sentinel Benchmarks

- [ ] Define a set of sentinel tailored cases (hard but biologically plausible hits).
- [ ] **Sentinel Strain Recovery = 100%:** Model must correctly identify known solutions for all sentinel strains.
- [ ] Require each major model version to report sentinel recovery performance.
- [ ] Track regressions in sentinel behavior across pipeline updates.

## Immediate Next Tasks

- [ ] Start Steel Thread v0 and complete ST0.2 through ST0.3 before any external-data ingest work.
- [ ] Finalize `score='n'` handling policy and document aggregation rules.
- [x] Define strict-confidence policy for ST0.1b and quantify retained coverage vs noise reduction.
- [ ] Lock denominator/cohort policy and publish metric definitions for Tier 1 vs Tier 2 benchmarks.
- [ ] Build canonical ID normalization and mismatch report script.
- [ ] Implement label builder for binary/strength/potency targets from raw interactions.
- [ ] Implement first calibrated joint host+phage baseline with fixed leakage-safe splits.
- [ ] Create `source_registry.csv` and populate initial entries for VHRdb, BASEL, KlebPhaCol, GPB, Virus-Host DB, NCBI.
- [ ] Implement first Tier A ingest path (VHRdb) and run internal-only vs +VHRdb ablation.
