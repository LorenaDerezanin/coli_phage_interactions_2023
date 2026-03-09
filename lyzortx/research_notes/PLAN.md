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
- [x] ST0.2 Build one canonical pair table with IDs, labels, uncertainty, and v0 feature blocks. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st02_build_pair_table.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json`.
- [x] ST0.3 Lock one leakage-safe split protocol and one fixed holdout benchmark for v0. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st03_build_splits.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json`.
- [x] ST0.4 Train one strong tabular baseline and one simple comparator baseline. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st04_train_baselines.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json`.
- [x] ST0.5 Calibrate probabilities and export ranked per-strain phage predictions. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st05_calibrate_rank.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st05_expected_metrics.json`.
- [x] ST0.6 Generate top-3 recommendations with policy-tuned defaults (`pred_logreg_platt`, no family cap). Implemented
      in `lyzortx/pipeline/steel_thread_v0/steps/st06_recommend_top3.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st06_expected_metrics.json`.
- [x] ST0.6b Compare ranking policy variants (`raw`, `platt`, `isotonic`; with/without family cap) to avoid
      recommendation-policy regressions. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st06b_compare_ranking_policies.py`.
- [x] ST0.7 Emit one reproducible report to `lyzortx/generated_outputs/steel_thread_v0/`. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st07_build_report.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st07_expected_metrics.json`.

### Required Artifacts

- `lyzortx/generated_outputs/steel_thread_v0/metrics_summary.csv`
- `lyzortx/generated_outputs/steel_thread_v0/top3_recommendations.csv`
- `lyzortx/generated_outputs/steel_thread_v0/calibration_summary.csv`
- `lyzortx/generated_outputs/steel_thread_v0/error_analysis.csv`
- `lyzortx/generated_outputs/steel_thread_v0/run_manifest.json`

### Go / No-Go Gates

- [ ] End-to-end command completes on a clean environment without manual patching.
- [x] No leakage violations detected by the v0 checks.
- [x] Top-3 hit-rate and calibration metrics are reported for the locked protocol.
- [ ] Top-3 and calibration metrics are reported on both slices: full-label and high-confidence (ST0.1b).
- [x] v0 model materially outperforms a naive baseline on the same split.
- [ ] Failure cases are documented with at least one concrete hypothesis per major error bucket.

### Expansion Rule After v0

- Only after v0 passes: add Track I Tier A datasets in strict order (`VHRdb -> BASEL -> KlebPhaCol -> GPB`) with
  one-source-at-a-time ablations.

### Steel Thread Findings (2026-02-15)

- Signal exists in current internal data: ST0.4 logistic baseline reached holdout ROC-AUC `0.826948` and top-3 hit rate
  `0.846154` (`55/65` strains), far above naive baseline (`0.015385` top-3).
- Label noise is a first-order issue: `8,917 / 35,424` pairs (`25.17%`) show conflicting interpretable observations.
- Strict-confidence filtering keeps `28,338 / 35,424` pairs (`79.9966%`) but remains class-imbalanced (`4,135` strict
  positives vs `24,203` strict negatives).
- Calibration quality improved materially at ST0.5 (logreg ECE: raw `0.176341`, isotonic `0.031802`, Platt `0.029253`),
  but best recommendation ranking was `platt/raw`, not isotonic.
- Raw and Platt top-3 tie by construction in current setup (monotonic remapping of the same raw model score), so top-k
  lift must come from new model signal, better labels, or new data, not calibration-only ranking changes.
- Recommendation policy choice is high-impact: `logreg_platt__none` yields top-3 `0.846154` vs prior
  `logreg_isotonic__max_family_2` at `0.784615`.
- ST0.7 error analysis now isolates `10` holdout miss strains for targeted follow-up, making next work item clear.

### Steel Thread Risks

- High replicate/dilution disagreement can cap achievable performance unless label uncertainty is modeled explicitly.
- Current v0 split mainly stress-tests host generalization; phage-family generalization risk is still under-tested.
- Current v0 features are metadata-heavy; missing mechanistic host/phage features may cap ceiling performance.
- Holdout denominator at strain level is modest (`65`), so metric variance can be non-trivial without confidence
  intervals.
- Tier 1 KPI targets remain stretch relative to current v0 (`84.6%` all-strain top-3), so short-cycle lift tracking is
  needed before major external-data integration.

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

- [x] Build a canonical ID map for bacteria and phages across all tables. Implemented in
      `lyzortx/generated_outputs/track_a/id_map/{bacteria_id_map.csv,phage_id_map.csv}`.
- [x] Resolve naming/alias mismatches (for example legacy phage names). Implemented in
      `lyzortx/generated_outputs/track_a/id_map/{bacteria_alias_resolution.csv,phage_alias_resolution.csv}` and
      candidate reports.
- [x] Add automated data integrity checks for row/column consistency. Implemented in
      `lyzortx/generated_outputs/track_a/integrity/{integrity_checks.csv,integrity_report.json}` and
      `lyzortx/pipeline/track_a/checks/check_track_a_integrity.py`.
- [x] Define and document handling policy for uninterpretable labels (`score='n'`). Implemented in
      `lyzortx/generated_outputs/track_a/labels/{label_set_v1_policy.json,label_set_v2_policy.json}`.
- [x] Add plaque-image-assisted QC pass for ambiguous/conflicting pairs using the core study raw image release.
      Implemented in `lyzortx/generated_outputs/track_a/qc/{plaque_image_qc_queue.csv,plaque_image_qc_summary.json}`.
- [x] Define cohort contracts and denominator rules (`raw369`, `matrix402`, `features404`) for all reports. Implemented
      in `lyzortx/generated_outputs/track_a/cohort/{cohort_contracts.csv,cohort_contracts.json}`.
- [x] Preserve replicate and dilution structure in intermediate tables. Implemented in
      `lyzortx/generated_outputs/track_a/labels/{track_a_observations_with_ids.csv,`
      `track_a_pair_dilution_summary.csv,track_a_pair_observation_grid.csv}`.
- [x] Create label set v1: `any_lysis`, `lysis_strength`, `dilution_potency`, `uncertainty_flags`. Implemented in
      `lyzortx/generated_outputs/track_a/labels/label_set_v1_pairs.csv`.
- [x] Create label set v2 with alternative aggregation assumptions and compare impact. Implemented in
      `lyzortx/generated_outputs/track_a/labels/{label_set_v2_pairs.csv,label_set_v1_v2_comparison.csv}`.
- [x] Add scripts that regenerate all derived labels from raw data in one command. Implemented in
      `lyzortx/pipeline/track_a/run_track_a.py` (plus docs in `lyzortx/pipeline/track_a/README.md`).

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
- [ ] Pilot structure-aware RBP embeddings (PHIStruct-style) for low-similarity generalization.
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
- [ ] Add LOGOCV-style grouped validation and mean-hit-ratio@k reporting for recommendation utility tracking.
- [ ] Add bootstrap confidence intervals for strain-level top-k metrics to quantify variance on small holdouts.
- [ ] Add dual-slice reporting for all benchmarks: full-label slice and strict-confidence slice.
- [ ] Add source-aware evaluation for external integration: leave-one-datasource-out and cross-source transfer
      benchmarks.
- [ ] Add source-aware leakage checks: ensure no duplicated isolates/genomes leak across datasource boundaries.
- [ ] Add pre-Tier-1 progress gates for internal-only iteration (delta vs locked ST0.6 baseline, not only absolute
      thresholds).
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
- [ ] Milestone G0.1: ship receptor-first enriched baseline (host adsorption proxies + phage RBP/depolymerase features)
      before broad weak-label expansion.
- [ ] External-data training order: internal-only baseline -> +Tier A supervised sources -> +Tier B weak-label sources.
- [ ] Stretch branch: Stage A model `P(adsorption)` from host-surface + phage-RBP + compatibility features.
- [ ] Stretch branch: Stage B model `P(productive_lysis | adsorption)` from post-entry features.
- [ ] Stretch branch: compose final probability `P(lysis) = P(adsorption) * P(productive_lysis | adsorption)`.
- [ ] Add multi-task formulation for binary + strength + potency targets.
- [ ] Add calibrated outputs (isotonic/Platt) and uncertainty intervals.
- [ ] Add label-noise-aware training variants (confidence-weighted or probabilistic labels from replicate structure).
- [ ] Add ablation matrix to measure where signal comes from: host-only, phage-only, pairwise-only, and no-identity
      controls.
- [ ] Add robust handling of class imbalance and label uncertainty.
- [ ] Add model interpretation outputs (global and per-sample).

## Track H: In-Silico Cocktail Recommendation

- [x] Benchmark policy variants for top-k recommendation and lock a non-regressing default (`ST0.6b` diagnostics).
- [ ] Add policy guardrail: do not expect top-k gains from monotonic score recalibration alone; require demonstrable
      ranking change from non-monotonic transformation, new model signal, or new constraints/objectives.
- [ ] Replace heuristic-only recommender with optimization-based recommender.
- [ ] Define objective: maximize expected coverage and potency under uncertainty.
- [ ] Add constraints: diversity, redundancy penalties, and risk-aware terms.
- [ ] Compare against baseline and generic recipes on held-out evaluation sets.
- [ ] Evaluate robustness under perturbations of uncertain interactions.
- [ ] Add recommendation explanations at per-strain and per-cocktail levels.

## Track I: External Data and Literature Integration

- [x] Create a curated reading list of closely related phage-host prediction papers. Reference:
      `lyzortx/research_notes/LITERATURE.md`.
- [x] Build `source_registry.csv` for all external sources: source type, label kind, host resolution, assay type,
      license, access path, last checked. Implemented in `lyzortx/research_notes/external_data/source_registry.csv`.
- [ ] For VHRdb ingest, keep source-fidelity fields: global response, datasource response, disagreement flag, and
      source-native reference link.
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

## Track K: Validation Benchmarks and Wet-Lab Readiness

- [ ] Define a set of held-out validation cases: known-answer pairs withheld from training that the model must recover
      convincingly before predictions are sent for external wet-lab testing.
- [ ] Report validation recovery rate and calibrated confidence for each model version — predictions below a confidence
      threshold are flagged as not ready for production/testing.
- [ ] Prepare a small batch of novel _E. coli_ strain predictions (strains unseen during training) formatted for
      external CDMO production and plaque-assay validation.
- [ ] Define a feedback protocol: wet-lab validation results feed back into the pipeline as ground-truth labels for the
      next training cycle.
- [ ] Track prediction-vs-reality concordance across validation batches to build empirical credibility.

## Immediate Next Tasks

- [x] Start Steel Thread v0 and complete ST0.7 before any external-data ingest work.
- [x] Bootstrap PLAN orchestration scaffold (task registry + runtime state + CLI + CI trigger workflow). Implemented in
      `lyzortx/orchestration/{tasks.json,orchestrator.py,README.md}`,
      `lyzortx/generated_outputs/orchestration/runtime_state.json`, and `.github/workflows/orchestrator.yml`.
- [x] Finalize `score='n'` handling policy and document aggregation rules. Implemented via Track A label policy
      artifacts in `lyzortx/generated_outputs/track_a/labels/`.
- [x] Define strict-confidence policy for ST0.1b and quantify retained coverage vs noise reduction.
- [x] Lock denominator/cohort policy and publish metric definitions for Tier 1 vs Tier 2 benchmarks. Implemented in
      `lyzortx/research_notes/TIER_BENCHMARK_DENOMINATOR_POLICY.md`.
- [x] Build canonical ID normalization and mismatch report script. Implemented in
      `lyzortx/pipeline/track_a/steps/build_track_a_foundation.py` with outputs under
      `lyzortx/generated_outputs/track_a/id_map/`.
- [x] Implement ST0.3b split suite with explicit phage-family holdout and host+phage dual-axis stress tests.
- [x] Implement ST0.4b ablations (host-only, phage-only, no-identity controls) to quantify true signal sources.
- [x] Implement ST0.5b/ST0.6c reporting on both slices (full-label and strict-confidence) with bootstrap CIs.
- [x] Implement label builder for binary/strength/potency targets from raw interactions. Implemented in
      `lyzortx/pipeline/track_a/steps/build_track_a_foundation.py` with v1/v2 outputs under
      `lyzortx/generated_outputs/track_a/labels/`.
- [x] Implement first mechanistic signal block from internal data: host receptor/defense proxies + phage
      RBP/depolymerase/domain proxies. Implemented in
      `lyzortx/pipeline/track_a/steps/build_mechanistic_proxy_features.py` with outputs under
      `lyzortx/generated_outputs/track_a/mechanistic_proxy_features/`.
- [x] Run first PHIStruct-style RBP embedding pilot on phage-family holdout split and compare to non-structural RBP
      features. Implemented in `lyzortx/pipeline/track_a/steps/run_phistruct_rbp_pilot.py` with outputs under
      `lyzortx/generated_outputs/track_a/phistruct_pilot/`.
- [x] Create `source_registry.csv` and populate initial entries for VHRdb, BASEL, KlebPhaCol, GPB, Virus-Host DB, NCBI.
      Implemented in `lyzortx/research_notes/external_data/source_registry.csv`.
- [x] Implement first Tier A ingest path (VHRdb) and run internal-only vs +VHRdb ablation.
