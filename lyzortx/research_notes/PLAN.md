# Lyzor Tx In-Silico Pipeline Plan

## Parallel Execution View

- Tracks in the same stage box can run in parallel unless blocked by their own incoming dependencies.

```mermaid
graph LR
  subgraph s0["Stage 0 (Serial Foundation)"]
    tst["Track ST: Steel Thread v0"]
    ta["Track A: Data Integrity and Labeling"]
  end

  subgraph s1["Stage 1 (Parallelizable Build-Out)"]
    tb["Track B: Exploratory Analysis and Signal Discovery"]
    tc["Track C: Feature Engineering (Host)"]
    td["Track D: Feature Engineering (Phage)"]
    te["Track E: Pairwise Compatibility Features"]
    ti["Track I: External Data and Literature Integration"]
  end

  subgraph s2["Stage 2 (Parallelizable Integration)"]
    tf["Track F: Evaluation Protocol"]
    tg["Track G: Modeling Pipeline"]
    th["Track H: In-Silico Cocktail Recommendation"]
    tk["Track K: External Data Lift Measurement"]
  end

  subgraph s3["Stage 3 (Release and Audit)"]
    tj["Track J: Reproducibility and Release Quality"]
  end

  ta --> tb
  ta --> tc
  ta --> td
  tc --> te
  td --> te
  tg --> tf
  tc --> tg
  td --> tg
  te --> tg
  tg --> th
  ta --> ti
  ti --> tk
  tg --> tk
  tg --> tj
```

## Track ST: Steel Thread v0

- **Guiding Principle:** Prove end-to-end viability with a minimal but honest pipeline using internal data only.
- [x] **ST01** Define v0 label policy and uncertainty flags from raw interactions. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`.
- [x] **ST01B** Add strict confidence tiering as a parallel output from ST0.1 to support dual-slice evaluation.
      Implemented in `lyzortx/pipeline/steel_thread_v0/steps/st01b_confidence_tiers.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json`.
- [x] **ST02** Build one canonical pair table with IDs, labels, uncertainty, and v0 feature blocks. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st02_build_pair_table.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json`.
- [x] **ST03** Lock one leakage-safe split protocol and one fixed holdout benchmark for v0. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st03_build_splits.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json`.
- [x] **ST04** Train one strong tabular baseline and one simple comparator baseline. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st04_train_baselines.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json`.
- [x] **ST05** Calibrate probabilities and export ranked per-strain phage predictions. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st05_calibrate_rank.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st05_expected_metrics.json`.
- [x] **ST06** Generate top-3 recommendations with policy-tuned defaults. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st06_recommend_top3.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st06_expected_metrics.json`.
- [x] **ST06B** Compare ranking policy variants to avoid recommendation-policy regressions. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st06b_compare_ranking_policies.py`.
- [x] **ST07** Emit one reproducible report to generated_outputs/steel_thread_v0/. Implemented in
      `lyzortx/pipeline/steel_thread_v0/steps/st07_build_report.py`. Regression baseline:
      `lyzortx/pipeline/steel_thread_v0/baselines/st07_expected_metrics.json`.
- [x] **ST08** Add dual-slice reporting (full-label and strict-confidence) to ST0.7
  - ST0.7 report includes separate metric rows for full-label and strict-confidence slices
  - Both slices report top-3 hit rate, calibration ECE, and Brier score
- [x] **ST09** Document failure case hypotheses for each major holdout miss error bucket
  - Each holdout miss strain in error_analysis.csv has at least one documented hypothesis
  - Hypotheses are written to the track lab notebook with actionable next steps

## Track A: Data Integrity and Labeling

- **Guiding Principle:** Canonical IDs, label policies, cohort contracts, and replicate-aware label sets from raw data.
- [x] **TA01** Build a canonical ID map for bacteria and phages across all tables. Implemented in
      `lyzortx/generated_outputs/track_a/id_map/{bacteria_id_map.csv,phage_id_map.csv}`.
- [x] **TA02** Resolve naming/alias mismatches (for example legacy phage names). Implemented in
      `lyzortx/generated_outputs/track_a/id_map/{bacteria_alias_resolution.csv,phage_alias_resolution.csv}`.
- [x] **TA03** Add automated data integrity checks for row/column consistency. Implemented in
      `lyzortx/pipeline/track_a/checks/check_track_a_integrity.py`.
- [x] **TA04** Define and document handling policy for uninterpretable labels (score='n'). Implemented in
      `lyzortx/generated_outputs/track_a/labels/{label_set_v1_policy.json,label_set_v2_policy.json}`.
- [x] **TA05** Add plaque-image-assisted QC pass for ambiguous/conflicting pairs. Implemented in
      `lyzortx/generated_outputs/track_a/qc/{plaque_image_qc_queue.csv,plaque_image_qc_summary.json}`.
- [x] **TA06** Define cohort contracts and denominator rules for all reports. Implemented in
      `lyzortx/generated_outputs/track_a/cohort/{cohort_contracts.csv,cohort_contracts.json}`.
- [x] **TA07** Preserve replicate and dilution structure in intermediate tables. Implemented in
      `lyzortx/generated_outputs/track_a/labels/track_a_observations_with_ids.csv`.
- [x] **TA08** Create label set v1: any_lysis, lysis_strength, dilution_potency, uncertainty_flags. Implemented in
      `lyzortx/generated_outputs/track_a/labels/label_set_v1_pairs.csv`.
- [x] **TA09** Create label set v2 with alternative aggregation assumptions and compare impact. Implemented in
      `lyzortx/generated_outputs/track_a/labels/{label_set_v2_pairs.csv,label_set_v1_v2_comparison.csv}`.
- [x] **TA10** Add scripts that regenerate all derived labels from raw data in one command. Implemented in
      `lyzortx/pipeline/track_a/run_track_a.py`.

## Track B: Exploratory Analysis and Signal Discovery

- **Guiding Principle:** Profile interactions, identify hard-to-lyse strains, rescuer phages, and dilution-response
  patterns.
- [x] **TB01** Profile raw interaction matrix composition and replicate consistency
- [x] **TB02** Quantify morphotype breadth and narrow-susceptibility patterns
- [x] **TB03** Characterize hard-to-lyse strains by known host traits
  - Identify strains with zero or very few lytic phages in the interaction matrix
  - Report which host metadata fields (serotype, phylogroup, ST) correlate with low susceptibility
  - Output a summary CSV and findings in the track lab notebook
- [x] **TB04** Characterize rescuer phages for narrow-susceptibility strains
- [x] **TB05** Analyze dilution-response patterns per phage and per bacterial subgroup

## Track C: Feature Engineering (Host)

- **Guiding Principle:** Defense-system subtypes, OMP receptor variants, capsule/LPS detail, and phylogenomic embeddings
  for host strains.
- [x] **TC01** Build defense-system subtype feature block from defense_finder annotations
  - Ingest 370+host_defense_systems_subtypes.csv (138 subtype columns, 404 strains)
  - Variance filter drops subtypes present in <5 or >395 strains
  - Derived features include defense diversity, CRISPR presence, Abi burden
  - Output CSV joinable on bacteria column with ~60-80 informative features
- [x] **TC02** Build OMP receptor variant feature block from BLAST cluster assignments
  - Ingest blast_results_cured_clusters=99_wide.tsv (12 receptor proteins, 404 strains)
  - Encode cluster IDs as categoricals (one-hot top-k, group rare clusters)
  - Output CSV joinable on bacteria column with ~20 receptor features
- [x] **TC03** Build extended host surface features (capsule detail, LPS core, UMAP embeddings)
  - Add Klebsiella-type capsule, LPS core type, and 8D UMAP phylogenomic embeddings
  - All features joinable on bacteria column
  - Missingness indicators added for features with incomplete coverage
- [x] **TC04** Integrate host feature blocks into v1 pair table
  - All host feature blocks merged into a single host feature matrix
  - Join completeness verified (no unexpected NaN increase vs source data)
  - Quick LightGBM sanity check on training fold confirms lift over v0

## Track D: Feature Engineering (Phage)

- **Guiding Principle:** RBP features, genome k-mer embeddings, and phage distance embeddings from existing genomic
  data.
- [x] **TD01** Build RBP feature block from RBP_list.csv annotations
  - Parse per-phage RBP count, has_fiber, has_spike, RBP type composition
  - Handle NAs with indicator features for missing RBP annotations
  - Output CSV joinable on phage column with ~5-8 features
- [x] **TD02** Build genome k-mer embedding features from phage FNA files
  - Compute tetranucleotide (k=4) frequency vectors from 97 FNA genomes
  - Reduce via SVD to 20-30 dimensions
  - Add GC content and continuous genome length
  - Output CSV joinable on phage column with ~25-30 features
- [x] **TD03** Build phage distance embedding from VIRIDIC phylogenetic tree
  - Extract pairwise distances from 96_viridic_distance_phylogenetic_tree_algo=upgma.nwk
  - Compute MDS embedding to 5-8 dimensions
  - Output CSV joinable on phage column

## Track E: Pairwise Compatibility Features

- **Guiding Principle:** RBP-receptor compatibility, defense evasion proxy, and phylogenetic distance features that
  break the popular-phage bias.
- [x] **TE01** Build RBP-receptor compatibility features from curated genus-receptor lookup
  - Curated lookup mapping phage genus/subfamily to known primary receptor targets
  - Per-pair features include target_receptor_present, receptor_cluster_matches,
    receptor_variant_seen_in_training_positives
  - Output CSV joinable on bacteria+phage pair with ~5-8 features
- [x] **TE02** Build defense evasion proxy features from training-fold collaborative filtering
  - For each phage family, compute average lysis rate against each defense subtype from training data only
  - Per-pair expected evasion score computed as sum of phage family success rates against host defense systems
  - Leakage verified by computing on training fold only, never holdout
- [x] **TE03** Build phylogenetic distance to isolation host features
  - UMAP Euclidean distance between target host and phage isolation host
  - Defense Jaccard distance between target host and phage isolation host
  - Output CSV joinable on bacteria+phage pair with ~3-4 features

## Track F: Evaluation Protocol

- **Guiding Principle:** Lock v1 benchmark split and add bootstrap confidence intervals. ST03 already provides
  leakage-safe host-group and phage-family holdouts. TF01/TF02 are done but their metrics are invalidated by the
  label-leakage fix — they will be re-run as part of TG06.
- [x] **TF01** Lock ST03 split as v1 benchmark and add bootstrap CIs for all metrics. Model: `gpt-5.4-mini`.
  - Existing ST03 split locked as the canonical v1 evaluation protocol
  - Bootstrap CIs (1000 resamples of holdout strains) for top-3 hit rate, AUC, Brier score, and ECE
  - Dual-slice reporting (full-label and strict-confidence) for all metrics
- [x] **TF02** Before/after comparison of v0 vs v1 with error bucket analysis. Model: `gpt-5.4-mini`.
  - Side-by-side metrics table for v0 (metadata logreg) vs v1 (genomic GBM)
  - Error bucket analysis showing which v0 holdout misses v1 fixed and why
  - Honest reporting of strains that remain unpredictable

## Track G: Modeling Pipeline

- **Guiding Principle:** LightGBM model on expanded genomic features with calibration, ablation, and SHAP
  interpretation.
- [x] **TG01** Train LightGBM binary classifier on v1 expanded feature set
  - LightGBM with hyperparameter tuning via 5-fold CV on existing leakage-safe cv_groups
  - Logistic regression kept as interpretable comparator
  - Target AUC 0.87-0.90 and top-3 hit rate 90%+
- [x] **TG02** Calibrate GBM outputs with isotonic and Platt scaling
  - Same calibration approach as ST05 applied to GBM
  - Report ECE, Brier, log-loss for both calibration methods
  - Target ECE < 0.03 on full-label
- [x] **TG03** Run feature-block ablation suite proving which features deliver lift
  - Ablation arms: v0 features only, +defense subtypes, +OMP receptors, +phage genomic, +pairwise compatibility, all
    features
  - Each arm reports AUC, top-3 hit rate, Brier on same holdout split
  - v0 baseline is reference point in all comparisons
- [x] **TG04** Compute SHAP explanations for per-pair and global feature importance. Model: `gpt-5.4`.
  - TreeExplainer SHAP values for GBM model
  - Per-pair explanations answering why each phage was recommended for each strain
  - Global feature importance ranking across the panel
  - Per-strain summary of what makes each strain hard or easy to predict
  - Concrete recommendation of which feature blocks to keep in final v1 model, based on SHAP evidence and TG03 ablation
    results
- [x] **TG05** Run feature-subset sweep to find best block combination for top-3 ranking. Model: `gpt-5.4`.
  - Train models on all 2-block and 3-block combinations of the 4 new feature blocks (defense, OMP, phage-genomic,
    pairwise)
  - Reuse the TG01 winning hyperparameters for all sweep arms — do NOT run per-arm hyperparameter search. The goal is to
    isolate the feature-block effect, not confound it with per-arm tuning differences.
  - Report top-3 hit rate, AUC, and Brier on the same ST03 holdout for each combo
  - Identify the winning subset that maximizes top-3 hit rate without degrading AUC
  - Compare winning subset against the TG01 all-features model
  - Include a deployment-realistic arm that excludes all features derived from training labels
    (legacy_label_breadth_count, legacy_receptor_support_count) to measure generalization to truly novel strains
  - Report both panel-evaluation and deployment-realistic metrics for the winning configuration
  - Lock the final v1 feature configuration for downstream Track F and H
- [x] **TG06** Delete label-leaked features from the feature pipeline. Model: `gpt-5.4-mini`.
  - Remove legacy_label_breadth_count: delete the (n_infections, legacy_label_breadth_count) rename in
    st02_build_pair_table.py and drop the column from ST02 output
  - Remove legacy_receptor_support_count: delete its construction in build_rbp_receptor_compatibility_feature_block.py
    (Track E) and drop it from the TE01 output schema
  - Remove the LABEL_DERIVED_COLUMNS list in run_feature_subset_sweep.py and the deployment-realistic arm logic that
    depends on it
  - Delete v1_config_keys.py and simplify v1_feature_configuration.json to a single flat feature config (no
    panel_default vs deployment_realistic_sensitivity split)
  - Grep the entire lyzortx/ tree for legacy_label_breadth_count and legacy_receptor_support_count — zero hits must
    remain
  - All existing tests pass after deletions
- [x] **TG07** Retrain, recalibrate, and re-run SHAP and ablation on the clean feature set. Model: `gpt-5.4-mini`.
  - Retrain LightGBM on the clean feature set (reuse TG01 hyperparameters)
  - Recalibrate (isotonic + Platt) and report AUC, top-3, Brier, ECE
  - Re-run SHAP explanations on the clean model
  - Re-run feature-block ablation on the clean feature set
  - Update v1_feature_configuration.json with the clean model metrics
- [x] **TG08** Re-run downstream tracks and verify end-to-end pipeline. Model: `gpt-5.4-mini`.
  - Re-run explained recommendations (Track H) against clean model outputs
  - Re-run v0-vs-v1 evaluation (Track F) against clean model metrics
  - Run python -m lyzortx.pipeline.track_j.run_track_j end-to-end and verify it completes without error on the clean
    pipeline
  - The old label-leaked metrics must not appear in any output
- [x] **TG09** Fix LightGBM determinism and lock defense + phage_genomic as v1 winner. Model: `gpt-5.4-mini`.
  - Add deterministic=True to make_lightgbm_estimator in train_v1_binary_classifier.py
  - Remove n_jobs=1 from make_lightgbm_estimator (deterministic=True handles thread safety, force_col_wise=True is
    already set)
  - Update v1_feature_configuration.json to lock defense + phage_genomic as the winner (exclude pairwise block — 5 of 13
    features are training-label-derived)
  - Remove the feature-subset-sweep step from Track J's run_track_j.py so the lock file is treated as a human decision,
    not a regenerated output
  - Verify two consecutive runs of run_track_g.py --step train-v1-binary produce identical outputs
- [x] **TG10** Re-run downstream tracks on the stable 2-block lock. Model: `gpt-5.4-mini`.
  - Re-run Track H explained recommendations against the 2-block model outputs
  - Re-run Track F v0-vs-v1 evaluation against the 2-block model metrics
  - Run python -m lyzortx.pipeline.track_j.run_track_j end-to-end and verify it completes without error
  - Verify v1_feature_configuration.json is unchanged after the Track J run (sweep no longer regenerates it)
- [x] **TG11** Investigate non-leaky features that close the calibration gap. Model: `gpt-5.4`.
  - Pairwise soft leakage context: TE02 defense_evasion_* features (4) and TE01
    receptor_variant_seen_in_training_positives (1) are training-label-derived via collaborative filtering. Do not
    include these in candidate features.
  - Clean pairwise candidates to evaluate individually: TE03 isolation_host distances (2 features) and TE01 curated
    lookup features (lookup_available, target_receptor_present, protein_target_present, surface_target_present,
    receptor_cluster_matches)
  - Propose and test at least two candidate features (from clean pairwise or other sources) that do not leak training
    labels
  - Report whether any candidate recovers >50% of the AUC gap between the 2-block model (~0.837) and the old leaked
    model (~0.911) without degrading top-3
  - If no candidate closes the gap, accept the 2-block calibration as the honest v1 baseline
- [x] **TG12** Delete soft-leaky training-label-derived features from Track E code. Model: `gpt-5.4-mini`.
  - Delete the legacy soft-leaky pairwise block from Track E code
  - Remove the exact-variant training-positive flag from the RBP-receptor compatibility block
  - Update downstream tests that assert on removed columns
  - Grep lyzortx/ for the removed pairwise feature names — zero hits outside lab notebooks
  - All existing tests pass after deletions

## Track H: In-Silico Cocktail Recommendation

- **Guiding Principle:** Top-k recommendations with SHAP-based explanations. TH01/TH02 are done but will be re-run as
  part of TG06 against the clean model.
- [x] **TH01** Benchmark policy variants for top-k recommendation and lock a non-regressing default
- [x] **TH02** Add explained recommendations with calibrated P(lysis), CI, and SHAP features. Model: `gpt-5.4-mini`.
  - Each top-3 recommendation includes calibrated P(lysis), 95% CI, and top-3 SHAP features
  - Output format suitable for clinician or CDMO operator review
  - Report covers all holdout strains

## Track I: External Data and Literature Integration

- **Guiding Principle:** Tier A supervised and Tier B weak-label ingestion with source-fidelity, ablations, and lift
  tracking. Track K consumes Track I outputs for per-source lift measurement.
- [x] **TI01** Create a curated reading list of closely related phage-host prediction papers. Implemented in
      `lyzortx/research_notes/LITERATURE.md`.
- [x] **TI02** Build source_registry.csv for all external sources. Implemented in
      `lyzortx/research_notes/external_data/source_registry.csv`.
- [x] **TI03** For VHRdb ingest, keep source-fidelity fields
  - Ingested VHRdb rows preserve raw global_response and datasource_response without case folding
  - source_datasource_id, source_disagreement_flag, and source_native_record_id columns populated
  - Unit tests verify source-fidelity preservation
- [x] **TI04** Tier A supervised ingestion priority: VHRdb, BASEL, KlebPhaCol, GPB
- [x] **TI05** Define harmonization protocol for Tier A datasets. Model: `gpt-5.4`.
- [x] **TI06** Tier B weak-label ingestion: Virus-Host DB and NCBI Virus/BioSample metadata. Model: `gpt-5.4-mini`.
- [x] **TI07** Define confidence tiers for external labels. Model: `gpt-5.4`.
- [x] **TI08** Integrate external data as non-blocking enhancer: internal-only baseline must remain runnable. Model:
      `gpt-5.4`.
- [x] **TI09** Run strict ablations in sequence: internal-only -> +VHRdb -> +BASEL -> +KlebPhaCol -> +GPB -> +Tier B.
      Model: `gpt-5.4-mini`.
- [x] **TI10** Track incremental lift and failure modes by datasource and confidence tier. Model: `gpt-5.4-mini`.

## Track K: External Data Lift Measurement

- **Guiding Principle:** Incrementally add Track I external sources to the v1 model and measure per-source lift. Each
  task adds exactly one source, retrains, and reports metrics against the internal-only baseline. This isolates the
  contribution of each external dataset.
- [x] **TK01** Add VHRdb to training and measure lift vs internal-only baseline. Model: `gpt-5.4-mini`.
  - Connect TI08 VHRdb cohort rows to Track G training pipeline
  - Retrain with internal + VHRdb on the locked ST03 holdout split
  - Report AUC, top-3, Brier delta vs the locked 2-block internal-only baseline
  - If lift is negative or negligible, document why and do not include VHRdb in subsequent arms
- [x] **TK02** Add BASEL to training and measure cumulative lift. Model: `gpt-5.4-mini`.
  - Add BASEL rows to the best-so-far cohort (internal-only or internal+VHRdb, depending on TK01 result)
  - Retrain and report AUC, top-3, Brier delta vs previous best
  - Document whether BASEL adds, hurts, or is neutral
- [ ] **TK03** Add KlebPhaCol to training and measure cumulative lift. Model: `gpt-5.4-mini`.
  - Add KlebPhaCol rows to the best-so-far cohort
  - Retrain and report AUC, top-3, Brier delta vs previous best
  - Document whether KlebPhaCol adds, hurts, or is neutral
- [ ] **TK04** Add GPB to training and measure cumulative lift. Model: `gpt-5.4-mini`.
  - Add GPB rows to the best-so-far cohort
  - Retrain and report AUC, top-3, Brier delta vs previous best
  - Document whether GPB adds, hurts, or is neutral
- [ ] **TK05** Add Tier B weak labels and measure cumulative lift. Model: `gpt-5.4-mini`.
  - Add TI07 confidence-weighted Tier B rows to the best-so-far cohort
  - Retrain and report AUC, top-3, Brier delta vs previous best
  - Document whether Tier B adds, hurts, or is neutral
  - If any source combination improved metrics, propose a new locked config; otherwise keep internal-only as the v1
    baseline
- [ ] **TK06** Synthesize per-source lift results and lock the external data decision. Model: `gpt-5.4`.
  - Summarize TK01-TK05 results in a single comparison table (source, delta AUC, delta top-3, delta Brier vs
    internal-only baseline)
  - Identify the best-performing source combination (may be internal-only if nothing helped)
  - If external data earned inclusion, update v1_feature_configuration.json with the new locked config and retrain the
    final model
  - If no external source improved metrics, document the finding and confirm internal-only remains the v1 baseline
  - Write a project notebook entry with the final decision and rationale

## Track J: Reproducibility and Release Quality

- **Guiding Principle:** One-command regeneration and environment freezing for v1 pipeline. TJ01/TJ02 are done but must
  be re-verified after TG06 retrains the clean model.
- [x] **TJ01** One command to regenerate all v1 outputs from raw data. Model: `gpt-5.4-mini`.
  - Single entry point regenerates feature blocks, model, calibration, recommendations, and report
  - Runs without error on a fresh clone with only phage_env dependencies
- [x] **TJ02** Freeze environment specs and seeds for v1 benchmark run. Model: `gpt-5.4-mini`.
  - requirements.txt and phage_env environment spec locked for exact versions used
  - Random seeds documented for reproducible model training
