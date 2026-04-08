# Project Knowledge Model

<!-- Last consolidated: 2026-04-09T00:30:00+00:00 -->
<!-- Source: lyzortx/research_notes/lab_notebooks -->

**41 knowledge units** across 7 themes (36 active, 5 dead ends, 0 superseded)

## Data & Labels

Labeling policy, data quality, split contracts, and training corpus.

- Binary lysis labels use any_lysis rule (matrix_score > 0); ~7% of positives are single-hit noise with matrix_score=0
  despite positive labels. [validated; source: ST0.1]
- Borderline score=0 pairs are downweighted (0.1) rather than excluded or flipped, preserving information while reducing
  noise impact. [validated; source: TA11; see also: label-policy-binary]
- Three confidence tiers (high/medium/low) from replicate agreement preserve 80% of data while narrowing positive
  fraction from 27.4% to 14.6%. [validated; source: ST0.1b]
- ST02 defines 294 training bacteria; ST03 reserves 65 cv_group-disjoint bacteria for sealed holdout. Deterministic
  cv_group hashing with salt ensures reproducibility. [validated; source: ST0.2, ST0.3, AR01; see also:
  raw-interactions-authority]
- raw_interactions.csv is the authoritative training corpus; all derived training cohorts and evaluation splits trace
  back to this file. [validated; source: ST0.2, AR01, DEPLOY01]

## Features & Feature Engineering

What works, what doesn't, leakage risks, and encoding decisions.

- Phylogroup, ST, and serotype structure hard-to-lyse behavior at field level (Kruskal-Wallis p=1.24e-09) but no
  individual trait values survive FDR correction. [preliminary; source: TB03]
- Narrow-susceptibility rescue is concentrated in 19/96 phages (19.8%); Myoviridae dominate, and no single universal
  rescuer exists. [validated; source: TB04; see also: family-bias-straboviridae]
- Receptors are richer than binary presence: OmpC spans 50 variants, BtuB spans 28; binary encoding discards this
  structure. [validated; source: TC01; see also: binary-encoding-waste]
- Phage tetranucleotide kmer SVD embeddings (24 dimensions, 99.42% variance retained) provide compact composition-based
  features, but the original paper achieves 86% AUROC without any kmer features. SVD is panel-fitted denoising
  incompatible with FASTA-only contracts. [validated; source: TD02, 2026-04-08 paper analysis; see also:
  raw-kmer-zero-signal]
  - *Raw 256-dim frequencies add zero signal; SVD denoises but requires panel fitting. The paper's success without kmers
    suggests they are not critical for prediction.*
- Only 77/96 phages have curated receptor data; 19 are explicitly uncovered rather than guessed, requiring explicit
  unknown handling in RBP-receptor compatibility features. [validated; source: TE01]
- 74% of numeric features are binary thresholds discarding continuous biological gradients in receptor scores, RBP
  identity, capsule HMM scores, and defense gene counts. [validated; source: TL18, DEPLOY track rationale; see also:
  receptor-variant-richness, defense-integer-counts]
- Defense system gene counts reflect real biological redundancy (e.g., 2 vs 1 MazEF copies); HMM detection scores
  reflect only tool confidence. Integer counts are the correct encoding. [validated; source: DEPLOY06, DEPLOY track
  design]
- Defense subtypes correlate with phylogroup, causing lineage confounding: defense features rerank borderline phages by
  host lineage rather than mechanistic evasion signal. [validated; source: 2026-04-08 defense ablation, 2026-04-08
  defense top-3 deep dive; see also: defense-ablation-autoresearch, adsorption-first-strategy]
  - *Specific mechanism: lytic phages dropped from top-3 when the host defense profile resembled phylogroups where those
    phages don't lyse. The top-3 regression only manifests in multi-seed aggregated predictions, not individual seeds.*
- The original paper confirms adsorption factors dominate prediction (30 significant traits vs 2 defense traits);
  per-phage models achieve 86% AUROC using only bacterial adsorption features, without kmer or defense features.
  [validated; source: 2026-04-08 paper analysis, Nature Microbiology 2024; see also: adsorption-first-strategy,
  defense-lineage-confounding]

## Model Architecture & Performance

Architecture choices, calibration, and performance bounds.

- LightGBM v1 baseline with defense + phage-genomic features (no pairwise) achieves clean AUC 0.837, top-3 hit rate
  0.908 on ST03 holdout. [validated; source: TG06, TG07, TG09; see also: tl18-improvement]
- TL18 model achieves +0.036 AUC over baseline (0.823 vs 0.787) with 98.5% probability of improvement; new TL15/16/17
  feature blocks account for 38.5% of total importance. [validated; source: TL18 audit; see also: lightgbm-v1-locked,
  autoresearch-parity]
- AUTORESEARCH raw-FASTA baseline achieves 0.810 AUC on ST03 holdout vs TL18's 0.823; the difference falls inside the
  95% bootstrap CI and is not statistically significant. [validated; source: 2026-04-08 AUTORESEARCH eval; see also:
  tl18-improvement, svd-bottleneck, defense-ablation-autoresearch, autoresearch-ceiling]
  - *Adding defense features narrows the gap further to 0.817 AUC (+0.7pp) but regresses top-3 from 90.8% to 86.2%. The
    remaining gap is architectural (pairwise features, calibration, per-phage modeling), not feature-driven.*
- AUTORESEARCH feature search is concluded: base slots (159 features, 0.810 AUC) are at ceiling for the current
  all-pairs LightGBM architecture. Defense adds noise, raw kmers add zero signal. Further improvement requires
  architectural changes. [validated; source: 2026-04-08 paper analysis, 2026-04-08 kmer ablation, 2026-04-08 defense
  ablation; see also: autoresearch-parity, adsorption-dominates-paper]
  - *Candidate architectural changes: pairwise interaction features, post-hoc calibration (isotonic/Platt), per-phage
    sub-models. All exceed the current train.py-only search surface.*
- SVD compression was the AUTORESEARCH bottleneck, not feature count: removing SVD and using 300-tree LightGBM on 159
  raw features improved from 0.765 to 0.810 AUC. [validated; source: 2026-04-08 AUTORESEARCH eval]
- Isotonic calibration outperforms Platt on ECE (0.021 vs 0.028); Platt outperforms on log-loss (0.333 vs 0.344).
  Neither dominates across all calibration metrics. [validated; source: TG02, ST0.5]
  - *Results are split-specific; other splits may rank differently.*
- Top-3 hit rate and AUC move in different directions across feature blocks: defense subtypes improved top-3 (+4.6pp)
  but degraded AUC (-0.001). [validated; source: TG03]
  - *On small holdout (65 strains), 1 strain flip = 1.5pp top-3.*
- LightGBM requires deterministic=True and force_col_wise=True for reproducibility across systems; nondeterminism was
  observed without these flags. [validated; source: TG08, TG09]
- Adding 79 DefenseFinder host defense features to AUTORESEARCH improves AUC from 0.810 to 0.817 (+0.7pp) but regresses
  top-3 hit rate from 90.8% to 86.2% (-4.6pp). [validated; source: 2026-04-08 defense ablation; see also:
  defense-lineage-confounding, autoresearch-parity, top3-auc-not-redundant]
  - *Defense features help discrimination (AUC) but hurt ranking (top-3) due to lineage confounding. LightGBM's native
    feature selection is not aggressive enough to ignore noisy defense subtypes.*
- Adsorption-first modeling (host surface + typing features) is the correct critical path; defense features contribute
  but are not gate-critical for first baseline. [validated; source: 2026-04-05 replan, antiphage-landscape reading; see
  also: autoresearch-parity]

## Evaluation & Benchmarking

Holdout protocol, benchmark methodology, and error analysis.

- ST03 grouped host split (65 cv_group-disjoint bacteria) is the canonical v1 benchmark; both TL18 and AUTORESEARCH are
  evaluated on it for honest comparison. [validated; source: TF01, 2026-04-08 AUTORESEARCH eval; see also:
  split-contract]
- Bootstrap CIs must be computed at holdout-strain level (not pair level) to align evaluation denominator with
  recommendation metric; 1000 resamples on 65 strains. [validated; source: TF01]
- Holdout misses split into four non-overlapping failure modes: abstention (2 strains, zero positives), narrow-recall
  (2, single positive outside top-3), family-collapse (2, wrong family ranked highest), and within-family ordering (4,
  right family wrong phage). [validated; source: ST09, TF02; see also: family-bias-straboviridae]
- Straboviridae prior collapse suppresses cross-family true positives: 4/10 holdout misses are cross-family blind spots
  where model top-3 remains all Straboviridae. [validated; source: ST09]

## Deployment & Train/Inference Parity

Feature derivation parity, raw-input pipeline, and pre-computation.

- Three systematic training/inference feature mismatches identified: DefenseFinder version drift (17.3% model
  importance), capsule detection sensitivity mismatch (3-5%), and extra phage in FNA directory (low impact). [validated;
  source: TL18 audit; see also: zero-delta-parity]
- Honest model packaging requires zero-delta parity: training and inference must call the exact same feature-derivation
  functions and produce identical features for any host. [validated; source: DEPLOY track rationale, DEPLOY09]
- Feature CSVs (defense gene counts, surface features for all 403 hosts) are versioned pre-computed artifacts that
  downstream tasks consume directly, decoupling expensive preprocessing from evaluation CI. [validated; source:
  DEPLOY06, DEPLOY07; see also: defensefinder-precompute]
- Picard Figshare assemblies (403 FASTAs, 1.9GB, CC BY 4.0) are the authoritative source for raw-input feature
  derivation; not all 403 have interaction labels. [validated; source: DEPLOY01]
- DefenseFinder on 403 hosts takes ~35-114 min depending on cores; incompatible with search-loop runtime budgets,
  requiring pre-computation and caching. [validated; source: DEPLOY06, 2026-04-05 replan]

## Dead Ends

Compressed lessons from approaches that didn't work.

- VHRdb, BASEL, KlebPhaCol, and GPB external interaction datasets showed neutral cumulative lift over the internal-only
  baseline; adding them did not improve predictions. [validated; source: TK01, TK02, TK03, TI09]
- Label-derived features (legacy_label_breadth_count, defense_evasion_*, receptor_variant_ seen_in_training_positives)
  caused severe leakage and were removed entirely from TL18. [validated; source: TG04, TG05, TG06, TG08, TG12; see also:
  pairwise-block-leaky]
- 5/13 pairwise compatibility features were soft-leaky (all 4 defense-evasion features + 1 receptor training-positive
  flag); the 8 clean ones individually cannot recover the AUC gap without degrading top-3. [validated; source: TG08,
  TG11, TG12]
- Mechanistic pairwise features (RBP-receptor, anti-defense pairs) showed no statistically significant lift on honest
  holdout rerun; kept in TL18 bundle but contribute only 3.5% and 2.0% to feature importance. [validated; source: TL12,
  TL18 audit; see also: adsorption-first-strategy]
- Raw 256-dim tetranucleotide kmer frequencies add zero extractable signal for LightGBM with 300 trees on 96 phages; the
  dimensionality/sample-size mismatch makes splits uninformative without SVD denoising. [validated; source: 2026-04-08
  kmer ablation; see also: kmer-embeddings, svd-bottleneck]

## Open Questions

Unresolved items that still matter for the project direction.

- Can host-compatibility features (receptor/defense) lift narrow-susceptibility recovery for the 12/36 resolved narrow
  strains not rescued by any panel phage? [preliminary; source: ST09, TB04; see also: narrow-susceptibility-rescue,
  error-buckets]
- Can within-family reranking using host-range evidence improve phage selection inside saturated score bands where the
  model knows the right family but not the right phage? [preliminary; source: ST09; see also: error-buckets,
  family-bias-straboviridae]
- Per-RBP-gene-level scores from FASTAs (beyond family presence) could close the remaining gap between AUTORESEARCH and
  TL18; aligns with adsorption-first strategy and paper's emphasis on RBPs as the key phage-side variable. [preliminary;
  source: antiphage-landscape reading in project.md, 2026-04-08 paper analysis; see also: autoresearch-parity,
  adsorption-first-strategy, adsorption-dominates-paper]
