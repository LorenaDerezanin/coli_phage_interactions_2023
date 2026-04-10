# Project Knowledge Model

<!-- Last consolidated: 2026-04-09T20:30:00+00:00 -->
<!-- Source: lyzortx/research_notes/lab_notebooks -->

**43 knowledge units** across 7 themes (34 active, 9 dead ends)

## Data & Labels

Labeling policy, data quality, split contracts, and training corpus.

- Binary lysis labels use any_lysis rule (matrix_score > 0); ~7% of positives are single-hit noise with matrix_score=0
  despite positive labels. [validated; source: ST0.1]
- Borderline score=0 pairs are downweighted (0.1) rather than excluded or flipped, preserving information while reducing
  noise impact. [validated; source: TA11; see also: label-policy-binary]
- Three confidence tiers (high/medium/low) from replicate agreement preserve 80% of data while narrowing positive
  fraction from 27.4% to 14.6%. Currently unused — the pipeline uses binary labels with sample weighting — but the tier
  structure is available in the pair table for future soft-label or curriculum-learning experiments. [validated; source:
  ST0.1b]
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
- Receptors are richer than binary presence: OmpC spans 50 variants, BtuB spans 28. The TL18 pipeline discarded this
  structure via binary thresholds; the AUTORESEARCH pipeline uses continuous HMM scores (e.g., OmpC: 79 unique values in
  [509, 818]). [validated; source: TC01, 2026-04-09 AUTORESEARCH host_surface audit; see also: defense-integer-counts]
- Only 77/96 phages have curated receptor data; 19 are explicitly uncovered rather than guessed, requiring explicit
  unknown handling in RBP-receptor compatibility features. [validated; source: TE01]
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

- TL18 model achieves +0.036 AUC over baseline (0.823 vs 0.787) with 98.5% probability of improvement; new TL15/16/17
  feature blocks account for 38.5% of total importance. [validated; source: TL18 audit; see also: autoresearch-parity]
- AUTORESEARCH raw-FASTA baseline achieves 0.810 AUC on ST03 holdout vs TL18's 0.823; the difference falls inside the
  95% bootstrap CI and is not statistically significant. [validated; source: 2026-04-08 AUTORESEARCH eval; see also:
  tl18-improvement, svd-bottleneck, defense-ablation-autoresearch, per-phage-blending-dominant]
  - *Adding defense features narrows the gap further to 0.817 AUC (+0.7pp) but regresses top-3 from 90.8% to 86.2%. The
    remaining gap is architectural, not feature-driven.*
- SVD compression was the AUTORESEARCH bottleneck, not feature count: removing SVD and using 300-tree LightGBM on 159
  raw features improved from 0.765 to 0.810 AUC. [validated; source: 2026-04-08 AUTORESEARCH eval]
- LightGBM requires deterministic=True and force_col_wise=True for reproducibility across systems; nondeterminism was
  observed without these flags. [validated; source: TG08, TG09]
- Adding 79 DefenseFinder host defense features to AUTORESEARCH improves AUC from 0.810 to 0.817 (+0.7pp) but regresses
  top-3 hit rate from 90.8% to 86.2% (-4.6pp). [validated; source: 2026-04-08 defense ablation; see also:
  defense-lineage-confounding, autoresearch-parity]
  - *Defense features help discrimination (AUC) but hurt ranking (top-3) due to lineage confounding. LightGBM's native
    feature selection is not aggressive enough to ignore noisy defense subtypes.*
- Per-phage LightGBM sub-models blended with all-pairs predictions are the dominant AUTORESEARCH architectural gain:
  +2.0pp AUC (0.810->0.830) on ST03 holdout, +3.1pp top-3 (90.8%->93.8%), and -2.3pp Brier (0.167->0.144). Surpasses
  TL18 on AUC (+0.7pp) and matches top-3 (93.8% vs 93.7%). [validated; source: 2026-04-09 APEX ablation, 2026-04-09 APEX
  holdout; see also: adsorption-dominates-paper, family-bias-straboviridae, autoresearch-parity,
  per-phage-not-deployable, deployment-goal]
  - *Each phage gets its own 32-tree LightGBM on host-only features (surface + typing + stats), blended 50/50 with
    all-pairs predictions. Phages with <3 positives fall back to all-pairs. All 96/96 phages qualify for per-phage
    models on the ST03 training set (min 3 positives in 221 bacteria). Bootstrap CIs overlap with TL18 — differences not
    statistically significant on 65-bacteria holdout.*
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
- The all-pairs model misses 6/65 holdout bacteria in top-3. Failure modes: Straboviridae prior collapse (2 strains
  rescued by per-phage, not deployable), abstention (2 strains with zero positives), needle-in-haystack (1 strain with
  1/96 positive), and narrow-host prior collapse (1 strain with 14 narrow-host positives). [validated; source: ST09,
  TF02, 2026-04-09 APEX holdout; see also: family-bias-straboviridae, narrow-host-prior-collapse,
  per-phage-not-deployable]
  - *Per-phage blend reduced misses to 4/65, but per-phage is not deployable for unseen phages. The 6/65 all-pairs error
    rate is the operationally relevant baseline for the deployment goal.*
- Straboviridae prior collapse suppresses cross-family true positives: broad-host Straboviridae (62-71% lysis rate)
  dominate model rankings, pushing narrow-host true positives (10-53% lysis rate) below the top-3 cutoff. [validated;
  source: ST09, 2026-04-09 APEX holdout NILS53 analysis]
- Per-phage models help broad-to-moderate phages distinguish their hosts but cannot override the broad-phage prior for
  narrow-host phages (<30% lysis rate) with few training positives; NILS53 (14 narrow-host TPs) remains a top-3 miss.
  [validated; source: 2026-04-09 APEX holdout NILS53 analysis; see also: family-bias-straboviridae, higher-res-rbp,
  per-phage-blending-dominant]
  - *The per-phage sub-model for a phage with 37 training positives and 10% lysis rate lacks discriminative power
    against broad phages scoring 0.80-0.94. Structural RBP embeddings are the path to resolving this — phage-side
    features that predict receptor binding specificity rather than relying solely on host-side features.*
- Inner-val top-3 predictions do not reliably transfer to holdout: per-phage blend predicted a top-3/AUC trade-off on
  inner-val (93.2% vs 94.6%) that did not replicate on holdout (93.8% vs 90.8% — both improved). [validated; source:
  2026-04-09 APEX holdout; see also: st03-canonical-benchmark, bootstrap-strain-level]
  - *On 65-74 bacteria evaluation sets, 1 strain flip = 1.4-1.5pp top-3. Inner-val bacteria overlap with training
    distribution; holdout bacteria are cv_group-disjoint. Holdout is the only honest top-3 test.*

## Deployment & Train/Inference Parity

Feature derivation parity, raw-input pipeline, and pre-computation.

- The overarching deployment goal is a model that produces reliable lysis-likelihood inference on unseen E. coli
  strains, ranks or recommends cocktails from a set of potentially unseen phages, and generalizes along both the host
  and phage axes simultaneously. [validated; source: 2026-04-09 project direction; see also: per-phage-not-deployable,
  higher-res-rbp]
- Per-phage sub-models are architecturally incompatible with the deployment goal: they require training-time interaction
  data for each phage and cannot produce predictions for unseen phages. The +2.0pp AUC gain is real on the fixed-panel
  holdout but non-transferable to the target deployment scenario. [validated; source: 2026-04-09 APEX holdout,
  2026-04-09 project direction; see also: per-phage-blending-dominant, deployment-goal, higher-res-rbp]
  - *Per-phage models generalize to unseen bacteria (the holdout tests this) but not to unseen phages (no sub-model
    exists to fall back on). For a system that must rank novel phage candidates, the all-pairs architecture with
    expressive phage-side features (e.g., PLM embeddings) is the only viable path.*
- Three systematic training/inference feature mismatches identified: DefenseFinder version drift (17.3% model
  importance), capsule detection sensitivity mismatch (3-5%), and extra phage in FNA directory (low impact). [validated;
  source: TL18 audit; see also: zero-delta-parity]
- Honest model packaging requires zero-delta parity: training and inference must call the exact same feature-derivation
  functions and produce identical features for any host. [validated; source: DEPLOY track rationale, DEPLOY09]
- Picard Figshare assemblies (403 FASTAs, 1.9GB, CC BY 4.0) are the authoritative source for raw-input feature
  derivation; not all 403 have interaction labels. [validated; source: DEPLOY01]

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
- Physicochemical RBP protein descriptors (AA composition + MW/GRAVY/pI/aromaticity/charge, 28 features from 80/96
  phages) add no predictive signal; bulk protein properties cannot capture binding-interface specificity. [validated;
  source: 2026-04-09 APEX ablation, 2026-04-09 APEX holdout; see also: higher-res-rbp, adsorption-dominates-paper]
  - *Confirmed on both inner-val and holdout: full combo (all features + per-phage) has identical AUC to per-phage-only
    (0.830). Physicochemical descriptors describe bulk protein properties, not binding geometry.*
- Phage functional gene repertoire features (PHROG category counts/fractions, anti-defense, depolymerase; 25 features)
  degrade top-3 from 94.6% to 87.8% on inner-val and are effectively noise for this prediction task. [validated; source:
  2026-04-09 APEX ablation; see also: defense-lineage-confounding, defense-ablation-autoresearch]
  - *PHROG categories may be redundant with existing phage_projection features that already encode family-level biology.
    Anti-defense features specifically confirmed as noise, consistent with the defense-lineage-confounding finding.*
- Raw 256-dim tetranucleotide kmer frequencies add zero extractable signal for LightGBM with 300 trees on 96 phages; the
  dimensionality/sample-size mismatch makes splits uninformative without SVD denoising. [validated; source: 2026-04-08
  kmer ablation; see also: svd-bottleneck]
- Label-free pairwise RBP x receptor cross-terms (24 features: has_rbp x receptor_score, rbp_count x receptor_score for
  12 OMP receptors) showed no improvement on ST03 holdout; the inner-val top-3 gain (+1.3pp) did not replicate.
  [validated; source: 2026-04-09 APEX ablation, 2026-04-09 APEX holdout; see also: inner-val-unreliable-top3,
  physicochemical-rbp-insufficient]
  - *Full combo AUC/top-3 identical to per-phage-only on holdout. The cross-terms are too coarse — they encode "phage
    has RBP and host has receptor" but not binding specificity.*
- ProstT5→SaProt PLM embeddings of RBP sequences (1280-dim, PCA to 32) achieve 33.9% LightGBM feature importance but
  zero predictive lift on ST03 holdout; they cannibalize existing phage family features without adding new
  discriminative signal. [validated; source: 2026-04-10 AX08 holdout; see also: physicochemical-rbp-insufficient,
  higher-res-rbp]
  - *PLM embeddings steal 28pp importance from phage_projection (22.7%→7.6%) and phage_stats (20.3%→7.3%). The
    embeddings encode protein-level similarity that correlates with genome-level family membership. Mean-pooling across
    RBPs further dilutes binding-specific signal. Same 4/65 holdout misses as baseline, NILS53 not rescued. The initial
    evaluation appeared to show zero lift due to a feature-filtering bug (PLM features silently dropped by old candidate
    SLOT_PREFIXES); after fixing, the result is confirmed: genuinely neutral.*

## Open Questions

Unresolved items that still matter for the project direction.

- Can host-compatibility features (receptor/defense) lift narrow-susceptibility recovery for the 12/36 resolved narrow
  strains not rescued by any panel phage? [preliminary; source: ST09, TB04; see also: narrow-susceptibility-rescue,
  error-buckets]
- Can within-family reranking using host-range evidence improve phage selection inside saturated score bands where the
  model knows the right family but not the right phage? [preliminary; source: ST09; see also: error-buckets,
  family-bias-straboviridae]
- Can the defense top-3 regression (-4.6pp on all-pairs) be mitigated on the deployable all-pairs architecture — e.g.,
  by feature selection, regularization, or phylogroup-aware encoding — to capture the +0.7pp AUC gain without the
  ranking penalty? [preliminary; source: 2026-04-08 defense ablation, 2026-04-09 APEX holdout, 2026-04-09 project
  direction; see also: defense-ablation-autoresearch, defense-lineage-confounding, deployment-goal]
  - *Originally framed as "does defense behave differently under per-phage blending" but per-phage is not deployable for
    unseen phages. The actionable question is whether defense can be made useful on the all-pairs model. LightGBM's
    native feature selection was not aggressive enough; explicit mitigation of lineage confounding may be needed.*
- General-purpose protein language model embeddings and structure predictions encode phage family similarity, not
  receptor binding specificity. Breaking the phage-side feature plateau requires binding-specific representations
  (fine-tuned PLMs, docking features, or receptor-type prediction from structure), not more expressive general protein
  features. [validated; source: 2026-04-08 paper analysis, 2026-04-09 APEX ablation, 2026-04-09 PHIStruct review,
  2026-04-10 AX08 holdout; see also: adsorption-first-strategy, adsorption-dominates-paper,
  physicochemical-rbp-insufficient, plm-rbp-redundant]
  - *ProstT5→SaProt PLM embeddings (1280-dim, PCA to 32) achieve 33.9% feature importance but zero holdout lift — they
    cannibalize phage_projection (22.7%→7.6%) by encoding the same family-level information in a more splittable
    continuous space. AlphaFold structural features would face the same redundancy unless binding-interface-specific
    features are extracted. With 96 phages (71 unique embeddings), the sample size is too small to learn
    embedding→binding mappings from general structural representations.*
