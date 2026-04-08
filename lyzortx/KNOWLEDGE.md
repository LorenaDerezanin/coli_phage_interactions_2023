# Project Knowledge Model

<!-- Last consolidated: 2026-04-08 -->
<!-- Source: lyzortx/research_notes/lab_notebooks -->

**56 knowledge units** across 8 themes (49 active, 6 dead ends, 1 superseded)

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
- Dilution response is a separable signal from binary lysis: 32.9% of lytic pairs show high-potency lysis, with
  Myoviridae skewed toward high potency and Siphoviridae toward low. [validated; source: TB05]
  - *Binary any_lysis label discards this structure; future models could treat potency as auxiliary task.*

## Features & Feature Engineering

What works, what doesn't, leakage risks, and encoding decisions.

- Phylogroup, ST, and serotype structure hard-to-lyse behavior at field level (Kruskal-Wallis p=1.24e-09) but no
  individual trait values survive FDR correction. [preliminary; source: TB03]
- Narrow-susceptibility rescue is concentrated in 19/96 phages (19.8%); Myoviridae dominate, and no single universal
  rescuer exists. [validated; source: TB04; see also: family-bias-straboviridae]
- Receptors are richer than binary presence: OmpC spans 50 variants, BtuB spans 28; binary encoding discards this
  structure. [validated; source: TC01; see also: binary-encoding-waste]
- OMP receptor clustering compressed 291 raw states to 22 indicators to fit feature budget; rare clusters grouped into
  single buckets. [validated; source: TC02]
- LPS core typing is complete (404/404 hosts); capsule typing is too sparse (5.7%) for standalone feature — missingness
  flag carries as much signal. [validated; source: TC03]
- Phage tetranucleotide kmer SVD embeddings (24 dimensions, 99.42% variance retained) provide compact composition-based
  features for all 96 panel phages. [validated; source: TD02]
  - *Composition-based only; does not capture gene-level or RBP-level signals.*
- VIRIDIC-tree MDS (8 coordinates from patristic distances) captures phage relatedness beyond identity or family labels.
  [validated; source: TD03]
- Only 77/96 phages have curated receptor data; 19 are explicitly uncovered rather than guessed, requiring explicit
  unknown handling in RBP-receptor compatibility features. [validated; source: TE01]
- Isolation-host distance is genuinely pairwise (same phage gets different values against each target) and not a
  phage-ID proxy. [validated; source: TE03]
- 74% of numeric features are binary thresholds discarding continuous biological gradients in receptor scores, RBP
  identity, capsule HMM scores, and defense gene counts. [validated; source: TL18, DEPLOY track rationale; see also:
  receptor-variant-richness, defense-integer-counts]
- 91+ wasted one-hot columns from exact duplicates: host_o_type (84 cols, duplicate of host_o_antigen_type),
  host_surface_lps_core_type (6 cols), host_capsule_abc_present (1 col), plus 4 derived summary features computable from
  their base features. [validated; source: DEPLOY track rationale, DEPLOY03]
- Defense system gene counts reflect real biological redundancy (e.g., 2 vs 1 MazEF copies); HMM detection scores
  reflect only tool confidence. Integer counts are the correct encoding. [validated; source: DEPLOY06, DEPLOY track
  design]
- Protein-level phmmer search for O-antigen features is 12x faster than nhmmer DNA search (4.3s vs 72s per host) with
  identical O-type calls and stronger E-values. [validated; source: DEPLOY04, DEPLOY07]

## Model Architecture & Performance

Architecture choices, calibration, and performance bounds.

- LightGBM v1 baseline with defense + phage-genomic features (no pairwise) achieves clean AUC 0.837, top-3 hit rate
  0.908 on ST03 holdout. [validated; source: TG06, TG07, TG09; see also: tl18-improvement]
- TL18 model achieves +0.036 AUC over baseline (0.823 vs 0.787) with 98.5% probability of improvement; new TL15/16/17
  feature blocks account for 38.5% of total importance. [validated; source: TL18 audit; see also: lightgbm-v1-locked,
  autoresearch-parity]
- AUTORESEARCH raw-FASTA baseline achieves 0.810 AUC on ST03 holdout vs TL18's 0.823; the difference falls inside the
  95% bootstrap CI and is not statistically significant. [validated; source: 2026-04-08 AUTORESEARCH eval; see also:
  tl18-improvement, svd-bottleneck]
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
- LogisticRegression with class_weight='balanced' is a strong baseline (AUC 0.827, top-3 0.846) that substantially
  outperforms naive dummy; LightGBM improves materially over it (AUC 0.908, top-3 0.933) on expanded features.
  [validated; source: ST0.4, TG01]
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
- Dual-slice reporting (full-label + strict-confidence) ensures recommendation quality and calibration quality are
  tracked separately; strict slice has narrower positive fraction (0.146 vs 0.274). [validated; source: ST0.7, ST0.8]
- Strict-confidence slice is materially harder to calibrate: ECE ~0.095 vs full-label ~0.020; root cause uncertain
  (feature ablations, class-balance, or group-specific error). [preliminary; source: TG02]
  - *Revisit after next round of feature engineering.*
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

## Infrastructure & Operations

Environment, CI, and operational knowledge not recoverable from code.

- V1 benchmark environment pinned to exact versions (python=3.12.12, lightgbm=4.6.0, scikit-learn=1.8.0, etc.); all
  random seeds centralized at 42 for training/calibration. [validated; source: TJ02]
- RunPod REST v1 API schema is unstable (field formats changed between calls); local execution is more reliable and
  default LightGBM trains in 0.4s on CPU. [validated; source: devops RunPod entries, AR08]
  - *Dedicated RunPod workflow exists for heavier search tasks, but default should remain local until complexity
    justifies cost.*
- Different tasks use different CI image profiles (full-bio for bioinformatics, basic for lightweight tests); 1.9GB
  assembly data cannot be baked into images due to disk limits. [validated; source: devops CI entries]
- External data endpoints differ from listed URLs: VHRdb at viralhostrangedb.pasteur.cloud (not CityU), BASEL via PLOS
  supplementary, KlebPhaCol at phage.klebphacol.soton.ac.uk, GPB via Nature source-data workbook. [validated; source:
  TI03, TI04]
  - *Endpoints subject to change; registry must be refreshed periodically.*

## Dead Ends

Compressed lessons from approaches that didn't work.

- VHRdb, BASEL, KlebPhaCol, and GPB external interaction datasets showed neutral cumulative lift over the internal-only
  baseline; adding them did not improve predictions. [validated; source: TK01, TK02, TK03, TI09]
  - *May be worth revisiting with different integration strategies or richer mapping.*
- Label-derived features (legacy_label_breadth_count, defense_evasion_*, receptor_variant_ seen_in_training_positives)
  caused severe leakage and were removed entirely from TL18. [validated; source: TG04, TG05, TG06, TG08, TG12; see also:
  pairwise-block-leaky]
- 5/13 pairwise compatibility features were soft-leaky (all 4 defense-evasion features + 1 receptor training-positive
  flag); the 8 clean ones individually cannot recover the AUC gap without degrading top-3. [validated; source: TG08,
  TG11, TG12]
  - *Single-feature tests exhausted; multi-feature combinations not tested.*
- Mechanistic pairwise features (RBP-receptor, anti-defense pairs) showed no statistically significant lift on honest
  holdout rerun; kept in TL18 bundle but contribute only 3.5% and 2.0% to feature importance. [validated; source: TL12,
  TL18 audit; see also: adsorption-first-strategy]
- All-features arm (top-3 0.877) underperformed best single-block arms (defense/phage-genomic top-3 0.908); feature
  combination does not always help. [validated; source: TG03]
- Family-cap diversity constraint (max_per_family=2) reduced top-3 hit rate by 3.1pp across all score variants;
  diversity constraint does not bind in current holdout. [validated; source: ST0.6b]
  - *Different phage panel or application may reorder priorities.*

### Superseded

- ~~Isotonic-calibration ranking was superseded by raw/Platt ranking; isotonic achieves 0.800 top-3 vs raw/Platt tied at
  0.846. [validated; source: ST0.6, ST0.6b]~~

## Open Questions

Unresolved items that still matter for the project direction.

- Can host-compatibility features (receptor/defense) lift narrow-susceptibility recovery for the 12/36 resolved narrow
  strains not rescued by any panel phage? [preliminary; source: ST09, TB04; see also: narrow-susceptibility-rescue,
  error-buckets]
- Can within-family reranking using host-range evidence improve phage selection inside saturated score bands where the
  model knows the right family but not the right phage? [preliminary; source: ST09; see also: error-buckets,
  family-bias-straboviridae]
- Should the recommendation interface support explicit abstention when no labeled susceptible phage exists (2/10 miss
  strains have zero positives)? [validated; source: ST09]
  - *Abstention is necessary but introduces false-negative tradeoff requiring user study.*
- DefenseFinder produces HMM scores, system completeness, and sub-threshold hits not yet mined; these could enable
  per-subtype detection-confidence features. [preliminary; source: Future note in project.md; see also:
  defense-integer-counts]
- Per-RBP-gene-level scores from FASTAs (beyond family presence) could close the remaining gap between AUTORESEARCH and
  TL18; aligns with adsorption-first strategy. [preliminary; source: antiphage-landscape reading in project.md; see
  also: autoresearch-parity, adsorption-first-strategy]
- Non-leaky pairwise candidates (curated receptor lookups + isolation-host distances) were tested individually but
  multi-feature combinations were not explored. [preliminary; source: TG11; see also: pairwise-block-leaky]
