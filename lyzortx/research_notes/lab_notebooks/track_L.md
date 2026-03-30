### 2026-03-28: TL01 Annotate all 97 phage genomes with Pharokka

#### Executive summary

Added Pharokka and its bioconda dependencies (mmseqs2, trnascan-se, minced, aragorn, mash, dnaapler) to
`environment.yml` and built a Track L pipeline (`lyzortx/pipeline/track_l/`) that runs Pharokka on all 97 phage FNA
files, parses the `_cds_final_merged_output.tsv` per phage, and produces three summary tables: PHROGs category counts,
RBP gene list, and anti-defense gene list. All 97 phages must produce >0 annotated CDS for the pipeline to succeed.

#### What was implemented

- `environment.yml`: added `pharokka=1.9.1`, `mmseqs2=18.8cc5c`, `trnascan-se=2.0.12`, `minced=0.4.2`,
  `aragorn=1.2.41`, `mash=2.3`, `dnaapler=1.3.0` as conda dependencies from the bioconda channel.
- `lyzortx/pipeline/track_l/run_track_l.py`: entry point with `--step annotate|parse|all`.
- `lyzortx/pipeline/track_l/steps/run_pharokka.py`: iterates over all 97 `.fna` files in `data/genomics/phages/FNA/`,
  runs `pharokka.py` per phage, verifies >0 CDS in each output. Supports `--force` to re-run and skips already-complete
  phages by default.
- `lyzortx/pipeline/track_l/steps/parse_annotations.py`: reads pharokka's `_cds_final_merged_output.tsv` per phage and
  produces:
  - `phrog_category_counts.csv` — phage x 10 PHROG functional categories.
  - `rbp_genes.csv` — per-phage receptor binding protein genes identified by annotation pattern matching (tail fiber,
    tail spike, receptor binding, host specificity, adhesin, etc.).
  - `anti_defense_genes.csv` — per-phage anti-defense genes (anti-CRISPR, anti-restriction, methyltransferases, Ocr,
    Ard, etc.).
  - `manifest.json` — run metadata.

#### Design decisions

- **Gene classification by annotation patterns**: Pharokka does not natively tag RBP or anti-defense genes as separate
  categories. We use regex pattern matching on the `annot` column of the merged CDS output. The patterns are defined as
  module-level constants and tested in `lyzortx/tests/test_track_l_parse_annotations.py`.
- **Skip mash and extra annotations**: The pharokka runner uses `--skip_mash --skip_extra_annotations` to avoid
  INPHARED matching and reduce runtime, since those outputs are not needed for TL01.
- **Per-phage output directories**: Each phage gets its own pharokka output directory under
  `lyzortx/generated_outputs/track_l/pharokka_annotations/<phage_name>/`, matching pharokka's native output structure.

#### Key findings from exploratory analysis

Post-annotation analysis comparing generalist phages (200-262 strains lysed) vs narrow-range phages (6-17 strains)
revealed strong discriminative signal in the RBP PHROG repertoire:

- **43 unique RBP PHROG families** across the 97-phage panel.
- Generalists carry **6-9 RBPs** with diverse subtypes (long/short tail fiber, tail spike, host specificity protein,
  baseplate wedge connector). Narrow-range phages carry **0-3 RBPs**, mostly generic tail spike or tail fiber.
- PHROG repertoires are **almost completely disjoint**: 19 PHROGs exclusive to generalists, 7 exclusive to narrow-range,
  only 1 shared (PHROG 817). This means the specific PHROG IDs, not just counts, carry strong signal.
- Track C provides 12 host OMP receptor types (22 variant clusters at 99% identity) and 82 defense system subtypes.

Host range distribution across the 96-phage interaction panel (369 bacteria): min=6 strains (411_P1), Q1=42, median=78,
Q3=164, max=262 (DIJ07_P2). No phages have ≤3 positive strains; 10 phages lyse ≤20 strains.

This suggests the most informative feature for TL02 is a **phage x RBP-PHROG binary matrix** (43 columns), not just RBP
count or diversity. Even better: cross with host receptor data via enrichment analysis to learn empirical PHROG → receptor
associations from the interaction matrix.

#### Generalization principle

The key design goal of Track L: PHROG-receptor associations are learned from the interaction matrix during training, but
at prediction time only genomes are needed. For a novel phage, run pharokka → extract RBP PHROGs → feature vector. For
a novel host, run BLAST against OMP DB + DefenseFinder → receptor/defense feature vector. The model applies the learned
PHROG-receptor weights to predict lysis without any interaction data for the new pair. This is what TL06-TL09 implement.

### 2026-03-29: TL02 Annotation-interaction enrichment analysis (PHROG x receptor/defense)

#### Executive summary

Built a permutation-based enrichment module (`annotation_interaction_enrichment.py`) and ran three analyses on the full
369×96 interaction matrix (9,720 lytic pairs, 27.4% base rate). The test conditions on the phage carrying the PHROG,
then permutes host labels to get calibrated p-values that respect the correlation structure of the interaction matrix.
Results: 380/3,424 significant RBP PHROG × OMP receptor associations, 24/160 RBP PHROG × LPS core associations, and
27/924 anti-defense PHROG × defense subtype associations (BH-corrected p < 0.05). These results support building
pairwise mechanistic features in TL03/TL04.

#### What was implemented

- `lyzortx/pipeline/track_l/steps/annotation_interaction_enrichment.py`: Reusable enrichment module. Takes any (phage
  binary feature matrix, host binary feature matrix, interaction matrix) triple. For each (phage_feature, host_feature)
  pair, computes the lysis rate difference conditioned on the phage having the feature (host_has rate − host_lacks rate).
  P-values are computed by permuting host labels (1000 permutations), which preserves the row/column correlation
  structure of the interaction matrix. Applies Benjamini-Hochberg FDR correction across all tests within each analysis.
- `lyzortx/pipeline/track_l/steps/run_enrichment_analysis.py`: Loads pharokka RBP/anti-defense annotations, host OMP
  receptor clusters, LPS core types, and defense system subtypes, constructs binary matrices, runs three enrichment
  analyses. Wired into `run_track_l.py` as `--step enrich`.
- `lyzortx/tests/test_annotation_interaction_enrichment.py`: 14 unit tests using a 20×10 slice of the real interaction matrix, covering BH correction, permutation
  test, contingency table arithmetic, resolved-mask exclusion, and main effect confounding.

#### Design decisions

- **Conditioning on phage feature**: The test conditions on the phage carrying the PHROG, then asks whether the host
  feature increases lysis within that subset. This controls for the phage main effect — generalist PHROGs do not show
  spurious enrichment for every host feature.
- **Permutation p-values over Fisher's exact test**: Fisher's exact test assumes independent observations, but
  interaction matrix entries are correlated (some hosts are generally susceptible, some phages are generalists). Null
  calibration with random features on the real interaction matrix showed Fisher's yielded 25% false positives at p < 0.05
  (expected: 5%). Permuting host labels (1000 permutations) preserves this correlation structure and gives calibrated
  p-values (3% at the 5% threshold in null calibration).

#### Data dimensions

- **Interaction panel**: 369 bacteria × 96 phages = 35,424 pairs (35,266 resolved: 9,720 lytic, 25,546 non-lytic;
  158 unresolved pairs excluded from enrichment).
- **RBP PHROGs**: 43 unique across 97 phages, 32 present in ≥2 phages (used in enrichment).
- **Anti-defense PHROGs**: 13 unique, 12 present in ≥2 phages.
- **OMP receptor features**: 107 (receptor, cluster) pairs with ≥5 bacteria across 12 receptor proteins.
- **LPS core types**: 5 types (R1=207, R3=61, K12=43, R4=35, R2=22 bacteria; No_waaL excluded).
- **Defense subtypes**: 77 subtypes with ≥5 bacteria (of 137 total).

#### Analysis 1: RBP PHROG × OMP receptor variant clusters

- **3,424 tests** (32 PHROGs × 107 receptor clusters), **380 significant** (11.1%).

#### Analysis 2: RBP PHROG × LPS core type

- **160 tests** (32 PHROGs × 5 LPS types), **24 significant** (15.0%).

#### Analysis 3: Anti-defense PHROG × defense system subtypes

- **924 tests** (12 PHROGs × 77 defense subtypes), **27 significant** (2.9%).

#### Caveats

- **Duplicate PHROG profiles**: The 32 RBP PHROGs reduce to ~25 unique phage-carrier patterns (e.g., 136/15437/4465/9017
  always co-occur, as do 1002/1154/967/972 and 2097/4277). The 380 significant associations are not 380 independent
  biological discoveries — TL03 should collapse identical PHROG profiles before building features.
- **P-value resolution**: With 1000 permutations, 313/380 significant OMP hits are at the p-value floor (0.001). The
  BH boundary is partly quantized: 95 hits fall in the BH 0.03–0.07 zone, and ~65 borderline significant hits could
  flip with more permutations. The 314 floor hits (where ≤1 permutation exceeded the observed statistic) are robust.
  For the screening purpose of this step (feeding candidate pairs to TL03), this resolution is sufficient. TL03 should
  not treat the counts or rankings from this screen as precise.
- **Residual confounding**: The permutation test conditions on phage feature and permutes host labels, but does not
  control for host phylogenetic lineage or correlated feature blocks. Some associations may reflect lineage correlation
  rather than specific molecular interactions. The anti-defense results (2.9% significance) are particularly susceptible
  — generic methyltransferases mapping to diverse defense subtypes are more plausibly explained by lineage than by
  specific evasion.

#### Implications for TL03/TL04

- **TL03 (RBP-receptor features)**: 380 significant PHROG-receptor candidate associations (after controlling for phage
  main effects and matrix correlation via host-label permutation). TL03 should collapse duplicate PHROG profiles and
  use enrichment odds ratios as candidate feature weights — not treat these as confirmed molecular mechanisms.
- **TL04 (defense evasion features)**: 27 significant anti-defense × defense associations. The lower rate and caveats
  about annotation specificity suggest these should be treated as weaker candidates than the RBP-receptor features.
- **No escape hatch needed**: With 380 significant associations (~285 unique after collapsing duplicate PHROG carrier
  profiles), enrichment-based features are viable for TL03.

#### Non-protein host factors considered

Beyond OMP protein receptors and defense systems, several non-protein surface structures affect phage adsorption:

- **LPS core type** (R1-R4, K12) — available in Track C, will be tested in TL02 enrichment (RBP PHROGs x LPS type).
- **K-antigen / polysaccharide capsule** — Track C has Klebsiella capsule type but 94% missing. Sparse signal.
- **O-antigen type** — determines phage adsorption for some phages. Available in Track C surface features.
- **Phase variation** — stochastic receptor expression switching (e.g., FimH, Ag43). Cannot be captured from genome
  alone — inherently epigenetic. Acknowledged as a limitation.
- **Phage depolymerases** — enzymes that degrade capsule/LPS. Pharokka found 18 "polysaccharide chain length
  determinant" genes but cannot distinguish capsule-type targets. Deferred to Future note in project.md.

#### Next steps

1. **Build a reusable enrichment module** (`annotation_interaction_enrichment.py`) that takes any (phage feature matrix,
   host feature matrix, interaction matrix) triple and produces a Fisher's exact test enrichment table with odds ratios,
   p-values, and BH-corrected significance. This module will be used by both TL02 and TL03.
2. **Run enrichment analysis on three pairings:**
   - RBP PHROGs (43) x OMP receptor variants (22) — core RBP-receptor signal
   - RBP PHROGs (43) x LPS core type (~5 types) — tail spikes that bind LPS directly
   - Anti-defense gene PHROGs x defense system subtypes (82) — for TL03
3. **TL02:** Use the significant PHROG-receptor associations to build pairwise features. For each phage-host pair:
   does the phage carry an RBP PHROG that is significantly associated with lysis of hosts carrying this receptor? The
   enrichment odds ratios become the feature weights.
4. **TL03:** Same logic for anti-defense genes x host defense systems.
5. **Depolymerase annotation (deferred):** Pharokka annotations are too coarse for capsule-depolymerase matching
   (18 "polysaccharide chain length determinant" hits, no capsule-type specificity). Dedicated tools exist (DepoScope,
   DePP, PDP-Miner) but none predict capsule-type specificity — they only classify binary depolymerase yes/no. Running
   Pfam/InterPro on tail spike sequences to get glycosyl hydrolase family would be more informative. Defer until
   enrichment analysis shows whether capsule features (Track C has LPS core type + Klebsiella capsule with 94% missing)
   carry enough signal to justify the effort.

### 2026-03-29: TL03 Mechanistic RBP-receptor compatibility features from annotations

#### Executive summary

Built `build_mechanistic_rbp_receptor_features.py`, a Track L step that turns TL02 enrichment hits plus Pharokka RBP
PHROG annotations into a pairwise feature matrix for the full 369×96 panel. The builder collapses the 32 panel RBP
PHROGs to 25 unique carrier profiles before feature construction, then emits two blocks: (1) 25 direct phage-level
collapsed-profile features copied onto each pair row, and (2) 302 weighted pairwise compatibility features where the
value is `lysis_rate_diff` when the phage carries the collapsed profile and the host carries the enriched OMP/LPS
feature, else 0. The output is joinable on `pair_id` / `bacteria` / `phage`.

#### What was implemented

- `lyzortx/pipeline/track_l/steps/build_mechanistic_rbp_receptor_features.py`: TL03 builder. It:
  - bootstraps Track A labels and TL02 enrichment outputs if they are missing from a fresh checkout,
  - rebuilds the panel RBP PHROG matrix from cached Pharokka TSVs,
  - collapses duplicate PHROG carrier profiles,
  - collapses duplicate significant enrichment hits onto those profiles,
  - writes the feature CSV, column metadata, profile metadata, sanity-check CSV, and manifest.
- `lyzortx/pipeline/track_l/run_track_l.py`: added `--step rbp-features` to run TL03 directly.
- `lyzortx/tests/test_track_l_mechanistic_rbp_receptor_features.py`: 5 unit tests covering duplicate-profile
  collapsing, collapsed-association merging, pairwise feature emission, curated-vs-Pharokka sanity rows, and a
  minimal end-to-end `main()` run.

#### Data dimensions and outputs

- **Panel pairs**: 35,424 bacteria-phage pairs (369 bacteria × 96 phages).
- **Collapsed direct phage block**: 25 columns from the 32 panel RBP PHROGs.
- **Collapsed pairwise block**: 302 weighted columns total.
  - 286 OMP receptor-cluster compatibility columns.
  - 16 LPS core compatibility columns.
- **Duplicate co-occurrence groups confirmed in the real data**:
  - `1002/1154/967/972` (11 carrier phages)
  - `136/15437/4465/9017` (6 carrier phages)
  - `2097/4277` (14 carrier phages)

#### Design decisions

- **Collapse before feature construction**: The builder collapses PHROGs by their 96-phage carrier vector before using
  TL02 outputs. This removes exact duplicates from both the direct phage block and the weighted pairwise block.
- **Use `lysis_rate_diff`, not odds ratio**: TL03 uses the same conditional effect size as TL02's test statistic
  (`lysis_rate_both - lysis_rate_phage_only`) because it is bounded, signed, and better behaved than `inf`/0 odds
  ratios from sparse contingency tables.
- **Bootstrap missing prerequisites**: Because generated outputs are absent on fresh CI checkouts, the TL03 builder
  regenerates `label_set_v1_pairs.csv` and the TL02 enrichment CSVs when the default inputs are missing, then fails
  loudly if those rebuilds do not produce the expected files.
- **Keep TL03 separate from TE01**: The new mechanistic block lives under Track L outputs and does not overwrite the
  existing Track E curated genus/subfamily lookup features. That avoids silently changing downstream experiments that
  already depend on TE01.

#### Sanity check against `RBP_list.csv`

- **Any-RBP presence agreement**: 74/96 phages.
- **Both curated and Pharokka call RBP present**: 74 phages.
- **Curated-only**: 15 phages.
- **Pharokka-only**: 7 phages.
- **Curated positives**: 89 phages.
- **Pharokka positives**: 81 phages.

The disagreement pattern is asymmetric but still useful as a sanity check. Pharokka recovers most curated positives,
but misses a minority of curated fiber/spike calls; conversely, Pharokka calls RBP-like tail proteins in seven phages
that are `NA` in the curated list (`536_P1`, `536_P6`, `536_P7`, `536_P9`, `DIJ07_P1`, `DIJ07_P2`, `LF31_P1`). This
supports using the annotation-derived block as an automated feature source while treating the curated list as an
imperfect reference rather than ground truth.

#### Interpretation

- **The mechanistic block is viable**: TL02's 404 significant RBP×OMP/LPS hits collapse to 302 unique pairwise
  features after duplicate-profile merging, so TL03 produces a non-trivial mechanistic feature space without label
  leakage at feature-build time.
- **Most signal is OMP-linked, not LPS-linked**: 286/302 collapsed pairwise features are OMP-based, with OmpC the
  single largest receptor family (42 collapsed columns). Only 16 collapsed LPS columns survive, mostly `LPS_R1`.
- **The direct phage block is sparse but not empty**: 78/96 phages carry at least one collapsed profile; the maximum is
  5 profiles on a single phage, and 43 phages carry exactly 1.
- **Curated-vs-Pharokka mismatch is tolerable for this use case**: the goal is not to recreate `RBP_list.csv` exactly,
  but to derive a reproducible genome-only feature block. The 74/96 agreement rate is good enough for a sanity check,
  but the disagreement counts should be acknowledged if TL05 shows surprising behavior.

### 2026-03-29: TL04 Mechanistic defense-evasion features from annotations

#### Executive summary

Built `build_mechanistic_defense_evasion_features.py`, a Track L step that turns TL02 anti-defense enrichment hits plus
Pharokka anti-defense PHROG annotations into a pairwise feature matrix for the full 369×96 panel. The builder
collapses the 12 panel anti-defense PHROGs to 11 unique carrier profiles, then emits a 36-column experimental block:
11 direct phage-level anti-defense profile indicators plus 25 weighted pairwise defense-evasion features. This output
is joinable on `pair_id` / `bacteria` / `phage` and is explicitly marked as an experimental candidate block for TL05,
not a confirmed mechanistic signal.

#### What was implemented

- `lyzortx/pipeline/track_l/steps/build_mechanistic_defense_evasion_features.py`: TL04 builder. It:
  - bootstraps Track A labels and the TL02 anti-defense enrichment CSV when they are missing from a fresh checkout,
  - rebuilds the panel anti-defense PHROG matrix from cached Pharokka TSVs,
  - collapses duplicate PHROG carrier profiles before feature construction,
  - writes the feature CSV, column metadata, profile metadata, and manifest.
- `lyzortx/pipeline/track_l/run_track_l.py`: added `--step defense-features` to run TL04 directly.
- `lyzortx/tests/test_track_l_mechanistic_defense_evasion_features.py`: 4 unit tests covering duplicate-profile
  collapsing, duplicate-association merging, pairwise feature emission, and a minimal end-to-end `main()` run with a
  realistic defense-subtype support threshold.

#### Data dimensions and outputs

- **Panel pairs**: 35,424 bacteria-phage pairs (369 bacteria × 96 phages).
- **Collapsed direct phage block**: 11 columns from the 12 panel anti-defense PHROGs.
- **Collapsed pairwise block**: 25 weighted columns total.
- **Full TL04 block**: 36 feature columns plus `pair_id` / `bacteria` / `phage`.
- **Phage coverage**: 66/96 phages carry at least one collapsed anti-defense profile; the maximum is 4 profiles on a
  single phage, and 52 phages carry exactly 1.
- **Most common collapsed profiles**:
  - `ANTIDEF_PHROG_2568` (`ocr-like anti-restriction`) on 31 phages.
  - `ANTIDEF_PHROG_757` (`DNA methyltransferase`) on 13 phages.
  - `ANTIDEF_PHROG_111` (`DNA methyltransferase`) on 12 phages.

#### Design decisions

- **Collapse before feature construction**: TL04 applies the same duplicate-profile logic as TL03 so exact co-carried
  anti-defense PHROGs do not inflate either the direct phage block or the pairwise block.
- **Use `lysis_rate_diff`, not odds ratio**: TL04 uses TL02's bounded conditional effect size as the feature weight for
  the same reasons as TL03: sparse anti-defense tables would otherwise produce unstable `inf`/0 odds ratios.
- **Carry the caveat into the metadata, not just the notebook**: the feature metadata, profile metadata, and manifest
  all mark this block as `experimental_candidate` so TL05 can include it as a separate optional block without confusing
  it with the stronger TL03 RBP-receptor features.
- **Bootstrap missing prerequisites, but fail loudly on real gaps**: on a fresh checkout the builder regenerates Track A
  labels and TL02 enrichment outputs if the default generated files are absent; custom paths must already exist.

#### Interpretation

- **The defense-evasion block is viable but materially weaker than TL03**: TL02's 27 significant anti-defense × defense
  hits collapse only slightly, to 25 unique weighted pairwise features after duplicate-profile merging. That is enough
  to evaluate in TL05, but it is an order of magnitude smaller than the 302-column TL03 pairwise block.
- **The signal is concentrated in a few defense families**: 8/25 pairwise columns target `Thoeris_II`, 4 target
  `CAS_Class1-Subtype-I-F`, and only 9 distinct defense subtypes appear at all. The largest single weights are
  `ANTIDEF_PHROG_363 × Thoeris_II` (`0.3392`) and `ANTIDEF_PHROG_4452 × Thoeris_II` (`0.3138`).
- **Generic methyltransferase annotations still dominate the phage side**: among the most common profiles, PHROGs 111,
  116/67, 1530, 2226, 56, and 757 map to `DNA methyltransferase` or `SAM-dependent methyltransferase` annotations.
  This is exactly the caveat from TL02: some apparent defense-evasion signal may reflect broad anti-restriction or
  lineage correlation rather than subtype-specific evasion.
- **The block is sparse enough to stay optional**: at most 6 weighted TL04 columns are non-zero on any single pair row,
  and at most 4 direct anti-defense profiles are present on any phage. That sparsity is compatible with evaluating TL04
  as a separate optional block in TL05 rather than folding it into the main feature surface by default.

### 2026-03-29: TL05 Retrain v1 model with mechanistic pairwise features and measure lift

#### Executive summary

Retrained the locked v1 LightGBM on the ST03 holdout split with TG01 hyperparameters and evaluated four arms
separately: the current locked defense + phage-genomic baseline, +TL03 only, +TL04 only, and +TL03+TL04. TL04 was the
best mechanistic add-on. It improved all three holdout metrics relative to the retrained baseline arm, so TL05 writes a
proposed lock config for `defense + phage_genomic + TL04`.

#### What was implemented

- `lyzortx/pipeline/track_l/steps/retrain_mechanistic_v1_model.py`: new TL05 retrain/eval step. It bootstraps TG01,
  TL03, and TL04 when the committed generated outputs are absent, retrains each arm with the fixed TG01 LightGBM
  hyperparameters, reports holdout metrics and deltas, and runs SHAP on the best mechanistic arm.
- `lyzortx/pipeline/track_l/run_track_l.py`: added `--step retrain-mechanistic-v1`.
- `lyzortx/tests/test_track_l_retrain_mechanistic_v1_model.py`: tests for arm construction, feature-block
  classification, proposal selection, and an end-to-end mocked TL05 run.
- `lyzortx/tests/test_track_l_run_track_l.py`: confirms the Track L runner dispatches the new TL05 step.

#### Holdout results

- Locked baseline `defense + phage_genomic`: AUC `0.835466`, top-3 `0.892308`, Brier `0.146153`.
- `+TL03` only: AUC `+0.000310`, top-3 `+0.046154`, Brier `-0.001684`.
- `+TL04` only: AUC `+0.003114`, top-3 `+0.015384`, Brier improvement `+0.002134`.
- `+TL03+TL04`: AUC `-0.000240`, top-3 `+0.015384`, Brier improvement `-0.001087`.

#### Interpretation

- TL03 is useful for ranking, but it does not improve calibration.
- TL04 is the best mechanistic block overall and is the only arm that clearly beats the locked baseline on AUC and
  Brier.
- The combined TL03+TL04 arm does not beat TL04 alone, so TL03 should stay optional rather than forced into the lock.
- TL05 therefore proposes a new v1 config that keeps `defense + phage_genomic` and adds TL04, while leaving TL03
  uncoupled.

#### SHAP check

- SHAP on the TL04 arm surfaced TL04 pairwise features in the global importance table and in per-pair explanations.
- The first TL04 feature appears at rank 114 overall (`tl04_pair_profile_005_x_defense_sanata_weight`), and several top
  pair explanations include `tl04_pair_profile_011_x_defense_rloc_weight` or
  `tl04_pair_profile_011_x_defense_cas_class1_subtype_i_f_weight`.
- The mechanistic block contributes signal, but phage-genomic features still dominate the ranking surface.

#### Caveat

- The committed `lyzortx/pipeline/track_g/v1_feature_configuration.json` still records the earlier TG09 lock metrics.
  TL05 retrained the same baseline arm on the current code path and used that live retrain as the comparison point for
  all deltas.

### 2026-03-29: TL06 Persist fitted transforms for novel-organism feature projection

#### Executive summary

Persisted the fitted Track D phage SVD transform and the Track C defense-subtype column mask as joblib artifacts, then
added reusable projection helpers for novel phage genomes and novel Defense Finder outputs. The projection helpers
reconstruct the same feature layout used by the panel tables: 24 phage SVD coordinates plus GC content and genome
length, and 79 retained defense-subtype indicators plus 3 derived defense summary columns.

#### What was implemented

- `lyzortx/pipeline/track_d/steps/build_phage_genome_kmer_features.py`: saves the fitted `TruncatedSVD` object to
  `phage_genome_kmer_svd.joblib` alongside the existing k-mer feature CSV and records the artifact path in the
  manifest.
- `lyzortx/pipeline/track_c/steps/build_v1_host_feature_pair_table.py`: saves the defense-subtype column mask to
  `defense_subtype_column_mask.joblib`, including the variance-filter thresholds, source subtype order, retained
  feature order, and derived-column list.
- `lyzortx/pipeline/track_l/steps/novel_organism_feature_projection.py`: adds
  `project_novel_phage(fna_path, svd_path)` and
  `project_novel_host(defense_finder_output_path, column_mask_path)` for novel-organism projection.
- `lyzortx/tests/test_track_l_novel_organism_feature_projection.py`: real-data round-trip checks against one panel
  phage and one panel host.

#### Real-data counts

- TD02 input genomes: 97 phage FASTA files.
- TD02 fitted embedding width: 24 dimensions.
- TC01 retained defense-subtype columns: 79.
- TC01 full host feature width: 82 columns after adding defense diversity, CRISPR presence, and Abi burden.

#### Round-trip sanity check

- Phage check used `409_P1` and matched the precomputed panel row within floating-point tolerance.
- Host check used `001-023` and matched the precomputed panel row within floating-point tolerance.
- The new helpers also accept the persisted artifacts directly, so TL07 can project novel inputs without rebuilding the
  panel transforms.

#### Interpretation

- This closes the training/prediction gap for the sequence-derived feature blocks: the panel fit is now reusable at
  inference time instead of being stranded in CSV-only outputs.
- Keeping the saved mask and SVD artifact next to their source tables preserves provenance and makes TL07 a straight
  projection step rather than a re-fit.

### 2026-03-30: TL07 Build Defense Finder runner for novel E. coli genomes

#### Executive summary

Implemented a Track L runner that takes a novel E. coli assembly FASTA, predicts CDS with Pyrodigal, runs Defense
Finder, and projects the result into the locked host-defense feature space used by the model. The runner also rebuilds
the TL06 defense mask from the committed panel subtype CSV when the generated joblib is absent, which is necessary in
fresh CI checkouts because generated artifacts are gitignored. A live smoke test on the public MG1655 reference genome
 (`NC_000913.3`) detected 9 defense systems and produced the expected 82-column projected feature row: 79 retained
 subtype indicators plus 3 derived defense summary features.

#### What was implemented

- Added `lyzortx/pipeline/track_l/steps/run_novel_host_defense_finder.py`.
  - Accepts one assembly FASTA and writes:
    - predicted proteins (`*.prt`)
    - raw retained-subtype counts (`defense_finder_subtype_counts.csv`)
    - projected model-ready host-defense features (`novel_host_defense_features.csv`)
    - a provenance manifest (`novel_host_defense_manifest.json`)
  - Uses explicit Pyrodigal preprocessing before Defense Finder instead of Defense Finder's nucleotide-input shortcut.
  - Rebuilds `defense_subtype_column_mask.joblib` from
    `data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv` when the TL06 artifact is missing.
- Added `lyzortx/tests/test_track_l_run_novel_host_defense_finder.py`.
  - Covers subtype aggregation from Defense Finder system rows.
  - Verifies mask regeneration when TL06 generated outputs are absent.
  - Confirms the end-to-end projected row has the exact training column names.
- Updated environment setup so the pipeline is runnable in `phage_env`.
  - `requirements.txt`: pinned `mdmparis-defense-finder==2.0.1`
  - `environment.yml`: added `hmmer=3.4`

#### Live smoke test

- Downloaded the public MG1655 reference FASTA from NCBI to `.scratch/tl07/NC_000913.3.fna`.
- Ran:
  `conda run -n phage_env python -m lyzortx.pipeline.track_l.steps.run_novel_host_defense_finder .scratch/tl07/NC_000913.3.fna --bacteria-id ecoli_k12_mg1655 --output-dir lyzortx/generated_outputs/track_l/novel_host_defense_finder/ecoli_k12_mg1655 --models-dir .scratch/defense_finder_models --force-run`
- Output paths:
  - `lyzortx/generated_outputs/track_l/novel_host_defense_finder/ecoli_k12_mg1655/NC_000913.3_defense_finder_systems.tsv`
  - `lyzortx/generated_outputs/track_l/novel_host_defense_finder/ecoli_k12_mg1655/novel_host_defense_features.csv`
  - `lyzortx/generated_outputs/track_l/novel_host_defense_finder/ecoli_k12_mg1655/novel_host_defense_manifest.json`

#### Observed counts

- Assembly length: `4,641,652` nt
- Pyrodigal CDS predictions: `4,319`
- Defense Finder systems detected: `9`
- Detected retained training subtypes: `6`
  - `CAS_Class1-Subtype-I-E`
  - `Hachiman`
  - `MazEF` (2 systems)
  - `RM_Type_I`
  - `RM_Type_IV` (2 systems)
  - `RnlAB`
- Detected but not projected because absent from the retained TL06 mask: `Lit`
- Final projected host-defense width: `82` columns (`79` retained subtype indicators + `3` derived features)
- Nonzero projected features for MG1655: `8`

#### External references used for the implementation

- Defense Finder PyPI documentation:
  `https://pypi.org/project/mdmparis-defense-finder/`
  Quote: "If input is a nucleotide fasta, DefenseFinder uses Pyrodigal to annotate the CDS."
- In this environment, `mdmparis-defense-finder==2.0.1` crashed on the documented nucleotide-input path with
  `AttributeError: 'str' object has no attribute 'decode'` while reading the FASTA headers. TL07 therefore runs
  Pyrodigal explicitly and passes the generated protein FASTA to Defense Finder instead of relying on the broken
  shortcut.
- CasFinder model-definition compatibility check:
  `https://raw.githubusercontent.com/macsy-models/CasFinder/3.1.0/definitions/CAS_Class2-Subtype-II-B.xml`
  Quote: `<model ... vers="2.0">`
- Newer CasFinder release for comparison:
  `https://raw.githubusercontent.com/macsy-models/CasFinder/3.1.1/definitions/CAS_Class2-Subtype-II-B.xml`
  Quote: `<model ... vers="2.1">`
- Defense Finder 2.0.1 rejected CasFinder 3.1.1 at runtime with:
  `The model definition CAS_Class2-Subtype-II-B.xml has not the right version. version supported is '2.0'.`
  TL07 therefore pins the installed models to `defense-finder-models==2.0.2` and `CasFinder==3.1.0`.

#### Interpretation

- The new runner closes the host side of the generalized inference path: a novel assembly can now be converted into the
  same defense-feature contract used during training without requiring the original 404-strain pair table.
- Rebuilding the TL06 mask from committed panel inputs keeps the pipeline fail-fast but CI-compatible: missing
  generated artifacts are regenerated explicitly rather than silently treated as empty data.
- The MG1655 smoke test confirms the workflow is biologically nontrivial on a real E. coli genome and that the output
  width matches the training schema exactly.

### 2026-03-30: TL08 Generalized inference bundle for arbitrary genomes

#### Executive summary

Built a deployable genome-only inference path for Track L. The new bundle builder trains a LightGBM model on the
canonical panel labels using only features that are available for arbitrary genomes at prediction time: host defense
subtypes from TL07 and phage tetranucleotide embedding features from TL06. It saves a self-contained bundle containing
the fitted estimator, `DictVectorizer`, isotonic calibrator, copied TD02 SVD artifact, copied defense mask, and a
reference panel-prediction CSV. The new `infer(host_genome_path, phage_fna_paths, model_path)` function runs TL07 for
the host, projects each phage FNA through the saved SVD, scores the host × phage cross-product, applies isotonic
calibration, and returns a ranked DataFrame with columns `phage`, `p_lysis`, and `rank`.

#### What was implemented

- Added `lyzortx/pipeline/track_l/steps/build_generalized_inference_bundle.py`.
  - Rebuilds missing ST0.2 / ST0.3 / TD02 / TG01 defaults when invoked through the Track L runner.
  - Trains a genome-only LightGBM bundle using the locked TG01 hyperparameters and ST0.3 weighting/split contract.
  - Fits isotonic calibration on the designated non-holdout calibration fold.
  - Copies the phage SVD and defense mask next to the saved model so inference does not depend on generated panel
    directories elsewhere in the repo.
  - Writes `tl08_locked_panel_predictions.csv` for regression testing and auditability.
- Added `lyzortx/pipeline/track_l/steps/generalized_inference.py`.
  - Exposes `infer(host_genome_path, phage_fna_paths, model_path)`.
  - Calls the TL07 host runner, projects phage genomes with TL06, vectorizes the feature rows with the saved
    `DictVectorizer`, scores with the saved LightGBM model, calibrates with the saved isotonic regressor, and ranks by
    calibrated probability.
- Updated `lyzortx/pipeline/track_l/run_track_l.py` with a `generalized-inference-bundle` step.
- Added `lyzortx/tests/test_track_l_generalized_inference.py`.
  - Covers the row-merging contract for the training table.
  - Builds a real panel-derived bundle in a temp directory and verifies `infer(...)` reproduces the saved per-phage
    calibrated predictions for panel host `001-023` when given the same phage genomes.

#### Design decision

- The current locked v1 panel model includes feature blocks that are not available for arbitrary genomes on a fresh
  host/phage pair. TL08 therefore locks a separate genome-only deployment bundle rather than pretending the full
  panel-only metadata stack can be generalized. Training still uses the canonical panel labels, splits, and weights, but
  the saved inference artifact itself depends only on genome-derivable inputs.

#### Test notes

- The phage side of the integration test is fully real: it uses the committed panel FNA files and the saved TD02 SVD
  projection path.
- The host side stubs the TL07 external-tool boundary in the regression test because this repository does not ship the
  underlying panel host FASTA assemblies. The stub writes the exact projected defense-feature row for panel host
  `001-023`, which is sufficient to verify that TL08 reproduces the locked calibrated panel predictions once the host
  features are available.

#### Interpretation

- TL08 now provides the missing deployment contract for Track L: a single saved bundle plus raw genomes are enough to
  score arbitrary host-phage combinations with the model's calibrated ranking surface.
- Saving bundle-local copies of the SVD and defense mask eliminates hidden dependencies on gitignored generated outputs,
  which was the main operational blocker for fresh-clone inference.
- The integration test is honest about the repo's current data gap. We can verify the full inference math against panel
  predictions now, but a true end-to-end host-genome regression on a panel strain will require committing or
  regenerating local host assemblies in a follow-up task.

### 2026-03-30: TL09 Virus-Host DB positive-only validation of generalized inference

#### Executive summary

Built a Track L validation step that mines the live Virus-Host DB, selects assembly-backed _E. coli_ hosts from NCBI,
downloads the host assemblies plus associated phage genomes, and runs the TL08 genome-only inference bundle on each host
against the union of its known Virus-Host DB phages plus the 96 panel phages. The current Virus-Host DB snapshot is
substantially larger than the original plan estimate: after filtering to strain-level _E. coli_ hosts (`tax_id != 562`)
with RefSeq accessions, it contained **82 hosts**, **1,304 phage accessions**, and **1,323 unique positive pairs**.

The validation result is negative for the current genome-only model. On the 10 selected novel hosts plus 1 round-trip
panel host, the overall median predicted `P(lysis)` for known positive pairs was **0.0264**, far below both the panel
base rate (**0.2756**) and the matched random-pair median (**0.2043**). The median host-level positive rank percentile
was **0.235**, meaning the known positives typically ranked in the bottom quartile of each host's candidate set rather
than above the midpoint. The one panel-host round-trip comparison that was actually comparable through the saved TL08
reference table (`EDL933`) also showed poor agreement: median absolute probability delta **0.1595** and only **10/96**
panel-phage ranks identical.

#### What was implemented

- Added `lyzortx/pipeline/track_l/steps/validate_vhdb_generalized_inference.py`.
  - Downloads the live Virus-Host DB TSV and RefSeq assembly summary.
  - Filters to strain-level _E. coli_ hosts with phage RefSeq accessions.
  - Resolves best host assemblies from RefSeq and downloads host genomic FASTAs.
  - Downloads associated phage genomes from NCBI `nuccore` FASTA with explicit Entrez rate limiting and retry/backoff.
  - Builds or reuses the TL08 inference bundle, projects external hosts/phages, scores each host against the union of
    its known positives plus the 96 panel phages, and writes per-host plus aggregate validation outputs under
    `lyzortx/generated_outputs/track_l/vhdb_generalized_inference_validation/`.
  - Computes positive-only metrics and a panel-host round-trip comparison table.
- Refactored `lyzortx/pipeline/track_l/steps/generalized_inference.py`.
  - Added reusable helpers to load the TL08 runtime, project hosts, project phages, and score projected feature rows.
  - Kept the public `infer(host_genome_path, phage_fna_paths, model_path)` contract unchanged.
- Updated `lyzortx/pipeline/track_l/run_track_l.py` with a `validate-vhdb-generalized-inference` step.
- Added `lyzortx/tests/test_track_l_vhdb_generalized_inference.py` covering host-name matching, assembly prioritization,
  cohort selection, and the positive-only metric calculations.

#### Cohort mining and selection

- **Live Virus-Host DB filter result**:
  - 82 strain-level _E. coli_ hosts
  - 1,304 phage accessions
  - 1,323 unique positive host-phage pairs
- **Hosts with >=5 associated phages and exact RefSeq assembly match**: 24
- **Novel-host validation cohort (10 hosts)**:
  - `E. coli O78` (5 phages)
  - `E. coli str. K-12 substr. DH10B` (5)
  - `E. coli BW25113` (7)
  - `E. coli CFT073` (7)
  - `E. coli O104:H4` (7)
  - `E. coli O157` (11)
  - `E. coli ATCC 25922` (17)
  - `E. coli O26:H11` (19)
  - `E. coli O121:H19` (52)
  - `E. coli O145:H28` (60)
- **Round-trip panel cohort actually comparable through the TL08 saved reference table**:
  - `EDL933` only. Virus-Host DB also contains `LF82`, `55989`, `536`, and `BL21` aliases, but the current
    `tl08_locked_panel_predictions.csv` artifact only had saved rows for `EDL933`, so the other panel-host examples
    could not be compared against a panel-path reference without rebuilding a different reference artifact.

To keep CI/runtime bounded while still satisfying the "at least 10 novel hosts" acceptance criterion, the selection
policy preferred **complete-genome assemblies with the smallest qualifying positive-set sizes first**, rather than
trying to score the largest hosts such as _E. coli_ C or MG1655 with hundreds of associated phages each.

#### Validation outputs and metrics

- **Evaluated positives**: 192 host-phage pairs across 11 hosts (10 novel + 1 round-trip panel host).
- **Candidate set sizes per host**: 98 to 156 phages (known Virus-Host DB positives for that host + 96 panel phages).
- **Overall metrics**:
  - Panel base rate from ST0.2 resolved rows: `0.2756`
  - Median predicted `P(lysis)` for known positives: `0.0264`
  - Median predicted `P(lysis)` for matched random candidate pairs: `0.2043`
  - Median host-level positive rank percentile: `0.2353`
  - Hosts with median positive rank percentile above `0.5`: `3 / 11`
  - Hosts with median positive `P(lysis)` above the panel base rate: `4 / 11`
- **Best novel-host slices by positive median `P(lysis)`**:
  - `E. coli ATCC 25922`: `0.5783`
  - `E. coli CFT073`: `0.4188`
  - `E. coli O157`: `0.3460`
  - `E. coli DH10B`: `0.2824`
- **Worst slices**:
  - `E. coli O121:H19`: `0.0264`
  - `E. coli O145:H28`: `0.0264`
  - `E. coli O26:H11`: `0.0264`
  - `EDL933` round-trip positives: `0.0264`

#### Round-trip sanity check

- Only `EDL933` could be compared directly against the saved TL08 panel-prediction artifact.
- On the 96 panel phages:
  - median absolute probability delta vs the saved TL08 panel-path predictions: `0.1595`
  - max absolute probability delta: `0.3183`
  - identical rank positions: `10 / 96`

This is not a successful round-trip. The genome-derived host projection for the downloaded `EDL933` assembly does not
recover the saved panel prediction surface with useful fidelity.

#### Interpretation

- **The current TL08 genome-only model does not generalize to this Virus-Host DB external-positive cohort.** The core
  validation expectations were missed in the wrong direction: known positives scored below the panel base rate, below
  matched random candidate pairs, and below the candidate-set midpoint by rank.
- The failure is not uniform. A few hosts (`ATCC 25922`, `CFT073`, `O157`, `DH10B`) showed some signal, but the cohort
  as a whole is dominated by poor ranking and collapsed probabilities near the isotonic floor.
- The negative result is biologically plausible. TL08 only uses host defense features plus phage tetranucleotide
  embeddings. It does **not** use receptor, surface, or annotation-derived mechanistic features at inference time, so
  external host-range positives that depend strongly on adsorption biology are not well captured.
- The round-trip miss on `EDL933` suggests the host-genome projection path itself is also part of the problem, not just
  external cohort shift. Differences between the downloaded assembly's defense profile and the internal panel row appear
  large enough to materially change the prediction surface.

#### Limitations

- This is a **positive-only** validation. There are no authoritative negatives for the Virus-Host DB cohort, so **AUC,
  ROC, PR curves, and top-3 hit rate cannot be computed honestly** here.
- The round-trip check was weaker than originally hoped because only `EDL933` overlapped with the saved TL08 reference
  prediction artifact. `LF82`, `55989`, `536`, and `BL21` were present in Virus-Host DB but not in the saved reference
  table generated by the current TL08 bundle build.
- The selection policy intentionally avoided the largest hosts (for example _E. coli_ C, MG1655, K-12, O157:H7) to keep
  runtime tractable in CI. That tradeoff preserves coverage of 10 novel hosts but does not exhaust the external cohort.

#### Conclusion

TL09 should be treated as a failed external validation for the current genome-only deployment path, not as supporting
evidence. If Track L continues, the next technically honest move is to revisit the deployable feature set rather than
trying to polish this evaluation. The obvious direction is to test whether the annotation-derived mechanistic blocks
from TL03/TL04 can be made available at inference time for arbitrary genomes, because the current defense + k-mer-only
bundle is not carrying enough cross-cohort signal.

### 2026-03-30: Replan — TL02 enrichment holdout leak identified

#### Executive summary

Post-completion review of Track L found that TL02's enrichment analysis uses the full 369×96 interaction matrix
including ST03 holdout strains. The enrichment weights therefore encode holdout outcomes, leaking test information into
TL03/TL04 features. TL10 has been added to fix this. TL03/TL04/TL05 will need re-evaluation after the fix.

#### Bug details

`run_enrichment_analysis.py` loads `label_set_v1_pairs.csv` (all 369 bacteria) at line 394 and builds the bacteria list
from all rows at lines 409–423 with no holdout filtering. `compute_enrichment()` at line 96 has no holdout parameter.
The TL02 acceptance criteria explicitly said "each analysis uses the full interaction matrix" — the implementing agent
followed the criteria literally.

The permutation test itself is statistically sound (host-label permutation, phage conditioning, BH correction, null
calibration at 3% FPR vs Fisher's 25%). Only the input data selection is wrong.

#### Impact on downstream tasks

- **TL03/TL04**: Feature weights are derived from enrichment associations computed on all 369 bacteria including holdout.
  The weights may partially encode holdout-strain patterns.
- **TL05**: Holdout metric deltas (+0 to +4.6pp top-3 across arms) were already within noise on 65 holdout strains.
  The local rerun produced different arm rankings than CI, confirming these deltas are not statistically robust. The
  enrichment leak adds a further validity concern on top of the power concern.
- **TL06–TL09**: The generalized inference bundle (TL08) uses only defense + k-mer features, not TL03/TL04 enrichment
  features. TL09's external validation failure is therefore NOT caused by this leak — it is caused by the genome-only
  feature set lacking compatibility signal. The leak is a separate problem.

#### What TL10 fixes

TL10 adds holdout filtering to `run_enrichment_analysis.py` (load ST03 split assignments, exclude holdout bacteria
before calling `compute_enrichment()`). It does not modify the enrichment module itself. After TL10, TL03/TL04/TL05
will need re-evaluation to determine whether the enrichment features provide any honest lift.

### 2026-03-30: TL10 Fix enrichment holdout leak

#### Executive summary

TL02 now excludes the 65 ST0.3 holdout bacteria before building any enrichment matrices, so the three PHROG x host
feature analyses run on the 304-bacteria non-holdout panel instead of the full 369-bacteria label set. The permutation
test itself was left unchanged. Updated CSVs were written to `lyzortx/generated_outputs/track_l/enrichment/`.

#### What was changed

- `lyzortx/pipeline/track_l/steps/run_enrichment_analysis.py`: added `--st03-split-assignments-path`, loaded the ST0.3
  split assignments, excluded holdout bacteria before assembling the interaction matrix and host feature matrices, and
  failed fast if the holdout set was inconsistent with the label table.
- `lyzortx/tests/test_track_l_run_enrichment_analysis.py`: verifies holdout bacteria are absent from the matrices
  passed to `compute_enrichment()`.
- `lyzortx/tests/test_annotation_interaction_enrichment.py`: adds a null-calibration regression test on random binary
  features and asserts BH-significant FPR stays below 10% at alpha 0.05.

#### Rerun results

- Panel size: `369` bacteria -> `304` after excluding `65` holdouts.
- Resolved pairs: `35,266` -> `29,031`.
- Lytic resolved pairs: `9,720` -> `8,149`.
- OMP host features: `107` -> `96`.
- Defense subtype features: `77` -> `70`.
- Significant enrichment hits:
  - RBP PHROG x OMP receptor: `380` -> `393` (`+13`)
  - RBP PHROG x LPS core: `24` -> `27` (`+3`)
  - Anti-defense PHROG x defense subtype: `27` -> `19` (`-8`)

#### Interpretation

The leaked holdout rows were enough to move the enrichment weights and the hit counts, but not enough to dominate the
signal completely. TL03, TL04, and TL05 all need to be re-evaluated against the holdout-excluded enrichment CSVs.

### 2026-03-30: Replan follow-up — acceptance-criteria hardening for TL11-TL14

#### Executive summary

After the initial replan, I reviewed the actual TL03-TL09 PRs plus the failed TL03 Codex implement run to separate
"bug in the implementation" from "task definition let the wrong thing count as done." The follow-on Track L tasks now
encode the lessons directly. The next pass must prove provenance of holdout-clean inputs, quantify uncertainty before
locking a mechanistic arm, make the deployable bundle honest about missing feature blocks, and require a real
round-trip cohort before claiming external validation says anything useful.

#### Why the original criteria were too weak

- **TL03/TL04**: good builders, but their downstream plan path never required manifests or regression tests proving that
  the rebuilt features came from holdout-clean enrichment outputs rather than stale leaked CSVs.
- **TL05**: the task asked for metric deltas and SHAP, but did not force bootstrap uncertainty or a decision rule for
  when a new lock is actually justified. That left room for a noisy +1 to +4 pp top-3 bump to look more decisive than
  it was.
- **TL08**: the bundle task required working inference plumbing, not an honest accounting of which training-time
  signals were dropped when moving to deployable genome-only inference.
- **TL09**: the validation task required scoring 10 novel hosts, but did not require pre-materializing the exact cohort
  or proving ahead of time that the round-trip panel hosts were actually comparable through saved reference artifacts.

#### What the new acceptance checks enforce

- **TL11**: rebuilt TL03/TL04 outputs must carry manifests listing the exact enrichment inputs, split file, excluded
  holdout IDs, and output hashes. Reusing pre-TL10 enrichment outputs is an explicit failure case.
- **TL11**: fixtures now need negative cases where the host lacks the target receptor/defense feature, and the emitted
  pairwise weight must be exactly zero. This closes the same kind of weak-fixture gap that PR review had to catch in
  TL04.
- **TL12**: the mechanistic re-evaluation must recompute all four arms from the same live code path and report
  bootstrap CIs. A new lock is forbidden unless the chosen arm beats the noise band and does not materially degrade the
  other metrics.
- **TL13**: the deployable bundle must begin with a feature-parity audit table and fail if it silently drops or
  substitutes feature blocks. Hardcoded repo-root paths and hidden dependencies on gitignored generated outputs are now
  explicit failures.
- **TL14**: the external cohort must be saved before scoring, keeping `positive_pair_count`, `unique_phage_count`,
  `host_count`, and `candidate_set_size` as separate validated quantities. Multi-host round-trip comparison is now a
  gate rather than a best-effort note in the limitations section.

#### Review-derived lessons encoded into the next tasks

- **TL04 PR review** caught that a test fixture with identical defense profiles could not prove absence behavior. TL11
  now requires a true zero-feature case.
- **TL07 PR review** caught that cache short-circuiting happened too late, after expensive preprocessing work. Future
  runtime tasks should treat "cache hit skips heavy work" as an acceptance requirement, not a review nit.
- **TL08 PR review** caught that the bundle still relied on hardcoded panel paths. TL13 now treats bundle-relative
  artifact resolution as part of correctness, not cleanup.
- **TL09 PR review** caught parser realism, runtime reuse, and cohort-count semantics. TL14 now requires those
  distinctions in the acceptance criteria themselves.

#### Codex run note

The first TL03 Codex implement run failed in CI before task code executed because `conda env create -f environment.yml`
was unsatisfiable on the runner (`openjdk` solver conflict). That is separate from the scientific mistakes above, but it
supports tightening future Track L tasks so environment solvability is validated explicitly whenever new bioinformatics
dependencies are part of the work.

### 2026-03-30: TL11 Rebuild TL03/TL04 mechanistic blocks from holdout-clean enrichment

#### Executive summary

Rebuilt the TL03 and TL04 mechanistic feature blocks from the TL10 holdout-excluded enrichment outputs and added
provenance checks so stale pre-TL10 enrichment artifacts fail fast. The rebuilt outputs now carry manifests that record
the exact enrichment CSV paths, the ST03 split file, the excluded holdout bacteria IDs, and SHA-256 hashes for the
emitted artifacts. Compared with the leaked rebuilds, TL03 gained 14 pairwise columns and TL04 lost 6 pairwise
columns, with the largest changes concentrated in a handful of OMP- and defense-linked associations.

#### What changed

- `lyzortx/pipeline/track_l/steps/run_enrichment_analysis.py` now writes a holdout-aware manifest for TL02 with the
  split file path, excluded ST03 bacteria IDs, and output hashes.
- `lyzortx/pipeline/track_l/steps/build_mechanistic_rbp_receptor_features.py` and
  `lyzortx/pipeline/track_l/steps/build_mechanistic_defense_evasion_features.py` now validate that TL02 manifest
  before rebuilding, record the TL02 provenance in their own manifests, and store output file hashes.
- The default TL02 bootstrap now recreates the missing ST01 -> ST01b -> ST02 -> ST03 Steel Thread prerequisites when
  the split file is absent, so the Track L rebuild works on a fresh checkout.
- `lyzortx/research_notes/ad_hoc_analysis_code/compare_tl11_mechanistic_rebuilds.py` captures the clean-vs-leaked
  comparison used for the notebook deltas.

#### Rebuilt output statistics

- TL03 holdout-clean output:
  - `25` direct profile columns
  - `316` pairwise columns
  - `341` total feature columns
  - `28,782` rows with any non-zero mechanistic signal
  - `20,703` rows with any non-zero pairwise signal
- TL03 leaked output:
  - `25` direct profile columns
  - `302` pairwise columns
  - `327` total feature columns
  - `28,782` rows with any non-zero mechanistic signal
  - `19,926` rows with any non-zero pairwise signal
- TL04 holdout-clean output:
  - `11` direct profile columns
  - `19` pairwise columns
  - `30` total feature columns
  - `24,354` rows with any non-zero mechanistic signal
  - `7,663` rows with any non-zero pairwise signal
- TL04 leaked output:
  - `11` direct profile columns
  - `25` pairwise columns
  - `36` total feature columns
  - `24,354` rows with any non-zero mechanistic signal
  - `9,425` rows with any non-zero pairwise signal

#### Most important changed associations

- TL03 gained `tl03_pair_profile_009_x_ompc_99_79_weight` at `0.5178` and `tl03_pair_profile_009_x_ompc_99_50_weight`
  at `0.4313`; both were absent from the leaked rebuild.
- TL03 dropped leaked-only weights such as `tl03_pair_profile_013_x_yncd_99_65_weight` at `0.4522`,
  `tl03_pair_profile_013_x_fhua_99_39_weight` at `0.4199`, and
  `tl03_pair_profile_002_x_nfra_99_93_weight` at `0.3718`.
- TL04 gained `tl04_pair_profile_004_x_defense_thoeris_ii_weight` at `0.3612` and
  `tl04_pair_profile_008_x_defense_septu_weight` at `0.1676`.
- TL04 dropped leaked-only weights such as `tl04_pair_profile_002_x_defense_rloc_weight` at `0.2827`,
  `tl04_pair_profile_011_x_defense_thoeris_ii_weight` at `0.2765`, and
  `tl04_pair_profile_010_x_defense_bsta_weight` at `0.2305`.

#### Interpretation

1. The holdout-clean rebuild is materially sparser for TL04 and slightly denser for TL03, which is what we want after
   removing leaked signal from the enrichment stage.
2. The largest shifts are not rounding noise; they are discrete association changes, especially around OmpC-linked RBP
   profiles and Thoeris/Rloc/BstA defense associations.
3. The TL03/TL04 manifests now make the provenance auditable enough to reject stale pre-TL10 artifacts in future
   rebuilds.

### 2026-03-30: TL12 Re-run mechanistic lift evaluation with holdout-clean features and explicit lock rules

#### Executive summary

Re-ran the mechanistic lift evaluation on the same label set and code path, but with the rebuilt TL11 holdout-clean
TL03/TL04 features. TL05 now validates the TL11 manifests, zero-fills missing TL11 pair rows for holdout bacteria
instead of crashing, and reports paired bootstrap confidence intervals over holdout strains for ROC-AUC, top-3 hit
rate, and Brier score.

#### Lock rule

- Predeclared rule: only lock a new mechanistic arm if the paired bootstrap 95% CI for ROC-AUC delta vs the locked
  baseline is entirely above zero, and the top-3 / Brier deltas do not materially degrade.
- SHAP is only supporting evidence. It cannot justify a lock when the holdout bootstrap deltas stay within noise.

#### Holdout results

- Locked baseline `defense + phage_genomic`:
  - ROC-AUC `0.835466` (`0.803481` to `0.864889`)
  - top-3 hit rate `0.892308` (`0.794872` to `0.933369`)
  - Brier score `0.146153` (`0.127844` to `0.165627`)
- `+TL03` RBP-receptor:
  - ROC-AUC `0.820052` (`0.786865` to `0.851195`)
  - top-3 hit rate `0.846154` (`0.720855` to `0.878049`)
  - Brier score `0.148296` (`0.128906` to `0.168768`)
  - ROC-AUC delta `-0.015414` (`-0.023869` to `-0.006673`)
- `+TL04` defense-evasion:
  - ROC-AUC `0.838029` (`0.804850` to `0.867451`)
  - top-3 hit rate `0.892308` (`0.794118` to `0.931856`)
  - Brier score `0.144594` (`0.126652` to `0.163641`)
  - ROC-AUC delta `0.002563` (`-0.002322` to `0.007935`)
- `+TL03+TL04` combined:
  - ROC-AUC `0.822875` (`0.793273` to `0.852281`)
  - top-3 hit rate `0.861538` (`0.756757` to `0.904762`)
  - Brier score `0.147016` (`0.128598` to `0.166644`)
  - ROC-AUC delta `-0.012591` (`-0.022734` to `-0.002589`)

#### Decision

- No mechanistic arm cleared the lock rule.
- TL03 was rejected because it was strictly worse on ROC-AUC and top-3, with ROC-AUC delta still below zero under
  bootstrap resampling.
- TL04 was the closest arm, but its ROC-AUC delta CI still crossed zero, so it never left the noise band.
- The combined arm also stayed inside the noise band and did not rescue the TL03 drop.
- Conclusion: **no honest lift**. The enrichment-derived pairwise path is a dead end for the current v1 lock.

#### Notebook notes

- The TL12 rerun controls for holdout-strain resampling and the current locked label set.
- It does not control for lineage confounding beyond the existing split design, so the bootstrap result is an honest
  holdout comparison, not a mechanistic proof.
- TL11 excluded holdout bacteria from the enrichment build; TL12 therefore zero-fills missing TL11 pair rows on the
  holdout side rather than pretending the features were learned from those strains.

### 2026-03-31: TL12 follow-up patch hardened the rerun mechanics without changing the verdict

#### Executive summary

The follow-up patch after PR `#283` fixed two real contract bugs in the rerun path:

- zero-fill for missing TL11 pair rows is now limited to `holdout_test` rows only; training/CV joins still fail fast;
- TL05 validates the actual CLI-provided TL11 manifest paths and rebuilds stale default TL11/TL02 artifacts before
  evaluating, rather than trusting whatever pre-existing generated outputs happen to be on disk.

It also removed duplicate CV-fold training in TL05 and added phase/progress logging so the rerun no longer goes quiet.
After rerunning on that hardened path, the verdict stayed the same: **no honest lift**.

#### Hardened-rerun holdout results

- Locked baseline `defense + phage_genomic`:
  - ROC-AUC `0.837060` (`0.806037` to `0.864883`)
  - top-3 hit rate `0.907692` (`0.809425` to `0.951220`)
  - Brier score `0.159486` (`0.142176` to `0.177996`)
- `+TL03` RBP-receptor:
  - ROC-AUC `0.822245` (`0.793170` to `0.850166`)
  - top-3 hit rate `0.907692` (`0.794768` to `0.945946`)
  - Brier score `0.156530` (`0.140885` to `0.173137`)
  - ROC-AUC delta `-0.014815` (`-0.024446` to `-0.005030`)
- `+TL04` defense-evasion:
  - ROC-AUC `0.839504` (`0.809889` to `0.868042`)
  - top-3 hit rate `0.892308` (`0.794872` to `0.944482`)
  - Brier score `0.156965` (`0.140532` to `0.174330`)
  - ROC-AUC delta `0.002444` (`-0.002404` to `0.007416`)
- `+TL03+TL04` combined:
  - ROC-AUC `0.823165` (`0.795989` to `0.848961`)
  - top-3 hit rate `0.892308` (`0.774980` to `0.926829`)
  - Brier score `0.155707` (`0.140804` to `0.171521`)
  - ROC-AUC delta `-0.013895` (`-0.023834` to `-0.002206`)

#### Interpretation

1. TL04 is still the least-bad mechanistic arm, but its ROC-AUC delta CI still crosses zero and its top-3 delta still
   crosses negative territory, so it remains outside the lockable region.
2. TL03 remains clearly non-competitive on ROC-AUC, and the combined arm still fails to rescue it.
3. The new result is more trustworthy than the earlier TL12 notebook entry because it was produced after fixing stale
   default-artifact recovery, manifest-path validation, and holdout-only zero-fill scope.

### 2026-03-31: Replan follow-up after TL12/TL12-hotfix — freeze panel-lift work, keep deployable-bundle work

#### Executive summary

The TL11/TL12 sequence answered the main question behind the mechanistic pairwise rebuild: does holdout-clean
annotation-derived pairwise signal produce an honest lift over the locked `defense + phage_genomic` baseline? The
answer is now clearly no. That means the rest of Track L should not keep spending effort as if TL03/TL04 are waiting to
become a lockable panel feature block after one more rerun.

The plan was updated accordingly:

- the Track L description now states that the mechanistic pairwise path is dead-ended for the current v1 lock;
- TL13 is now explicitly framed as a deployable-bundle audit plus compatibility-signal experiment, not another panel
  lift attempt; and
- TL14 is now contingent on TL13 clearing a round-trip gate, so broad external validation does not run by inertia on
  another feature-impoverished bundle.

#### Why this replan is warranted

1. TL12 and the follow-up hardened rerun both reached the same conclusion: **no honest lift**.
2. The least-bad mechanistic arm (TL04) still sits inside the noise band and still gives up top-3 ranking quality.
3. The remaining high-value open problem from TL08/TL09 is different: generalized inference is missing deployable
   compatibility signal and therefore needs a richer feature-parity audit, not another attempt to relitigate the panel
   lock.

#### What stays alive

- TL13 remains worth doing, but only as a go/no-go decision on deployable features.
- TL14 remains worth doing only if TL13 first proves that the richer bundle materially improves round-trip behavior on
  panel hosts with saved references.

#### What is now dead-ended

- Further TL03/TL04/TL12-style panel-lift work aimed at replacing the current locked v1 configuration.
