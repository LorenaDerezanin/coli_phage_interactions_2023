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
