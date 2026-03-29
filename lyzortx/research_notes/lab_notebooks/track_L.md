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
