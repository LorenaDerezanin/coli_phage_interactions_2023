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
Results: 379/3,424 significant RBP PHROG × OMP receptor associations, 25/160 RBP PHROG × LPS core associations, and
39/924 anti-defense PHROG × defense subtype associations (BH-corrected p < 0.05). These results support building
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
- `lyzortx/tests/test_annotation_interaction_enrichment.py`: 13 unit tests using a 20×10 slice of the real interaction
  matrix, covering BH correction, permutation test, contingency table arithmetic, and main effect confounding.

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

- **Interaction panel**: 369 bacteria × 96 phages = 35,424 pairs (9,720 lytic, 25,704 non-lytic).
- **RBP PHROGs**: 43 unique across 97 phages, 32 present in ≥2 phages (used in enrichment).
- **Anti-defense PHROGs**: 13 unique, 12 present in ≥2 phages.
- **OMP receptor features**: 107 (receptor, cluster) pairs with ≥5 bacteria across 12 receptor proteins.
- **LPS core types**: 5 types (R1=207, R3=61, K12=43, R4=35, R2=22 bacteria; No_waaL excluded).
- **Defense subtypes**: 77 subtypes with ≥5 bacteria (of 137 total).

#### Analysis 1: RBP PHROG × OMP receptor variant clusters

- **3,424 tests** (32 PHROGs × 107 receptor clusters), **379 significant** (11.1%).

#### Analysis 2: RBP PHROG × LPS core type

- **160 tests** (32 PHROGs × 5 LPS types), **25 significant** (15.6%).

#### Analysis 3: Anti-defense PHROG × defense system subtypes

- **924 tests** (12 PHROGs × 77 defense subtypes), **39 significant** (4.2%).

#### Caveats

- **Duplicate PHROG profiles**: The 32 RBP PHROGs reduce to ~25 unique phage-carrier patterns (e.g., 136/15437/4465/9017
  always co-occur, as do 1002/1154/967/972 and 2097/4277). The 379 significant associations are not 379 independent
  biological discoveries — TL03 should collapse identical PHROG profiles before building features.
- **P-value resolution**: With 1000 permutations, 281/379 significant OMP hits are censored at the minimum p-value
  (0.001). This is sufficient for significance screening (all survive BH) but does not rank top hits.
- **Residual confounding**: The permutation test conditions on phage feature and permutes host labels, but does not
  control for host phylogenetic lineage or correlated feature blocks. Some associations may reflect lineage correlation
  rather than specific molecular interactions. The anti-defense results (4.2% significance) are particularly susceptible
  — generic methyltransferases mapping to diverse defense subtypes are more plausibly explained by lineage than by
  specific evasion.

#### Implications for TL03/TL04

- **TL03 (RBP-receptor features)**: 379 significant PHROG-receptor associations provide candidate interaction features.
  TL03 should collapse duplicate PHROG profiles and use the enrichment odds ratios as feature weights.
- **TL04 (defense evasion features)**: 39 significant anti-defense × defense associations. The lower rate and caveats
  about annotation specificity suggest these should be treated as weaker candidates than the RBP-receptor features.
- **No escape hatch needed**: With 379+ significant associations (after duplicate collapse: ~250 unique), enrichment-based
  features are viable for TL03.

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
