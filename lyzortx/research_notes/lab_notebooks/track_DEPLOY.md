### 2026-04-02: Track rationale — Deployment-Paired Feature Pipeline

#### Executive summary

This track eliminates the training/inference feature mismatch discovered in the TL18 post-merge audit. The model is
currently trained on curated panel metadata but scored at inference time on features derived from raw genome assemblies
by bioinformatics tools. These two paths produce systematically different feature values. This track makes them
identical by re-deriving all training features from raw assemblies using the same pipeline that runs at inference time.
It also replaces binary feature thresholds with continuous scores where the gradient carries biological signal, and
removes ~95 redundant or derived features.

#### Motivation from the TL18 audit

The TL18 audit (2026-04-02, documented in `track_L.md`) walked through every inference step on 3 validation hosts and
found:

- **Training/inference parity bugs**: DefenseFinder version drift (17.3% of model importance affected), capsule HMM
  sensitivity mismatch (3-5% importance), extra phage in FNA directory.
- **Binary thresholds discarding gradients**: 141 of 190 numeric features are binary. Receptor phmmer bit scores, RBP
  family mmseqs percent identity, capsule HMM profile scores, and defense gene counts all carry biological information
  that the current encoding throws away.
- **Feature redundancy**: 91 wasted one-hot features from exact duplicates (host_o_type duplicates host_o_antigen_type,
  etc.) plus 4 derived summary features (defense_diversity, has_crispr, abi_burden, rbp_family_count).

#### Design decisions

**What switches to continuous scores (and why):**

- **Receptors** (12 features): phmmer bit scores replace binary. A receptor at 99% identity is biologically different
  from one at 60% — variant proteins may not function as phage receptors. The score is a proxy for functional integrity.
- **RBP families** (32 features): mmseqs percent identity replaces binary. A phage with a 90% identity match to an RBP
  family is more likely to use that receptor-binding strategy than one at 31%.
- **Capsule** (~30 features): per-profile HMM scores replace binary capsule_present + categorical serotype. The paper's
  CapsuleFinder required syntenic gene co-localization to call a capsule present; our HMM scan detects individual gene
  hits. Rather than reproducing the paper's threshold (which requires a tool we don't ship), we expose the raw scores
  and let the model learn what patterns of capsule gene presence matter. Scattered capsule gene fragments that don't
  form a functional locus will have weak, dispersed scores that the model can learn to discount.
- **Defense subtypes** (79 features): integer gene counts replace binary. Two copies of MazEF is biologically different
  from one — the redundancy makes it harder for a phage to neutralize both simultaneously. The HMM detection *score* is
  a tool artifact (detection confidence, not defense strength), but the *count* is real biology.

**What stays categorical:**

- Phylogroup, serotype, sequence type, O-antigen type, LPS core type — these are genuinely categorical biological
  classifications with no meaningful continuous encoding.

**What gets dropped:**

- `host_o_type` (exact duplicate of `host_o_antigen_type`): 84 wasted one-hot columns
- `host_surface_lps_core_type` (exact duplicate of `host_lps_core_type`): 6 wasted columns
- `host_capsule_abc_present` (exact duplicate of `host_capsule_abc_proxy_present`): 1 wasted column
- `host_o_antigen_present`, `host_lps_core_present` (derivable from whether the categorical is non-empty)
- `host_k_antigen_type_source` (metadata about feature provenance, not biology)
- `host_defense_has_crispr` (max of CRISPR-Cas subtype columns)
- `host_defense_diversity` (count of nonzero defense subtypes)
- `host_defense_abi_burden` (count of Abi-family subtypes)
- `tl17_rbp_family_count` (sum of RBP family presence columns)
- `host_capsule_abc_proxy_present` and `host_abc_serotype_proxy` (replaced by continuous capsule profile scores)

#### Infrastructure benchmarks

- **Figshare assemblies**: 403 FASTA files, 1.9GB zip, CC BY 4.0 license. Download: ~7 min via the "Download all" zip
  endpoint (`https://ndownloader.figshare.com/articles/25941691/versions/1`). Unzip: ~7 seconds. Downloaded on demand,
  stored in gitignored `lyzortx/data/assemblies/picard/`, skipped if already present.
- **CI image**: 9.5GB compressed (`full-bio` profile). The free GitHub Actions runner (14GB disk) cannot accommodate
  the assemblies baked into the image. DEPLOY tasks download at runtime instead.
- **Pharokka meta mode**: all 97 phage genomes annotated in 3 min 13s using `pharokka.py --meta --split` on a
  concatenated multi-FASTA, vs 1-2 hours with the current per-phage approach. The 40x speedup comes from running
  mmseqs2 indexing and profile search once instead of 97 times. This is a deferred optimization — the existing
  committed pharokka annotations (4.5MB) are small enough to keep.

#### Task structure and gating

7 tasks (DEPLOY01-07), reorganized from an initial 8-task plan based on review of acceptance criteria quality:

- **DEPLOY01** (download): assembly download script, no manifest
- **DEPLOY02** (defense, **gate**): re-derive defense features from raw assemblies; if DefenseFinder disagreement
  with panel annotations exceeds 3 systems/host on average, stop and investigate before proceeding
- **DEPLOY03** (surface): re-derive host surface features with continuous scores; depends on DEPLOY02 gate clearing
- **DEPLOY04** (typing): re-derive host typing; depends on DEPLOY03 (sequential to avoid merge conflicts on shared
  host feature code)
- **DEPLOY05** (phage RBP): switch to continuous mmseqs scores; independent, can run in parallel with DEPLOY02-04
- **DEPLOY06** (retrain): 3-way comparison (TL18 baseline vs parity-only vs parity+gradient), lock decision
- **DEPLOY07** (wire inference): make training and inference call the exact same functions, validate zero-delta parity

Each feature task (DEPLOY02-05) outputs a schema manifest (JSON) listing column names and dtypes. DEPLOY06 validates
the joint schema before assembling the feature matrix.

#### Expected outcomes

The primary outcome is deployment integrity: zero delta between training features and inference features for any host
whose assembly is available. The secondary outcome is potentially richer signal from continuous scores. DEPLOY06
explicitly separates these with a 3-way ablation (TL18 baseline vs parity-only vs parity+gradients) so we can measure
whether gradients help independently of the parity fix.

### 2026-04-02: DEPLOY01 assembly download implementation

#### Executive summary

Implemented `download_picard_assemblies()` in `lyzortx/pipeline/deployment_paired_features/download_picard_assemblies.py`.
The function derives the ST02 validation host set from `data/interactions/raw/raw_interactions.csv`, downloads the
figshare "Download all" archive to `.scratch/`, extracts it into `lyzortx/data/assemblies/picard/`, and fails loudly if
any of the 369 ST02 bacteria are missing their assembly file.

#### Interpretation

- The committed raw interactions table contains 369 unique bacteria, which matches the ST02 host count referenced in
  the plan.
- `data/genomics/bacteria/picard_collection.csv` contains 403 Picard collection hosts, so the download target and the
  ST02 validation set are different but overlapping cohorts.
- The download path is guarded by a complete-cache check: if `lyzortx/data/assemblies/picard/` already contains 403
  FASTA files, the function skips the network request and only validates coverage.

#### Tests

- Unique ST02 bacteria extraction from a semicolon-delimited raw interactions table.
- Zip extraction and validation on a small synthetic archive.
- Cache-skip behavior when the expected FASTA count is already present.
- Loud failure when a required ST02 assembly is absent.

### 2026-04-02: DEPLOY02 host defense integer-count gate

#### Executive summary

Implemented `derive_host_defense_features()` and `run_validation_subset()` in
`lyzortx/pipeline/deployment_paired_features/derive_host_defense_features.py`.
The deployment host-defense block now runs the existing pinned DefenseFinder runner on raw assemblies, emits integer
gene counts for the retained 79 defense subtypes using the original panel subtype names, and writes a schema manifest
at `lyzortx/generated_outputs/deployment_paired_features/host_defense/schema_manifest.json`.

The 3-host gate clears. On the committed validation FASTAs (`55989`, `EDL933`, `LF82`), the average disagreement
against the panel annotations is **0.667 systems/host** across the retained 79-column feature schema, below the
track's 3-systems/host stop threshold. The observed mismatches look like minor subtype-label/version drift within
restriction-modification families, not a fundamental methodology mismatch.

#### Implementation

- Reused the Track L DefenseFinder wrapper for the pinned CLI/model installation, Pyrodigal protein calling, and
  systems TSV parsing rather than creating a second runner.
- Built the deployment schema from the same Track C support mask logic (`min_present_count=5`, `max_present_count=395`)
  so the block keeps the established 79 retained defense subtypes, but the output columns stay in raw panel naming
  form (`AbiD`, `RM_Type_IV`, etc.) instead of `host_defense_subtype_*` slugs.
- Dropped the derived summary columns `host_defense_has_crispr`, `host_defense_diversity`, and
  `host_defense_abi_burden` from the deployment schema entirely.
- Wrote per-host count CSVs and manifests under
  `lyzortx/generated_outputs/deployment_paired_features/host_defense/<host>/`, plus a combined validation CSV and
  validation disagreement report at the block root.

#### Validation comparison

Comparison scope: retained 79 subtype columns from the panel CSV, with integer counts and no derived summary features.

- `55989`: exact match on all 79 retained subtype columns. Raw unmatched detections were `DS-13`, `DS-27`, and
  `PARIS_II`; none of these affect the retained deployment feature schema.
- `EDL933`: one lost system. Panel annotation has `RM_Type_IIG=1`; the raw DefenseFinder output reported
  `RM_Type_IIG_2=1`, which did not normalize onto the retained `RM_Type_IIG` column. All other retained subtype counts
  matched exactly.
- `LF82`: one count change. Panel annotation has `RM_Type_IV=2`; the raw DefenseFinder output matched one
  `RM_Type_IV` system and emitted one unmatched `RM_Type_IV_1`, so the derived retained count is `RM_Type_IV=1`.

#### Interpretation

- The disagreement pattern is narrow and family-localized: both mismatches are restriction-modification subtype naming
  variants (`RM_Type_IIG_2`, `RM_Type_IV_1`) rather than broad system loss, broad system gain, or widespread count
  compression.
- `55989` matching exactly across all 79 retained columns is a strong sanity check that the pipeline is not globally
  misconfigured.
- Because the gate discrepancy rate is low and the misses are explainable as subtype-label drift rather than systematic
  detection failure, DEPLOY02 should proceed to DEPLOY03 instead of stopping at the disagreement report.

#### Tests

- Unit tests for schema construction, ensuring raw panel subtype names are preserved and the three derived summary
  columns are excluded.
- Unit tests for disagreement reporting (`systems_gained`, `systems_lost`, `count_changes`).
- Unit tests for the single-assembly derivation path, validating integer counts and schema manifest contents.
- Unit tests for the 3-host validation subset runner, validating combined CSV/report generation.

### 2026-04-02: DEPLOY03 host surface continuous-score derivation

#### Executive summary

Implemented `derive_host_surface_features()` and `run_validation_subset()` in
`lyzortx/pipeline/deployment_paired_features/derive_host_surface_features.py`.
The deployment host-surface block now reuses the existing TL15 raw-genome O-antigen, receptor, and capsule scans but
emits a reduced continuous schema: categorical `host_o_antigen_type` and `host_lps_core_type`, one continuous
`host_o_antigen_score`, 12 continuous receptor phmmer bit scores, and 99 continuous capsule-profile HMM scores. The
schema manifest is written to `lyzortx/generated_outputs/deployment_paired_features/host_surface/schema_manifest.json`.

The 3-host validation run completed cleanly on the committed FASTAs (`55989`, `EDL933`, `LF82`). All three derived
O-antigen calls matched the legacy surface labels exactly, all three derived LPS proxy calls matched exactly, and the
continuous receptor block preserved perfect binary parity with the old presence/absence panel (`0.0` receptor
zero/nonzero mismatches per host on average).

#### Implementation

- Reused the TL15 raw-genome assets and CLI paths instead of inventing a second surface-calling stack:
  `nhmmer` for O-antigen alleles, `phmmer` for receptor proteins, and `hmmscan` for the vendored ABC capsule HMMs.
- Kept O-antigen typing categorical but added `host_o_antigen_score`, defined as the summed best-family `nhmmer` bit
  score for the top-supported O-type candidate. Zero means no supporting allele hits.
- Replaced the 12 receptor binary columns with 12 continuous `host_receptor_*_score` columns, using the best
  `phmmer` bit score per receptor reference. Missing receptors are encoded as `0.0`.
- Replaced thresholded capsule presence/serotype outputs with 99 continuous
  `host_capsule_profile_<profile>_score` columns, one per vendored capsule HMM profile, keeping only the best score
  per profile and deliberately avoiding locus-synteny interpretation.
- Dropped the audited duplicate/derived/proxy columns:
  `host_o_type`, `host_surface_lps_core_type`, `host_capsule_abc_present`, `host_o_antigen_present`,
  `host_lps_core_present`, `host_k_antigen_type_source`, `host_capsule_abc_proxy_present`,
  `host_abc_serotype_proxy`, `host_k_antigen_present`, `host_k_antigen_type`, and `host_k_antigen_proxy_present`.
  The last three were removed because the acceptance criterion explicitly forbids thresholding capsule profiles into
  present/absent or typed-locus calls.

#### Validation comparison

Schema size: 115 columns total = `bacteria` + 2 categorical feature columns + 1 continuous O-antigen score + 12
continuous receptor scores + 99 continuous capsule-profile scores.

- `55989`: `host_o_antigen_type=O104`, `host_lps_core_type=R3`, `host_o_antigen_score=2206.1`. All 12 receptor scores
  were nonzero. Fourteen capsule profiles were nonzero; strongest signals were `cluster_53=564.0`,
  `cluster_48=527.2`, and `cluster_97=136.7`.
- `EDL933`: `host_o_antigen_type=O157`, `host_lps_core_type=R3`, `host_o_antigen_score=2428.7`. All 12 receptor
  scores were nonzero. Fourteen capsule profiles were nonzero; strongest signals were `cluster_53=562.0`,
  `cluster_48=527.5`, and `cluster_97=77.3`.
- `LF82`: `host_o_antigen_type=O83`, `host_lps_core_type=R1`, `host_o_antigen_score=2529.6`. All 12 receptor scores
  were nonzero. Twenty-two capsule profiles were nonzero; strongest signals were `cluster_27=935.3`,
  `cluster_25=927.3`, `cluster_26=893.8`, and `cluster_19=674.3`.

#### Interpretation

- The categorical part of the block is stable on the validation hosts: exact agreement for all 3 O-types and all 3
  LPS proxy calls indicates the raw-genome O-antigen path is aligned with the legacy labels on the committed gate set.
- The receptor validation outcome is informative: zero binary mismatches means the continuous block preserves the old
  presence/absence support, but every validation host had all 12 receptor scores nonzero. The old receptor binaries are
  therefore saturated on this gate set, while the new score magnitudes expose variation the model can actually learn.
- The capsule profile block is intentionally richer and noisier than the legacy capsule proxies. The nonzero profile
  counts (14, 14, and 22) show why reducing this biology to a single binary proxy or serotype call throws away a large
  amount of information.

#### Tests

- Unit tests for schema construction, including the continuous receptor/capsule columns and the dropped legacy fields.
- Unit tests for O-antigen score aggregation and unresolved-call behavior.
- Unit tests for receptor best-hit score selection and zero-filling.
- Unit tests for feature-row construction and validation-report generation.
