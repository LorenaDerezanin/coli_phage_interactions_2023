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

8 tasks (DEPLOY01-08), reorganized from an initial plan based on review of acceptance criteria quality and CI runtime
constraints:

- **DEPLOY01** (download): assembly download script, no manifest
- **DEPLOY02** (defense, **gate**): re-derive defense features from raw assemblies; if DefenseFinder disagreement
  with panel annotations exceeds 3 systems/host on average, stop and investigate before proceeding
- **DEPLOY03** (surface): re-derive host surface features with continuous scores; depends on DEPLOY02 gate clearing
- **DEPLOY04** (typing): re-derive host typing; depends on DEPLOY03 (sequential to avoid merge conflicts on shared
  host feature code)
- **DEPLOY05** (phage RBP): switch to continuous mmseqs scores; independent, can run in parallel with DEPLOY02-04
- **DEPLOY06** (pre-compute defense): run DefenseFinder on all 403 hosts locally, check in aggregated CSV
- **DEPLOY07** (pre-compute surface): run pyhmmer surface scans on all 403 hosts locally, check in aggregated CSV
- **DEPLOY08** (retrain): 3-way comparison (TL18 baseline vs parity-only vs parity+gradient), lock decision
- **DEPLOY09** (wire inference): make training and inference call the exact same functions, validate zero-delta parity

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

### 2026-04-03: DEPLOY04 host typing categorical derivation

#### Executive summary

Implemented `derive_host_typing_features()` in `lyzortx/pipeline/deployment_paired_features/derive_host_typing_features.py`.
The deployment host-typing block now runs the pinned Clermont phylogroup caller, ECTyper serotype caller, and MLST
caller on raw assemblies and emits a purely categorical schema: `bacteria`, `host_clermont_phylo`,
`host_st_warwick`, `host_o_type`, `host_h_type`, and `host_serotype`. The schema manifest is written to
`lyzortx/generated_outputs/deployment_paired_features/host_typing/schema_manifest.json`.

#### Validation comparison

Validation scope: the 3 committed FASTAs (`55989`, `EDL933`, `LF82`) compared against `data/genomics/bacteria/picard_collection.csv`.

- `55989`: exact match on phylogroup `B1`, O-type `O104`, H-type `H4`, ST `678`, and derived serotype `O104:H4`.
- `EDL933`: exact match on phylogroup `E`, O-type `O157`, H-type `H7`, and derived serotype `O157:H7`, but MLST
  returned an unresolved `ST = -` in the raw caller output, so `host_st_warwick` stayed empty and did not match the
  Picard metadata ST `11`.
- `LF82`: exact match on phylogroup `B2`, O-type `O83`, H-type `H1`, ST `135`, and derived serotype `O83:H1`.

Aggregate field matches across the 3 validation hosts:

- phylogroup: `3/3`
- O-type: `3/3`
- H-type: `3/3`
- ST: `2/3`
- derived serotype consistency: `3/3`

#### Interpretation

- The typing path is aligned with Picard metadata on the categorical biology that was called successfully.
- The only miss is the unresolved MLST assignment for `EDL933`; the raw caller output is auditable in
  `lyzortx/generated_outputs/deployment_paired_features/host_typing/EDL933/sequence_type/mlst_legacy.tsv` and shows
  `ST = -`. That is a caller/data limitation, not a schema or parsing failure.
- The feature block is ready for the full 403-host DEPLOY07 run, with the `EDL933` MLST caveat documented for review.

#### Tests

- Unit tests for schema construction and categorical column typing.
- Unit tests for direct feature-row projection from raw caller outputs.
- Unit tests for panel-metadata comparison and validation aggregation.

### 2026-04-03: DEPLOY05 phage RBP continuous-score projection

#### Executive summary

Implemented the TL17 phage RBP projection update in
`lyzortx/pipeline/track_l/steps/deployable_tl17_runtime.py` and wired it through
`lyzortx/pipeline/track_l/steps/build_tl17_phage_compatibility_preprocessor.py`.
The projected phage feature block now emits 32 continuous mmseqs percent-identity columns, one per retained RBP
family, plus the non-derivable `tl17_rbp_reference_hit_count`. The old binary family presence columns are gone, and
`tl17_rbp_family_count` is no longer written. A schema manifest is now written to
`lyzortx/generated_outputs/track_l/tl17_phage_compatibility_preprocessor/schema_manifest.json`.

#### Validation comparison

The projection logic now uses query coverage as the acceptance gate and stores the best percent identity per family.
That means:

- zero means no accepted hit above the query-coverage threshold
- scores are continuous percent identities, not presence flags
- `tl17_rbp_reference_hit_count` still tracks total accepted reference hits and cannot be reconstructed from the family
  score columns alone

The deployment bundle integration now carries the new 33-column phage feature block directly into TL18 without any
assembly download step, because the committed phage FNAs in `data/genomics/phages/FNA/` are already sufficient.

#### Interpretation

- This fixes the last TL17 training/inference mismatch for phage-side features: the bundle now trains and scores on the
  same continuous family scores that the runtime projects.
- Dropping `tl17_rbp_family_count` is correct because it is derivable from the family score block, while keeping
  `tl17_rbp_reference_hit_count` preserves extra information about how many hits survived the coverage filter.
- The schema manifest makes the TL17 contract explicit for downstream bundle assembly and review.

#### Tests

- Unit tests for schema construction and schema-manifest emission.
- Unit tests for single-phage and batched TL17 projection with continuous family scores.
- Unit tests for TL17 validation-summary aggregation on the family score columns.

### 2026-04-03: DEPLOY06 pre-compute 403-host defense features

#### Executive summary

Ran DefenseFinder on all 403 Picard collection hosts locally (10 parallel workers, ~114 min total, 0 failures) and
checked in the aggregated integer gene counts CSV at
`lyzortx/data/deployment_paired_features/403_host_defense_gene_counts.csv`. This task was created after the first
DEPLOY07 (originally DEPLOY06) Codex CI attempt failed because DefenseFinder is too slow for a 4-core CI runner with
a 10-minute Codex timeout.

#### Why this task exists

The first Codex attempt at the full retrain task (GitHub Actions run 23935738906) spent its entire 77-minute window
debugging DefenseFinder environment issues and waiting for HMM searches, without reaching the training/evaluation
phase. The root cause is architectural: DefenseFinder runs `hmmsearch` once per HMM profile (~1,178 profiles) against
each host's predicted proteome (~2,500 proteins). At ~50 seconds per host with `--workers 1`, the full 403-host run
takes ~35 minutes on 10 cores — far beyond what a 4-core CI runner can deliver within the Codex session timeout.

Specific failures observed in the CI run:

- **DefenseFinder model installation race**: parallel per-host workers all tried to install models simultaneously
- **Anonymous GitHub API rate limit**: model downloads hit the unauthenticated rate limit in CI
- **Source vs release tarball confusion**: the GitHub source tarball contains only 1 HMM profile; the release asset
  contains the full 1,178
- **Stale `.old` model directories**: `macsydata` leaves rename artifacts that confuse the CLI's package counter
- **Per-host HMM scaling**: 403 separate DefenseFinder runs × 1,178 hmmsearch calls each is infeasible on a 4-core
  runner even after fixing all environment issues

The Codex agent attempted several increasingly sophisticated fixes (authenticated model fetch, batch mode over a
combined gembase, parallel Pyrodigal), all of which improved things incrementally but could not make the fundamental
compute requirement fit the CI window.

#### Implementation

`lyzortx/pipeline/deployment_paired_features/run_all_host_defense.py` — parallel driver that:

- Pre-installs DefenseFinder models once before fanning out workers
- Runs `derive_host_defense_features()` per host with `--workers 1` across `max_workers` parallel processes
  (defaults to `os.cpu_count()`)
- Skips hosts that already have `host_defense_gene_counts.csv` (resume-safe)
- Aggregates all per-host CSVs into a single checked-in CSV sorted by bacteria ID
- Supports `--aggregate-only` to re-aggregate without re-running derivation

#### Runtime profile

- **Machine**: 10-core macOS, `phage_env` conda environment with DefenseFinder + hmmsearch
- **Total hosts**: 403 (402 processed + 1 pre-existing from a test run)
- **Failures**: 0
- **Wall time**: 6,819 seconds (~114 minutes)
- **Throughput**: ~6.1 hosts/min (10 parallel workers)
- **Per-host average**: ~17 seconds effective (including Pyrodigal gene prediction + DefenseFinder HMM search)

#### Output

`lyzortx/data/deployment_paired_features/403_host_defense_gene_counts.csv`:

- 403 rows, 80 columns (bacteria key + 79 retained defense subtype integer counts)
- Schema matches the DEPLOY02 manifest exactly
- Integer counts (not binary): e.g., `RM_Type_IV=2` for hosts with two copies
- Sorted by bacteria ID

Per-host intermediate outputs (protein FASTAs, raw DefenseFinder TSVs, hmmer TSVs, manifests) remain in gitignored
`lyzortx/generated_outputs/deployment_paired_features/host_defense/` and are not checked in.

#### Unused DefenseFinder outputs — future feature candidates

The checked-in CSV captures only integer gene counts per defense subtype. DefenseFinder produces richer per-host
outputs that are preserved locally but not yet mined for features:

- **HMM hit scores** (`*_defense_finder_genes.tsv`): per-gene `hit_score`, `i_eval`, `hit_profile_cov`,
  `hit_seq_cov`. These are continuous detection-confidence signals. A CRISPR system with HMM score 500 vs 50 may
  reflect different system integrity, though the score is a detection artifact rather than a direct measure of defense
  strength.
- **System completeness** (`sys_wholeness` in genes TSV): ranges 0.0-1.0, indicating what fraction of the expected
  subunits were found. A complete RM system (all subunits) behaves differently from a partial one (just the REase).
- **Sub-threshold HMM hits** (`*_defense_finder_hmmer.tsv`): all hits including those that did not pass
  DefenseFinder's co-localization filter. May contain degenerate or incomplete defense systems.
- **Total defense gene count**: raw number of defense-associated genes (not subtypes) per host, a proxy for overall
  defense investment.

These are candidates for richer defense features in a future task, but extracting them is a feature-engineering
decision for DEPLOY07 or beyond — not a pre-compute blocker.

#### Tests

- Aggregation of per-host CSVs into a single sorted output with correct schema columns.
- Integer counts preserved through aggregation (not collapsed to binary).
- Missing columns in per-host CSVs zero-filled during aggregation.
- Empty output directory raises `FileNotFoundError`.
- Test fixtures are real DefenseFinder outputs from 3 completed hosts (001-023, 003-026, 013-008).

### 2026-04-03 23:24 UTC: DEPLOY07 full 403-host surface pre-compute

#### Executive summary

Completed the full 403-host surface-feature derivation and checked in
`lyzortx/data/deployment_paired_features/403_host_surface_features.csv`. The final CSV has 403 unique bacteria rows
and 115 columns, and its header matches the DEPLOY03 schema manifest exactly (`bacteria`, O-antigen type/score, LPS
core type, 12 receptor scores, 99 capsule-profile scores). The only code change needed during the run was removing an
unnecessary Biopython dependency from the O-antigen allele translation helper so the runner works in the declared
`phage_env`.

#### Run result

- Command: `micromamba run -n phage_env python -m lyzortx.pipeline.deployment_paired_features.run_all_host_surface
  --max-workers 4`
- Runner environment: GitHub Actions `full-bio` image, 4 CPUs available (`os.cpu_count() == 4`)
- Picard input set: 403 assemblies downloaded into `lyzortx/data/assemblies/picard/`
- Successful scan result: 403/403 hosts completed, 0 failures
- Aggregated CSV: `lyzortx/data/deployment_paired_features/403_host_surface_features.csv` (`275292` bytes)
- Schema validation: exact header match against `build_host_surface_schema(...)`, `403` unique bacteria IDs, sorted
  from `001-023` through `colF12g`

#### Runtime notes

The first full attempt on this runner spent `201s` predicting proteins for all 403 hosts, then failed before the scan
phase because `run_all_host_surface.py` imported `Bio.SeqIO` and `phage_env` does not ship Biopython. After replacing
that helper with a small built-in FASTA parser and codon-table translator, the rerun resumed from the cached
`predicted_proteins.faa` files and completed the pyhmmer scan phase in `1802s` (`30.0 min`), for an end-to-end cold
cache runtime of roughly `2003s` (`33.4 min`) on 4 cores.

This is slower than the earlier 10-core local benchmark recorded below, but it still fits the purpose of this task:
the expensive surface derivation can now be run once outside the downstream retrain PR, and DEPLOY08 can consume the
checked-in CSV directly without launching 403 HMMER scans in CI.

#### Interpretation

- The runner now matches the environment contract more honestly: no hidden Biopython dependency, no manual package
  install, and no need to weaken validation.
- The checked-in CSV satisfies the deployability contract for DEPLOY08. Downstream code can load one stable artifact
  instead of re-running O-antigen/receptor/capsule searches on every CI attempt.
- Per-host intermediates remain gitignored under
  `lyzortx/generated_outputs/deployment_paired_features/host_surface/`; only the aggregated deployment artifact is
  versioned.

#### Tests

- `micromamba run -n phage_env pytest -q lyzortx/tests/test_run_all_host_surface.py`
- Schema/header verification against `build_host_surface_schema(...)` on the checked-in aggregated CSV

### 2026-04-04 22:18 UTC: DEPLOY07 pre-compute 403-host surface features (code + plan)

#### Executive summary

Added `run_all_host_surface.py` to pre-compute continuous surface features (O-antigen, receptor, capsule) for all 403
Picard hosts locally and check in the aggregated CSV. This task was created after the DEPLOY08 (formerly DEPLOY07) Codex
CI attempt failed because the DEPLOY03 surface derivation (nhmmer O-antigen DNA scan at ~72s/host single-threaded) is
too slow for a 4-core CI runner. The runner uses pyhmmer for in-process HMMER searches and replaces the nhmmer DNA scan
with a protein phmmer search (~4.3s/host, 12x faster) by translating O-antigen alleles to protein.

#### Problem: nhmmer is the bottleneck

The original DEPLOY03 surface derivation calls nhmmer to search 84 O-antigen DNA allele queries against each host
assembly. On a 10-core Mac:

| Scan | Tool | Per-host (1 CPU) | 403 hosts / 10 workers |
|------|------|-----------------|----------------------|
| O-antigen | nhmmer | 72s | ~48 min |
| Receptor | phmmer | 0.7s | ~28s |
| Capsule | hmmscan | 5.4s | ~3.6 min |

nhmmer accounts for 92% of per-host compute. It uses profile HMMs for DNA-vs-DNA search — no prefilter, pure
Forward/Backward over the full target. BLAST is not viable (zero significant hits — seed-and-extend cannot match
nhmmer's profile sensitivity for divergent alleles). mmseqs2 nucleotide mode was killed after 20 min of prefiltering
against the 2GB concatenated assembly database.

#### Solution: translate alleles to protein, use phmmer

The O-antigen allele sequences are protein-coding genes (wzx, wzy, wzm, wzt). Translating them to protein and using
phmmer (protein-vs-protein) gives the same O-type calls with stronger E-values (1e-28 vs 1e-9 for nhmmer) and 12x
faster execution. Biological justification: protein-level conservation is higher than DNA-level for these
transmembrane/transporter genes, so protein search is actually more sensitive for homolog detection.

Measured timings with pyhmmer (in-process, no subprocess overhead, per-worker query caching):

| Scan | Per-host | 403 hosts / 10 workers |
|------|---------|----------------------|
| O-antigen phmmer (protein) | 4.3s | ~4.5 min |
| Receptor phmmer | 0.5s | included |
| Capsule hmmscan | 1.4s | included |
| **Total scans** | **6.2s** | **~10 min** |

The `micromamba run` subprocess approach was 2-3x slower due to 6-10s per-call environment activation overhead. pyhmmer
eliminates that entirely.

#### Output

- Runner: `lyzortx/pipeline/deployment_paired_features/run_all_host_surface.py`
- Output CSV: `lyzortx/data/deployment_paired_features/403_host_surface_features.csv` (268K, 403 rows x 115 columns)
- Schema: bacteria key + O-antigen type/score + LPS core type + 12 receptor scores + 99 capsule profile scores

#### Plan changes

- Inserted new DEPLOY07 (surface pre-compute, `executor: human`)
- Renumbered old DEPLOY07 -> DEPLOY08 (retrain/evaluate), old DEPLOY08 -> DEPLOY09 (wire inference)
- DEPLOY08 now depends on both DEPLOY06 (defense CSV) and DEPLOY07 (surface CSV) — no HMMER scans needed in CI
- Added `blast=2.16.0` to `environment.phage-annotation-tools.yml` (tested but not used in final approach)

#### Tests

- `build_surface_feature_row_from_scan_results`: full row structure, empty O-antigen result, unknown O-type LPS
  fallback, all 12 receptors present, realistic capsule profile names (KfiA, cluster_94), unknown capsule profile
  ignored, optional LPS-core exclusion.
- `_capsule_score_column_name`: simple name, cluster with number, empty name raises ValueError.
- `_translate_o_antigen_alleles`: DNA to protein translation with stop codon stripping, empty input.
- `aggregate_host_surface_csvs`: empty rows (header-only), sorted output with correct schema columns and values.
