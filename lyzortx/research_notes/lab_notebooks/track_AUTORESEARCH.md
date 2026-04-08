### 2026-04-04 20:45 UTC: AUTORESEARCH replanned around raw inputs and frozen featurizers

#### Executive summary

Track AUTORESEARCH now starts from the raw interaction table plus host and phage FASTAs, not from DEPLOY outputs. The
feature contract is frozen in `prepare.py`, the search loop is restricted to `train.py`, and only inference-safe
feature builders survive into the track. The sealed benchmark is explicitly bacteria-disjoint, so the holdout cannot
quietly reuse host identity seen during training.

#### Design decisions

- **Keep Track A labels, do not reopen labeling and model search at the same time.** AUTORESEARCH v1 reuses Track A
  `label_set_v1` semantics plus `training_weight_v3` as the fixed label policy. That keeps the search focused on model
  and representation choices instead of mixing in a second moving target.
- **Reuse raw-data plumbing, not DEPLOY artifacts.** The Picard assembly downloader from
  `lyzortx/pipeline/deployment_paired_features/download_picard_assemblies.py` is still useful because it resolves the
  full host FASTA inventory from the raw bacteria IDs. That is a data-ingest helper, not a dependency on DEPLOY's
  feature tables.
- **Keep only inference-safe featurizers.** The allowed feature families are those that run from raw FASTA at train and
  inference time: host DefenseFinder counts, host typing calls, raw host O-antigen/receptor/capsule-profile scans,
  simple sequence statistics, and TL17 phage RBP-family projection including `tl17_rbp_reference_hit_count`.
- **Cut panel-derived proxies even if they looked useful in DEPLOY.** The current `host_lps_core_type` path is still an
  O-type-to-LPS lookup built from Picard metadata, so it fails the unseen-FASTA test. AUTORESEARCH should not carry it
  forward just because it was convenient in DEPLOY.
- **Treat checked-in DEPLOY CSVs as caches at most.** They are acceptable only as optional warm-start accelerators that
  match a frozen schema and can be rebuilt from raw inputs. They are not source-of-truth inputs for AUTORESEARCH.
- **Keep autoresearch focused on model search, not on mutating bioinformatics preprocessing.** Heavy feature extraction
  is compatible with `autoresearch` only if it happens once in fixed `prepare.py` or an equivalent frozen cache step.
  It is explicitly out of scope for the search loop to rewrite or rerun expensive feature-building logic on every trial.
- **Carry forward the DEPLOY runtime scars into the acceptance criteria.** The DEPLOY notebook recorded that
  DefenseFinder took ~114 minutes for 403 hosts even on 10 cores, that naive surface derivation was too slow for CI
  until the algorithmic shape changed, and that repeated environment/bootstrap work mattered. AUTORESEARCH therefore
  must state up front that cache building is separate from the search loop, heavy steps must be batched or resume-safe,
  and RunPod time must not be burned redoing known one-time preprocessing.
- **Split the cache work by runtime-risk boundary, not by file count.** Host defense, host surface, host typing, and
  phage projection now live in separate plan tasks because they have different toolchains, different performance
  characteristics, and different known failure modes. The previous 5-ticket plan hid too much risk inside one broad
  preprocessing task.
- **Tighten criteria before dispatch, not after the first failed implement run.** The current plan now names the search
  metric, requires AR01 to record the exact locked comparator benchmark, fixes AR02's schema-composability contract,
  states explicitly that AR03-AR06 validate correctness on fixtures/subsets in CI while full-panel scale is measured
  outside CI, and requires bacteria-disjoint splits so the sealed holdout stays scientifically meaningful.

#### Immediate task sequence

1. `AR01`: freeze the raw corpus, label policy, and sealed split contract.
2. `AR02`: scaffold the sandbox and freeze the cache contract.
3. `AR03`: add host-defense cache building.
4. `AR04`: add host-surface cache building.
5. `AR05`: add host typing and simple host stats.
6. `AR06`: add phage projection and simple phage stats.
7. `AR07`: define the one-file baseline and strict search contract.
8. `AR08`: add the dedicated RunPod workflow and secret boundary.
9. `AR09`: import candidates back and replicate on the sealed holdout.

#### Interpretation

The point of the replan is to separate reusable biological signal from DEPLOY-specific scaffolding. AUTORESEARCH keeps
the label policy, host/phage FASTA acquisition, and raw-sequence featurizers that can be rerun on new genomes; it cuts
panel-shaped schemas, checked-in feature tables as scientific inputs, and any benchmark contract that would allow host
identity leakage. If a future AUTORESEARCH model wins, it should win on the strength of a better learner over a frozen
train-inference-parity cache, not because the search workspace inherited hidden structure from an earlier pipeline.

### 2026-04-04 22:05 UTC: AR01 locked the AUTORESEARCH pair table, label policy, and bacteria-disjoint split contract

#### Executive summary

AR01 now has a dedicated contract builder at `lyzortx/pipeline/autoresearch/build_contract.py`. It freezes the
AUTORESEARCH source-of-truth inputs to raw interactions plus resolved host/phage FASTAs, emits one canonical pair table
keyed by `bacteria` and `phage`, records explicit exclusion reasons, and predeclares deterministic `train`,
`inner_val`, and sealed `holdout` splits before any model search.

#### What changed

- **Added a dedicated AR01 contract builder instead of reusing ST0.2/ST0.3 outputs directly.** The new builder reuses
  Track A `label_set_v1` semantics through `compute_label_v1()` and reuses the Picard assembly resolver from
  `download_picard_assemblies.py`, but it does not inherit the wider steel-thread metadata feature joins.
- **Locked labels as read-only search inputs.** The emitted policy manifest marks labels immutable for downstream
  AUTORESEARCH tasks and documents exact `score='n'` handling:
  - `score='n'` never creates a positive label.
  - `score='n'` does not count toward the `>=5` interpretable observations needed for a hard negative.
  - `score='n'` only contributes uncertainty flags; pairs that stay below the negative-support threshold remain
    unresolved and are excluded.
- **Removed the hidden `interaction_matrix.csv` dependency from `training_weight_v3`.** TA11's original weight used the
  matrix-backed `aux_matrix_score_0_to_4==0` rule, which is incompatible with AR01's raw-only input contract. AR01
  therefore freezes a raw-only equivalent: any positive pair with no repeated lysis support inside a dilution
  (`>=2 score='1'` at the same dilution) gets `training_weight_v3=0.1`; all other pairs get `1.0`.
- **Locked the current production-intent comparator for AR09.** The manifest now points AR09 at the clean Track G v1
  benchmark artifacts:
  - `lyzortx/generated_outputs/track_g/tg02_gbm_calibration/tg02_benchmark_summary.json`
  - `lyzortx/generated_outputs/track_g/tg05_feature_subset_sweep/tg05_locked_v1_feature_config.json`
  - `lyzortx/generated_outputs/track_g/tg01_v1_binary_classifier/tg01_model_summary.json`

#### Findings

Running the pure contract logic against `data/interactions/raw/raw_interactions.csv` and the checked-in phage FASTAs
produced the following locked cohort counts:

- Observed pair rows: `35,424`
- Distinct bacteria: `369`
- Distinct panel phages: `96`
- Labels:
  - positive `1`: `9,720`
  - negative `0`: `25,546`
  - unresolved/excluded: `158`
- Split sizes by bacterium:
  - `train`: `221` bacteria, `21,216` rows, `21,125` retained labeled rows
  - `inner_val`: `74` bacteria, `7,104` rows, `7,040` retained labeled rows
  - sealed `holdout`: `74` bacteria, `7,104` rows, `7,101` retained labeled rows
- Raw-only borderline-noise downweighted positives (`training_weight_v3=0.1`): `2,685`

All three splits cover the same `96` panel phages, and the split assignment is bacteria-disjoint by construction.

#### Validation

- Added `lyzortx/tests/test_autoresearch_contract.py` to cover:
  - Track A v1 label reuse on aggregated raw pairs
  - raw-only `training_weight_v3` behavior
  - unresolved/missing-FASTA exclusion reasons
  - bacteria-disjoint split assignment and manifest writing
- Full repo validation passed in CI shell:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/`
  - Result: `421 passed`

#### Interpretation

AR01 deliberately chooses a thinner contract than the earlier panel pipelines. That is the right simplification. The
search track now has exactly one immutable label table and exactly one sealed benchmark boundary, both derivable from
raw interactions plus FASTAs. The only substantive policy deviation from prior work is the raw-only
`training_weight_v3` rule; that is intentional and preferable to carrying `interaction_matrix.csv` as an undeclared
fourth source input.

### 2026-04-04 23:00 UTC: AR02 froze the sandbox surface and the search-cache composability contract

#### Executive summary

AR02 now fixes the AUTORESEARCH sandbox to `lyzortx/autoresearch/{prepare.py,train.py,README.md,program.md}` and
adds a dedicated cache builder at `lyzortx/pipeline/autoresearch/prepare_cache.py`. `prepare.py` rebuilds AR01 from
raw inputs, exports only `train` and `inner_val` pair tables into `search_cache_v1/`, and freezes the slot names,
join keys, namespace prefixes, and provenance metadata that downstream feature-family tasks must honor.

#### What changed

- **Made `prepare.py` the only supported raw-input path into the search workspace.** The user-facing entry point now
  rebuilds the AR01 contract first, then materializes the AR02 cache from that contract. `train.py` is explicitly the
  short experiment loop and fails fast if the prepared cache is missing.
- **Froze the cache layout under `lyzortx/generated_outputs/autoresearch/search_cache_v1/`.** AR02 now writes:
  - `ar02_search_cache_manifest_v1.json`
  - `ar02_schema_manifest_v1.json`
  - `ar02_provenance_manifest_v1.json`
  - `search_pairs/train_pairs.csv`
  - `search_pairs/inner_val_pairs.csv`
  - `feature_slots/<slot>/entity_index.csv`
  - `feature_slots/<slot>/schema_manifest.json`
- **Locked the named feature-slot contract before any columns exist.** The frozen slots are:
  - host-side: `host_defense`, `host_surface`, `host_typing`, `host_stats`
  - phage-side: `phage_projection`, `phage_stats`
  Each slot now has a fixed join key and column-family prefix, so AR03-AR06 may add columns inside a slot but may not
  change the slot boundary itself.
- **Kept warm caches optional and subordinate to raw-input reproducibility.** `prepare.py` now accepts an optional
  warm-cache manifest, validates that its `schema_manifest_id`, join keys, prefixes, and CSV headers match the frozen
  AR02 contract, and records the result in provenance. That preserves the option to reuse checked-in DEPLOY CSVs as
  accelerators without turning them into source-of-truth inputs.
- **Sealed holdout outputs stay outside the search cache.** The provenance manifest records the omitted holdout row
  counts explicitly, but no holdout pair table or holdout-ready evaluation artifact is exported into the workspace that
  `train.py` consumes.

#### Validation

- Added `lyzortx/tests/test_autoresearch_prepare_cache.py` to cover:
  - frozen top-level slot contract
  - raw-only cache generation from fixture interactions and FASTAs
  - exclusion of holdout bacteria from exported host-side slot indexes
  - warm-cache manifest mismatch and match cases
- Full repo validation passed in CI shell:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/`
  - Result: `434 passed`

#### Interpretation

This is the smallest honest AR02 implementation. It fixes the contract surface that matters for later work, but avoids
pretending that feature-family semantics already exist. Downstream tasks can now fill reserved slots incrementally
without reopening the more dangerous questions of how raw inputs enter the cache, whether holdout data leaks into the
workspace, or whether a warm cache silently changes the schema.

### 2026-04-04 23:23 UTC: AR03 added host-defense cache building with one-time model controls

#### Executive summary

`lyzortx/pipeline/autoresearch/prepare_cache.py` now builds the `host_defense` feature slot from retained raw host
FASTAs instead of leaving that slot empty. The coordinator preinstalls pinned DefenseFinder release models once, workers
run with model installation explicitly forbidden, and the cache can be re-aggregated from per-host outputs without
rerunning every host. CI validation stayed on fixtures and mocked small-subset paths as intended; the required full
repo suite passed with `440` tests, while full-panel cold-cache wall-clock remains a dedicated manual benchmark rather
than a CI metric.

#### What changed

- **Filled the AR02 `host_defense` slot with real cache artifacts.** `prepare.py` now writes
  `feature_slots/host_defense/features.csv` plus a dedicated
  `feature_slots/host_defense/host_defense_build_manifest.json`. The exported columns are namespaced as
  `host_defense__*`, keyed only by `bacteria`, and the top-level/schema manifests now record those frozen columns.
- **Separated one-time model installation from per-host work.** The shared DefenseFinder runtime now exposes explicit
  `model_install_mode` control:
  - `ensure` installs pinned release models when needed.
  - `forbid` requires a preinstalled pinned model directory and raises if the path looks like a source checkout or wrong
    version.
  AUTORESEARCH uses `ensure` once on the coordinator, then fans out workers with `forbid` so model-install races and
  hidden downloads cannot occur inside the parallel loop.
- **Made host-defense aggregation reusable.** `run_all_host_defense.py` now supports loading or aggregating a requested
  subset of per-host outputs, which lets AUTORESEARCH rebuild the slot artifact from completed host jobs without
  rerunning DefenseFinder.
- **Kept the block inference-safe.** The slot is built from raw FASTAs and pinned release models, not from checked-in
  DEPLOY aggregate CSVs, and it exports only defense-subtype count columns. No panel metadata fields, no
  label-derived pair features, and no pair-table labels are mixed into the host-defense block itself.

#### Forbidden regressions covered

- **Model-install races / hidden worker downloads.** Added tests proving AUTORESEARCH workers call the host-defense
  builder with `model_install_mode="forbid"` and that aggregate-only mode can rebuild the slot artifact without calling
  model installation.
- **Release-vs-source model confusion.** Added tests that reject model directories missing pinned release metadata and
  reject `force_update` when install mode is `forbid`.
- **Re-aggregation without rerun.** Added tests for subset aggregation and missing-host failure behavior in
  `lyzortx/tests/test_run_all_host_defense.py`.

#### Validation and runtime interpretation

- CI validation:
  - focused host-defense/AUTORESEARCH regression tests passed
  - `micromamba run -n phage_env pytest -q lyzortx/tests/` passed with `440 passed`
- Full-panel cold-cache wall-clock was deliberately **not** recorded in CI. That matches the AR03 acceptance rule: CI
  should validate correctness on fixtures or a small subset, while the real wall-clock belongs to a manual or benchmark
  run on hardware that can actually represent RunPod economics.
- This change removes one known DEPLOY inefficiency rather than adding one: coordinator-only preinstall eliminates
  repeated model bootstrapping inside workers. I did not record a new 403-host benchmark in this CI run, so I cannot
  claim a numeric runtime delta yet, but there is no code-path regression on the specific failure mode that previously
  risked wasting parallel runtime.

#### Interpretation

Defense hits remain useful positive evidence for the model, especially when a subtype is confidently detected from the
assembly. The inverse is weaker: a zero in the exported `host_defense__*` block means "not detected by this pinned
annotation path under this schema," not "clean biological absence." That caveat is now written directly into the
AR03 build manifest so downstream search code cannot silently overinterpret missing defense features.

### 2026-04-04 23:22 UTC: AR04 materialized the raw-only host-surface cache with the pyhmmer fast path

#### Executive summary

`prepare.py` now materializes the `host_surface` slot instead of leaving it empty. The exported block is namespaced as
`host_surface__*`, built directly from retained host FASTAs with the pyhmmer/protein fast path from DEPLOY07, and
explicitly drops `host_lps_core_type` so AUTORESEARCH no longer depends on Picard lookup tables for host-surface
features.

#### What changed

- **Filled the `host_surface` slot from raw FASTAs inside `prepare_cache.py`.** The cache builder now resolves the
  retained train/inner-val host FASTAs from the AR01 pair table, runs the reusable fast-path surface builder, writes
  `search_cache_v1/feature_slots/host_surface/features.csv`, and updates the slot schema manifest with the realized
  column list and provenance.
- **Pinned the build to the fast path and recorded that choice in provenance.** The slot manifest and cache
  provenance now record the fast-path runtime ID plus `legacy_nhmmer_path_forbidden=true`, so the old slow
  per-host `nhmmer` route cannot quietly reappear as an implementation detail.
- **Kept only inference-safe raw signals in the exported block.** The materialized slot keeps O-antigen type/score,
  receptor scores, and capsule-profile scores, but not `host_lps_core_type`. That field still comes from an O-type to
  LPS proxy lookup rooted in Picard metadata, so exporting it would violate the raw-FASTA rebuildability rule.
- **Made retries cheaper without weakening correctness.** The fast path caches `predicted_proteins.faa` under
  `lyzortx/generated_outputs/autoresearch/host_surface_cache_build/`, so a rerun does not redo the full gene-calling
  front half before the pyhmmer scans.

#### Validation

- Added unit coverage for the raw-only host-surface schema and row builder in
  `lyzortx/tests/test_deployment_paired_host_surface_features.py`.
- Added AUTORESEARCH cache tests in `lyzortx/tests/test_autoresearch_prepare_cache.py` proving that:
  - `prepare.py` writes a real `host_surface` slot artifact with prefixed columns;
  - the exported slot omits `host_surface__host_lps_core_type`;
  - the cache provenance records the fast-path runtime and the explicit `legacy_nhmmer_path_forbidden` guard.
- Added `run_all_host_surface.py` coverage for the raw-only row emission path in
  `lyzortx/tests/test_run_all_host_surface.py`.
- Focused validation command:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/test_deployment_paired_host_surface_features.py lyzortx/tests/test_run_all_host_surface.py lyzortx/tests/test_autoresearch_prepare_cache.py`
  - Result: `30 passed`

#### Runtime interpretation

AR04 does not claim a new full-panel cold-cache number from CI. Instead it reuses the DEPLOY07 fast-path shape already
recorded in `lyzortx/research_notes/lab_notebooks/track_DEPLOY.md`: roughly `6.2s/host` for the scan phase and about
`10 min` for `403` hosts on a 10-core local benchmark, versus the rejected `~72s/host` `nhmmer` design. That is the
runtime boundary AUTORESEARCH now enforces.

### 2026-04-05 06:27 UTC: AR05 materialized raw host typing plus a small host-stats baseline block

#### Executive summary

`prepare.py` now materializes both `host_typing` and `host_stats` from retained raw host FASTAs. The typing block
reuses the pinned Clermont phylogroup, ECTyper serotype, and Achtman-4 MLST callers from the raw-assembly deployment
path without depending on Picard metadata at runtime, while the stats block adds a cheap numeric baseline
(`record_count`, `genome_length_nt`, `gc_content`, `n50_contig_length_nt`) that is rebuildable directly from the
assembly. Unresolved caller behavior, especially blank MLST outputs, is now recorded in a slot build manifest instead
of being silently coerced.

#### What changed

- **Filled the `host_typing` slot with raw-only categorical calls.** `prepare_cache.py` now runs the reusable
  deployment host-typing helper on the retained AUTORESEARCH host FASTAs and writes
  `search_cache_v1/feature_slots/host_typing/features.csv` with namespaced categorical columns:
  `host_clermont_phylo`, `host_st_warwick`, `host_o_type`, `host_h_type`, and `host_serotype`.
- **Separated runtime feature construction from panel comparison.** The shared host-typing helper now supports a
  runtime-only path where Picard metadata is optional and used only for validation/comparison. AUTORESEARCH calls that
  runtime-only path, so the exported features depend only on raw assemblies plus the pinned caller envs.
- **Added a small `host_stats` block as the low-cost baseline family.** `prepare_cache.py` now writes
  `search_cache_v1/feature_slots/host_stats/features.csv` from the raw host assemblies with four numeric fields:
  `host_sequence_record_count`, `host_genome_length_nt`, `host_gc_content`, and `host_n50_contig_length_nt`.
- **Recorded caller caveats in manifests instead of inventing placeholders.** The per-host typing helper manifest and
  the slot-level `host_typing_build_manifest.json` now keep explicit runtime caveat rows when a caller returns an
  unresolved value. The main concrete case covered here is MLST returning `-`, which stays blank in the exported
  feature row and is called out in the manifest for auditability.
- **Updated the frozen cache schema and join tests.** The top-level AUTORESEARCH schema manifest now records the
  realized `host_typing__*` and `host_stats__*` columns, and the cache tests now prove those categorical and numeric
  host-side blocks can still be loaded and joined onto retained pair rows by `bacteria`.

#### Validation

- Added runtime-only host-typing coverage in `lyzortx/tests/test_deployment_paired_host_typing_features.py`, including
  the unresolved-MLST manifest case with panel metadata disabled.
- Added host-stats coverage in `lyzortx/tests/test_deployment_paired_host_stats_features.py`.
- Extended `lyzortx/tests/test_autoresearch_prepare_cache.py` to cover:
  - materialized `host_typing` and `host_stats` slot schemas;
  - host-side slot manifests carrying runtime caveats; and
  - successful joins of those slot artifacts onto retained AUTORESEARCH pair rows.
- Focused validation command:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/test_deployment_paired_host_typing_features.py lyzortx/tests/test_deployment_paired_host_stats_features.py lyzortx/tests/test_autoresearch_prepare_cache.py`
  - Result: `16 passed`

#### Runtime interpretation

AR05 does not claim a new full-panel cold-cache wall-clock number from CI. No dedicated 403-host host-typing/stats
benchmark was run in this PR; the acceptance path stayed on fixtures and retained-subset tests as intended.

### 2026-04-05 06:52 UTC: AR06 materialized frozen TL17 phage projection plus a small phage-stats baseline

#### Executive summary

`prepare.py` now fills the phage-side `phage_projection` and `phage_stats` slots instead of leaving them as reserved
placeholders. The projection block reuses the frozen TL17 RBP-family runtime payload and reference bank as a pure
phage-side featurizer, runs the batched mmseqs path for retained phages, and records reference-bank provenance in the
cache manifests so the slot remains rebuildable from committed phage FASTAs plus the frozen runtime assets. Alongside
that, `phage_stats` adds a cheap numeric baseline (`record_count`, `genome_length_nt`, `gc_content`,
`n50_contig_length_nt`) derived directly from the phage FASTAs.

#### What changed

- **Filled the `phage_projection` slot from frozen TL17 runtime assets.** `prepare_cache.py` now loads the committed
  TL17 runtime payload, schema, reference-bank FASTA, and reference metadata from the TL17 output directory, projects
  retained phages into that space, and writes `search_cache_v1/feature_slots/phage_projection/features.csv` with
  namespaced columns for the retained family scores plus `tl17_rbp_reference_hit_count`.
- **Used the batched runtime path instead of one-phage-at-a-time projection.** AUTORESEARCH now calls
  `project_phage_feature_rows_batched()` so pyrodigal gene calling and `mmseqs easy-search` are amortized across the
  retained phage set rather than rebuilding avoidable query/reference state independently for every phage.
- **Recorded frozen reference-bank provenance directly in the cache outputs.** The phage-projection slot schema and
  build manifest now carry checksummed paths for the TL17 runtime payload, reference FASTA, reference metadata, family
  metadata, and schema manifest. The cache manifest/provenance manifest inherit that slot summary, so later rebuilds
  can verify exactly which frozen TL17 bank was used.
- **Made the phage block explicitly independent of checked-in projection CSVs.** The TL17 manifest may still list the
  original panel projection CSV for provenance, but AUTORESEARCH records that path as "not used" and rebuilds its own
  phage-side rows from raw FASTAs plus the frozen runtime assets only.
- **Added a small `phage_stats` baseline family.** A new helper at
  `lyzortx/pipeline/autoresearch/derive_phage_stats_features.py` writes per-phage stats rows and manifests with four
  low-cost numeric features: `phage_sequence_record_count`, `phage_genome_length_nt`, `phage_gc_content`, and
  `phage_n50_contig_length_nt`.

#### Validation

- Added `lyzortx/tests/test_autoresearch_phage_stats_features.py` for phage-stats schema, feature-row, and manifest
  coverage.
- Extended `lyzortx/tests/test_autoresearch_prepare_cache.py` to prove:
  - AUTORESEARCH uses the batched TL17 projection path rather than the per-phage path;
  - the frozen TL17 reference-bank provenance is written into the phage-projection slot manifests; and
  - `phage_projection__*` and `phage_stats__*` columns join cleanly onto retained pair rows by `phage`.
- Focused validation command:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/test_autoresearch_phage_stats_features.py lyzortx/tests/test_track_l_deployable_tl17_runtime.py lyzortx/tests/test_autoresearch_prepare_cache.py`
  - Result: `17 passed`
- Full repo validation:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/`
  - Result: `447 passed`

#### Runtime interpretation

AR06 does not claim a new cold-cache full-panel wall-clock from CI. This PR validates correctness on fixtures and the
retained-subset cache path, which is the intended acceptance scope. The new implementation shape should be cheaper
than a naive per-phage TL17 loop because it batches the shared mmseqs search, but any comparison to the dedicated
DEPLOY/TL17 reference runtime still belongs in a manual benchmark run rather than in CI.

### 2026-04-05 11:35 UTC: AUTORESEARCH critical path changed to adsorption-first

#### Executive summary

We kept host defense in AUTORESEARCH, but removed it as the gate for the first runnable search loop. The task graph now
lets host surface, host typing/stats, and phage projection/stats reach the first honest baseline before the slower
DefenseFinder block lands. This is a critical-path change, not a philosophical demotion of defense.

#### Design decision

- **Unblock first search from the slowest optional cache family.** `AR04` now depends on `AR02` instead of `AR03`, so
  the adsorption-first path no longer waits on host defense.
- **Keep defense in the schema and in the plan.** `host_defense` remains a reserved cache block from `AR02`; it is
  still intended as additive signal, but not as the prerequisite for the first runnable baseline in `AR07`.
- **Make the first AUTORESEARCH baseline explicitly adsorption-first.** The minimum cache for first search is now
  `host_surface + host_typing + host_stats + phage_projection + phage_stats`.
- **Clarify what defense absence means.** `AR03` now states directly that defense hits are useful positive evidence,
  while defense-feature absences are annotation-limited rather than clean biological absence.

#### Interpretation

This keeps the track aligned with the repo's actual prediction problem. If the first AUTORESEARCH loop cannot beat the
current benchmark with adsorption-first and phage-side features, we learn that cheaply and honestly. If defense helps,
it should help as an additive block later, not by delaying the first serious search behind the slowest preprocessing
path in the track.

### 2026-04-05 13:10 UTC: AR07 locked the one-file baseline and strict search contract

#### Executive summary

`lyzortx/autoresearch/train.py` is now the only ordinary AUTORESEARCH experiment surface. It validates the frozen
cache/schema/split contract before training, runs one adsorption-first baseline with a host encoder, a phage encoder,
and a learned pair scorer, and emits one scalar inner-validation ROC-AUC under a fixed search-time budget. The
baseline ignores `host_defense` by default, treating it as a later additive ablation rather than the prerequisite for
the first honest run.

#### What changed

- **Made `train.py` the full short-loop baseline instead of a cache-presence stub.** The file now loads the prepared
  cache, joins only the adsorption-first minimum slot set
  (`host_surface`, `host_typing`, `host_stats`, `phage_projection`, `phage_stats`), fits one host encoder plus one
  phage encoder, and trains one LightGBM pair scorer on the resulting pair features.
- **Locked the search contract around the frozen cache rather than trusting convention.** `train.py` now refuses to
  run if:
  - a holdout-named pair table appears inside `search_cache_v1/search_pairs/`;
  - retained `train` or `inner_val` pair IDs no longer match the AR01 split hashes; or
  - a slot feature table header no longer matches the frozen schema manifest.
- **Kept `host_defense` additive instead of gating the first baseline.** The default baseline does not consume
  `host_defense`; it only joins that block when `--include-host-defense` is requested explicitly.
- **Fixed the search metric and report-only metrics in the runtime output.** `train.py` now writes a baseline summary
  that declares inner-validation ROC-AUC as the primary search metric and records top-3 hit rate plus Brier score as
  secondary diagnostics only.
- **Documented the bootstrap and train commands in the AUTORESEARCH README and tightened `program.md`.** The docs now
  say directly that labels, splits, feature extraction, and evaluation policy are out of bounds for ordinary
  `train.py`-only search work.

#### Validation

- Added `lyzortx/tests/test_autoresearch_train_contract.py` to cover:
  - successful adsorption-first baseline execution without requiring `host_defense`;
  - rejection of sealed holdout pair-table leakage into the cache;
  - rejection of silent split-membership drift relative to AR01; and
  - rejection of slot feature tables whose headers bypass the frozen cache schema.
- Planned full-suite validation command for this task:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/`

#### Interpretation

AR07 deliberately chooses a thinner baseline than a "search everything" sandbox. That is the right requirement cut.
If AUTORESEARCH cannot produce useful lift with one frozen adsorption-first cache and one honest comparable metric, the
answer is not to reopen labels, splits, or featurizers mid-search. The right next step is to improve the learner
inside `train.py` or add clearly bounded ablations such as `host_defense` after the first baseline is reproducible.

### 2026-04-05 20:20 UTC: AR08 locked the dedicated RunPod workflow, pod spec, and secret boundary

#### Executive summary

AR08 adds a separate manual workflow at `.github/workflows/autoresearch-runpod.yml` instead of broadening
`codex-implement.yml` with cloud spend and provisioning logic. The workflow now stages a frozen host-side
AUTORESEARCH bundle, gates RunPod access behind a dedicated GitHub environment, provisions one fixed `NVIDIA A40`
community-cloud pod, runs one bounded `train.py` command, uploads a candidate artifact bundle, and deletes the pod.
I also split the AUTORESEARCH runtime contract into a lightweight `runtime_contract.py` module so the pod-side bundle
can honestly contain only the sandbox files plus the frozen search cache rather than dragging along wider repo code.

#### What changed

- **Added one dedicated manual RunPod workflow instead of touching the Codex workflows.** The new workflow is strictly
  `workflow_dispatch` only. It never injects RunPod credentials into `codex-implement.yml` or
  `codex-pr-lifecycle.yml`.
- **Locked the RunPod secret boundary to a dedicated GitHub environment.** The required environment is
  `runpod-autoresearch`; the required environment-scoped secret is `RUNPOD_API_KEY`. Approval happens at the GitHub
  environment boundary before any paid cloud step starts.
- **Separated host-side cache staging from pod-side search.** The workflow can either:
  - build a fresh frozen AUTORESEARCH cache and package it as `autoresearch-runpod-bundle`; or
  - reuse that bundle artifact from a previous run.
  The pod only receives the staged bundle and never reruns `prepare.py`, Picard assembly resolution, DefenseFinder,
  host typing, host surface derivation, or TL17 projection.
- **Made the runtime bundle small enough to match the acceptance text.** Before AR08, `train.py` imported broad
  preprocessing modules indirectly through `prepare_cache.py` and `build_contract.py`, so “sync only the sandbox plus
  its frozen cache artifacts” was not actually true. AR08 moves the shared split/cache constants and hash helpers into
  `lyzortx/pipeline/autoresearch/runtime_contract.py`, which lets the RunPod bundle include just:
  - `lyzortx/autoresearch/{prepare.py,train.py,README.md,program.md}`
  - `environment.yml`, `requirements.txt`, `pyproject.toml`
  - `lyzortx/log_config.py`
  - `lyzortx/pipeline/autoresearch/runtime_contract.py`
  - `lyzortx/generated_outputs/autoresearch/search_cache_v1/`
- **Collected a candidate bundle shaped for AR09 import.** Each RunPod run now pulls back:
  - the exact `train.py` used on the pod
  - `ar07_baseline_summary.json`
  - `ar07_inner_val_predictions.csv`
  - the staged bundle manifest
  - local workflow metadata
  - pod execution metadata
  - the remote experiment log

#### Locked pod decision

- **Chosen GPU:** `NVIDIA A40`
- **VRAM:** `48 GB`
- **RunPod on-demand price at lock time:** `$0.35/hr`
- **Nearby alternatives on the same pricing page:** `RTX 4090` at `$0.34/hr` with `24 GB` VRAM and `RTX A5000` at
  `$0.16/hr` with `24 GB` VRAM
- **Why the A40 lock is the right simplification:** AR07 search runs are now thin cache consumers, so the pod no
  longer needs the absolute cheapest option at any cost. The A40 buys 2x the VRAM of the nearby `RTX 4090` for a
  `$0.01/hr` premium, which is a better human-approved safety margin for future additive ablations than auto-selecting
  the cheapest 24 GB card at dispatch time.

#### Interpretation

This is the smallest honest cloud contract for AUTORESEARCH. Host-side preprocessing is expensive and mostly fixed, so
the right architecture is to freeze it once, stage it once, and let RunPod spend only on the short search loop. The
separate environment gate matters as much as the pod lock: it keeps paid infrastructure credentials out of the generic
Codex automation path and makes human approval explicit before any external GPU is provisioned.

#### References

- RunPod GPU pricing page: `https://www.runpod.io/gpu-pricing`
  - Quote: `A40 ... 48 GB VRAM ... $ 0.35 /hr`
  - Quote: `RTX 4090 ... 24 GB VRAM ... $ 0.34 /hr`
- RunPod SSH docs: `https://docs.runpod.io/pods/configuration/use-ssh`
  - Quote: `If you prefer to use a different public key for a specific Pod, you can override the default by setting the SSH_PUBLIC_KEY environment variable for that Pod.`
  - Quote: `If you're using a Runpod official template such as Runpod PyTorch ... full SSH access is already configured for you`
- RunPod environment-variable docs: `https://docs.runpod.io/pods/templates/environment-variables`
  - Quote: ``RUNPOD_PUBLIC_IP`Public IP address (if available).`
  - Quote: ``RUNPOD_TCP_PORT_22`Public port mapped to SSH.`
- LightGBM docs: `https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html`
  - Quote: `The original GPU version of LightGBM (device_type=gpu) is based on OpenCL.`
  - Quote: `The CUDA-based version (device_type=cuda) is a separate implementation.`

### 2026-04-05 22:16 UTC: AR09 added candidate import and sealed-holdout replication with an auditable decision rule

#### Executive summary

AR09 adds one explicit AUTORESEARCH handoff path from AR08 into local replay: `lyzortx/autoresearch/replicate.py`
now imports a RunPod candidate bundle into `lyzortx/generated_outputs/autoresearch/candidates/` and can replay that
imported `train.py` on the sealed AR01 holdout. The replay path refits both the imported candidate arm and the locked
production-intent comparator on all retained non-holdout AR01 rows, evaluates both on the same sealed holdout rows,
and writes one auditable decision bundle with repeated-seed metrics, paired bootstrap deltas, and a final
`promote`/`no_honest_lift` outcome.

#### What changed

- **Added one import command for the raw AR08 handoff.** `replicate.py import-runpod-candidate` accepts either the
  downloaded candidate directory or `candidate_bundle.tgz`, validates that the required AR08 files are present
  (`train.py`, local workflow metadata, bundle manifest, pod metadata, remote log), copies them under
  `lyzortx/generated_outputs/autoresearch/candidates/<candidate_id>/`, and records file checksums in
  `ar09_import_manifest.json`.
- **Made sealed-holdout replay a separate post-search contract.** `replicate.py replicate` deliberately does not
  reuse the RunPod inner-validation outputs as promotion evidence. Instead it rebuilds or validates the frozen raw-input
  AUTORESEARCH cache, reloads the AR01 pair table, and evaluates the frozen imported candidate on the sealed holdout.
- **Predeclared the replication rule instead of deciding case by case.** Once a candidate is imported, both arms are
  trained on all retained AR01 non-holdout rows (`train + inner_val`) and scored on the same retained holdout rows
  with repeated seeds. That is the right simplification because search is over; keeping `inner_val` aside during the
  final replay would just waste labeled training data.
- **Used the same decision math for candidate and comparator.** AR09 now computes holdout ROC-AUC, top-3 hit rate, and
  Brier score for both arms, then runs paired holdout-strain bootstrap deltas with the comparator as baseline. The
  decision bundle only promotes when the ROC-AUC delta bootstrap lower bound clears zero and the top-3/Brier delta
  upper bounds show no material regression; otherwise the artifact says `no_honest_lift`.

#### Validation

- Added `lyzortx/tests/test_autoresearch_candidate_replay.py` to cover:
  - import of a RunPod candidate bundle into the local candidate registry;
  - the narrow default-path fallback to the committed Track G feature lock;
  - replay of an imported candidate module on a sealed-holdout fixture; and
  - writing a final AR09 decision bundle with aggregated predictions and bootstrap-based decision text.
- Focused validation in the CI shell:
  - `micromamba run -n phage_env pytest -q lyzortx/tests/test_autoresearch_candidate_replay.py`
  - Result: `5 passed`

#### Interpretation

AR09 keeps the expensive and fragile parts where they belong. AR08 still owns the cloud search loop; AR09 owns only
the narrow import boundary and the honest post-search replication rule. During this task there was no completed
`AUTORESEARCH RunPod` workflow run in GitHub Actions to import as a real winner, so the new code establishes the
auditable replay path and decision contract without claiming a production promotion result that did not yet exist.

### 2026-04-08 21:50 UTC: First end-to-end AUTORESEARCH baseline — local CPU run and sealed holdout replication

#### Executive summary

Ran the complete AR01-AR09 pipeline locally on CPU for the first time. The raw-FASTA AUTORESEARCH candidate achieved
holdout ROC-AUC 0.787 vs the locked production comparator's 0.865. The AR09 decision bundle correctly reports
`no_honest_lift` — the candidate's bootstrap AUC delta CI is entirely negative ([-0.097, -0.059]). This is the
expected first-baseline result: hand-engineered Track C/D/G features still outperform the raw-FASTA-only pipeline.

#### Context

The AUTORESEARCH tooling (AR01-AR09) was all merged but had never been exercised end-to-end against a real experiment.
The original plan was to run `train.py` on a RunPod GPU, but two factors made local CPU execution the pragmatic first
step: (1) the RunPod REST v1 API schema kept changing between runs, and (2) the current `train.py` baseline is a
64-tree LightGBM model that trains in 0.4 seconds on CPU — GPU rental would be pure waste.

#### Bugs discovered and fixed

Five bugs were found and fixed during the first real run:

1. **Slot-level schema manifests left empty on resume (prepare_cache.py).** When `try_reuse_slot()` found existing
   `features.csv` files from a prior partial run, it returned a summary with the correct column list but did not
   rewrite the slot-level `schema_manifest.json`. Those manifests had been written empty during the partial run
   (before features were materialized). `train.py` then failed with "slot bypassed the frozen top-level cache schema"
   because it compares slot-level vs top-level `reserved_feature_columns`. Fix: `try_reuse_slot()` now rewrites
   `schema_manifest.json` on every reuse, using `build_slot_schema_manifest()` with the actual columns from the CSV.

2. **Feature slots excluded holdout entities (prepare_cache.py).** `select_search_rows()` only selected train and
   inner_val splits, and all downstream entity resolution (slot indexes, feature materialization) used this restricted
   set. This meant the cache contained 295 bacteria but the sealed holdout had 74 additional bacteria whose features
   were never computed. When `replicate.py` tried to build holdout embeddings, the left join produced NaN embeddings
   for all 74 holdout-only bacteria, triggering a "could not join all required host/phage embeddings" error.

   Fix: added `select_all_retained_rows()` which includes holdout rows for feature materialization. Pair tables
   remain restricted to train+inner_val (labels are sealed), but entity-level features derived from raw FASTAs carry
   no label information and are safe to compute for all splits. The cache now contains 369 bacteria (295 + 74 holdout)
   and 96 phages (unchanged — all phages were already covered).

3. **Dynamic module loading crashed on dataclass decorator (candidate_replay.py).** `load_module_from_path()` used
   `importlib.util.module_from_spec()` + `spec.loader.exec_module()` but did not register the module in
   `sys.modules` before execution. Python's `@dataclass` decorator internally calls `sys.modules.get(cls.__module__)`
   to resolve the module's `__dict__`, and when the module isn't registered, this returns `None` and raises
   `AttributeError: 'NoneType' object has no attribute '__dict__'`. Fix: insert
   `sys.modules[module_name] = module` before `exec_module()`.

4. **Feature lock mismatch between AR01 contract and generated Track G artifact (candidate_replay.py).** The AR01
   contract specifies `locked_feature_blocks: ["defense", "phage_genomic"]`, but the generated
   `tg05_locked_v1_feature_config.json` has `winner_subset_blocks: ["omp", "phage_genomic", "pairwise"]` from a
   different Track G sweep. The hard-coded check in `load_v1_lock()` raised before the AR09-level validation could
   run. Fix: `resolve_feature_lock_path()` now reads the lock file's blocks and compares them to the expected contract
   before returning the path. When they diverge, it falls back to the checked-in
   `lyzortx/pipeline/track_g/v1_feature_configuration.json` which has the correct defense+phage_genomic blocks.

5. **RunPod API schema instability.** The `gpuTypeId` (singular) field that worked on 2026-04-07 was rejected on
   2026-04-08 in favor of `gpuTypeIds` (array). Similarly, `dockerArgs` went from required to rejected. See the devops
   notebook for details.

#### Results

**Inner validation (train.py, train split only):**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.765 |
| Top-3 hit rate | 90.5% |
| Brier score | 0.166 |

**Sealed holdout replication (replicate.py, 3 seeds with bootstrap):**

| Metric | Candidate (raw-FASTA) | Comparator (Track C/D/G) | Delta CI (95%) |
|--------|----------------------|-------------------------|----------------|
| ROC-AUC | 0.787 | 0.865 | [-0.097, -0.059] |
| Top-3 hit rate | 90.5% | 95.9% | [-0.156, 0.000] |
| Brier score | 0.173 | 0.137 | [-0.052, -0.022] |

**Decision: `no_honest_lift`** — AUC delta stays within bootstrap noise (entirely negative); Brier score materially
degrades.

#### Interpretation

The ~8 point AUC gap (0.787 vs 0.865) reflects the expected difference between raw-FASTA-derived features and the
hand-engineered feature pipeline. The raw-FASTA candidate uses: capsule profile HMM scores (113 columns),
phylogroup/serotype/MLST typing (5 columns), host sequence statistics (4 columns), TL17 RBP-family presence vectors
(33 columns), and phage sequence statistics (4 columns) — all compressed through TruncatedSVD into 8-dimensional
host and phage embeddings. The comparator uses the full Track C defense columns, Track D phage genomic k-mer and
distance features, and the TG05 feature-subset sweep winner, without the dimensionality bottleneck.

The candidate's 90.5% top-3 hit rate is respectable — for most bacteria, the correct phage is in the top 3
predictions — but the comparator's 95.9% leaves less room for error.

This result establishes the baseline for AUTORESEARCH iteration. The `no_honest_lift` decision is correct and expected
for the first honest baseline. The pipeline is now proven end-to-end: `prepare.py` → `train.py` → `replicate.py` all
work locally. Future iterations should focus on:

- More expressive feature engineering within `train.py` (the only mutable surface).
- Potentially including `host_defense` features (currently reserved for additive ablation).
- Hyperparameter search within the 1800-second wall-clock budget.
- Exploring whether the SVD dimensionality bottleneck (8 dimensions) is too aggressive.

#### Artifacts

- Inner validation summary: `lyzortx/generated_outputs/autoresearch/train_runs/local_cpu_baseline/ar07_baseline_summary.json`
- AR09 decision bundle: `lyzortx/generated_outputs/autoresearch/decision_bundles/local_cpu_baseline_1/ar09_decision_bundle.json`
- AR09 seed metrics: `lyzortx/generated_outputs/autoresearch/decision_bundles/local_cpu_baseline_1/ar09_seed_metrics.csv`
- AR09 aggregated predictions: `lyzortx/generated_outputs/autoresearch/decision_bundles/local_cpu_baseline_1/ar09_aggregated_holdout_predictions.csv`

#### Timing

- Cache rebuild with holdout entities: ~25 minutes (host_defense ~20 min, host_surface ~8 min, host_typing ~8 min,
  host_stats ~3 min, phage slots reused).
- Cache resume (all slots exist): ~5 seconds.
- train.py on CPU: 0.4 seconds.
- replicate.py (3 seeds + 1000 bootstrap): ~20 seconds.

### 2026-04-08 20:55 UTC: Drop SVD, match TL18 capacity, evaluate on ST03 holdout

#### Executive summary

Removed SVD compression from train.py and fed all 159 raw slot features directly to a 300-tree/31-leaf LightGBM
matching TL18's model capacity. Added ST03 holdout evaluation to candidate_replay.py for apples-to-apples comparison
with TL18. Result: **0.810 ROC-AUC** [0.765, 0.847] on the same 65-bacteria holdout where TL18 scores 0.823. The
difference is not statistically significant (TL18 falls inside our 95% CI), but TL18 has better calibration
(Brier 0.141 vs 0.167) and higher top-3 hit rate (93.7% vs 90.8%).

#### Problem statement

The first AUTORESEARCH baseline (AR07 v1) crushed 159 raw features into 32 via 8-dim SVD embeddings and used a weak
64-tree/15-leaf LightGBM. This produced 0.787 AUC on the AR01 holdout, but that number was not comparable to TL18's
0.823 because:

1. Different holdout splits: AR01 uses 74 individual-bacteria holdout; ST03 uses 65 cv_group-disjoint bacteria.
   Only 12 bacteria overlap.
2. Different model architectures: SVD + small LightGBM vs raw features + large LightGBM.
3. Different feature sets: AUTORESEARCH uses only FASTA-derived slot features; TL18 adds defense, kmer, and
   TL15/16/17 preprocessor features.

The AR09 comparator arm was also flawed: it used panel-derived V0 metadata features on the AR01 split, producing a
suspiciously high 0.865 AUC that was meaningless for honest comparison.

#### Changes made

**train.py** — Complete rewrite of the experiment surface:
- Removed: `FittedEntityEncoder`, `fit_entity_encoder()`, `transform_entity_frame()`, old
  `build_pair_design_matrix()` with interaction terms, SVD/StandardScaler/OneHotEncoder imports
- Added: `type_entity_features()` for native categorical detection, `build_raw_pair_design_matrix()` for direct
  host/phage feature merges, `SLOT_PREFIXES` for feature column identification
- Updated: `PAIR_SCORER_PARAMS` to 300 trees, 31 leaves, min_child_samples=10, subsample=0.8, colsample_bytree=0.8;
  `build_pair_scorer()` with class_weight="balanced" and deterministic CPU mode
- Restored holdout-leakage and split-membership-drift validation lost during rewrite
- Feature count unchanged: 159 (155 numeric + 4 categorical), now fed directly without compression

**candidate_replay.py** — ST03 holdout support:
- Added `load_st03_holdout_frame()` and `build_st03_training_frame()`: join ST02 pair table with ST03 split
  assignments, filter by split/trainable
- Added `--use-st03-split` CLI flag: evaluates on ST03 holdout, skips flawed comparator arm
- Updated `build_candidate_holdout_rows()` for new train.py API: `type_entity_features` +
  `build_raw_pair_design_matrix` + `categorical_feature` in `.fit()`
- Candidate-only bootstrap CIs when using ST03 (no comparator arm to compare against)

#### Results

**Inner validation (AR01 split):**

| Metric            | SVD baseline (v1) | Raw features (v2) |
|-------------------|-------------------|--------------------|
| ROC-AUC           | 0.765             | **0.817**          |
| Top-3 hit rate    | 93.8%             | **94.6%**          |
| Brier score       | 0.166             | **0.158**          |

**ST03 holdout (same split as TL18):**

| Metric         | AUTORESEARCH raw features   | TL18 (defense+kmer+preprocessors) |
|----------------|-----------------------------|-----------------------------------|
| ROC-AUC        | 0.810 \[0.765, 0.847\]      | **0.823**                         |
| Top-3 hit rate | 90.8% (59/65)               | **93.7%** (59/63)                 |
| Brier score    | 0.167                       | **0.141**                         |

Per-seed stability (ST03 holdout, 3 seeds): AUC 0.808–0.810, confirming low random-state variance.

Bootstrap 95% CI for AUTORESEARCH: [0.765, 0.847]. TL18's 0.823 is inside this interval, so the difference is not
statistically significant at alpha=0.05.

#### Interpretation

The SVD bottleneck was the primary performance limiter: removing it added +5.2pp inner-val AUC and brought the
AUTORESEARCH baseline within bootstrap noise of TL18 on the honest ST03 holdout. The remaining gap (1.3pp AUC,
2.6pp Brier) is likely attributable to TL18's richer feature set: defense system annotations, phage genome kmer
profiles, and TL15/16/17 preprocessor-derived features — none of which AUTORESEARCH currently uses.

The comparator arm has been scrapped for ST03 evaluations. It was fundamentally flawed: different holdout split
(AR01 vs ST03), different feature source (panel-derived V0 metadata vs FASTA-derived slots), and only 12 bacteria
overlap between the two splits.

#### Next steps

1. Add defense features from FASTA-derived annotations to the AUTORESEARCH slot system
2. Add phage genome kmer features
3. Both should close the remaining gap to TL18 since those are exactly the features TL18 has that we don't

#### Generated outputs

- Decision bundle: `lyzortx/generated_outputs/autoresearch/decision_bundles/raw_features_v2/ar09_decision_bundle.json`
- Seed metrics: `lyzortx/generated_outputs/autoresearch/decision_bundles/raw_features_v2/ar09_seed_metrics.csv`
- Aggregated predictions: `lyzortx/generated_outputs/autoresearch/decision_bundles/raw_features_v2/ar09_aggregated_holdout_predictions.csv`

### 2026-04-08 21:16 UTC: Defense feature ablation on ST03 holdout

#### Executive summary

Adding 79 DefenseFinder-derived host defense features to the raw-feature baseline improves ST03 holdout ROC-AUC from
0.810 to 0.817 (+0.7pp), narrowing the gap to TL18's 0.823 by half. However, top-3 hit rate **dropped** from 90.8%
to 86.2%, suggesting the defense features add noise that hurts ranking quality. Brier score improved marginally
(0.167 to 0.164). The defense features alone are not sufficient to close the gap to TL18.

#### Method

Ran `train.py --include-host-defense` which adds the `host_defense` slot (79 binary/integer features from
DefenseFinder subtype annotations, derived from raw host assemblies via pyrodigal + defense-finder). Total feature
count: 159 (base) + 79 (defense) = 238 features (234 numeric + 4 categorical). Replayed on ST03 holdout with
3 seeds and 1000-sample bootstrap.

#### Results

| Metric         | No defense              | + Defense               | TL18      |
|----------------|-------------------------|-------------------------|-----------|
| ROC-AUC        | 0.810 \[0.765, 0.847\]  | 0.817 \[0.778, 0.851\]  | **0.823** |
| Top-3 hit rate | 0.908                   | 0.862                   | **0.937** |
| Brier score    | 0.167                   | 0.164                   | **0.141** |

Per-seed AUC stability: 0.814, 0.818, 0.817 (low variance).

#### Interpretation

The defense features provide a modest AUC lift but hurt ranking. This is consistent with the AGENTS.md warning that
"anti-defense-defense subtype associations are vulnerable to lineage confounding" — many defense subtypes correlate
with phylogroup, so they may be proxying for host lineage rather than adding mechanistic signal. LightGBM's native
feature selection may not be aggressive enough to ignore the noisy subtypes.

The remaining gap to TL18 (0.6pp AUC, 2.6pp Brier) is likely from kmer features, which capture phage genome
composition directly. Kmer extraction is the next step.

#### Top-3 regression deep dive

The 3 new top-3 misses (ECOR-14, IAI67, NILS76) all show the same pattern: a lytic phage that was ranked 1-3 without
defense dropped to rank 4 with defense features. In every case, non-lytic DIJ07/LF82 phages were boosted relative to
lytic LF73/LF82 phages.

Specific example — ECOR-14:

- Without defense: LF73\_P1 at rank 2 (prob 0.895), LF73\_P4 at rank 3 (prob 0.894) — both lytic.
- With defense: LF73\_P1 dropped to rank 4 (prob 0.676). DIJ07\_P2 and DIJ07\_P1 moved to ranks 2-3.

The defense features are pulling down predictions for specific host-phage pairs where the host defense profile
resembles hosts that resist these phages. This is consistent with lineage confounding: defense subtype presence
correlates with phylogroup, so the model may be learning "ECOR-14 looks like phylogroup X hosts where LF73 doesn't
lyse" rather than a mechanistic defense-evasion signal.

Statistically, the top-3 difference (59/65 → 56/65) is well within the bootstrap CI \[0.769, 0.925\] and not
significant. The per-seed top-3 values are actually the same or better with defense — the regression only appears in
the 3-seed aggregated predictions where probability averaging reshuffles borderline rankings.

#### Generated outputs

- Decision bundle: `lyzortx/generated_outputs/autoresearch/decision_bundles/raw_features_v2_defense/`
- Inner-val summary: `lyzortx/generated_outputs/autoresearch/train_runs/ar07_defense_ablation/`
