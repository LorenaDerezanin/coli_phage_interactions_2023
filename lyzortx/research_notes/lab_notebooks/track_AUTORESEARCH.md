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
