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
