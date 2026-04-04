### 2026-04-04 20:45 UTC: AUTORESEARCH replanned around raw inputs and frozen featurizers

#### Executive summary

Track AUTORESEARCH was rewritten from a DEPLOY-artifact sandbox into a raw-input search track. The new contract is:
start from raw interactions, host FASTAs, and phage FASTAs; freeze all preprocessing in `prepare.py`; let the search
loop mutate `train.py` only; and keep only those feature builders that can be rerun on unseen genomes at inference
time. The practical outcome is that AUTORESEARCH now depends on Track A labels but no longer depends on DEPLOY outputs
or checked-in feature CSVs as scientific inputs.

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

#### Immediate task sequence

1. `AR01`: freeze the raw corpus, label policy, and sealed split contract.
2. `AR02`: build `prepare.py` around the inference-safe raw-FASTA featurizer allowlist.
3. `AR03`: define the one-file baseline and strict search contract.
4. `AR04`: add the dedicated RunPod workflow and secret boundary.
5. `AR05`: import candidates back and replicate on the sealed holdout.

#### Interpretation

This is a cleaner AUTORESEARCH track than the earlier "strict readiness from DEPLOY artifacts" version because it
matches the real scientific question: can a small search loop find a better learner over a frozen, deployable
raw-genome feature contract? It also makes the failure modes legible. If a future winner is real, it won because the
model got better on train-inference-parity features, not because the search workspace quietly inherited panel-only
metadata or preblessed artifact tables.
