### 2026-03-22: TE01 RBP-receptor compatibility feature block

#### What was implemented

- Added a dedicated Track E builder:
  `lyzortx/pipeline/track_e/steps/build_rbp_receptor_compatibility_feature_block.py`.
- Added a Track E runner and README:
  - `lyzortx/pipeline/track_e/run_track_e.py`
  - `lyzortx/pipeline/track_e/README.md`
- Checked in a narrowed curated lookup at
  `lyzortx/pipeline/track_e/curated_inputs/genus_receptor_lookup.csv`.
  The lookup prefers exact `phage_genus` matches and falls back to `phage_subfamily` only when genus-level curation is
  intentionally absent.
- Defined the TE01 pairwise feature contract around the canonical ST0.2/ST0.3 grid:
  - output rows stay joinable on `pair_id`, `bacteria`, and `phage`
  - leakage-safe training-positive views reuse ST0.3 split logic
  - holdout rows only see `train_non_holdout` positives
  - non-holdout rows use out-of-fold positives from the other CV folds
- Emitted `7` per-pair compatibility features:
  - `lookup_available`
  - `target_receptor_present`
  - `protein_target_present`
  - `surface_target_present`
  - `receptor_cluster_matches`
  - `receptor_variant_seen_in_training_positives`
  - `receptor_variant_training_positive_count`
- Added regression tests in `lyzortx/tests/test_rbp_receptor_compatibility_feature_block.py` covering:
  - lookup loading and target validation
  - leakage-safe exact-phage and taxon-level training aggregation
  - end-to-end file emission for the feature matrix, metadata, lookup summary, and manifest

#### Output summary

- The generated TE01 output directory is
  `lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/`.
- The main joinable artifact is `rbp_receptor_compatibility_features_v1.csv`.
- Supporting outputs are:
  - `rbp_receptor_compatibility_feature_metadata_v1.csv`
  - `rbp_receptor_lookup_summary_v1.csv`
  - `rbp_receptor_compatibility_manifest_v1.json`
- Final matrix size: `35,424` rows (`369` bacteria x `96` phages) with `7` engineered compatibility features plus the
  pair join keys.
- Curated lookup coverage:
  - `77 / 96` phages covered (`80.2%`)
  - `19 / 96` phages uncovered (`19.8%`)
  - Covered target-family split:
    - `62` phages mapped to surface-glycan targets
    - `11` phages mapped to mixed `OMPC|LPS_CORE`
    - `4` phages mapped to single protein receptors (`NFRA` or `LAMB`)
  - Covered target split:
    - `44` phages: `O_ANTIGEN|LPS_CORE`
    - `18` phages: `CAPSULE`
    - `11` phages: `OMPC|LPS_CORE`
    - `3` phages: `NFRA`
    - `1` phage: `LAMB`
- Feature prevalence across the full pair grid:
  - `lookup_available = 28,413 / 35,424` pairs (`80.2%`)
  - `target_receptor_present = 28,391 / 35,424` pairs (`80.1%`)
  - `protein_target_present = 5,502 / 35,424` pairs (`15.5%`)
  - `surface_target_present = 26,937 / 35,424` pairs (`76.0%`)
  - `receptor_cluster_matches = 4,820 / 35,424` pairs (`13.6%`)
  - `receptor_variant_seen_in_training_positives = 24,963 / 35,424` pairs (`70.5%`)
- Leakage-safe training positives used for the out-of-fold / holdout aggregations: `8,149`.
- Total exact-phage variant support accumulated in `receptor_variant_training_positive_count`: `819,067`.
- The uncovered taxa are explicit rather than silently guessed:
  - `Kagunavirus / Guernseyvirinae`: `11` phages
  - `Dhillonvirus / Other`: `4` phages
  - `Sashavirus / Other`: `2` phages
  - `Wanchaivirus / Other`: `1` phage
  - `Wifcevirus / Other`: `1` phage

#### Interpretation

1. The correct simplification for TE01 was to ship a narrow curated lookup with explicit unknowns, not a fake-complete
   receptor table. The repository does not currently contain enough in-repo evidence to justify exhaustive genus-level
   receptor claims for every phage in the panel.
2. Leakage-safe aggregation matters here. `receptor_variant_seen_in_training_positives` would be optimistic noise if it
   were computed globally, so the builder treats holdout rows and CV rows as different training views.
3. The most defensible first-cut signal is a mixed receptor contract: exact OMP-cluster matches where protein receptors
   are known, and surface-glycan presence/variant reuse where the curated target is LPS, O-antigen, or capsule.
4. This TE01 block is now ready to join onto the canonical pair grid and should be a cleaner way to attack the current
   popular-phage bias than adding still more raw phage identity features.

#### Next steps

Use this TE01 block alongside the pending Track E defense-evasion and isolation-host-distance features, then measure
whether the combined pairwise stack improves phage-family holdouts and rescues the strain-specific miss cases already
documented in the steel-thread notebook.

### 2026-03-22: TE02 Defense-evasion proxy feature block

#### What was implemented

- Added a dedicated Track E builder:
  `lyzortx/pipeline/track_e/steps/build_defense_evasion_proxy_feature_block.py`.
- Wired the builder into the Track E entrypoint and README:
  - `lyzortx/pipeline/track_e/run_track_e.py`
  - `lyzortx/pipeline/track_e/README.md`
- Reused the audited Track C host-defense encoding instead of reparsing raw defense-finder inputs in Track E:
  - input pair grid:
    `lyzortx/generated_outputs/track_c/v1_host_feature_pair_table/pair_table_v1.csv`
  - split assignments:
    `lyzortx/generated_outputs/steel_thread_v0/intermediate/st03_split_assignments.csv`
- Defined the TE02 collaborative-filtering contract around phage-family x defense-subtype averages:
  - detect all `host_defense_subtype_*` columns from the Track C pair table
  - normalize missing families with the same `__MISSING_PHAGE_FAMILY__` policy used in ST0.3b
  - for each leakage scenario, compute family-specific average `label_hard_any_lysis` only on
    `train_non_holdout` rows with the training flag enabled
  - holdout rows see the full non-holdout training slice
  - non-holdout rows see out-of-fold training rows from the other CV folds only
- Emitted `4` per-pair TE02 features:
  - `defense_evasion_expected_score`
  - `defense_evasion_mean_score`
  - `defense_evasion_supported_subtype_count`
  - `defense_evasion_family_training_pair_count`
- Added supporting artifacts for traceability:
  - `family_defense_lysis_rates_v1.csv` with long-form scenario/family/subtype averages
  - `defense_evasion_proxy_feature_metadata_v1.csv`
  - `defense_evasion_proxy_manifest_v1.json`
- Added regression tests in `lyzortx/tests/test_defense_evasion_proxy_feature_block.py` covering:
  - fold-exclusion leakage protection for CV rows
  - holdout-only training views
  - end-to-end emission of the feature matrix, metadata, rate table, and manifest

#### Output summary

- The generated TE02 output directory is
  `lyzortx/generated_outputs/track_e/defense_evasion_proxy_feature_block/`.
- The main joinable artifact is `defense_evasion_proxy_features_v1.csv`.
- Supporting outputs are:
  - `defense_evasion_proxy_feature_metadata_v1.csv`
  - `family_defense_lysis_rates_v1.csv`
  - `defense_evasion_proxy_manifest_v1.json`
- Final matrix size: `35,424` rows (`369` bacteria x `96` phages) with `4` engineered defense-evasion features plus
  the pair join keys.
- The builder consumed:
  - `79` retained host defense subtype columns from the Track C v1 pair table
  - `5` phage families across the panel
  - `29,031` leakage-safe non-holdout training pairs for the holdout scenario
  - `395` holdout family-subtype rate cells (`5` families x `79` subtypes)
  - `2,365` total scenario-specific rate rows across `6` scenarios (`holdout` + `5` CV folds)
- Fold-specific sparsity was minimal:
  - `cv_fold_3` dropped `5` family/subtype cells because `host_defense_subtype_gao_hhe` only appeared in that fold's
    excluded training rows
  - all other scenarios retained the full `395` family/subtype matrix
- Feature distribution across the full pair grid:
  - `defense_evasion_expected_score` mean `2.216`, median `1.964`, max `7.534`
  - `defense_evasion_mean_score` mean `0.284`, median `0.321`
  - `defense_evasion_supported_subtype_count` mean `7.80`, median `8`, max `15`
  - all `35,424 / 35,424` pairs had non-zero expected score and non-zero subtype support in the holdout view
- Family-level score ordering was strong and stable:
  - `Straboviridae`: mean expected score `4.048`
  - `Other`: `2.773`
  - `Autographiviridae`: `1.365`
  - `Schitoviridae`: `0.937`
  - `Drexlerviridae`: `0.652`
- Highest holdout family/subtype rates were concentrated in `Straboviridae`, for example:
  - `host_defense_subtype_thoeris_i`: `0.848`
  - `host_defense_subtype_fs_hsd_r_like`: `0.818`
  - `host_defense_subtype_thoeris_ii`: `0.800`

#### Interpretation

1. The implementation is genuinely leakage-safe with respect to the ST0.3 host-group split. Each evaluation row uses
   only non-holdout training rows, and non-holdout rows exclude their own CV fold before computing family/subtype
   averages.
2. TE02 ended up dense rather than sparse on the current panel. Because all `5` phage families have training support
   for almost every retained defense subtype, the feature block behaves more like a family-conditioned weighting of host
   defense burden than a missingness-heavy collaborative filter.
3. The strongest evasion priors belong to `Straboviridae`, which is exactly the family where a defense-evasion proxy is
   most likely to matter. The rank ordering also shows that TE02 is carrying real phage-family identity signal rather
   than only host-side defense counts.
4. The open limitation is scope: this builder is keyed to the ST0.3 host-group split, not the ST0.3b phage-family
   holdout suite. When Track F benchmarks the phage-family holdout explicitly, we should decide whether TE02 is
   disabled there or given an unseen-family fallback instead of pretending those family priors are available.

#### Next steps

Join TE01 and TE02 into the Track F ablation stack, then decide the correct behavior for TE02 under explicit
phage-family holdouts before using it in the final dual-axis evaluation suite.
