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
