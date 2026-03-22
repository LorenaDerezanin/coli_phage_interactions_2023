# Track E Pairwise Compatibility Features

`python lyzortx/pipeline/track_e/run_track_e.py`

This command runs the implemented Track E pairwise feature builder:

1. `rbp-receptor-compatibility`: write leakage-safe RBP-receptor compatibility features to
   `lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/`
   `rbp_receptor_compatibility_features_v1.csv`
2. `defense-evasion-proxy`: write leakage-safe family-by-defense collaborative-filtering features to
   `lyzortx/generated_outputs/track_e/defense_evasion_proxy_feature_block/defense_evasion_proxy_features_v1.csv`

The TE01 builder reads the canonical ST0.2 pair table, ST0.3 split assignments, the host OMP receptor-cluster table,
and a checked-in curated genus/subfamily lookup. Lookup matching prefers `phage_genus` and falls back to
`phage_subfamily` when the genus has no explicit entry. Training-positive aggregates are leakage-safe:

1. Holdout rows only look at `train_non_holdout` positives.
2. Non-holdout rows use out-of-fold positives from all other CV folds.

The final feature CSV is joinable on `pair_id`, `bacteria`, and `phage`, and the output directory also includes a
column-level metadata CSV, a per-phage lookup summary CSV, and a manifest with input/output hashes.

The TE02 builder reads the Track C v1 pair table so it can reuse the audited `host_defense_subtype_*` host features,
plus ST0.3 split assignments for leakage-safe training views. It computes phage-family average lysis rates against each
defense subtype from training data only:

1. Holdout rows only use `train_non_holdout` rows with the training flag enabled.
2. Non-holdout rows use out-of-fold training rows from the other CV folds only.

Per-pair features are then derived by summing the relevant family-by-subtype success rates for the host defense systems
present on that pair. The output directory includes the joinable feature CSV, column metadata, a long-form
family/subtype rate table across leakage scenarios, and a manifest.
