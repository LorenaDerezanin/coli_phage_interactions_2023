# Track E Pairwise Compatibility Features

`python lyzortx/pipeline/track_e/run_track_e.py`

This command runs the implemented Track E pairwise feature builder:

1. `rbp-receptor-compatibility`: write leakage-safe RBP-receptor compatibility features to
   `lyzortx/generated_outputs/track_e/rbp_receptor_compatibility_feature_block/rbp_receptor_compatibility_features_v1.csv`

The TE01 builder reads the canonical ST0.2 pair table, ST0.3 split assignments, the host OMP receptor-cluster table,
and a checked-in curated genus/subfamily lookup. Lookup matching prefers `phage_genus` and falls back to
`phage_subfamily` when the genus has no explicit entry. Training-positive aggregates are leakage-safe:

1. Holdout rows only look at `train_non_holdout` positives.
2. Non-holdout rows use out-of-fold positives from all other CV folds.

The final feature CSV is joinable on `pair_id`, `bacteria`, and `phage`, and the output directory also includes a
column-level metadata CSV, a per-phage lookup summary CSV, and a manifest with input/output hashes.
