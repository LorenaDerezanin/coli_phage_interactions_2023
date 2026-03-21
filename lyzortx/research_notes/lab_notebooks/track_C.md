### 2026-03-21: TC01 receptor and surface feature block

#### What was implemented

- Added a reproducible Track C builder:
  `lyzortx/pipeline/track_c/steps/build_receptor_surface_feature_block.py`.
- Configured the step to write generated outputs under
  `lyzortx/generated_outputs/track_c/receptor_surface_feature_block/`:
  - `host_receptor_surface_features_v1.csv`
  - `host_receptor_surface_feature_metadata_v1.csv`
  - `host_receptor_surface_feature_manifest_v1.json`
- Defined the emitted host set as the intersection
  `interaction_hosts ∩ lps_core_hosts ∩ receptor_cluster_hosts`, which resolves to the requested 369-host genomic
  subset from the current repository snapshot.
- Emitted three surface-antigen columns with explicit provenance and missingness tracking:
  - O-antigen presence and type
  - K-antigen typed call plus source field and capsule-proxy aggregate
  - LPS core presence and type
- Emitted known receptor locus features for `BtuB`, `FhuA`, `LamB`, `OmpA`, and `OmpC` as paired
  `*_present` / `*_variant` columns from the BLAST cluster table.
- Reserved `TonB` columns in the matrix and marked them fully missing with explicit provenance because the repository
  does not currently contain an in-repo TonB locus source analogous to the OMP cluster table.
- Added regression tests in `lyzortx/tests/test_receptor_surface_feature_block.py` covering host-set intersection,
  feature construction, metadata missingness, and end-to-end file emission.

#### Output summary

- Final matrix size: `369` host rows.
- O-antigen coverage: `369 / 369` typed (`0.0%` missing).
- LPS core coverage: `369 / 369` typed (`0.0%` missing).
- Typed K-antigen coverage: `141 / 369` typed (`38.2%` observed, `61.8%` missing).
- Typed K-antigen source split:
  - `124` hosts from `ABC_serotype`
  - `17` hosts from `Klebs_capsule_type`
- K-antigen proxy aggregate coverage: `369 / 369` non-missing because capsule-locus proxy flags are available across
  the genomic subset.
- LPS core distribution:
  - `R1`: `187`
  - `R3`: `54`
  - `K12`: `40`
  - `R4`: `34`
  - `No_waaL`: `32`
  - `R2`: `22`
- Receptor variant missingness:
  - `BtuB`: `0 / 369` missing
  - `FhuA`: `1 / 369` missing (`0.27%`)
  - `LamB`: `4 / 369` missing (`1.08%`)
  - `OmpA`: `0 / 369` missing
  - `OmpC`: `1 / 369` missing (`0.27%`)
  - `TonB`: `369 / 369` missing (`100%`) because no direct source is currently available in-repo
- Receptor variant diversity:
  - `BtuB`: `28` observed clusters
  - `FhuA`: `18` observed clusters
  - `LamB`: `11` observed clusters
  - `OmpA`: `11` observed clusters
  - `OmpC`: `50` observed clusters

#### Interpretation

1. The 369-host target is a data-contract issue, not a modeling mystery: it is the overlap between the interaction
   panel and the current LPS/receptor genomic annotations, with one extra LPS-only genome in the source table and 33
   interaction hosts outside that genomic subset.
2. O-antigen and LPS core are effectively complete on this subset, so they are ready for immediate downstream
   integration without auxiliary missingness indicators.
3. Typed K-antigen labels are sparse enough (`61.8%` missing) that models should not treat the raw typed call as a
   stand-alone high-confidence feature. The capsule-proxy aggregate is the safer default signal until capsule typing is
   improved.
4. The receptor block is materially richer than a simple presence mask. Even on this 369-host subset, `OmpC` alone
   spans `50` cluster variants and `BtuB` spans `28`, which is enough heterogeneity to justify variant-aware encoding in
   later pairwise compatibility work.
5. `TonB` is still an explicit source gap. Leaving it silently absent would hide a real repository limitation, so the
   current builder surfaces that gap directly in both the matrix and the metadata manifest.
