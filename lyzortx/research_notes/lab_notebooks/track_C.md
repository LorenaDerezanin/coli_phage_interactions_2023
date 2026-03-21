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

### 2026-03-21: TC02 OMP receptor variant feature block

#### What was implemented

- Added a dedicated Track C OMP-cluster builder:
  `lyzortx/pipeline/track_c/steps/build_omp_receptor_variant_feature_block.py`.
- Configured the step to write generated outputs under
  `lyzortx/generated_outputs/track_c/omp_receptor_variant_feature_block/`:
  - `host_omp_receptor_variant_features_v1.csv`
  - `host_omp_receptor_variant_feature_metadata_v1.csv`
  - `host_omp_receptor_variant_feature_manifest_v1.json`
- Ingested the full `blast_results_cured_clusters=99_wide.tsv` table directly, preserving the requested `404`-strain
  host panel and the full set of `12` receptor proteins (`BTUB`, `FADL`, `FHUA`, `LAMB`, `LPTD`, `NFRA`, `OMPA`,
  `OMPC`, `OMPF`, `TOLC`, `TSX`, `YNCD`).
- Implemented a bounded categorical encoding policy:
  - receptor clusters observed in fewer than `5` hosts are grouped into receptor-specific `rare` buckets
  - one grouped category is always retained per receptor
  - additional grouped categories are added by descending Bernoulli indicator variance until the feature budget is
    exhausted
- Added regression tests in `lyzortx/tests/test_omp_receptor_variant_feature_block.py` covering rare-cluster grouping,
  feature-budgeted selection, row-level encoding, and end-to-end file emission.

#### Output summary

- Final matrix size: `404` host rows x `22` receptor features, plus the `bacteria` join key.
- Receptor diversity in the source BLAST table:
  - `BTUB`: `29` observed clusters, `1` missing host
  - `FADL`: `12` observed clusters, `0` missing hosts
  - `FHUA`: `20` observed clusters, `1` missing host
  - `LAMB`: `11` observed clusters, `4` missing hosts
  - `LPTD`: `11` observed clusters, `0` missing hosts
  - `NFRA`: `62` observed clusters, `7` missing hosts
  - `OMPA`: `11` observed clusters, `0` missing hosts
  - `OMPC`: `52` observed clusters, `1` missing host
  - `OMPF`: `14` observed clusters, `7` missing hosts
  - `TOLC`: `10` observed clusters, `2` missing hosts
  - `TSX`: `4` observed clusters, `2` missing hosts
  - `YNCD`: `55` observed clusters, `11` missing hosts
- Selected grouped categories by receptor:
  - `BTUB`: `99_6`, `99_15`
  - `FADL`: `99_1`, `99_17`
  - `FHUA`: `99_5`
  - `LAMB`: `99_10`, `99_9`, `99_19`
  - `LPTD`: `99_3`, `99_8`
  - `NFRA`: `99_14`, `99_18`, `rare`
  - `OMPA`: `99_13`, `99_16`
  - `OMPC`: `99_24`
  - `OMPF`: `99_4`, `99_12`
  - `TOLC`: `99_0`
  - `TSX`: `99_2`, `99_11`
  - `YNCD`: `99_7`
- Highest-support emitted indicators:
  - `host_omp_receptor_tolc_cluster_99_0`: `367` hosts
  - `host_omp_receptor_fadl_cluster_99_1`: `290` hosts
  - `host_omp_receptor_tsx_cluster_99_2`: `267` hosts
  - `host_omp_receptor_lptd_cluster_99_3`: `235` hosts
  - `host_omp_receptor_ompf_cluster_99_4`: `209` hosts
- Grouped rare buckets were mostly compressed away by the global feature budget; only `NFRA`'s grouped `rare` bucket
  survived as a final indicator (`67` hosts).

#### Interpretation

1. The acceptance target of `~20` receptor features is only feasible with aggressive categorical compression. The raw
   source table contains `291` non-missing receptor-cluster states across the 12 proteins, so a naive full one-hot
   expansion would be mostly noise and would violate the requested block size.
2. Most of the usable signal is concentrated in a small number of common receptor variants. The selected `22` columns
   capture the dominant structure in each receptor while still preserving one explicit grouped-rare signal for the most
   diverse locus (`NFRA`).
3. Receptor heterogeneity is highly uneven across loci. `TSX`, `TOLC`, and `LPTD` are dominated by one or two major
   variants, while `NFRA`, `OMPC`, and `YNCD` remain much more fragmented even after support-based rare clustering.
4. The emitted block is ready for downstream joins on `bacteria`, but the manifest should remain the source of truth
   for interpretation because grouped categories are a lossy compression of the original BLAST cluster table.
