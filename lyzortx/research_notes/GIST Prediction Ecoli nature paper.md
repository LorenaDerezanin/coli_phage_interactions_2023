# Gist + Reproducibility Audit: Prediction of Strain-Level Phage-Host Interactions in *Escherichia*

Last updated: 2026-02-14 (local audit of repo + paper text + git history + issue #4)

## Citation
- Title: *Prediction of strain level phage-host interactions across the Escherichia genus using only
  genomic information*
- Journal: Nature Microbiology (Vol. 9, Nov 2024, pp. 2847-2861)
- DOI: https://doi.org/10.1038/s41564-024-01832-5

## One-paragraph scientific gist
The paper builds a large *Escherichia*-phage interaction resource and shows that strain-level lytic prediction is
feasible from genomic features alone, with adsorption-related host traits carrying most of the predictive signal and
antiphage systems contributing less to infection/no-infection classification. They report AUROC around 86% for binary
lytic prediction and then apply a 4-step recommender to propose 3-phage cocktails for 100 unseen pathogenic strains,
with tailored recommendations outperforming generic and baseline in lysis outcomes.

## Paper details that are implementation-critical
1. Prediction target is binary lytic vs non-lytic, trained per-phage.
2. Group 10-fold CV groups strains by core-genome distance threshold 1e-4 substitutions/site.
3. Four model families are compared per phage: two logistic regressions (L1/L2) and two random forests
   (depth 3/6, 250 estimators), with class-weight schedule based on positive-class prevalence.
4. Cocktail recommender has 4 sequential steps and stops once 3 phages are selected.
5. Baseline cocktail is a single uninformed top-coverage triplet; paper names it as
   `(LF82_P8, CLB_P2, LF73_P1)` and reports 63% productive lysis on original Picard collection.
6. Test-set claims: 100 unseen strains, 18 distinct recommended cocktails, split into tailored
   (15 cocktails for 24 strains) and generic (3 cocktails for 76 strains).

# What is currently in this repository (as of 2026-02-15)
1. Prediction training entry point exists: `dev/predictions/predict_all_phages.py`.
2. CV grouping helper exists: `dev/predictions/build_cv_clusters_from_distance.py`.
3. CV grouping CSV now exists: `data/metadata/370+host_cross_validation_groups_1e-4.csv`.
4. Cocktail analysis notebook exists:
   `dev/cocktails/analyze_cocktail_interactions_tailored_vs_generic_vs_baseline.ipynb`.
5. `recommend_cocktail.py` source is missing from git history (only `.pyc` artifact exists under
   `dev/cocktails/__pycache__/`).
6. README now explicitly states the missing recommender script and partial reproducibility limitations.

## Data/state mismatches observed during audit
1. `data/interactions/interaction_matrix.csv` has 402 strains x 96 phages (not 403 x 96).
2. `data/genomics/bacteria/picard_collection.csv` has 403 strains.
3. `data/metadata/370+host_cross_validation_groups_1e-4.csv` and
   `data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv` each have 404 strains.
4. IDs are not perfectly aligned:
   - `LF110` is present in features but absent from interaction matrix.
   - `H1-005-0065-L-P` and `H27` are in CV/UMAP but absent from matrix/features.
5. Current interaction matrix values are `0.0..4.0`; script currently binarizes with `y = (y > 0)` at training time.
6. Paper baseline phage `CLB_P2` is not present in current `data/genomics/phages/guelin_collection.csv`;
   likely a naming/snapshot drift.
7. On current matrix (`>0` treated as lytic), the unique max-coverage 3-phage triplet is
   `(LF82_P8, NIC06_P2, 536_P7)` with `352/402` strains covered (`87.56%`), which does not match
   the paper's baseline description/coverage.
8. Current `dev/cocktails/data/cocktails_composition.csv` has `100` strains and `18` distinct cocktails,
   but the three most frequent cocktails cover `72` strains (not `76`).

## Git-history pattern: what kinds of fixes were needed
Across 45 commits (all branches), recurring work was reproducibility hardening rather than new science.

1. Path portability and local-machine de-hardcoding.
   - Example: `4a78b18`, `5090765`.
   - Effect: replaced absolute Windows paths with repo-relative paths.
2. Missing metadata for CV grouping.
   - Example: `02efa6d`.
   - Effect: added `370+host_cross_validation_groups_1e-4.csv` and a generator script.
3. Label/binning logic drift for interaction scores.
   - Examples: `93a7885`, `5b09cc3`.
   - Effect: explicit conversions added/changed to binary training labels.
4. Feature engineering correctness bug (ABC host comparison).
   - Example: `cb6fdc0` and earlier branch work.
   - Effect: fixed self-comparison bug to compare strain ABC serotype against host ABC feature.
5. CV/model robustness against edge cases.
   - Examples: `8b353c1`, `6a0944a`, `ceb6aa3`.
   - Effect: handling single-class folds, model-fit errors, alias/model bookkeeping, excluded-phage reporting.
6. Documentation and environment onboarding.
   - Examples: `ea605d0`, `95a0a30`, `63b300b`.
   - Effect: micromamba setup, research notes, explicit repro caveats.
7. Upstream data/docs changes that alter reproducibility context.
   - Examples: `7160a5a`, `39857dc`, `99656d8`.
   - Effect: README tightened, notebook output cleaned, raw interaction annotations added.

## Issue #4 summary and relevance
Issue URL: https://github.com/mdmparis/coli_phage_interactions_2023/issues/4

The issue describes the same reproducibility gap observed in this audit:
1. `predict_all_phages.py` can be run after local fixes and CV file restoration, but outputs still diverge from
   paper-committed results.
2. Reconstructed cocktail recommender logic (from Methods + notebook) gives near-but-not-exact outputs.
3. Recomputed outputs are `99` strains instead of expected `100` in that attempt.
4. Baseline/generic/tailored split and baseline identity are close but not exact relative to paper numbers.
5. A key remaining dependency is access to `recommend_cocktail.py` and exact data/snapshot provenance.

## Additional clarification
Current reproducibility assumptions used in this audit:
1. The exact four-step recommender script used for Figure 6 is not currently available in the public repository.
2. The script appears to be stored on prior hardware and was not immediately retrievable at that time.
3. The README was updated by the author to document this and to indicate the script may be shared if recovered.
4. Exact cocktail-by-cocktail matching may not always be expected because parts of the pipeline use
   stochastic algorithms (for example random forests).
5. Inputs identified as required for cocktail prediction are:
   - `data/interactions/interaction_matrix.csv`
   - `data/genomics/bacteria/picard_collection.csv`
   - `data/genomics/phages/guelin_collection.csv`
6. Phage identity is treated as a one-hot feature, and alternative cocktails may still be valid.
7. Priority should be given to reproducing non-trivial tailored recommendations, with a concrete example: strain `AN13`
   receiving `BDX03_P1` (despite broad host-range not being maximal), linked to shared host-feature similarity
   (including outer-core LPS type).

## Current reproducibility status (pragmatic)
1. Prediction pipeline rerun capability:
   - Partially reproducible at code level (script and inputs present).
   - Not proven bitwise-reproducible to publication outputs because of unresolved snapshot/preprocessing drift.
2. Cocktail recommendation pipeline:
   - Not end-to-end reproducible from source in this repo.
   - Current limitation: recommender implementation source is not present here; only derived outputs and analysis
     notebook are available.
3. Paper-level Figure 6 narrative:
   - Directionally reproducible (same conceptual categories and many same cocktail motifs).
   - Numerically not fully locked to paper claims with current public snapshot.
4. Given these constraints, "exact cocktail identity match" is best treated as a secondary criterion unless RNG seeds,
   software versions, and the exact script snapshot are pinned.
5. A stronger near-term criterion is recovery of tailored/feature-driven recommendations and step-level
   behavior, not strict row-wise identity.

## Interpretation note on baseline/generic definitions
- Baseline: one fixed max-coverage 3-phage cocktail computed on the training interaction matrix.
- Generic: not "top 3 most frequent cocktails" in abstract; in paper semantics these are cocktails
  mostly assembled from later, less specific recommender steps.
- Tailored: cocktails containing phages selected via earlier, more specific steps.

The distinction matters because frequency-based post hoc labeling can be close to the paper labels but not identical.
