# Gist: 2025 Noonan — Phylogeny-Agnostic Strain-Level Prediction of Phage-Host Interactions (GenoPHI)

Last updated: 2026-04-11

## Citation

- Title: _Phylogeny-agnostic strain-level prediction of phage-host interactions from genomes_
- Authors: Noonan, Moriniere, Mutalik, Arkin et al.
- Status: bioRxiv preprint, November 2025; under review
- Preprint: https://www.researchsquare.com/article/rs-8176565/v1
- GenoPHI code: https://github.com/Noonanav/GenoPHI
- Local PDF: `.scratch/Phylogeny-agnostic strain-level prediction of phage-host interactions from genomes.pdf`

## Executive summary

GenoPHI is a phylogeny-agnostic framework for predicting whether a specific phage infects a specific bacterial strain,
using binary protein-family and k-mer features from both genomes. Across 5 datasets (128,357 interactions, 1,058
strains, 560 phages), it achieves AUROC 0.67-0.94 in cross-validation and 0.84 on 1,328 novel experimental
interactions. A 13.2 million training run sweep shows that ML pipeline configuration (algorithm, training strategy,
feature selection) matters 5-18x more than genomic representation for prediction quality.

This is our most direct competitor/comparator: same task, same E. coli dataset (94/96 of our Guelin phages), AUROC
0.869 on E. coli.

## What the paper does

### Datasets

Five published interaction datasets:

| Dataset | Strains | Phages | Interactions | Positive % |
|---|---|---|---|---|
| E. coli (subset, for CV) | 177 | 94 | 16,638 | 22.9% |
| E. coli (full, for validation) | 402 | 94 | 37,788 | 20.7% |
| Klebsiella mixed | 62 | 59 | 3,658 | 4.9% |
| K. pneumoniae | 138 | 46 | 6,348 | 2.4% |
| Pseudomonas | 23 | 19 | 437 | 36.2% |
| Vibrionaceae (held-out) | 256 | 248 | 63,488 | 2.1% |

The E. coli dataset uses 94 of our 96 Guelin phages (missing T4LD and T7_Portugal) x 402 strains (232 overlapping with
our 369). This confirms direct data sharing, likely via the Brisse lab.

### Feature representation

Two complementary binary presence-absence approaches:

- **Protein family features**: MMseqs2 clustering (identity 0.4, coverage 0.8) of all proteins from all genomes.
  Binary: does this genome contain a member of cluster X?
- **k-mer features**: amino acid k-mers (k=4 proteome-wide, k=6 standalone). Binary presence-absence. Filtered to
  remove single-genome k-mers.

Both host AND phage genomes are featurized -- the model sees genetic content from both sides.

### ML pipeline optimization (13.2M training runs)

Tested >200 parameter configurations across all datasets:

- 8 ML algorithms (CatBoost, Random Forest, KNN, LR, MLP, SVM, naive Bayes, gradient boosting)
- 6 feature selection methods (RFE, SHAP, SHAP-RFE, LASSO, chi-squared, SelectKBest)
- 14 training strategies (class weighting, feature filtering)
- Multiple k-mer lengths, clustering thresholds

**Winner**: CatBoost with RFE, inverse-frequency class weighting per phage, hierarchical 20-cluster strain splitting,
feature filtering requiring presence across >=2 strain clusters.

### The critical finding on what matters

| Factor | DMCC | Interpretation |
|---|---|---|
| Modeling algorithm | 0.365 | Most important by far |
| Training strategy | 0.203 | Class weighting and data handling |
| Feature selection method | 0.184 | RFE dominates |
| Genomic representation | 0.020-0.072 | Least important |

**Implication for our project**: we have been optimizing the least important factor (feature engineering via PLM
embeddings, physicochemical descriptors, cross-terms) while using an unoptimized ML pipeline (LightGBM without RFE,
without per-phage class weighting, without systematic algorithm comparison).

### Cross-validation performance

| Dataset | AUROC | MCC |
|---|---|---|
| Vibrionaceae | 0.941 | 0.533 |
| Klebsiella-2 | 0.878 | 0.492 |
| E. coli | 0.869 | 0.536 |
| Pseudomonas | 0.745 | 0.326 |
| Klebsiella-1 | 0.674 | 0.130 |

Performance scales with dataset size. E. coli AUROC 0.869 matches Gaborieau et al. 2024 benchmark (0.86) which
required prior mechanistic knowledge -- GenoPHI achieves the same without mechanistic assumptions.

### Experimental validation

**Novel interaction matrix (1,328 interactions):**
- 56 BASEL phages x 25 ECOR strains (not in training)
- AUROC = 0.84 on genuinely unseen phages (honestly below CV)

**RB-TnSeq validation in ECOR27:**
- 157,637 barcoded mutants, 4,032 genes, screened against 21 phages
- 51 high-scoring host genes; 68.8% linked to model's predictive features (5.0x above random, p = 2.8e-6)
- Identified known receptors (tsx, fadL, nfrA) and cell wall biosynthesis pathways

### Cocktail design

5-phage cocktails using HDBSCAN clustering of predictive feature profiles:
- 97.0% strain coverage for E. coli (vs 80.7% promiscuity-based baseline)
- Up to 6.2x improvement over promiscuity selection in one-shot phage selection

### Biological insights from SHAP

Top predictive features include:
- Defense systems (RM): negative impact when present (as expected)
- Cell wall biosynthesis (LPS, O-antigen, capsule): major strain-side determinants
- Putrescine catabolism (goaG/puuE, gabT): novel finding
- 30 adsorption factors significant vs only 2 defense systems (consistent with Gaborieau 2024 finding)

### Stated limitation and where PLMs fit

Direct quote from their discussion:

> "In the future, representations that capture structural information, such as protein language model embeddings, may
> improve cross-genus transfer learning."

They explicitly position PLM embeddings as the next step for cross-genus generalization, while acknowledging that
binary k-mer/protein-family features already work well within-genus.

## Relevance to our project

### Direct comparison

GenoPHI AUROC 0.869 on E. coli vs our 0.830 on ST03 holdout. Not directly comparable (different holdout strategies,
different bacteria counts), but directionally they outperform us with simpler features and a more optimized ML
pipeline.

### What we should learn from this

1. **ML pipeline optimization may matter more than features.** Try CatBoost, RFE feature selection, per-phage class
   weighting on our existing features before adding new feature engineering.
2. **Binary features work.** Their protein-family presence-absence captures information our continuous scores may
   overcomplicate. Consider whether simpler encodings of our features could help.
3. **Both-sides featurization is key.** They featurize both phage and host genomes with the same representation. Our
   approach uses different feature types per side, which may be fine but makes interaction modeling harder.
4. **The cocktail design framework is directly reusable** on top of our models.

### What we have that they don't

- **Continuous receptor variant scores** (79 OmpC values vs binary presence): finer host-side discrimination
- **Per-phage blending** (our biggest gain, +2.0pp AUC): orthogonal to their approach
- **Mechanistically grounded features**: receptor HMM scores, defense counts, typing -- vs their annotation-free k-mers
- **The receptor specificity predictions** from the companion Moriniere paper can be added as directed features

### Complementary, not competitive

The most promising path is combining their ML pipeline insights (CatBoost, RFE, class weighting) with our richer
mechanistic features and receptor-directed cross-terms. Their features and ours encode different information; combining
both feature sets could outperform either alone.

## Sources

- Preprint: https://www.researchsquare.com/article/rs-8176565/v1
- GenoPHI repo: https://github.com/Noonanav/GenoPHI
- Local PDF: `.scratch/Phylogeny-agnostic strain-level prediction of phage-host interactions from genomes.pdf`
- Companion receptor paper: Moriniere et al. 2026 (see separate gist)
- Gaborieau et al. 2024 (our core data source): https://www.nature.com/articles/s41564-024-01832-5
