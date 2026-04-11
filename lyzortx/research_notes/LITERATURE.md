# Phage-Host Data and Literature Map (Performance-First)

Last updated: 2026-04-11

## Objective

- Maximize model performance for phage-lysis prediction and Top-k cocktail utility.
- Matching the exact schema of `raw_interactions.csv` is not required.
- Prefer sources that improve ranking, calibration, and generalization.

## Decision Rule for New External Inputs

- Prioritize data that provides direct supervised signal at strain level.
- Use weak-label metadata at large scale only after confidence-tiering.
- Use large genome/protein resources for representation learning and feature augmentation.
- Keep all external integrations measurable through ablation against internal-only baseline.

## Tier A: Highest-Value Supervised Interaction Data and Directly Actionable Methods

### 1) GenoPHI / Noonan et al. 2025 — Strain-level phage-host prediction

- Why it matters:
  - Most direct competitor/comparator: same task, same Guelin E. coli phages (94/96), AUROC 0.869.
  - 13.2M training run sweep shows ML pipeline (algorithm, class weighting, feature selection) matters 5-18x more than
    genomic representation -- directly actionable for our pipeline.
  - CatBoost + RFE + inverse-frequency class weighting is the empirically optimal classical ML configuration.
  - Contains our interaction matrix (402 strains x 94 phages) in its repo, confirming data overlap.
  - Experimentally validated: AUROC 0.84 on 1,328 novel interactions (56 BASEL phages x 25 ECOR strains).
  - 5-phage cocktail design achieves 97% E. coli strain coverage.
- Usable assets:
  - Expanded interaction matrix (402 strains, 170 strains beyond our 369).
  - Optimal ML pipeline configuration (CatBoost, RFE, class weighting parameters).
  - Feature importance rankings and SHAP-identified host genes.
  - Cocktail design framework (HDBSCAN on predictive feature profiles).
  - 1,328 novel interactions for independent validation.
- Access:
  - Preprint: `https://www.researchsquare.com/article/rs-8176565/v1`
  - Code: `https://github.com/Noonanav/GenoPHI`
- Gist: `GIST 2025 Noonan GenoPHI strain-level prediction.md`
- Recommended role:
  - ML pipeline benchmark and optimization reference. Expanded training data source.

### 2) Moriniere et al. 2026 — Phage receptor specificity prediction

- Why it matters:
  - Solves the phage-to-receptor mapping our AX03 cross-terms failed to provide.
  - 19 receptor classes for 193/255 E. coli phages; k-mer classifiers at median AUROC 0.99.
  - Receptor specificity is discrete, localized in short sequence motifs (not diffuse structural features).
  - Single amino acid changes can switch receptor specificity (Q206L: OmpF to OmpW).
  - GenoPHI framework already contains our Guelin phages; predictions may be obtainable by contacting authors.
  - Independent validation: 83.7% OMP/NGR accuracy, 96% LPS, zero false positives.
- Usable assets:
  - 260-phage receptor annotations (Table S1) for genus-level receptor mapping.
  - Receptor predictions for 18,398 NCBI phages (Dataset S7).
  - k-mer feature sets per receptor class (Datasets S6-S7).
  - GenoPHI v0.1 framework for retraining on our phage panel.
- Access:
  - Local PDF: `.scratch/Prediction of phage receptor specificity from genom data.pdf`
  - GenoPHI code: `https://github.com/Noonanav/GenoPHI`
  - Supplementary (Figshare): `https://doi.org/10.6084/m9.figshare.31930314`
  - PhageDataSheets: `https://iseq.lbl.gov/PhageDataSheets/Ecoli_phages/`
- Gist: `GIST 2026 Moriniere receptor specificity from genomes.md`
- Recommended role:
  - Receptor-class features for directed cross-terms. Mechanistic interpretability layer.
- Caveat:
  - Training on BW25113/BL21 (no O-antigen or capsule). O-antigen/capsule-mediated specificity not covered.

### 3) ViralHostRangeDB (VHRdb)

- Why it matters:
  - Purpose-built host-range resource with experimental host-range data.
  - Has structured API with interaction responses and datasource metadata.
  - Includes this project's E. coli dataset as one datasource.
- Usable labels:
  - Interaction response values (`0`, `1`, `2`) from API.
  - Per-datasource and aggregated host-range records.
- Caveats:
  - Some original study score scales are compressed to `0/1/2` in VHRdb representation.
  - External datasets (VHRdb, BASEL, KlebPhaCol, GPB) showed neutral cumulative lift in Track I ablations (TI09). Value
    may be in expanded strain coverage rather than direct label addition.
- Access:
  - Web: `https://viralhostrangedb.pasteur.cloud/`
  - API docs: `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
  - Datasources: `https://viralhostrangedb.pasteur.cloud/data-source/`
- Recommended role:
  - Expanded strain coverage source. Cross-reference for label quality.

### 4) BASEL phage collection (E. coli, original and completion)

- Why it matters:
  - High-quality E. coli phage-host phenotyping with strong mechanistic annotations.
  - Includes receptor usage, immunity sensitivity, and host-range outcomes.
  - Used by Noonan et al. for independent validation (56 BASEL phages x 25 ECOR strains).
- Usable labels:
  - Host-range phenotypes and related mechanistic metadata.
- Caveats:
  - External datasets showed neutral cumulative lift in Track I ablations (TI09).
  - Cohorts and host panels differ from this project and require careful harmonization.
- Access:
  - Original BASEL paper (2021): `https://pubmed.ncbi.nlm.nih.gov/34784345/`
  - BASEL completion paper (2025): `https://pubmed.ncbi.nlm.nih.gov/40193529/`
  - Dataset record: `https://zenodo.org/records/15736582`
- Recommended role:
  - Independent validation cohort. Mechanistic feature cross-reference.

### 5) PhageHostLearn / Klebsiella strain-level matrix

- Why it matters:
  - Strain-level interaction matrix and corresponding genomic resources.
  - ESM-2 embeddings of RBPs + K-locus proteins with XGBoost; ROC AUC 0.818 for Klebsiella.
  - Lab-validated: 93.8% top-5 hit ratio on 28 clinical isolates.
  - Pairs RBP embeddings with host surface protein embeddings -- the pairwise approach our project considered.
- Usable labels:
  - Spot-test-derived host range labels between phages and bacterial strains.
- Evaluation signals to reuse:
  - Leave-one-group-out cross-validation (LOGOCV).
  - Mean hit ratio @ `k` for practical recommendation quality.
- Access:
  - Paper: `https://www.nature.com/articles/s41467-024-48675-6`
  - Data package: `https://zenodo.org/records/11061100`
- Recommended role:
  - Methodological reference for pairwise embedding approaches. Cross-genus transfer benchmark.

### Additional collections

- KlebPhaCol:
  - Priority: `Low` (external data showed neutral lift in Track I; revisit if Klebsiella expansion becomes a goal)
  - Access: `https://www.klebphacol.org/` | `https://pubmed.ncbi.nlm.nih.gov/41261852/`
- Gut Phage Biobank:
  - Priority: `Low` (restricted to academic use; out-of-domain for E. coli)
  - Access: `https://www.nature.com/articles/s41467-025-61946-0`
- Felix d'Herelle Reference Center:
  - Priority: `Watchlist`
  - Access: `https://www.phage.ulaval.ca/en`

## Tier B: Large-Scale Weak-Label Host Links

### 6) Virus-Host DB

- Why it matters:
  - Broad virus-host association coverage from public genomes.
- Usable labels:
  - Virus-host associations, mainly positive links (not dense interaction matrices).
- Access:
  - Main: `https://www.genome.jp/virushostdb/`
- Recommended role:
  - Weak-label positive pool with strict confidence flags.

### 7) NCBI Virus + NCBI Datasets + BioSample host metadata

- Why it matters:
  - Largest practical source for scalable host metadata extraction.
- Access:
  - NCBI Virus metadata: `https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/virus/virus-metadata/`
  - BioSample attributes: `https://www.ncbi.nlm.nih.gov/biosample/docs/attributes/`
- Recommended role:
  - High-scale weak supervision and candidate-pair mining.

## Tier C: Feature and Representation Resources

### 8) INPHARED

- Access: `https://github.com/RyanCook94/inphared` | `https://pubmed.ncbi.nlm.nih.gov/36159887/`
- Role: Phage representation pretraining and feature enrichment.

### 9) PHROGs

- Access: `https://phrogs.lmge.uca.fr/` | `https://pubmed.ncbi.nlm.nih.gov/33538820/`
- Role: Domain/protein family features.

### 10) PHIStruct (structure-aware RBP embeddings)

- Why it matters:
  - Structure-aware PLM (SaProt) embeddings of RBPs show +7-9% F1 at <40% sequence identity over sequence-only PLMs.
  - Relevant as methodological reference, but general PLM embeddings proved redundant with phage family features in our
    project (AX07/AX08 dead end). The Moriniere receptor paper shows k-mers on specific loci outperform global
    embeddings for receptor prediction.
- Access:
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/39804673/`
  - Code: `https://github.com/bioinfodlsu/PHIStruct`
  - RBP structures: `https://zenodo.org/records/11202338`
- Recommended role:
  - Background methodological reference. Low priority for direct integration given AX07/AX08 findings.

### 11) Antiphage Landscape 2025 (defense-discovery LM resource)

- Access:
  - Preprint: `https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1`
  - Repo: `https://github.com/mdmparis/antiphage_landscape_2025`
  - Interactive map: `https://mdmparis.github.io/antiphage-landscape/`
- Gist: `GIST 2025 antiphage defense landscape preprint.md`
- Recommended role:
  - Defense-feature discovery reference. Not directly actionable for strain-level prediction.

### 12) Shang et al. 2025 — 27-tool phage-host prediction benchmark

- Why it matters:
  - Most comprehensive benchmark of phage-host prediction tools (27 tools, systematic feature analysis).
  - Key findings: CRISPR and prophage methods are complementary (only 30.3% agreement); protein-derived features
    outperform DNA-based; tail protein sequences alone match full-genome performance; k-mer distributions overlap at
    intra-genus level ("hard negatives"); over half of published tools are unusable.
- Access:
  - Paper: `https://academic.oup.com/bib/article/26/6/bbaf626/8341158`
- Recommended role:
  - Benchmark design reference. Calibration for where our approach sits in the broader landscape.

### 13) Malajczuk et al. 2026 — Strain-level AI review

- Why it matters:
  - Focused review concluding "predictive performance is tightly coupled to outcome definition, label resolution, and
    negative handling rather than model complexity." Recommends precision-recall AUC and MCC over AUROC for sparse
    matrices.
- Access:
  - Paper: `https://doi.org/10.1093/bib/bbag085`
- Recommended role:
  - Evaluation methodology reference. Supports our bootstrap CI approach.

## Literature Shortlist with Direct Project Relevance

### A) Strain-level prediction and treatment utility

- Gaborieau et al., Nature Microbiology (2024): `https://www.nature.com/articles/s41564-024-01832-5`
- Noonan et al. (2025) GenoPHI: `https://www.researchsquare.com/article/rs-8176565/v1`
- Moriniere et al. (2026) receptor specificity: see local PDF and Figshare
- Boeckaerts et al., Nature Communications (2024): `https://www.nature.com/articles/s41467-024-48675-6`
- BASEL collection (2021): `https://pubmed.ncbi.nlm.nih.gov/34784345/`
- BASEL completion (2025): `https://pubmed.ncbi.nlm.nih.gov/40193529/`

### B) Benchmarks and reviews

- Shang et al., Briefings in Bioinformatics (2025): `https://academic.oup.com/bib/article/26/6/bbaf626/8341158`
- Malajczuk et al., Briefings in Bioinformatics (2026): `https://doi.org/10.1093/bib/bbag085`

### C) Data and feature resources

- ViralHostRangeDB (2021): `https://pubmed.ncbi.nlm.nih.gov/33594411/`
- INPHARED (2022): `https://pubmed.ncbi.nlm.nih.gov/36159887/`
- PHIStruct (2025): `https://pubmed.ncbi.nlm.nih.gov/39804673/`

## Gap-to-Solution Map

Updated 2026-04-11 based on Moriniere 2026, Noonan/GenoPHI 2025, and APEX Phase 1-2 dead ends.

- Gap: ML pipeline may be suboptimal (LightGBM without feature selection or class weighting).
  - Fix: benchmark CatBoost + RFE + inverse-frequency class weighting on existing features (GenoPHI's optimal config).
  - Source: Noonan et al. 2025 (13.2M training runs: algorithm choice DMCC=0.365, representation DMCC=0.020-0.072).
- Gap: pairwise cross-terms are biologically meaningless without knowing which receptor each phage targets.
  - Fix: receptor-class predictions from Moriniere 2026 as directed features (receptor=OmpC x host_OmpC_score).
  - Source: Moriniere et al. 2026 (19 receptor classes, AUROC 0.99, GenoPHI v0.1).
- Gap: label noise and conflicting replicate/dilution signals.
  - Fix: image-assisted QC from core Nature dataset raw plaque images plus uncertainty-aware label tiers.
  - Sources: Gaborieau et al. 2024 data availability; VHRdb conflict handling.
- Gap: narrow-host prior collapse (NILS53 and similar).
  - Fix: receptor-directed features may help by giving the model mechanistic reasons to prefer specific narrow-host
    phages for specific receptor configurations. Per-phage class weighting (GenoPHI approach) may also help.
  - Sources: Moriniere et al. 2026; Noonan et al. 2025 (inverse-frequency weighting).
- Gap: defense features hurt top-3 ranking due to lineage confounding.
  - Fix: RFE feature selection (GenoPHI's winning strategy) may automatically exclude confounded defense subtypes.
    Alternatively, phylogroup-aware encoding or residualization.
  - Source: Noonan et al. 2025 (RFE as best feature selection method, DMCC=0.184).

### Retired gaps (dead ends confirmed)

- ~~Weak performance on novel/low-similarity phage contexts → PHIStruct structure-aware embeddings.~~
  Retired: general PLM embeddings (ProstT5+SaProt) proved redundant with phage family features (AX07/AX08).
  Moriniere shows k-mers on specific RBP loci outperform global embeddings for receptor prediction.
- ~~Limited mechanistic signal → receptor-first feature program (RBP/tailspike + host adsorption factors).~~
  Partially retired: bulk RBP features (physicochemical, PLM) are dead ends. Replaced by directed receptor-class
  features from Moriniere.

## Source-Quality Rubric (for Go / No-Go Decisions)

- Signal strength: matrix labels from assays > curated positive pairs > inferred metadata links.
- Label polarity: sources with both positive and negative evidence are preferred.
- Resolution: strain-level labels are preferred over species/genus-level labels.
- Practicality: scriptable download/API and clear licensing beat manual-only resources.
- Compatibility: better alignment when host/phage naming and genome identifiers can be normalized reliably.
