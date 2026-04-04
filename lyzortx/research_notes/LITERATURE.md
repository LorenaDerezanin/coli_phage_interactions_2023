# Phage-Host Data and Literature Map (Performance-First)

Last updated: 2026-04-05

## Objective

- Maximize model performance for phage-lysis prediction and Top-k cocktail utility.
- Matching the exact schema of `raw_interactions.csv` is not required.
- Prefer sources that improve ranking, calibration, and generalization.

## Decision Rule for New External Inputs

- Prioritize data that provides direct supervised signal at strain level.
- Use weak-label metadata at large scale only after confidence-tiering.
- Use large genome/protein resources for representation learning and feature augmentation.
- Keep all external integrations measurable through ablation against internal-only baseline.

## Tier A: Highest-Value Supervised Interaction Data

### 1) ViralHostRangeDB (VHRdb)

- Why it matters:
  - Purpose-built host-range resource with experimental host-range data.
  - Has structured API with interaction responses and datasource metadata.
  - Includes this project's E. coli dataset as one datasource.
- Usable labels:
  - Interaction response values (`0`, `1`, `2`) from API.
  - Per-datasource and aggregated host-range records.
- Caveats:
  - Some original study score scales are compressed to `0/1/2` in VHRdb representation.
  - API responses are in the VHRdb global response scheme (not raw source-native responses).
  - For highest-fidelity labels, cross-reference original publications or source repositories.
- Access:
  - Web: `https://viralhostrangedb.pasteur.cloud/`
  - API docs: `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
  - Datasources: `https://viralhostrangedb.pasteur.cloud/data-source/`
- Recommended role:
  - Primary external supervised source for Track I.

### 2) PhageHostLearn / Klebsiella strain-level matrix

- Why it matters:
  - Strain-level interaction matrix and corresponding genomic resources.
  - Explicitly designed for host-range machine learning.
- Usable labels:
  - Spot-test-derived host range labels between phages and bacterial strains.
- Evaluation signals to reuse:
  - Leave-one-group-out cross-validation (LOGOCV).
  - Mean hit ratio @ `k` for practical recommendation quality.
- Access:
  - Paper: `https://www.nature.com/articles/s41467-024-48675-6`
  - Data package: `https://zenodo.org/records/11061100`
- Recommended role:
  - External supervised training domain for transfer learning and robustness tests.

### 3) BASEL phage collection (E. coli, original and completion)

- Why it matters:
  - High-quality E. coli phage-host phenotyping with strong mechanistic annotations.
  - Includes receptor usage, immunity sensitivity, and host-range outcomes.
  - Directly relevant to feature hypotheses for Tracks C, D, and E.
- Usable labels:
  - Host-range phenotypes and related mechanistic metadata from publications and dataset releases.
- Access:
  - Original BASEL paper (2021): `https://pubmed.ncbi.nlm.nih.gov/34784345/`
  - BASEL completion paper (2025): `https://pubmed.ncbi.nlm.nih.gov/40193529/`
  - Dataset record: `https://zenodo.org/records/15736582`
- Caveats:
  - Cohorts and host panels differ from this project and require careful harmonization.
  - Some key assets are distributed across paper supplements and linked repositories.
- Recommended role:
  - Tier A source for E. coli external supervision and mechanistic feature transfer.

### 4) Gut Phage Biobank (GPB) and CNGB-linked resources

- Why it matters:
  - Large cultured phage collection with infection matrix data in gut-associated systems.
  - Valuable for cross-cohort validation and broad host-range priors.
- Usable labels:
  - Infection matrix and associated isolate metadata from the publication resources.
- Access:
  - Paper: `https://www.nature.com/articles/s41467-025-61946-0`
  - Data accessions in paper data-availability section (CNGBdb / public repositories).
- Caveats:
  - Data and resources are restricted to academic research use per paper statement.
- Recommended role:
  - Secondary supervised source and out-of-domain stress-test cohort.

### Additional BASEL-like collections (priority-tagged)

- KlebPhaCol:
  - Priority: `Medium`
  - Why it matters: open host-range-tested phage/strain collection with searchable data and manuscript.
  - Access:
    - Collection site: `https://www.klebphacol.org/`
    - Manuscript (NAR 2025): `https://pubmed.ncbi.nlm.nih.gov/41261852/`
  - Recommended role: next supervised external cohort after VHRdb/BASEL ingestion.
- Felix d'Herelle Reference Center:
  - Priority: `Medium`
  - Why it matters: large historical and reference phage collection with searchable catalog.
  - Access:
    - Center site: `https://www.phage.ulaval.ca/en`
    - Catalog search: `https://www.phage.ulaval.ca/en/rechercher-catalogue-phages`
  - Recommended role: discovery and curation source for candidate phage-host pairs and metadata.
- Phage Collection Project Portal:
  - Priority: `Watchlist`
  - Why it matters: emerging searchable portal connected to community phage collections.
  - Access:
    - Portal: `https://portal.phage-collection.org/`
  - Recommended role: monitor maturity and API/export support before heavy ingestion investment.

## Tier B: Large-Scale Weak-Label Host Links

### 5) Virus-Host DB

- Why it matters:
  - Broad virus-host association coverage from public genomes.
  - Good for positive-pair expansion and pretraining priors.
- Usable labels:
  - Virus-host associations, mainly positive links (not dense interaction matrices).
- Access:
  - Main: `https://www.genome.jp/virushostdb/`
  - About: `https://www.genome.jp/virushostdb/about`
- Recommended role:
  - Weak-label positive pool with strict confidence flags.

### 6) NCBI Virus + NCBI Datasets + BioSample host metadata

- Why it matters:
  - Largest practical source for scalable host metadata extraction.
  - Supports automated retrieval pipelines through APIs and CLI.
- Usable labels:
  - Host-related fields from virus reports and BioSample metadata.
  - Mostly positive-only or metadata-derived signals.
- Access:
  - NCBI Virus metadata how-to: `https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/virus/virus-metadata/`
  - Virus report schema: `https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/data-reports/virus/`
  - BioSample attributes: `https://www.ncbi.nlm.nih.gov/biosample/docs/attributes/`
- Recommended role:
  - High-scale weak supervision, pretraining, and candidate-pair mining.

### 7) Public read archives for future high-effort extraction

- Why it matters:
  - May contain unpublished interaction screens and host-range experiments.
- Usable labels:
  - Potential matrix reconstruction from project metadata and raw assays.
- Access:
  - NCBI SRA / ENA / CNGBdb projects (case-by-case).
- Recommended role:
  - Long-term program, not an immediate blocker for baseline delivery.

## Tier C: Feature and Representation Resources

### 8) INPHARED

- Why it matters:
  - Curated phage genome resource useful for sequence feature engineering.
- Access:
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/36159887/`
  - Repo: `https://github.com/RyanCook94/inphared`
- Recommended role:
  - Phage representation pretraining and feature enrichment.

### 9) PHROGs

- Why it matters:
  - Protein-family annotation resource tailored to phages.
- Access:
  - Site: `https://phrogs.lmge.uca.fr/`
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/33538820/`
- Recommended role:
  - Domain/protein family features for Track D and Track E.

### 10) PhageScope

- Why it matters:
  - Integrative phage genome and metadata resource with analysis tooling.
- Access:
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/37904614/`
- Recommended role:
  - Taxonomy, quality checks, and auxiliary phage metadata features.

### 11) PhageDive

- Why it matters:
  - Curated phage metadata explorer and downloadable resource.
- Access:
  - Site: `https://www.phagedive.com/`
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/39373542/`
- Recommended role:
  - Phage metadata harmonization and candidate feature augmentation.

### 12) PHIStruct (structure-aware RBP embeddings)

- Why it matters:
  - Targets a core steel-thread gap: low-similarity generalization for new phages.
  - Uses structure-aware embeddings of RBPs and reports gains at lower train-test similarity.
- Access:
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/39804673/`
  - Full text: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11783280/`
  - Code: `https://github.com/bioinfodlsu/PHIStruct`
  - Accompanying RBP structures dataset: `https://zenodo.org/records/11202338`
- Recommended role:
  - Pilot representation-learning branch for Track D/Track G, especially for phage-family holdouts.

### 13) Raw plaque image assets from the core E. coli study

- Why it matters:
  - Directly addresses steel-thread label-noise risk by enabling image-assisted QC of uncertain pairs.
  - Supports uncertainty flags beyond hard aggregation of replicate labels.
- Access:
  - Core paper (Data availability): `https://www.nature.com/articles/s41564-024-01832-5`
- Recommended role:
  - Optional but high-value label-audit enhancer before large-scale weak-label ingestion.

### 14) Antiphage Landscape 2025 (defense-discovery LM resource)

- Why it matters:
  - Strong recent defense-discovery paper from the same broader group/community as the core _Escherichia_ work.
  - Provides a public code repo, supplementary candidate tables, and an interactive map of predicted antiphage protein
    space.
  - Relevant mainly as a feature-discovery and hypothesis-generation resource, not as a direct supervised host-range
    matrix.
- Access:
  - Preprint: `https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1`
  - Code/data repo: `https://github.com/mdmparis/antiphage_landscape_2025`
  - Interactive map: `https://mdmparis.github.io/antiphage-landscape/`
- Caveats:
  - As of 2026-04-05 this still appears to be a preprint rather than a verified journal publication.
  - The paper does not provide a new strain-level _E. coli_ lysis matrix for direct training.
  - The models are discovery-oriented and low-precision enough that manual curation remains central.
- Recommended role:
  - Use for candidate defense-feature mining and background biological calibration, after adsorption-side features and
    same-host-genus supervised data are addressed.

## Literature Shortlist with Direct Project Relevance

### A) Strain-level prediction and treatment utility

- Gaborieau et al., Nature Microbiology (2024): `https://www.nature.com/articles/s41564-024-01832-5`
- BASEL collection, PLOS Biology (2021): `https://pubmed.ncbi.nlm.nih.gov/34784345/`
- BASEL completion, PLOS Biology (2025): `https://pubmed.ncbi.nlm.nih.gov/40193529/`
- KlebPhaCol, Nucleic Acids Research (2025): `https://pubmed.ncbi.nlm.nih.gov/41261852/`
- Boeckaerts et al., Nature Communications (2024): `https://www.nature.com/articles/s41467-024-48675-6`
- Gut Phage Biobank, Nature Communications (2025): `https://www.nature.com/articles/s41467-025-61946-0`

### B) Foundational host prediction methods

- WIsH (2017): `https://pubmed.ncbi.nlm.nih.gov/28961777/`
- VirHostMatcher (2016): `https://pubmed.ncbi.nlm.nih.gov/27634843/`
- PHIST (2022): `https://pubmed.ncbi.nlm.nih.gov/35231951/`
- PHIStruct (2025): `https://pubmed.ncbi.nlm.nih.gov/39804673/`
- Digital phagograms review (2022): `https://pubmed.ncbi.nlm.nih.gov/34952265/`

### C) Data and benchmarking resources

- ViralHostRangeDB paper (2021): `https://pubmed.ncbi.nlm.nih.gov/33594411/`
- INPHARED paper (2022): `https://pubmed.ncbi.nlm.nih.gov/36159887/`
- PhageScope paper (2024): `https://pubmed.ncbi.nlm.nih.gov/37904614/`
- PhageDive paper (2025): `https://pubmed.ncbi.nlm.nih.gov/39373542/`

## Integration Strategy for Best Results (Track I)

### Immediate (now)

- Build a `source_registry.csv` with one row per datasource and these fields: `source_name`, `source_type`,
  `label_kind`, `host_resolution`, `assay_type`, `confidence_tier`, `license`, `access_path`, `last_checked`.
- Implement VHRdb and BASEL ingestion first for fastest high-signal supervised lift.
- Queue KlebPhaCol as the next supervised ingest after VHRdb/BASEL normalization pass.
- In parallel, define confidence tiers for weak labels from Virus-Host DB and NCBI metadata.

### Near-term (after baseline refresh)

- Train with internal-only data first and freeze a strong calibrated baseline.
- Add Tier A external supervised datasets and run strict ablations.
- Add Tier B weak labels only with confidence filtering and noise-robust training.
- Use Tier C resources to improve phage/host feature blocks and pairwise compatibility features.

### Guardrails

- Do not mix all external data at once; add one source family at a time.
- Report all key metrics with cohort denominators (`raw369`, `matrix402`, `features404`, `external`).
- Keep cross-source leakage checks mandatory before accepting any benchmark gains.
- For VHRdb ingestion, keep source-fidelity metadata: global scheme response, per-datasource response, disagreement
  flag, and link back to source-native labels when available.

## Gap-to-Solution Map (Steel Thread -> Literature-Backed Fixes)

- Gap: label noise and conflicting replicate/dilution signals.
  - Fix: image-assisted QC from the core Nature dataset raw plaque images plus uncertainty-aware label tiers.
  - Sources:
    - Core paper data availability and raw image release: `https://www.nature.com/articles/s41564-024-01832-5`
    - VHRdb conflict/disagreement handling and response schemes:
      `https://hub.pages.pasteur.fr/viralhostrangedb/explore.html`
      `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
- Gap: weak performance on novel/low-similarity phage contexts.
  - Fix: structure-aware RBP embeddings (PHIStruct) as a Track D/Track G pilot branch.
  - Source:
    - `https://pubmed.ncbi.nlm.nih.gov/39804673/`
- Gap: recommender objective drift vs practical top-k utility.
  - Fix: LOGOCV-style grouped validation and mean hit ratio @ `k` as first-class evaluation signals.
  - Source:
    - PhageHostLearn practical validation setup: `https://www.nature.com/articles/s41467-024-48675-6`
- Gap: limited mechanistic signal in metadata-heavy baseline features.
  - Fix: receptor-first feature program (RBP/tailspike/depolymerase + host adsorption factors).
  - Sources:
    - Core E. coli study on adsorption determinants: `https://www.nature.com/articles/s41564-024-01832-5`
    - BASEL collection and completion: `https://pubmed.ncbi.nlm.nih.gov/34784345/`
      `https://pubmed.ncbi.nlm.nih.gov/40193529/`

## Source-Quality Rubric (for Go / No-Go Decisions)

- Signal strength:
  - Matrix labels from assays > curated positive pairs > inferred metadata links.
- Label polarity:
  - Sources with both positive and negative evidence are preferred.
- Resolution:
  - Strain-level labels are preferred over species/genus-level labels.
- Practicality:
  - Scriptable download/API and clear licensing beat manual-only resources.
- Compatibility:
  - Better alignment when host/phage naming and genome identifiers can be normalized reliably.

## Inference Notes

- Inference from sources: the strongest near-term performance gains should come from Tier A supervised matrices plus
  Tier C feature resources, then Tier B weak labels.
- Inference from sources: weak-label scale is valuable, but only after confidence tiering and robust ablation design to
  prevent noise-driven regressions.
