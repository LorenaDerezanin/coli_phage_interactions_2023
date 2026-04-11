# Phage-Host Literature and Data Map

Last updated: 2026-04-11

## Tier A: Supervised Interaction Data and Directly Actionable Methods

### 1) GenoPHI / Noonan et al. 2025 — Strain-level phage-host prediction

- Key findings:
  - Same task, same Guelin E. coli phages (94/96), AUROC 0.869 on 402 strains.
  - 13.2M training run sweep: algorithm choice DMCC=0.365, training strategy DMCC=0.203, feature selection DMCC=0.184,
    but genomic representation only DMCC=0.020-0.072. ML pipeline matters 5-18x more than feature representation.
  - CatBoost + RFE + inverse-frequency class weighting is the empirically optimal classical ML configuration.
  - Experimentally validated: AUROC 0.84 on 1,328 novel interactions (56 BASEL phages x 25 ECOR strains).
  - 5-phage cocktail design achieves 97% E. coli strain coverage.
- Usable assets:
  - Expanded interaction matrix (402 strains x 94 phages, contains our data).
  - Optimal ML pipeline configuration (CatBoost, RFE, class weighting parameters).
  - Feature importance rankings and SHAP-identified host genes.
  - Cocktail design framework (HDBSCAN on predictive feature profiles).
  - 1,328 novel interactions for independent validation.
- Access:
  - Preprint: `https://www.researchsquare.com/article/rs-8176565/v1`
  - Code: `https://github.com/Noonanav/GenoPHI`
- Gist: `GIST 2025 Noonan GenoPHI strain-level prediction.md`

### 2) Moriniere et al. 2026 — Phage receptor specificity prediction

- Key findings:
  - 19 receptor classes for 193/255 E. coli phages; k-mer classifiers at median AUROC 0.99.
  - Receptor specificity is discrete, localized in short hypervariable sequence motifs at RBP tip domains.
  - Single amino acid changes can switch receptor specificity (Q206L: OmpF to OmpW).
  - Independent validation: 83.7% OMP/NGR accuracy, 96% LPS, zero false positives.
  - Only 13 of ~60-180 E. coli K-12 OMPs serve as phage receptors -- convergence on conserved, highly expressed OMPs.
- Usable assets:
  - 260-phage receptor annotations (Table S1).
  - Receptor predictions for 18,398 NCBI phages (Dataset S7).
  - k-mer feature sets per receptor class (Datasets S6-S7).
  - GenoPHI v0.1 framework (retrainable).
- Caveat:
  - Trained on BW25113/BL21 (no O-antigen or capsule). O-antigen/capsule-mediated specificity not covered.
- Access:
  - Local PDF: `.scratch/Prediction of phage receptor specificity from genom data.pdf`
  - GenoPHI code: `https://github.com/Noonanav/GenoPHI`
  - Supplementary (Figshare): `https://doi.org/10.6084/m9.figshare.31930314`
  - PhageDataSheets: `https://iseq.lbl.gov/PhageDataSheets/Ecoli_phages/`
- Gist: `GIST 2026 Moriniere receptor specificity from genomes.md`

### 3) ViralHostRangeDB (VHRdb)

- What it provides:
  - Purpose-built host-range resource with experimental host-range data.
  - Structured API with interaction responses and datasource metadata.
  - Includes this project's E. coli dataset as one datasource.
- Usable labels:
  - Interaction response values (`0`, `1`, `2`) from API.
  - Per-datasource and aggregated host-range records.
- Caveat:
  - Some original study score scales are compressed to `0/1/2`.
- Access:
  - Web: `https://viralhostrangedb.pasteur.cloud/`
  - API docs: `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
  - Datasources: `https://viralhostrangedb.pasteur.cloud/data-source/`

### 4) BASEL phage collection (E. coli, original and completion)

- What it provides:
  - High-quality E. coli phage-host phenotyping with strong mechanistic annotations.
  - Includes receptor usage, immunity sensitivity, and host-range outcomes.
  - Used by Noonan et al. for independent validation (56 BASEL phages x 25 ECOR strains).
- Usable labels:
  - Host-range phenotypes and related mechanistic metadata.
- Caveat:
  - Cohorts and host panels differ from this project; requires careful harmonization.
- Access:
  - Original BASEL paper (2021): `https://pubmed.ncbi.nlm.nih.gov/34784345/`
  - BASEL completion paper (2025): `https://pubmed.ncbi.nlm.nih.gov/40193529/`
  - Dataset record: `https://zenodo.org/records/15736582`

### 5) PhageHostLearn / Klebsiella strain-level matrix

- Key findings:
  - ESM-2 embeddings of RBPs + K-locus proteins with XGBoost; ROC AUC 0.818 for Klebsiella.
  - Lab-validated: 93.8% top-5 hit ratio on 28 clinical isolates.
  - Pairwise approach: RBP embeddings paired with host surface protein embeddings.
  - LOGOCV evaluation; mean hit ratio @ k for practical recommendation quality.
- Usable assets:
  - Strain-level interaction matrix and corresponding genomic resources.
  - Spot-test-derived host range labels.
- Access:
  - Paper: `https://www.nature.com/articles/s41467-024-48675-6`
  - Data package: `https://zenodo.org/records/11061100`

### Additional collections

- KlebPhaCol:
  - Access: `https://www.klebphacol.org/` | `https://pubmed.ncbi.nlm.nih.gov/41261852/`
- Gut Phage Biobank:
  - Access: `https://www.nature.com/articles/s41467-025-61946-0`
  - Note: restricted to academic use; gut phage focus, not E. coli-specific.
- Felix d'Herelle Reference Center:
  - Access: `https://www.phage.ulaval.ca/en`

## Tier B: Large-Scale Weak-Label Host Links

### 6) Virus-Host DB

- What it provides:
  - Broad virus-host association coverage from public genomes.
  - Mainly positive links (not dense interaction matrices).
- Access:
  - Main: `https://www.genome.jp/virushostdb/`

### 7) NCBI Virus + NCBI Datasets + BioSample host metadata

- What it provides:
  - Largest practical source for scalable host metadata extraction.
- Access:
  - NCBI Virus metadata: `https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/virus/virus-metadata/`
  - BioSample attributes: `https://www.ncbi.nlm.nih.gov/biosample/docs/attributes/`

## Tier C: Feature and Representation Resources

### 8) INPHARED

- Access: `https://github.com/RyanCook94/inphared` | `https://pubmed.ncbi.nlm.nih.gov/36159887/`
- What it provides: Curated phage genome database from NCBI GenBank, updated monthly.

### 9) PHROGs

- Access: `https://phrogs.lmge.uca.fr/` | `https://pubmed.ncbi.nlm.nih.gov/33538820/`
- What it provides: Prokaryotic virus protein families clustered by remote homology.

### 10) PHIStruct (structure-aware RBP embeddings)

- Key findings:
  - Structure-aware PLM (SaProt) embeddings of RBPs show +7-9% F1 at <40% sequence identity over sequence-only PLMs.
  - Structural features help most when sequence similarity is low.
- Access:
  - Paper: `https://pubmed.ncbi.nlm.nih.gov/39804673/`
  - Code: `https://github.com/bioinfodlsu/PHIStruct`
  - RBP structures: `https://zenodo.org/records/11202338`

### 11) DepoScope / Concha-Eloko et al. 2024 (depolymerase annotation)

- Key findings:
  - State-of-the-art tool for phage depolymerase detection and amino-acid-level enzymatic domain delineation.
  - Fine-tuned ESM-2 + CNN; MCC 0.455 on independent benchmark (>2x next best: PhageDPO 0.178, DePP 0.131).
  - Classifies fold type (right-handed beta-helix, n-bladed beta-propeller, triple helix) and pinpoints domain
    boundaries per residue.
  - Same group as PhageHostLearn (Boeckaerts/Briers, Ghent).
- Usable assets:
  - Pre-trained ESM-2 models (6L, 12L, 30L) for binary classification and domain delineation.
  - Curated PD fold database and HMM profiles for polysaccharide-degrading domains.
  - Training dataset of 1,926 positive + 1,409 negative proteins with domain annotations.
- Caveats:
  - Recall weakness on triple helix folds (18/26 missed due to intrinsic disorder before chaperone-assisted
    trimerization). Union with DePP (91.6% recall) recommended for comprehensive screening.
  - Detects depolymerases and delineates domains but does not predict substrate specificity. No tool predicts E. coli
    K-antigen specificity from depolymerase sequence. For Klebsiella, Gittrich et al. 2025 (Nature Comms) demonstrated
    a prophage-mining approach using DAG models and sequence clustering to predict capsular tropism.
- Access:
  - Paper: `https://doi.org/10.1371/journal.pcbi.1011831`
  - Code: `https://github.com/dimiboeckaerts/DepoScope`
  - Data: `https://zenodo.org/records/10957073`
- Gist: `GIST 2024 Concha-Eloko DepoScope depolymerase annotation.md`

### 12) Antiphage Landscape 2025 (defense-discovery LM resource)

- Access:
  - Published: `https://doi.org/10.1126/science.adv8275` (Science, April 2026)
  - Preprint: `https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1`
  - Repo: `https://github.com/mdmparis/antiphage_landscape_2025`
  - Interactive map: `https://mdmparis.github.io/antiphage-landscape/`
- Gist: `GIST 2025 antiphage defense landscape.md`
- What it provides: Comprehensive defense system catalog and protein language model for defense gene discovery.

### 13) Shang et al. 2025 — 27-tool phage-host prediction benchmark

- Key findings:
  - Most comprehensive benchmark of phage-host prediction tools (27 tools, systematic feature analysis).
  - CRISPR and prophage methods are complementary (only 30.3% agreement).
  - Protein-derived features outperform DNA-based; tail protein sequences alone match full-genome performance.
  - k-mer distributions overlap at intra-genus level ("hard negatives").
  - Over half of published tools are unusable.
- Access:
  - Paper: `https://academic.oup.com/bib/article/26/6/bbaf626/8341158`

### 14) Malajczuk et al. 2026 — Strain-level AI review

- Key findings:
  - "Predictive performance is tightly coupled to outcome definition, label resolution, and negative handling rather
    than model complexity."
  - Recommends precision-recall AUC and MCC over AUROC for sparse matrices.
- Access:
  - Paper: `https://doi.org/10.1093/bib/bbag085`

## Literature Shortlist

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
- DepoScope / Concha-Eloko et al., PLoS Comp Bio (2024): `https://doi.org/10.1371/journal.pcbi.1011831`
- PHIStruct (2025): `https://pubmed.ncbi.nlm.nih.gov/39804673/`

## Source-Quality Rubric

- Signal strength: matrix labels from assays > curated positive pairs > inferred metadata links.
- Label polarity: sources with both positive and negative evidence are preferred.
- Resolution: strain-level labels are preferred over species/genus-level labels.
- Practicality: scriptable download/API and clear licensing beat manual-only resources.
- Compatibility: better alignment when host/phage naming and genome identifiers can be normalized reliably.
