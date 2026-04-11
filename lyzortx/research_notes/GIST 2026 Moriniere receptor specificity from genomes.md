# Gist: 2026 Moriniere — Prediction of Phage Receptor Specificity from Genome Data

Last updated: 2026-04-11

## Citation

- Title: _Prediction of phage receptor specificity from genome data_
- Authors: Moriniere, Noonan, Johnson, Mutalik, Arkin et al.
- Status: 2026, presumed published or late-stage preprint (companion to Noonan et al. 2025 GenoPHI)
- Local PDF: `.scratch/Prediction of phage receptor specificity from genom data.pdf`
- GenoPHI code: https://github.com/Noonanav/GenoPHI
- PhageDataSheets browser: https://iseq.lbl.gov/PhageDataSheets/Ecoli_phages/
- Supplementary data (Figshare): https://doi.org/10.6084/m9.figshare.31930314
- Patent filed: US Provisional 63/830,352

## Executive summary

This paper experimentally identifies which bacterial surface receptor each phage binds for 193 of 255 E. coli phages,
across 19 receptor classes (8 OMP, 4 LPS core sugars, 1 polysaccharide, plus secondary receptors). It then builds
k-mer-based binary classifiers (GenoPHI v0.1) that predict receptor class from phage proteome alone with median AUROC
0.99. Independent validation on 49 new phages achieves 83.7% accuracy for OMP/NGR and 96% for LPS, with zero false
positives. Receptor-switching experiments confirm the molecular basis down to single amino acid resolution.

This is the most directly actionable external paper for our project: it solves the phage-to-receptor mapping that our
AX03 cross-terms failed to provide, and it uses a framework (GenoPHI) that already contains our Guelin phage panel.

## What the paper does

### Phage collection

- 255 unique E. coli dsDNA phages, whole-genome sequenced
- 224 from labs worldwide + 31 isolated in-house
- 129 myoviruses, 101 siphoviruses, 25 podoviruses; 10 families, 19 subfamilies, 50 genera
- Spans nearly all genomic clusters in a vConTACT2 network of 1,875 NCBI E. coli phages

### Experimental screens

- 1,050 genome-wide RB-TnSeq + Dub-seq screens on E. coli BW25113 and BL21
- Two host strains chosen because they lack O-antigen and capsule, so OMP receptors are directly exposed
- Validated with EOP assays on KEIO knockouts and ASKA overexpression strains (~2,000 individual observations)

### 19 receptor classes

Receptors assigned to 193/255 phages:

- 8 OMP receptors: Tsx, FadL, OmpA, OmpF, OmpC, OmpW (novel), NupG (novel), plus HepI-requiring secondary
- 4 LPS core sugar classes: Kdo, HepI, HepII, GluI
- 1 polysaccharide: NGR (N4-glycan receptor)
- Only 13 of estimated 60-180 E. coli K-12 OMPs serve as phage receptors -- convergence on broadly conserved, highly
  expressed, centrally positioned OMPs

### Receptor-binding protein identification

**Straboviridae (T-even-like, 73 phages):**
- Gp38 adhesins on long tail fibers with 4 C-terminal hypervariable sequence domains (HVS1-4)
- Allelic variation in HVSs correlates with porin specificity
- Single amino acid change Q206L in HVS3 switches RB68 from OmpF to OmpW
- Gp12 short tail fibers: C-terminal allelic variability correlates with LPS core sugar targeting

**Drexlerviridae and related siphoviruses (101 phages, 81 assigned):**
- Central tail fiber gene gpJ with hypervariable downstream locus
- 4 unrelated protein families in this locus target FhuA, BtuB, LptD, or YncD (>30% identity within, <30% between)
- AlphaFold3 confirmed these as bona fide RBPs

**NGR-targeting phages:**
- Perfectly conserved 5-aa 'GMSHY' motif across all NGR-targeting phages
- Convergent evolution (not HGT) -- present in morphologically diverse myoviruses and podoviruses

### ML modeling: GenoPHI v0.1

- Input: amino acid 5-mers (presence-absence) across the entire phage proteome -- annotation-free
- 13 independent binary classifiers (gradient-boosted decision trees), one per receptor class with >=4 positives
- Recursive feature elimination (RFE) per model, restricted to training partitions
- Cross-validation (20-fold, 10% balanced holdouts): median AUROC = 0.99 (range 0.62-1.00), median AUPR = 0.91
- OmpA is the weak point (AUROC ~0.62) due to insufficient allelic diversity in training
- k-mer features map to the same RBP domains identified by comparative genomics in 11/13 classes
- For 6 classes with AlphaFold3 models, features mapped to interface-proximal residues

### Independent validation (49 new phages)

- Trained on initial 206 phages, predicted on 49 acquired based on model predictions
- OMP/NGR accuracy: 41/49 (83.7%); LPS core sugar: 47/49 (96%)
- Zero false positives -- perfect precision
- All failures were false negatives from insufficient RBP diversity in training set
- Final models retrained on all 255 phages, applied to 18,398 NCBI Caudoviricetes genomes (S7)

### Receptor-switching experiments

- Gp38 gene swap between T6-like phages: complete receptor switch; double mutant created novel phenotype absent from
  parents
- Q206L single amino acid substitution: necessary and sufficient to switch OmpF <-> OmpW
- Models correctly predicted every engineered variant's phenotype

## Key limitation

Training on BW25113/BL21 which lack O-antigen and capsule. O-antigen and capsular polysaccharide receptors are
explicitly not covered. The authors cite their companion Klebsiella preprint (Gittrich et al. 2026) as proof of concept
for extending to capsule-mediated specificity.

## Relevance to our project

### Direct overlap

GenoPHI GitHub contains `ecoli_interaction_matrix.csv` with 94 of our 96 Guelin phages x 402 strains -- our data is
already in their framework, likely via the Brisse lab. However, our phages are NOT in the S7 NCBI prediction table
(no NCBI accessions), and pre-trained receptor models are not released.

### What this solves for us

Our AX03 pairwise cross-terms failed because they paired "any RBP" with "any receptor" -- biologically meaningless.
This paper gives us the phage-to-receptor mapping: "this phage targets OmpC." Combined with our continuous host OMP
HMM scores, this enables directed cross-terms: predicted_receptor=OmpC x host_OmpC_score. This is the mechanistic
prior our model has been missing.

### What it doesn't solve

Receptor compatibility is necessary but not sufficient for lysis. The gap between receptor prediction (AUROC 0.99) and
strain-level lysis prediction (AUROC 0.87, Noonan et al.) represents post-adsorption biology (defense systems,
intracellular barriers) that receptor predictions alone cannot capture.

### Practical paths to receptor predictions for our panel

1. Genus-level receptor mapping from their 260-phage Table S1 (hours, coarse)
2. Run GenoPHI framework on our FNA files (days, requires retraining on their labeled data)
3. Contact authors -- they already have our data, may have predictions ready (one email)

## Sources

- Local PDF: `.scratch/Prediction of phage receptor specificity from genom data.pdf`
- GenoPHI repo: https://github.com/Noonanav/GenoPHI
- Figshare supplementary: https://doi.org/10.6084/m9.figshare.31930314
- PhageDataSheets: https://iseq.lbl.gov/PhageDataSheets/Ecoli_phages/
- Companion strain-level paper: Noonan et al. 2025 (see separate gist)
