# Gist: 2025 Antiphage Defense Landscape

Last updated: 2026-04-11

## Citation

- Published title: _Protein and genomic language models uncover the unexplored diversity of bacterial immunity_
- Preprint title: _Protein and genomic language models chart a vast landscape of antiphage defenses_
- Journal: Science, Vol. 392, Issue 6793, eadv8275 (2 April 2026)
- DOI: https://doi.org/10.1126/science.adv8275
- Preprint: https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1
- Paper repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive map: https://mdmparis.github.io/antiphage-landscape/
- Full text: not available on Sci-Hub as of 2026-04-11; bioRxiv preprint is the best freely available version

## Executive summary

This is a relevant paper for the repo, but it is not a new strain-level host-range study. It is a defense-discovery
paper that uses protein and genomic language models to find novel antiphage systems. The published Science version is
substantially expanded from the preprint: it adds a third model (GeneCLR-DefenseFinder, combining sequence and context),
validates twelve systems (up from six in the preprint), and applies the models to 120M+ proteins across 30,000+ genomes.
The paper has a public code repository and an interactive map of predicted defense-related protein space.

## What the paper actually does

The paper targets a different question from the 2024 Nature _Escherichia_ study used by this repo. Instead of
predicting whether phage P lyses host H, it asks how much antiphage defense biology is still undiscovered in bacterial
genomes and whether language models can accelerate that discovery.

The authors build three complementary systems (the published version adds a third not in the preprint):

1. `ESM-DefenseFinder` (ESM_DF)
   - Fine-tunes the ESM2 protein language model to classify proteins as defense-related.
   - Main strength: picks up distant homology to known defense proteins.
   - Main weakness: still leans heavily on homology and has low precision.

2. `ALBERT-DefenseFinder` (ALBERT_DF)
   - Trains a genomic language model on stretches of neighboring genes in Actinomycetota.
   - Main strength: finds defense candidates from genomic context even when sequence homology is weak or absent.
   - Main weakness: computationally heavier and still low precision, so manual curation remains necessary.

3. `GeneCLR-DefenseFinder` (GeneCLR_DF) — **new in the published version**
   - Combines sequence and genomic context via contrastive learning.
   - Bridges the two single-modality models.

The published version validates twelve systems experimentally (in _E. coli_ and _S. albus_), up from six in the
preprint. The preprint named six: Ceres, Geb, Veles, Prithvi, Ukko, and Oshun. The Science version adds further
validated systems. Applied to 120M+ proteins from 30,000+ genomes, the models predict ~2.39M antiphage proteins
(~23,000 operon families), 85% with no previously known link to immunity.

## Publication status

Published in Science on 2 April 2026 as "Protein and genomic language models uncover the unexplored diversity of
bacterial immunity" (DOI: 10.1126/science.adv8275). The title changed from the preprint. The published version is
substantially expanded (third model, more validations, larger genomic sweep).

### Code and data

- GitHub repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive UMAP: https://mdmparis.github.io/antiphage-landscape/
- Supplementary data reportedly on Zenodo (linked from the Science paper).

## Scientific gist

The paper's main scientific claim is that the known defense catalog is still a small slice of the full antiviral
landscape. Using rarefaction-style analyses, the authors argue that the number of defense-related protein families in
the bacterial pangenome is far larger than what current curated systems capture. Their lower-bound estimate is on the
order of tens of thousands of protein families, and their upper bound is much larger again.

The important nuance is that the paper is strongest on:

- discovery of new candidate defense proteins and systems;
- proof that genomic-context models find systems missed by homology-based tools;
- experimental validation that some of those candidates are real defenses.

It is weaker as direct evidence for:

- pairwise phage-host prediction in _E. coli_;
- quantitative dominance of defense over adsorption in lysis outcomes;
- deployable high-precision annotation from raw genomes without manual review.

## Sources

- Published paper (Science): https://doi.org/10.1126/science.adv8275
- bioRxiv preprint: https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1
- Paper repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive UMAP: https://mdmparis.github.io/antiphage-landscape/
- Nature News coverage: https://www.nature.com/articles/d41586-026-01011-y
- 2024 Nature _Escherichia_ paper: https://doi.org/10.1038/s41564-024-01832-5
- VHRdb docs: https://hub.pages.pasteur.fr/viralhostrangedb/api.html
- BASEL completion: https://pubmed.ncbi.nlm.nih.gov/40193529/
