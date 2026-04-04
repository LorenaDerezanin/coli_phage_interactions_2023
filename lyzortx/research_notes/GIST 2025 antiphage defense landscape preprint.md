# Gist: 2025 Antiphage Defense Landscape Preprint

Last updated: 2026-04-04

## Citation

- Title: _Protein and genomic language models chart a vast landscape of antiphage defenses_
- Status checked: 2026-04-04
- Preprint: https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1
- Paper repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive map: https://mdmparis.github.io/antiphage-landscape/

## Executive summary

This is a relevant paper for the repo, but it is not a new strain-level host-range study. It is a defense-discovery
paper that uses a protein language model plus a genomic-context transformer to find novel antiphage systems, then
validates six new systems experimentally in Actinomycetota. The paper also has a public code repository and an
interactive map of predicted defense-related protein space, but as of 2026-04-04 I did not find a journal or PubMed
version of this specific work, so it still appears to be a preprint rather than a peer-reviewed publication.

## What the paper actually does

The paper targets a different question from the 2024 Nature _Escherichia_ study used by this repo. Instead of
predicting whether phage P lyses host H, it asks how much antiphage defense biology is still undiscovered in bacterial
genomes and whether language models can accelerate that discovery.

The authors build two complementary systems:

1. `ESM-DefenseFinder`
   - Fine-tunes the ESM2 protein language model to classify proteins as defense-related.
   - Main strength: picks up distant homology to known defense proteins.
   - Main weakness: still leans heavily on homology and has low precision.

2. `ALBERT-DefenseFinder`
   - Trains a genomic language model on stretches of neighboring genes in Actinomycetota.
   - Main strength: finds defense candidates from genomic context even when sequence homology is weak or absent.
   - Main weakness: computationally heavier and still low precision, so manual curation remains necessary.

They then manually curate candidates from the context model, synthesize ten systems, and test them in
_Streptomyces albus_. Two were toxic under the expression setup. Six showed antiphage activity with at least
100-fold plaque reduction against at least one phage. They named the validated systems Ceres, Geb, Veles, Prithvi,
Ukko, and Oshun.

## Current status: repo, code, and publication

### Do they have a GitHub repo?

Yes. The preprint's data-availability section points to:

- https://github.com/mdmparis/antiphage_landscape_2025

The repo exists publicly and, at minimum, exposes README plus supplementary tables. The preprint also exposes an
interactive projection:

- https://mdmparis.github.io/antiphage-landscape/

### Why is it still a preprint? Was it published?

As of 2026-04-04, I did not find a journal landing page, PubMed entry, or replacement DOI for this exact title. The
safest statement is:

- it has public code/resources;
- it is still indexed as a bioRxiv preprint;
- I cannot verify a later peer-reviewed version.

I would not infer anything stronger than "likely still under review, unpublished, or not yet indexed." The GitHub
README saying "publication" is not evidence of journal publication.

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

- bioRxiv preprint: https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1
- Paper repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive UMAP: https://mdmparis.github.io/antiphage-landscape/
- 2024 Nature _Escherichia_ paper: https://doi.org/10.1038/s41564-024-01832-5
- VHRdb docs: https://hub.pages.pasteur.fr/viralhostrangedb/api.html
- BASEL completion: https://pubmed.ncbi.nlm.nih.gov/40193529/
