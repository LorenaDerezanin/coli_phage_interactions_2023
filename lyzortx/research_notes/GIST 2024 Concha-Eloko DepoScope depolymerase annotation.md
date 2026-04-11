# Gist: 2024 Concha-Eloko — DepoScope: Depolymerase Annotation and Domain Delineation

Last updated: 2026-04-11

## Citation

- Title: _DepoScope: Accurate phage depolymerase annotation and domain delineation using large language models_
- Authors: Concha-Eloko R, Stock M, De Baets B, Briers Y, Sanjuan R, Domingo-Calap P, Boeckaerts D
- Journal: PLoS Computational Biology 20(8): e1011831
- Published: August 5, 2024
- DOI: `https://doi.org/10.1371/journal.pcbi.1011831`
- Code: `https://github.com/dimiboeckaerts/DepoScope`
- Data: `https://zenodo.org/records/10957073`
- Local PDF: `paper/Deposcope 2024.pdf`

## Executive summary

DepoScope is the current state-of-the-art tool for phage depolymerase detection and enzymatic domain delineation. It
fine-tunes ESM-2 protein language models for token classification (which amino acids belong to the polysaccharide-
degrading domain and what fold class) and adds a CNN binary classifier (is this protein a depolymerase). On an
independent benchmark (Pires et al. 2016 dataset), it achieves MCC 0.455 -- more than double the next best tool
(PhageDPO 0.178, DePP 0.131) -- driven by far fewer false positives (123 FPs vs 2103/966). Its unique feature is
amino-acid-level domain delineation with fold classification, which no other tool provides.

For our project, DepoScope is the right tool if we ever need to annotate depolymerases on our phage panel. However,
depolymerase features were tested as part of the phage functional gene repertoire (PHROG category counts) and proved
to be noise for strain-level lysis prediction (top-3 degradation from 94.6% to 87.8%).

## What the tool does

### Architecture (two stacked models)

1. **Token classification model**: Fine-tuned ESM-2 (tested 6L, 12L, 30L configs). Each amino acid is classified as
   "none", "right-handed beta-helix", "n-bladed beta-propeller", or "triple helix". ESM-2 30L is the final model
   (MCC 0.903 on token task).

2. **Binary classification model**: Two 1D convolutional layers + two dense layers on top of the token classifier
   outputs. Predicts whether the whole protein is a depolymerase (MCC 0.987 on binary task with ESM-2 30L).

### Training data curation

- 554,981 proteins from INPHARED (>200 aa), filtered through:
  - Custom HMM profile database from 221 InterPro carbohydrate catalytic domains (EC 4.4.2 and EC 3.2.1)
  - HHblits screening with bit score >20 and >=30 aligned positions
  - CD-HIT clustering at 95% identity, ESM-2 affinity propagation (389 clusters)
  - ESMFold 3D structure prediction + FoldSeek against curated PD fold database
- Final training set: 1,926 positive proteins (602 right-handed beta-helix, 96 n-bladed beta-propeller, 146 triple
  helix) + 1,409 negative proteins (catalytically active but not depolymerases)
- 70/20/10 split for token/binary/evaluation tasks

### Fold types detected

- Right-handed beta-helix (dominant: 844/984 structural hits, 85.8%)
- N-bladed beta-propeller (119/984, 12.1%)
- Alpha/alpha toroid (11/984, 1.1%) -- too few for training
- TIM beta/alpha-barrel (10/984, 1.0%) -- too few for training
- Triple helix (146 manually annotated, added separately)

## Benchmark results (Pires et al. 2016 dataset, Table 2)

| Model             | Precision | Recall | Specificity | Accuracy | F1    | MCC   | PR AUC |
|-------------------|-----------|--------|-------------|----------|-------|-------|--------|
| DePP              | 3.5%      | 91.6%  | 60.9%       | 61.4%    | 6.7%  | 0.131 | 10.2%  |
| PhageDPO          | 6.0%      | 74.7%  | 82.0%       | 81.9%    | 11.2% | 0.178 | 27.8%  |
| DepoScope         | 32.0%     | 69.0%  | 98.0%       | 97.2%    | 43.4% | 0.455 | 42.3%  |
| DepoScope FP adj. | 81.4%     | 69.0%  | 99.8%       | 99.3%    | 74.5% | 0.744 | 51.6%  |

- "FP adjusted": 78/123 FPs have PD folds on manual inspection (likely missed by Pires), 32 more confirmed by
  domain analysis. Only 13 FPs are true false positives.
- DepoScope's recall weakness is concentrated in triple helix folds (18/26 missed) due to intrinsic disorder before
  chaperone-assisted trimerization.

## Key technical details

- **Domain delineation**: Uses SWORD2 to identify protein units from ESMFold 3D structures, then screens each unit
  against the PD fold database. This gives exact domain boundaries (start/end residue) for the enzymatic domain.
- **Structure-level filtering**: FoldSeek against 7 known PD-associated folds (beta-propeller, beta-helix, TIM barrel,
  alpha/alpha toroid, alpha/beta hydrolase, flavodoxin-like, triple helix). Probability thresholds: >0.5 for most
  folds, >0.2 for right-handed beta-helix (more divergent).
- **Generalization**: Group Shuffle Split by CD-HIT clusters across a range of identity thresholds (0.25-0.85). No
  significant performance impact, indicating the model generalizes beyond training sequence similarity.
- **Computational cost** (ESM-2 12L, chosen config): 89.7 sec / 100k amino acids scanned, 600 MB memory. The 30L
  model is 229.5 sec / 100k AA, 1769 MB.

## Competitive landscape (as of April 2026)

DepoScope remains state of the art. No citing paper has superseded it:

- **PDP-Miner** (2025, Bioinformatics): Wraps DePP + Pharokka for prophage mining in bacterial genomes. Different
  use case (prophage depolymerases in bacterial genomes, not lytic phage annotation). Benchmarked alongside DepoScope
  but does not improve on it.
- **PhageDPO** (published Feb 2025, Computers in Biology and Medicine): SVM-based. Self-reports 96% accuracy / 94%
  precision, but on an easier self-curated test set. DepoScope's independent benchmark shows only 6% precision / MCC
  0.178 for PhageDPO on the Pires dataset.
- **DepoRanker** (Jan 2025, arXiv): Klebsiella-specific ranking tool (AUROC 0.99). Not general-purpose.
- **DepoCatalog** (Sep 2025 preprint, same Ghent group): Experimental catalog of 105 recombinant Klebsiella
  depolymerases mapping specificity across 58 KL-types. Complementary resource, not a prediction tool replacement.

## Relevance to our project

### Direct relevance: low for prediction, moderate for annotation

Our project tested depolymerase-related features as part of phage functional gene repertoire (PHROG category counts
including depolymerase presence). These features degraded top-3 from 94.6% to 87.8% on inner-val and were classified
as noise (knowledge unit `phage-functional-noise`). However, the negative result may be attributable to a detection
bug: the patterns used to identify depolymerases did not match Pharokka's actual "tail spike protein" labels, so the
feature was likely miscounted rather than truly uninformative.

Our host panel (369 diverse clinical E. coli from Gaborieau et al. 2024) has rich capsule variation: 99 capsule
feature columns, all 369 bacteria with nonzero capsule features, and wide variation in individual capsule profiles.
This means directed cross-terms (depolymerase domain cluster x host capsule profile) have real host-side signal to
learn from -- unlike Moriniere's BW25113/BL21 receptor screens which used capsule-negative K-12 strains.

### Potential near-term relevance

- With ~34 tail-spike-bearing phages in our panel and 99 host capsule feature columns with real variation,
  DepoScope domain clusters could serve as the phage-side input for directed cross-terms against host capsule
  profiles. This implicit learning approach (cluster depolymerase domains, let LightGBM discover substrate-capsule
  associations) is viable because the host-side capsule signal is rich.
- DepoScope's domain delineation enables more nuanced features than binary presence: fold type, domain length, and
  domain-level sequence clustering for grouping phages by likely substrate specificity.
- The same Ghent group (Boeckaerts/Briers) authored both DepoScope and PhageHostLearn (Klebsiella pairwise
  prediction), suggesting potential integration paths. Pharokka (Bouras, Adelaide) is the independent upstream
  gene-calling step.
- **Substrate specificity gap**: DepoScope detects depolymerases and delineates domains but does not predict which
  capsule type they target. No existing tool predicts E. coli K-antigen specificity from depolymerase sequence.
  For Klebsiella, Gittrich et al. 2025 (Nature Comms) demonstrated a prophage-mining approach using DAG models and
  sequence clustering to predict capsular tropism from 8,105 prophages -- this method is transferable to lytic phages
  and could in principle be adapted for E. coli (~80 K-antigen types), but nobody has done it yet. DepoCatalog
  (Ghent, Sep 2025 preprint) provides an experimental catalog of 105 Klebsiella depolymerases across 58 KL-types
  but is not a computational predictor.

### Tool applicability

If depolymerase annotation is needed for our phage panel:
- Run DepoScope (ESM-2 12L config for speed, 30L for accuracy) on all predicted proteins from Pharokka output.
- Union with DePP hits for recall (DePP has 91.6% recall vs DepoScope's 69%).
- Domain boundaries from DepoScope are unique and useful for downstream structural analysis.

## Authors and connections

- Dimitri Boeckaerts (Ghent) is senior author on both DepoScope and PhageHostLearn (Boeckaerts et al. 2024, Nature
  Communications) -- the Klebsiella strain-level prediction paper already in our literature list.
- Yves Briers (Ghent) leads the Applied Biotechnology lab focused on phage enzymes.
- Rafael Sanjuan and Pilar Domingo-Calap (Valencia) contribute the phage evolution perspective.

## Sources

- Paper: `https://doi.org/10.1371/journal.pcbi.1011831`
- Code: `https://github.com/dimiboeckaerts/DepoScope`
- Data: `https://zenodo.org/records/10957073`
- Local PDF: `paper/Deposcope 2024.pdf`
