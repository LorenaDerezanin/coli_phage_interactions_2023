# Repo Note: Antiphage Landscape 2025 Implications for Modeling Strategy

Last updated: 2026-04-05

## Executive summary

The 2025 antiphage-landscape preprint does not give this repo a new strain-level host-range matrix or a better direct
classifier for _E. coli_ lysis. Its value is strategic: it reinforces that known defense annotations are incomplete,
which means defense-feature absence is weak evidence, while adsorption-side biology remains the strongest first-order
signal for our actual task. The right response is not to pivot toward defense-heavy pairwise prediction, but to keep
the adsorption-first architecture, preserve defense features as asymmetric positive evidence, and invest next in
higher-resolution phage RBP FASTA features and higher-fidelity external _E. coli_ interaction data.

For problem fit, our approach is better aligned than theirs to the repo's goal, but not globally "better." Their
method is better for novel defense-system discovery; our pipeline is better for predicting which phages lyse which
_E. coli_ strains from inference-time inputs. That is a scope claim, not a benchmark win claim: we should not say we
beat them head-to-head because we have not run a like-for-like comparison on the same task.

## Why this matters

The preprint and the 2024 Nature _Escherichia_ paper answer different questions:

- the 2024 paper asks which genomic features predict strain-level lysis in _Escherichia_;
- the 2025 preprint asks how much unknown antiphage defense biology remains to be discovered and whether language
  models can uncover it.

Those results are compatible. The 2024 paper can still be right that adsorption factors dominate present-day
supervised prediction in _E. coli_, while the 2025 preprint is also right that the defense universe is vastly larger
than what current annotation tools recover.

## Modeling implications

### 1. Keep adsorption and RBP features as the main axis

Nothing in the preprint justifies demoting host surface features, receptor compatibility, phage RBP features, or
isolation-host features. Those remain the most direct path to improving pairwise _E. coli_ lysis prediction.

### 2. Reinterpret defense negatives

The biggest update is semantic, not architectural. A missing DefenseFinder/PadLoc call should be interpreted as
"not recognized by the current catalog," not "host lacks relevant antiviral function." That means:

- defense-feature presence remains useful evidence;
- defense-feature absence is weak evidence;
- mechanistic write-ups should stop reading sparse defense zeros too literally.

### 3. The next likely gain is phage-side RBP resolution

The repo already captures much of the host-side surface biology the 2024 paper cares about. The more obvious remaining
gap is phage-side resolution beyond coarse PHROG-family bins:

- finer RBP sequence similarity summaries;
- tail-fiber / tail-spike tip-domain features;
- depolymerase-like domain summaries;
- per-RBP embedding or nearest-neighbor features against curated references.

### 4. Defense-discovery models are candidate generators, not ready-made core features

`ALBERT-DefenseFinder` and `ESM-DefenseFinder` are compelling discovery tools, but they are not strong evidence that
we should center the main classifier on LM-predicted defense burden. Their precision is still low enough that manual
curation remains central in the paper itself.

## External data implications

### What this preprint does not provide

It does not provide a new 403 x 96-like strain-by-strain host-range matrix. The experimental section validates
candidate defense systems in _Streptomyces albus_; that is biologically interesting but not a direct supervised
training set for our pairwise lysis task.

### What we should still prioritize

1. Highest-fidelity ingestion of the 2024 _Escherichia_ source package, not only the current repo snapshot.
2. BASEL plus the 2025 BASEL completion cohort as the strongest same-host-genus external supervision.
3. PhageHostLearn as transfer-learning and robustness data, not as same-domain substitute training data.
4. GPB and broader collections as secondary stress-test cohorts.

## FASTA-extractable feature implications

### High-priority additions

1. Finer RBP sequence similarity features from phage FASTAs.
2. Tail-tip and depolymerase domain features tied to adsorption specificity.
3. Better phage-side adsorption architecture summaries built from phage proteins rather than generic whole-genome
   embeddings.

### Medium-priority additions

1. Softer host defense-burden summaries from raw outputs, treated as auxiliary host-state priors.
2. Limited exploration of defense-island-adjacent host protein representations, only after adsorption upgrades.

### Low-priority additions right now

1. Direct reuse of `ALBERT-DefenseFinder` on our core _E. coli_ pipeline.
2. Promoting LM-derived defense predictions into the main pairwise classifier before adsorption-side upgrades.

## Recommendations

1. Keep the repo adsorption-first.
2. Keep defense features, but interpret absences conservatively in analyses and prose.
3. Do not describe our approach as generically better than theirs; describe it as better aligned to the repo's task.
4. Make the next phage-side feature branch about higher-resolution RBP FASTA features.
5. Prioritize external _E. coli_ matrices and source-fidelity ingestion before any major defense-feature expansion.

## Sources

- 2025 preprint: https://www.biorxiv.org/content/10.1101/2025.01.08.631966v1
- Paper repo: https://github.com/mdmparis/antiphage_landscape_2025
- Interactive UMAP: https://mdmparis.github.io/antiphage-landscape/
- 2024 Nature _Escherichia_ paper: https://doi.org/10.1038/s41564-024-01832-5
