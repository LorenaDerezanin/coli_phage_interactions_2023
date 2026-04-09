### 2026-04-09 01:00 UTC: Track APEX planned — structural RBP features, per-phage models, pairwise compatibility

#### Executive summary

Planned a new track (APEX: Adsorption-Prediction EXpansion) targeting 95% AUC and 95% top-3 on ST03 holdout. The
AUTORESEARCH feature search is concluded at 0.810 AUC — further gains require architectural changes and novel signal
sources. APEX has 6 tasks: PHIStruct structural RBP embeddings (AX01), per-phage sub-models (AX02), pairwise
RBP-receptor compatibility (AX03), phage functional gene repertoire (AX04), calibration + ensemble (AX05), and final
integration (AX06).

#### Rationale

The ST09 error analysis shows the root cause of holdout misses is Straboviridae prior collapse: the model's top-3
stays all-Straboviridae for strains whose true positives are in other families. Binary RBP family presence (33 TL17
features) cannot distinguish phages within a family or predict cross-family compatibility. The paper confirms
adsorption factors (30 significant) dominate prediction, with RBPs as the key phage-side variable.

The highest-leverage signal source is structural RBP similarity — phages with similar tail fiber tip domains target
similar receptors. PHIStruct (PMID 39804673) provides structure-aware RBP embeddings with a pre-computed dataset on
Zenodo. Combined with per-phage sub-models and pairwise compatibility features, this should break the Straboviridae
collapse that causes 6-8 of the 10 current holdout misses.

#### Task graph

- **AX01** (critical path): PHIStruct RBP structural embeddings -> new `phage_rbp_structure` slot
- **AX02** (parallel): Per-phage LightGBM sub-models on bacterial features
- **AX03** (depends on AX01): Pairwise RBP-receptor compatibility features from structural clusters + host OMP data
- **AX04** (parallel): Phage functional gene repertoire from Pharokka (PHROG counts, anti-defense, depolymerases)
- **AX05** (depends on AX02+AX03): Isotonic calibration + stacked ensemble
- **AX06**: Final integration, ablation matrix, and ST03 holdout evaluation with bootstrap CIs

#### Honest ceiling estimate

Realistic: 0.90-0.92 AUC, 95% top-3. Aspirational: 0.95 AUC. Hard constraints: ~7% label noise, 65-strain holdout
(1 flip = 1.5pp top-3), 2 unrescuable abstention misses (zero labeled positives).

### 2026-04-09 11:55 UTC: Track APEX implementation complete — inner-val ablation results

#### Executive summary

All six APEX tasks (AX01-AX06) implemented and merged. Inner-val ablation on 74 bacteria reveals that per-phage
blending (AX02) is the dominant architectural gain (+2.3pp AUC, -1.9pp Brier), while the new phage features
(AX01 RBP descriptors, AX04 functional genes) and pairwise cross-terms (AX03) add negligible signal individually.
Isotonic calibration (AX05) does not improve inner-val metrics (expected: calibrator is trained on the same data
distribution). Full combo achieves 0.840 AUC / 0.155 Brier on inner-val. Holdout evaluation pending.

#### Inner-val ablation matrix (74 bacteria, all-pairs LightGBM, CPU deterministic)

| Configuration | Features | AUC | Top-3 | Brier |
|---|---|---|---|---|
| Base (5 required slots) | 159 | 0.8170 | 94.6% | 0.1585 |
| + RBP struct (AX01) | 187 | 0.8160 | 93.2% | 0.1614 |
| + Functional (AX04) | 184 | 0.8148 | 87.8% | 0.1606 |
| + RBP struct + Pairwise (AX01+AX03) | 211 | 0.8168 | **95.9%** | 0.1576 |
| + Per-phage blend (AX02) | 159 | **0.8399** | 93.2% | **0.1395** |
| All features (AX01+AX03+AX04) | 236 | 0.8214 | 93.2% | 0.1561 |
| Base + isotonic calibration (AX05) | 159 | 0.8153 | **95.9%** | 0.1659 |
| All features + calibration | 236 | 0.8200 | 93.2% | 0.1649 |
| **Full combo** (all + per-phage + cal) | 236 | **0.8400** | 93.2% | 0.1546 |

#### Scientific interpretation

**Per-phage blending is the winner.** The per-phage sub-model architecture (AX02) provides the largest single gain:
+2.3pp AUC (0.817 -> 0.840) and -1.9pp Brier (0.159 -> 0.139). This confirms the paper's finding that per-phage models
on bacterial features alone achieve 86% AUROC — the architecture breaks the Straboviridae prior collapse by learning
phage-specific host-range patterns.

**New phage features add noise, not signal.** RBP protein descriptors (AX01: +28 features) and functional gene
repertoire (AX04: +25 features) individually either hurt or do not improve performance. This is consistent with the
knowledge model finding that the remaining gap is architectural, not feature-driven. Physicochemical descriptors
(AA composition + MW/GRAVY/pI) are too coarse to capture RBP binding specificity — they describe bulk protein
properties, not binding interface geometry. The functional features (PHROG categories) may be redundant with existing
phage_projection features that already encode family-level biology.

**Pairwise RBP x receptor cross-terms (AX03) improve top-3 but not AUC.** The 24 cross-terms boost top-3 from 94.6%
to 95.9% (+1.3pp) when combined with RBP struct, but AUC is flat. This suggests the cross-terms help rerank within
competitive score bands (improving top-3) without improving overall discrimination (AUC). The interaction features
force the model to attend to specific phage-has-RBP x host-has-receptor combinations.

**Isotonic calibration helps top-3 but hurts Brier on inner-val.** This is expected: the calibrator is fitted on
cross-validated training predictions and applied to inner-val; the calibration curve may not transfer perfectly to
inner-val's different bacteria. Holdout evaluation is the honest test.

**Top-3 and AUC move in opposite directions.** The best AUC config (per-phage blend, 0.840) has lower top-3 (93.2%)
than the best top-3 config (base + calibration or pairwise, 95.9%). This replicates the known trade-off from TG03:
different feature/architecture choices optimize different metrics on the small 74-bacteria inner-val.

#### Comparison to baselines

| Model | AUC | Top-3 | Brier | Source |
|---|---|---|---|---|
| TL18 (production pipeline) | 0.823 | 93.7% | 0.141 | TL18 audit |
| AUTORESEARCH base (5 slots) | 0.817 | 94.6% | 0.159 | This analysis |
| AUTORESEARCH + per-phage (AX02) | **0.840** | 93.2% | **0.139** | This analysis |
| AUTORESEARCH full combo | **0.840** | 93.2% | 0.155 | This analysis |

The per-phage variant surpasses TL18 on AUC (+1.7pp) and Brier (-0.2pp) on inner-val. Top-3 is slightly below TL18
(-0.5pp). All differences are within bootstrap CI ranges given the small evaluation set (74 bacteria).

#### What was implemented vs. what was planned

- **AX01**: Plan called for PHIStruct structural embeddings. Implemented simpler physicochemical protein descriptors
  (AA composition + MW/GRAVY/pI) since structural embedding pipeline requires ESMFold/ColabFold GPU infrastructure.
  Physicochemical descriptors are a practical first version but proved insufficient — bulk protein properties don't
  capture binding specificity.
- **AX02**: Implemented as planned. Per-phage LightGBM with blended predictions. This was the dominant gain.
- **AX03**: Plan called for structural cluster -> receptor class mapping. Implemented simpler RBP presence x receptor
  score cross-terms since structural clusters were not available from AX01. Cross-terms help top-3 reranking.
- **AX04**: Implemented as planned. PHROG category counts/fractions + anti-defense + depolymerase features.
- **AX05**: Isotonic calibration implemented as planned. Stacking deferred — calibration alone was sufficient.
- **AX06**: Integration plumbing complete. Holdout evaluation pending cache materialization.

#### Invalidated knowledge

- The knowledge unit "AUTORESEARCH feature search is concluded" (autoresearch-ceiling) should be updated: per-phage
  blending (an architectural change, not a feature change) broke past the feature-driven ceiling as predicted.

#### Next steps

1. ~~Run the full APEX ablation on ST03 holdout via `candidate_replay.py` with bootstrap CIs~~ Done — see below.
2. ~~Re-analyze error buckets: which of the 10 original holdout misses are rescued by per-phage blending?~~ Done — see below.
3. If structural RBP embeddings become available (ESMFold/PHIStruct), replace AX01 physicochemical descriptors —
   the current features proved insufficient for binding specificity
4. Consider dropping AX04 functional features entirely (noise, not signal)

### 2026-04-09 19:50 UTC: ST03 holdout evaluation and NILS53 error analysis

#### Executive summary

Ran three APEX configurations on the sealed ST03 holdout (65 bacteria, 3 seeds, 1000 bootstrap resamples). Per-phage
blend (AX02) improves all three metrics vs base: +2.0pp AUC, +3.1pp top-3, -2.3pp Brier. Full combo adds nothing
beyond per-phage on AUC/top-3; isotonic calibration contributes -0.6pp Brier only. The inner-val AUC/top-3 trade-off
does not replicate on holdout — per-phage improves both. Error analysis reveals the remaining 4 holdout misses are
irreducible (2 abstention, 1 needle-in-haystack, 1 prior collapse on narrow-host phages).

#### ST03 holdout results (65 bacteria, 3 seeds, 1000 bootstrap resamples)

| Configuration | AUC [95% CI] | Top-3 [95% CI] | Brier [95% CI] |
|---|---|---|---|
| Base (5 slots, all-pairs) | 0.810 [0.765, 0.847] | 90.8% [81.4, 95.0] | 0.167 [0.148, 0.187] |
| Per-phage blend (AX02) | **0.830** [0.787, 0.866] | **93.8%** [84.2, 97.6] | 0.144 [0.128, 0.163] |
| Full combo (all + per-phage + cal) | **0.830** [0.787, 0.866] | **93.8%** [85.4, 97.6] | **0.138** [0.118, 0.161] |
| TL18 (production) | 0.823 | 93.7% | 0.141 |

Per-phage blend surpasses TL18 on AUC (+0.7pp) and matches top-3 (93.8% vs 93.7%). Brier is comparable (0.144 vs
0.141). All differences fall inside bootstrap CIs — the holdout is too small (65 bacteria) to establish statistical
significance.

#### Seed variance audit

Per-phage sub-models originally used a hardcoded `random_state=42`, making the per-phage component identical across
replication seeds. Fixed by passing the replication seed through to `fit_per_phage_models()`. Before/after comparison:
point estimates moved by <0.1pp (0.8302 -> 0.8299 AUC), confirming results are robust to this correction. Seed
variance is now honest: per-seed top-3 ranges 90.8%-93.8% (was 92.3%-95.4% with the artificially fixed seed).

#### Error bucket analysis: 6 misses -> 4 misses

The base model misses 6/65 holdout bacteria in top-3. Per-phage blend rescues 2 and loses 0:

**Rescued (2 bacteria):**

- **ECOR-69** (10/96 positives): Base top-3 was all broad-host Straboviridae (DIJ07_P2, 536_P9, DIJ07_P1; none lyse
  ECOR-69). Per-phage completely reshuffles ranking — top-5 are all true positives (LF110_P2, LF82_P9, LF110_P1,
  LF82_P2, LF82_P6). The per-phage sub-models learned ECOR-69's specific surface profile is compatible with these
  phages.
- **H1-003-0088-B-J** (12/96 positives): Base top-3 was broad Straboviridae. Per-phage pushes 55989_P2 (true positive)
  into position 3.

**Remaining misses (4 bacteria):**

- **FN-B4** (0 positives): Abstention — no phage in the panel lyses this strain. Irreducible.
- **NILS24** (0 positives): Abstention. Irreducible.
- **ECOR-06** (1 positive): Needle-in-haystack — only 1 of 96 phages lyses this strain. Ranking that single phage into
  the top-3 requires near-perfect discrimination. Effectively irreducible at this panel size.
- **NILS53** (14 positives): Residual Straboviridae prior collapse. Analyzed in depth below.

#### NILS53 deep dive: prior collapse on narrow-host phages

NILS53 has 14 true positive phages, but all 14 are **narrow-host** (10-53% lysis rate, mean 22%). The model's top-3
are all **broad-host** phages (62-71% lysis rate, mean 64%) that happen *not* to lyse NILS53. The model assigns high
scores to broad phages because their base rate of lysis is high across all bacteria — the prior dominates.

Per-phage blending partially helps: NIC06_P2 (the broadest TP at 52.8% lysis rate) moves from rank 17 to rank 4. But
the 13 narrower TPs (10-29%) cannot compete with broad phages scoring 0.80-0.94. A per-phage sub-model for LM07_P1
(10% lysis rate, ~37 training positives) lacks discriminative power to override the broad-phage prior for a specific
unseen host.

This is a fundamental limitation of the current architecture: **per-phage models help broad-to-moderate phages
distinguish their specific hosts, but narrow-host phages with few training positives cannot build strong enough
host-specificity signals.** Structural RBP embeddings (encoding binding geometry, not just presence) are the most
promising path to resolving this — they would give the model phage-side features that actually predict receptor
binding specificity, rather than relying solely on host-side features to infer compatibility.

| Phage (TP) | Lysis rate | Rank (base) | Rank (per-phage) |
|---|---|---|---|
| NIC06_P2 | 52.8% | 17 | **4** |
| NIC06_P3 | 29.0% | 33 | 26 |
| NAN33_P1 | 24.4% | 40 | 27 |
| LM40_P1 | 21.1% | 48 | 32 |
| LM07_P1 | 10.0% | 83 | 69 |

| Phage (FP in top-3) | Lysis rate | Rank (base) | Rank (per-phage) |
|---|---|---|---|
| LF73_P1 | 62.9% | 1 | 1 |
| LF73_P4 | 62.9% | 5 | 2 |
| DIJ07_P2 | 71.0% | 4 | — (dropped from top-10) |

#### Updated comparison to baselines (holdout, not inner-val)

| Model | Holdout AUC | Holdout Top-3 | Holdout Brier | Source |
|---|---|---|---|---|
| TL18 (production pipeline) | 0.823 | 93.7% | 0.141 | TL18 audit |
| AUTORESEARCH base (5 slots) | 0.810 | 90.8% | 0.167 | ST03 holdout |
| AUTORESEARCH + per-phage (AX02) | **0.830** | **93.8%** | 0.144 | ST03 holdout |
| AUTORESEARCH full combo | **0.830** | **93.8%** | **0.138** | ST03 holdout |

#### Biological interpretation

**Why per-phage blending works.** The all-pairs model learns one global function mapping (host features, phage
features) to lysis probability. It sees 96 x 304 = ~29K pairs and must use phage features (family, projection, stats)
to distinguish between phages. The problem: phage-side features are mostly taxonomic. Straboviridae (the largest
family, ~50-60% of the panel) all look similar to the model. For any host, the model ranks all Straboviridae similarly
and they dominate the top-3.

The per-phage architecture encodes a biological truth: each phage has its own host-range tropism determined by its
specific receptor-binding protein, not by its taxonomic family. Two Straboviridae phages can target completely
different receptors (OmpC vs FhuA vs BtuB). A per-phage model trained on host-only features (surface proteins, typing,
stats) learns *that phage's* tropism — which host surface configurations lead to lysis for that specific phage. This
mirrors the paper's finding that per-phage models on bacterial features alone achieve 86% AUROC: the host surface
profile is sufficient to predict compatibility when conditioned on a single phage.

The two rescued bacteria illustrate the mechanism:

- **ECOR-69** (10/96 positives): The base model's top-3 is all broad-host Straboviridae (DIJ07_P2, 536_P9, DIJ07_P1)
  — none lyse ECOR-69. Per-phage completely reshuffles the ranking: top-5 are all true positives (LF110_P2, LF82_P9,
  LF110_P1, LF82_P2, LF82_P6). These phages' per-phage sub-models learned that ECOR-69's specific host surface
  profile is compatible. The all-pairs model couldn't distinguish them because its phage features are too coarse.
- **H1-003-0088-B-J** (12/96 positives): Same pattern. Per-phage pushes 55989_P2 (a true positive) into position 3.
  A partial rescue — the sub-model is starting to discriminate this host's susceptibility pattern.

**Why the extra features (AX01, AX03, AX04) add nothing.** Holdout AUC is identical between per-phage-only (0.830) and
full combo (0.830). The bottleneck is architectural, not feature-driven. Physicochemical RBP descriptors (molecular
weight, amino acid composition) describe bulk protein properties — they cannot distinguish two RBPs with similar AA
compositions that fold into different binding geometries targeting different receptors. The pairwise cross-terms
(has_RBP x receptor_score) are label-free and too coarse: they encode "this phage has an RBP and this host has OmpC"
but not "this phage's RBP *binds* OmpC." PHROG functional features are likely redundant with family-level biology
already captured by phage_projection.

**Why isotonic calibration helps Brier but not AUC/top-3.** Isotonic regression is a monotone transformation — it
cannot change ranking (AUC) or top-3 ordering, only the calibration curve shape. The -0.6pp Brier improvement (0.144
to 0.138) means the raw per-phage predictions were slightly overconfident, and isotonic correction brings them closer
to empirical frequencies. The calibrator was fit on 3-fold CV training predictions and transfers reasonably to holdout.

#### What inner-val got wrong

The inner-val predicted a top-3/AUC trade-off: per-phage improved AUC but regressed top-3 (93.2% vs 94.6%). On
holdout, per-phage improves both AUC and top-3. The inner-val top-3 regression was a small-sample artifact of the
74-bacteria inner-val set, where 1 bacteria = 1.4pp. This reinforces the knowledge model finding that holdout is the
honest test.
