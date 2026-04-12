### 2026-04-12 00:15 CEST: Track GIANTS launched — three-layer biological prediction model

#### Executive summary

Launched Track GIANTS (Standing on the Shoulders of GenoPHI and Moriniere) with 4 tasks. The hypothesis: lysis requires
passing adsorption gates (capsule penetration OR receptor binding) then surviving host defenses. We build features for
each layer using the best available tools (DepoScope for depolymerases, Moriniere Table S1 for OMP receptors,
DefenseFinder for defense systems), then let LightGBM + RFE learn the gating logic. Baseline: AUTORESEARCH all-pairs
0.810 AUC on ST03 holdout.

#### Biological motivation

Two papers from the Arkin/Mutalik group reshape our approach:

1. **Moriniere et al. 2026** — Receptor specificity is a near-solved discrete classification problem (19 classes, AUROC
   0.99 from amino acid k-mers). The signal is in localized hypervariable sequence motifs at RBP tip domains, not global
   protein embeddings. This explains why our PLM approach (AX07/AX08) failed — mean-pooled whole-protein embeddings are
   the wrong tool.

2. **Noonan et al. 2025 (GenoPHI)** — ML pipeline configuration matters 5-18x more than feature representation. CatBoost
   + RFE + inverse-frequency class weighting is the optimal classical ML config. GenoPHI achieves AUROC 0.869 on our
   exact phages with binary features.

Together they suggest a three-layer architecture matching the actual infection mechanism:

```
Gate 1: Can phage degrade host's capsule/O-antigen? (depolymerase x capsule)
Gate 2: Can phage bind host's OMP receptor? (RBP receptor class x host OMP variant)
   Adsorption = Gate 1 OR Gate 2 (or both)
Gate 3: Can phage survive host's defenses? (all defense systems, unrestricted)
   Lysis = Adsorption AND survives Gate 3
```

#### Why previous approaches failed at the feature level

+ **AX03 pairwise cross-terms** — paired "any RBP" with "any receptor." Biologically meaningless without knowing WHICH
  receptor the phage targets.
+ **AX07/AX08 PLM embeddings** — encoded family similarity, not binding specificity. The signal is in short motifs at
  specific loci (Moriniere), not global protein embeddings.
+ **Defense features flat** — applied defense globally, confounding it with phylogroup. Defense should only matter AFTER
  adsorption succeeds through either gate.

#### Key data enabling this track

+ **Host capsule variation is real**: 99 capsule feature columns across 369 diverse clinical E. coli isolates (not K-12
  lab strains). All 369 have nonzero capsule profiles.
+ **34/97 phages have tail spike annotations** from Pharokka (39 genes), but the DEPOLYMERASE_PATTERNS detection bug
  means they're currently classified as generic RBPs, not depolymerases. DepoScope will provide high-confidence calls
  with domain boundaries.
+ **Moriniere Table S1** gives genus-level OMP receptor mapping for ~60% of our phages. High-confidence for some genera
  (Justusliebigvirus→NGR, Lambdavirus→LamB), ambiguous for others (Tequatrovirus spans 5 receptors), and "Resistant"
  for capsule-dependent genera (Vectrevirus, Kagunavirus).

#### Baseline

AUTORESEARCH all-pairs: 0.810 AUC, 90.8% top-3, 0.167 Brier on ST03 holdout. This is the single clean baseline — TL18
(0.823) has feature integrity issues, per-phage (0.830) is not deployable.

#### Task graph

```
GT01 (depolymerase x capsule) ──┐
                                 ├── GT03 (integration + RFE + class weighting) ── GT04 (holdout eval)
GT02 (receptor x OMP)     ──────┘
```

GT01 and GT02 are parallel. GT03 combines all layers with RFE and class weighting. GT04 (HPO) and GT05 (CatBoost)
are incremental follow-ups.

### 2026-04-12 08:41 CEST: GT03 — Three-layer integration with RFE and inverse-frequency weighting

#### Executive summary

Combined all three gate feature sets (depolymerase × capsule, receptor × OMP, defense) with the 5-slot AUTORESEARCH
baseline and evaluated on ST03 holdout. The all_gates_rfe arm achieves 0.823 AUC [0.781, 0.858], a statistically
significant +1.2pp over the 0.810 baseline. Gate 1 (depolymerase × capsule) is the primary driver, contributing +0.7pp
alone with 22% feature importance. RFE selects 252/502 numeric features. Inverse-frequency weighting hurts AUC but
improves Brier from 0.165 to 0.145.

#### Ablation results

| Arm | AUC | AUC 95% CI | Top-3 | Brier | AUC delta CI vs baseline |
|-----|-----|-----------|-------|-------|--------------------------|
| baseline | 0.810 | [0.767, 0.848] | 92.3% | 0.165 | — |
| +gate1 (depo×capsule) | 0.818 | [0.772, 0.856] | 93.8% | 0.162 | [+0.003, +0.012] *|
| +gate2 (receptor×OMP) | 0.814 | [0.769, 0.851] | 89.2% | 0.165 | [-0.001, +0.007] |
| +gate3 (defense) | 0.816 | [0.777, 0.850] | 89.2% | 0.165 | [-0.003, +0.016] |
| all_gates | 0.822 | [0.781, 0.857] | 92.3% | 0.161 | [+0.002, +0.022]* |
| all_gates_rfe | 0.823 | [0.781, 0.858] | 90.8% | 0.162 | [+0.003, +0.024] * |
| all_gates_rfe_ifw | 0.809 | [0.758, 0.849] | 89.2% | 0.145 | [-0.014, +0.015] |

`*` = 95% CI excludes zero. 3 seeds, 1000 bootstrap resamples on 65 holdout bacteria.

#### Feature importance (all_gates_rfe arm, seed-averaged)

+ pair_depo_capsule: 21.5% — the dominant new signal
+ host_surface: 21.2% — OMP/capsule HMM scores
+ phage_projection: 15.6% — TL17-frozen phage features
+ phage_stats: 13.9% — genome statistics
+ host_typing: 12.8% — phylogroup/serotype/ST
+ host_stats: 8.4% — genome statistics
+ host_defense: 4.6% — defense system counts
+ pair_receptor_omp: 2.0% — directed receptor cross-terms

#### Interpretation

**Gate 1 (depolymerase × capsule) is the primary discovery.** The 242 features (41 cluster memberships + has_depo/count
× 99 capsule scores) capture 22% of total feature importance and produce a statistically significant +0.7pp AUC lift
alone. This validates the three-layer hypothesis at the capsule penetration layer: DepoScope-predicted depolymerase
presence interacted with host capsule HMM profiles provides discriminative signal that LightGBM can exploit.

**Gate 2 (receptor × OMP) is marginal.** Only 8/96 phages have clean OMP receptor assignments from the genus-level
Table S1 lookup (Tequatrovirus→Tsx, Lambdavirus→LamB, Dhillonvirus→FhuA). The +0.3pp AUC is not statistically
significant. GenoPHI per-phage receptor prediction (AUROC 0.99) would assign receptors to all 96 phages and likely
strengthen this gate.

**Gate 3 (defense) replicates prior findings.** The +0.6pp AUC is consistent with the previously observed +0.7pp
(defense-ablation-autoresearch). The CI is wide [-0.003, +0.016], confirming it's not a reliable contributor at this
sample size.

**RFE helps marginally.** Pruning from 507 to ~257 features adds +0.1pp over all_gates (0.823 vs 0.822). The pruning
primarily removes redundant capsule cross-terms and low-importance defense subtypes.

**Inverse-frequency weighting hurts AUC, helps calibration.** The IFW arm drops AUC by 1.4pp but improves Brier from
0.162 to 0.145 — the model becomes better calibrated by upweighting narrow-host phage positives, but at the cost of
discrimination. This suggests IFW might be better applied as a post-hoc calibration adjustment rather than a training
weight.

#### Error bucket analysis

The baseline has 6/65 holdout misses (90.3% top-3 after seed averaging). The all_gates_rfe arm has comparable top-3
(90.8%). Individual seeds show high variance: top-3 ranges from 87.7% to 95.4% across seeds, reflecting the small
holdout (1 strain = 1.5pp). A proper error bucket analysis requires comparing per-strain predictions, deferred to
after the bootstrap summary.

#### Next steps

+ GT04: HPO with Optuna over LightGBM params — the new feature families may benefit from different tree depth and
  regularization than the default config.
+ GT05: CatBoost comparison — handles categoricals natively, found optimal by GenoPHI.
+ Flag for future: GenoPHI per-phage receptor prediction to strengthen Gate 2 beyond the 8-phage genus-level mapping.
+ Knowledge model update: Gate 1 depolymerase × capsule is a validated signal; IFW calibration tradeoff is a new
  finding.

### 2026-04-12 10:01 CEST: GT04 — HPO with Optuna on three-layer feature set

#### Executive summary

Ran 50-trial Optuna HPO over key LightGBM hyperparameters using 5-fold stratified CV on the GT03 RFE-selected feature
set (257 features). The tuned params achieve 0.828 AUC vs 0.823 default — a marginal +0.4pp that is not statistically
significant (delta CI [-0.010, +0.015]). The GT03 default LightGBM configuration is near-optimal for this feature set.

#### Optuna best params vs GT03 defaults

| Parameter | GT03 Default | Optuna Best |
|-----------|-------------|-------------|
| n_estimators | 300 | 450 |
| learning_rate | 0.05 | 0.077 |
| num_leaves | 31 | 108 |
| min_child_samples | 10 | 9 |
| subsample | 0.8 | 0.75 |
| colsample_bytree | 0.8 | 0.83 |
| reg_lambda | (default) | 0.003 |
| reg_alpha | (default) | 0.19 |

#### Holdout results

| Arm | AUC | 95% CI | Top-3 | Brier |
|-----|-----|--------|-------|-------|
| gt03_default | 0.823 | [0.782, 0.859] | 89.2% | 0.161 |
| optuna_tuned | 0.828 | [0.775, 0.867] | 90.8% | 0.161 |

Delta (tuned vs default): [-0.010, +0.015] — not significant.

#### Interpretation

The HPO found a more complex model (108 leaves, 450 trees) that marginally improves CV AUC but doesn't translate to a
significant holdout gain. This is consistent with GenoPHI's finding that algorithm choice matters more than
hyperparameter tuning once the algorithm family is fixed. The default params (31 leaves, 300 trees) are a better
tradeoff: simpler, faster, and within noise of the tuned config.

The feature importance shift is minor: depo×capsule increases from 21.5% to 24.1% with tuned params (deeper trees
capture more of the cross-term signal), while host_typing drops from 12.8% to 5.5%.

#### Next steps

+ GT05: CatBoost comparison — the GenoPHI-optimal algorithm, not just tuned LightGBM.
+ GT06: GenoPHI per-phage receptor prediction to strengthen Gate 2.
