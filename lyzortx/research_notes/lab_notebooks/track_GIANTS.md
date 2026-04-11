### 2026-04-12 00:30 UTC: Track GIANTS launched — three-layer biological prediction model

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

GT01 and GT02 are parallel. GT03 combines all layers. GT04 evaluates.
