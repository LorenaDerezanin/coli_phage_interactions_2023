### 2026-02-15: PLAN update after external-data literature deep dive

#### Summary

We updated `PLAN.md` to make Track I execution source-prioritized and evaluation source-aware. The main principle
remains unchanged: internal paper data is the baseline foundation, and external data is integrated incrementally as a
measured enhancer.

#### What changed in the plan

1. Track I now has explicit ingestion order for supervised external datasets: `VHRdb -> BASEL -> KlebPhaCol -> GPB`.
2. Track I now separates Tier A supervised matrices from Tier B weak-label sources (`Virus-Host DB`,
   `NCBI Virus/BioSample`).
3. Track I now requires a `source_registry.csv` capturing source metadata, label type, host resolution, assay type,
   license, and access path.
4. Track F now includes source-aware evaluation and leakage controls: leave-one-datasource-out and cross-source transfer
   checks.
5. Track G now includes explicit training sequence: internal-only baseline first, then Tier A, then optional Tier B.
6. Track J now includes external data-license and use-restriction tracking in manifests.

#### Why this improves execution quality

1. Prevents mixing noisy and high-quality sources too early.
2. Makes performance gains attributable to specific sources.
3. Reduces hidden leakage risk across merged external datasets.
4. Preserves reproducibility and compliance when datasets have different use terms.

### 2026-02-15: Defining a Meaningful Model and Product-Oriented Benchmarks

Based on a review of the project plan and the available data, we've established a strategic framework for what
constitutes a "meaningful model" and how to evaluate its utility for a clinical product that recommends phage cocktails
for new _E. coli_ strains.

#### 1. Data Sufficiency Assessment

We have a sufficient foundation of raw data (`raw_interactions.csv`, genomic assemblies) to build a predictive model.
However, success is contingent on executing the data integrity and feature engineering tracks outlined in `PLAN.md`
(Tracks A, C, D, E). The raw data is not yet model-ready, and a significant effort is required to create clean labels
and potent predictive features.

#### 2. Defining a Meaningful Model

A "meaningful model" in this context is not a simple classifier. It is a system that produces a **calibrated probability
of lysis** for any given phage-bacterium pair.

- **Input:** The genomic assembly of a new _E. coli_ strain.
- **Output:** A ranked list of all phages in our panel, sorted by their predicted probability of successfully lysing the
  new strain, e.g., `P(Lysis | new_strain, phage_i)`.

This probabilistic output is essential for the downstream cocktail recommendation algorithm (Track H), which needs to
weigh evidence and confidence, not just binary predictions.

#### 3. Proposed Product-Oriented Evaluation Benchmarks

To be considered "very useful" for a clinical product, the model must meet a set of rigorous benchmarks that go beyond
standard academic metrics.

- **Level 1: Foundational Model Performance (Offline)**

  - **Top-3 Lytic Hit Rate > 95%:** The 3-phage cocktail recommended by the model must contain at least one truly lytic
    phage for over 95% of new strains.
  - **Precision @ P(Lysis) > 0.9 must be > 99%:** When the model is highly confident, it must be extremely reliable to
    build clinical trust.

- **Level 2: Cocktail Recommendation Utility (Simulation)**

  - **Simulated 3-Phage Cocktail Coverage > 98%:** An end-to-end simulation must show that the final recommended
    cocktails are effective against over 98% of strains in the held-out test set.

- **Level 3: Safety and Robustness (Negative Controls)**
  - **Sentinel Strain Recovery = 100%:** The model must correctly identify known effective phages for a predefined set
    of challenging "sentinel" strains, ensuring it doesn't fail on critical edge cases.

#### 4. Rationale for 3-Phage Cocktails

The choice of a 3-phage cocktail is a deliberate trade-off between efficacy and complexity.

- **Why Not Fewer (1-2 Phages)?**

  - **Higher Risk of Resistance:** Bacteria can rapidly evolve resistance to a single phage. Using multiple phages
    targeting different receptors or using different lytic mechanisms makes this much harder.
  - **Lower Coverage:** A single phage is less likely to be effective against a new, unknown strain.

- **Why Not More (4+ Phages)?**
  - **Manufacturing & Regulatory Complexity:** Each phage in a cocktail must be individually produced, characterized,
    and proven safe and stable. The regulatory burden and cost increase exponentially with each added phage.
  - **Risk of Antagonistic Interference:** Phages can compete for host cell resources or even block each other's entry,
    reducing the overall efficacy of the cocktail.
  - **Diminishing Returns:** The marginal benefit of adding a fourth or fifth phage is often small and does not justify
    the significant increase in complexity and cost.

#### 3-phage cocktail rationale note

A 3-phage cocktail is considered the current industry "sweet spot," providing robust defense against bacterial
resistance while remaining tractable from a manufacturing and regulatory perspective.

### 2026-02-15: Research on External Phage-Host Interaction Datasets

#### External Data Summary

To improve model performance, we must expand our training data beyond the internal `raw_interactions.csv`. Research was
conducted to identify public databases that contain phage-host interaction data. The findings indicate that while a
direct equivalent to our raw interaction matrix is rare, a significant volume of valuable data can be compiled from
various sources.

#### Key Data Sources Identified

1. **Dedicated Phage-Host Databases:** These are the highest-value targets for finding experimentally confirmed
   interaction pairs.

   - **Relevant Databases:** `Virus-Host DB`, `ViralHostRangeDB`.
   - **Content:** Aggregate interaction data from literature, providing "Phage X infects Bacterium Y" pairs. They
     typically lack the fine-grained dilution/replicate data but are excellent for expanding our core training set.

2. **Public Sequence Archives (NCBI, CNCB):** These contain the raw data from sequencing experiments and associated
   metadata.

   - **NCBI BioSample Database:** A good source for "known positive" interactions. The `isolation_host` field in a
     phage's BioSample record provides a confirmed host. This is a low-effort way to augment our dataset with thousands
     of positive examples.
   - **NCBI Sequence Read Archive (SRA) / China National Center for Bioinformation (CNCB) Genome Warehouse (GWH):** A
     potential goldmine, but high-effort. These archives contain raw data from high-throughput screening projects.
     Extracting an interaction matrix would require a significant bioinformatics effort (re-processing raw reads,
     mapping metadata), making it a long-term strategic goal.

3. **International Databases (Focus on China):**
   - **China National Center for Bioinformation (CNCB):** This is a key resource, hosting databases like the **Virus &
     Host (V&H) database**, which aggregates data from Chinese research and is a priority for exploration.

#### Next Steps & Strategy

The most pragmatic approach is to start with the lowest-effort, highest-value tasks.

1. **Short-Term:** Systematically query the APIs of **Virus-Host DB** and the **NCBI BioSample database**. The goal is
   to script a process to download all available phage-host pairs and integrate them as "known positive" in our dataset.
2. **Mid-Term:** Investigate the curated databases at the CNCB.
3. **Long-Term:** Scope the effort required to re-process a full screening project from the SRA or GWH.

### 2026-03-21: Strategic plan revision — v1 push

#### Context

Steel thread v0 is complete and all go/no-go gates pass. The v0 logistic regression on metadata features achieves 84.6%
top-3 hit rate (AUC 0.827) on the holdout set. However, the model has a "popular phage" bias: it recommends the same
broad-range Myoviridae for almost every strain because it has zero compatibility signal between specific phage RBPs and
specific host receptors. The strict-confidence slice drops to 62.5%.

Meanwhile, the repo contains rich genomic data that is completely unused: 138 defense system subtypes
(`370+host_defense_systems_subtypes.csv`), 12 outer-membrane receptor variant clusters
(`blast_results_cured_clusters=99_wide.tsv`), per-phage RBP annotations (`RBP_list.csv`), 97 complete phage genomes
(`FNA/`), UMAP phylogenomic embeddings (`coli_umap_8_dims.tsv`), and a pangenome matrix (`unique_host_genes.csv`).

This revision refocuses the plan to maximize prediction accuracy by exploiting this untapped data, targeting mid-May 2026
as an aspirational deadline for discussion-ready results.

#### What changed in the plan

**Tracks kept as-is (complete):** ST (11/11), A (10/10).

**Tracks cut or radically downsized:**

- **B** (EDA): marked done. TB06 (uncertainty map) and TB07 (mechanistic hypotheses) cut — TB07 is subsumed by actually
  building features in C/D/E; TB06 is nice-to-have, not blocking.
- **F** (Splits/Eval): 12 tasks → 2 tasks. ST03 already provides leakage-safe host-group and phage-family holdouts. Keep
  only: lock existing split as v1 benchmark + bootstrap CIs.
- **G** (Modeling): 14 tasks → 4 tasks. Two-stage mechanistic decomposition (P(adsorption) × P(lysis|adsorption))
  requires labeled adsorption outcomes that don't exist. Multi-task learning for strength/potency is lower-ROI than
  fixing the core binary prediction. Keep: LightGBM model + calibration + ablation suite + SHAP explanations.
- **H** (Cocktail): 8 tasks → 2 tasks. The heuristic recommender works at 84.6%. Optimization-based cocktail design is a
  later concern. Keep: existing top-3 + explained recommendations with SHAP features.
- **J** (Reproducibility): 7 tasks → 2 tasks. Keep: one-command regeneration + environment freeze.
- **K** (Wet-Lab): eliminated. No wet-lab access exists. Held-out strain evaluation serves the same credibility purpose.

**Tracks refocused (critical path):**

- **C** (Host Features): 6 tasks → 4 tasks. Defense subtypes (138 binary cols from defense_finder), OMP receptor variants
  (12 proteins with cluster IDs), capsule/LPS detail, and UMAP embeddings.
- **D** (Phage Features): 7 tasks → 3 tasks. RBP features from `RBP_list.csv`, k-mer embeddings from 97 FNA genomes
  (tetranucleotide SVD), phage distance embedding from VIRIDIC tree.
- **E** (Pairwise Features): 4 tasks → 3 tasks, moved to stage 1. RBP×receptor compatibility, defense evasion proxy
  (collaborative filtering from training data), phylogenetic distance to isolation host.

**Tracks kept at full scope:** I (External Data) — full Tier A pipeline: VHRdb + BASEL + KlebPhaCol + GPB ingestion with
strict ablation sequence.

**New track added:** P (Presentation) — 3 tasks: digital phagogram visualization, panel coverage heatmap, feature lift
visualization.

**Net effect:** 101 tasks → ~37 tasks. The cut tasks are done, deferred, or eliminated as low-ROI.

#### Execution timeline (aspirational, ~8 weeks)

- **Weeks 1–3 (Phase 1):** Feature engineering sprint. Expand pair table from ~28 metadata features to ~160–200 genomic
  features across C, D, E.
- **Weeks 2–6 (Parallel):** External data integration (Track I). Full Tier A ingestion with ablations.
- **Weeks 3–5 (Phase 2):** Model upgrade. LightGBM replacing logistic regression, calibration, ablation suite, SHAP.
- **Weeks 5–7 (Phase 3):** Evaluation and presentation artifacts. Bootstrap CIs, before/after comparison, explained
  recommendations, visualizations.
- **Week 8 (Phase 4):** Buffer. One-command reproducibility, environment freeze.

#### Expected performance targets

| Metric | v0 (current) | v1 (target) | Source of lift |
|--------|-------------|-------------|----------------|
| Top-3 hit rate (full-label) | 84.6% | 90–93% | OMP receptor + RBP compatibility resolves "popular phage" bias |
| Top-3 hit rate (strict-conf) | 62.5% | 72–78% | Defense subtypes provide discriminative signal |
| AUC | 0.827 | 0.87–0.90 | GBM captures nonlinear defense×phage interactions |
| Brier score | 0.171 | 0.12–0.15 | Better feature set + GBM calibration |

#### Risk factors

1. RBP data is sparse — not all phages have annotations. Handle with indicator features.
2. Defense subtype sparsity — many subtypes in <5 strains. Aggressive variance filtering needed.
3. E2 (defense evasion proxy) leakage risk — must compute strictly on training fold per CV split.
4. RBP-receptor lookup curation — requires ~2–3 days of manual literature work.
5. GBM overfitting — with ~200 features and 29K training pairs, need careful regularization.
6. Pangenome data deferred — `unique_host_genes.csv` (7,511 records) parked unless defense+receptor features plateau.
