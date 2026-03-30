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

### 2026-03-22: v1 model results — ablation paradox and feature-selection plan adjustment

#### What happened

Tracks C, D, E, and most of G landed in a single sprint. The v1 LightGBM model on 191 features (21 categorical + 170
numeric) was trained, calibrated, and ablated against the locked ST03 holdout (65 strains, 6,235 pairs).

#### Headline v0 → v1 comparison

| Metric | v0 (logreg, metadata) | v1 (LightGBM, all features) | Delta |
|--------|----------------------|---------------------------|-------|
| ROC-AUC | 0.827 | **0.910** | +0.083 |
| Top-3 hit rate (all strains) | 84.6% | **89.2%** | +4.6% |
| Top-3 hit rate (susceptible only) | 87.3% | **92.1%** | +4.8% |
| Brier score | 0.171 | **0.113** | -0.058 |
| ECE (isotonic, full-label) | 0.032 | **0.020** | -0.012 |
| ECE (isotonic, strict-conf) | 0.124 | **0.094** | -0.030 |

AUC exceeded the 0.87–0.90 target at 0.910. Calibration is excellent (ECE 0.020 isotonic on full-label). Top-3 hit rate
at 89.2% is close to the 90% target but does not clear it.

#### The ablation paradox

The TG03 ablation suite revealed an unexpected pattern: individual feature blocks outperform the combined model on the
top-3 ranking metric.

| Arm (v0 baseline + one block) | Top-3 hit rate | ROC-AUC | Brier |
|-------------------------------|---------------|---------|-------|
| v0 only (metadata baseline) | 86.2% | 0.908 | 0.114 |
| **+defense subtypes** | **90.8%** | 0.907 | 0.114 |
| +OMP receptors | 87.7% | **0.910** | **0.112** |
| **+phage genomic** | **90.8%** | 0.909 | 0.112 |
| +pairwise compatibility | 87.7% | 0.905 | 0.117 |
| all features combined | 87.7% | 0.909 | 0.113 |

Key observations:

1. **Defense subtypes and phage genomic features are the clear winners.** Each independently pushes top-3 hit rate to
   90.8% (+3 holdout strains recovered). These are the features that break the "popular phage" bias — they encode *which
   specific defense systems* a strain carries and *which specific phage genome architecture* can evade them.

2. **OMP receptors win on discrimination but not ranking.** Best AUC (0.910) and Brier (0.112) as a single block, but
   only 87.7% top-3 hit rate. The receptor variants help separate lytic from non-lytic pairs more precisely, but that
   precision does not translate into moving the *right* phages into the top-3 slots.

3. **Pairwise compatibility features hurt.** Adding them alone degrades both AUC (0.905, worst of all arms) and Brier
   (0.117, worst of all arms). The genus-level receptor lookup covers 80% of phages, but the compatibility signal may
   be too coarse, or the defense evasion proxy may be partially redundant with the raw defense subtype block.

4. **The all-features model does not dominate.** At 87.7% top-3 / 0.909 AUC, it underperforms the single-block defense
   and phage-genomic arms on ranking. This means feature interactions are creating noise when all blocks are thrown
   together without selection.

#### Why this happens

The most likely explanation: when all 191 features are present, the GBM splits on pairwise-compatibility and
OMP-receptor features that have good binary discrimination (high AUC) but that *dilute the ranking signal* from defense
and phage-genomic features. Top-3 hit rate is a ranking metric sensitive to the relative ordering of the top few phages
per strain, not just the classification boundary. A feature that improves average-case AUC can still hurt the top-k
ranking for specific strains by pushing a marginally-higher-scoring wrong phage above a correct one.

#### What this means for the plan

The 90.8% top-3 hit rate from defense-only or phage-genomic-only arms proves the target is reachable — we just need
to find the right feature combination that preserves it.

**Plan adjustment (two changes):**

1. **Added TG05: feature-subset sweep.** Train models on all 2-block and 3-block combinations of the 4 new feature
   blocks (defense, OMP, phage-genomic, pairwise). Identify the winning subset that maximizes top-3 hit rate without
   degrading AUC. Lock the final v1 feature configuration for downstream Tracks F, H, and P.

2. **Extended TG04 acceptance criteria.** SHAP analysis must now also produce a concrete recommendation of which feature
   blocks to keep in the final v1 model, informed by both SHAP evidence and TG03 ablation results. This ensures SHAP
   is not just descriptive but prescriptive.

No other plan changes needed. The pairwise compatibility work (Track E) was still worth doing — the features may become
useful after refinement (e.g., finer-grained receptor lookup, per-genus rather than per-family evasion rates) or in the
optimization-based recommender (Track H). We just should not force them into the v1 model if they hurt ranking.

#### Next steps (priority order)

1. **TG04** (SHAP): understand *which* pairwise features cause the ranking degradation and whether any are worth keeping.
2. **TG05** (feature-subset sweep): find the combination that clears 90%+ top-3 on holdout.
3. **TF01** (bootstrap CIs): run on the winning feature configuration, not necessarily the all-features model.
4. **TF02** (before/after error analysis): compare the winning v1 model against v0 on the specific holdout miss strains.

### 2026-03-22: Per-task model selection for Codex CI — cost analysis and assignments

#### Problem

The orchestrator dispatches all tasks to `gpt-5.4` via `codex-implement.yml`, regardless of task complexity. After the
first full day of automated v1 sprint execution (8 implementation runs for TD03, TE01–TE03, TG01–TG03, and TG04
attempt), total token consumption reached **856,463 tokens** with an average of **106,752 tokens per successful run**.

At gpt-5.4 pricing ($2.50/1M input, $15.00/1M output), this represents significant cost. Meanwhile, gpt-5.4-mini
($0.75/1M input, $4.50/1M output — approximately **70% cheaper**) scores 54.4% on SWE-Bench Pro and handles
straightforward coding tasks well. The key trade-off: gpt-5.4-mini has a 400K context window (vs 1.05M for gpt-5.4)
and shows a 5–10% accuracy gap on complex reasoning tasks.

#### Token usage breakdown from today's runs

| Task | Tokens | Status | Complexity |
|------|--------|--------|------------|
| TD03 (phage distance embedding) | 58,488 | success | Low — single MDS embedding from Newick tree |
| TE01 (RBP-receptor compatibility) | 134,626 | success | Medium — curated lookup + pair feature generation |
| TE02 (defense evasion proxy) | 103,687 | success | Medium — collaborative filtering with leakage guard |
| TE03 (phylogenetic distance) | 114,781 | success | Low-Medium — UMAP distance computation |
| TG01 (LightGBM training) | 129,532 | success | High — hyperparameter tuning, CV, comparator model |
| TG02 (calibration) | 85,193 | success | Medium — isotonic/Platt scaling, metric reporting |
| TG03 (ablation suite) | 120,957 | success | Medium — parameterized training loop, metric collection |
| TG04 (SHAP explanations) | 109,199 | failure | High — TreeExplainer, interpretation, cross-referencing |

Observations: TD03 (simplest task) used 58K tokens. TG01 and TE01 (most complex) used 130K+ tokens. The failed TG04
attempt still consumed 109K tokens. There is no correlation between token usage and task complexity strong enough to
predict model needs from token count alone — the decision must be based on task characteristics.

#### Waste patterns observed

Analysis of shell command patterns across today's 8 runs revealed systematic waste:

- **12 environment discovery issues**: Codex attempts `micromamba activate` in CI where micromamba does not exist. This
  is partially addressed in AGENTS.md but the agent still fumbles on first attempts.
- **30 git config attempts**: Git identity is pre-configured by the workflow, but Codex tries to set it again.
- **24 failed commands**: Various command failures across runs (exit codes 1, 127, 128).

These waste patterns account for an estimated 5–10% of total tokens. Fixing them helps but does not change the
fundamental cost picture — the bulk of tokens are legitimate implementation work.

#### Model capability comparison

**gpt-5.4** (full model):
- 1,050,000 token context window
- Highest accuracy on complex reasoning, architectural decisions, multi-file coordination
- Best for tasks requiring deep domain knowledge, novel algorithmic design, or cascading design decisions
- $2.50/1M input, $15.00/1M output

**gpt-5.4-mini**:
- 400,000 token context window
- 54.4% on SWE-Bench Pro (vs gpt-5.4 full score)
- 2x+ faster inference
- Strong on targeted edits, standard library usage, parameterized loops, visualization code
- $0.75/1M input, $4.50/1M output (~70% cheaper)

The 400K context window is sufficient for all tasks in this repo — the largest single-step context needed is the pair
table (~30K pairs × ~200 features), which is loaded from disk, not from the prompt. The deciding factor is reasoning
quality, not context size.

#### Assignment methodology

Each of the 16 pending tasks was evaluated on four axes:

1. **Novelty**: Does the task require implementing a pattern not yet in the codebase, or does it follow established
   patterns (TG01 training loop, TG03 ablation structure, ST05 calibration)?
2. **Domain criticality**: Do design errors cascade to downstream tasks? A wrong harmonization protocol (TI05) poisons
   all external data work; a wrong bar chart color (TP03) is trivially fixable.
3. **Reasoning depth**: Does the task require multi-step logical reasoning (combinatorial search, leakage analysis,
   SHAP interpretation) or straightforward data assembly (bootstrap resampling, metric aggregation)?
4. **Established patterns**: Can the task largely reuse existing code structure? Tasks that follow TG01/TG03 patterns
   (train model, collect metrics, output CSV) are good candidates for gpt-5.4-mini.

#### Assignments: gpt-5.4 (5 tasks)

**TG04 — Compute SHAP explanations for per-pair and global feature importance**
- TreeExplainer integration is new to the codebase (no existing SHAP usage to copy)
- Must cross-reference TG03 ablation results with per-feature SHAP values to produce prescriptive recommendations
- Per-strain narrative synthesis ("what makes strain X hard to predict") requires domain interpretation
- Acceptance criteria explicitly require a concrete recommendation of which feature blocks to keep — this is a design
  decision, not a computation

**TG05 — Run feature-subset sweep to find best block combination for top-3 ranking**
- 10 combinatorial model runs (C(4,2) + C(4,3) = 6 + 4) with careful feature-block bookkeeping
- Must identify the winning subset, compare against the TG01 all-features model, and lock the final v1 configuration
- The "lock" decision affects all downstream tracks (F, H, P) — getting it wrong means rework
- Unlike TG03 (sequential ablation), this requires combinatorial logic and winner-selection heuristics

**TI05 — Define harmonization protocol for Tier A datasets**
- Multi-source schema alignment across VHRdb, BASEL, KlebPhaCol, and GPB — each has different column semantics, label
  types, and confidence levels
- Design decisions here cascade to TI06 (Tier B ingestion), TI07 (confidence tiers), TI08 (integration), and TI09
  (ablation sequence)
- Requires domain understanding of what "lysis" means in each source's assay context
- Errors are silent and expensive to detect (wrong label mapping looks correct until model evaluation)

**TI07 — Define confidence tiers for external labels**
- Subjective tier design: what confidence level does a VHRdb curated record deserve vs an NCBI BioSample isolation_host
  record?
- Weighting strategy for training (how much to down-weight low-confidence labels) requires balancing coverage vs noise
- Cascades to TI08 (integration uses tier weights) and TI09 (ablation sequence is ordered by tier)
- The ST01b strict-confidence concept provides a pattern, but extending it to external sources with heterogeneous assay
  types is a design challenge

**TI08 — Integrate external data as non-blocking enhancer**
- Architectural pattern: external data must be truly optional — the internal-only pipeline must remain runnable
- Leakage prevention: external datasets may contain organisms that overlap with holdout strains — careful ID validation
  needed
- Fallback handling: graceful degradation when external data files are absent (CI runs without them)
- The "non-blocking" constraint means conditional imports, feature-flag-like behavior, and defensive coding — harder to
  get right than it looks

#### Assignments: gpt-5.4-mini (11 tasks)

**TF01 — Lock ST03 split as v1 benchmark and add bootstrap CIs**
- Bootstrap resampling is a standard NumPy pattern (np.random.choice with replacement, 1000 iterations)
- Dual-slice filtering (full-label vs strict-confidence) already implemented in TG02
- Metric functions (AUC, top-3 hit rate, Brier, ECE) already exist in the codebase
- No design decisions — just apply existing patterns with confidence intervals

**TF02 — Before/after comparison of v0 vs v1 with error bucket analysis**
- Side-by-side metric table is a DataFrame join between ST05 and TG02 outputs
- Error bucket identification is algorithmic: for each holdout strain, did v0 miss and v1 hit (or vice versa)?
- The "honest reporting" requirement is met by listing strains that remain unpredictable — no narrative synthesis needed

**TH02 — Add explained recommendations with calibrated P(lysis), CI, and SHAP features**
- Top-3 recommendation assembly reuses ST06 logic with TG02 predictions
- SHAP feature extraction is a DataFrame merge with TG04 output (pair_id → top-3 SHAP features)
- CI computation is percentile-based from bootstrap or calibration data
- Output formatting for "clinician-ready" display is light presentation logic

**TI06 — Tier B weak-label ingestion (Virus-Host DB, NCBI BioSample)**
- ID cross-referencing against canonical maps from Track A (already built)
- Follows TI03/TI04 ingestion patterns (source fidelity preservation, standardized output format)
- Confidence tier assignment uses rules defined in TI07 (upstream dependency)

**TI09 — Run strict ablations in sequence**
- Parameterized loop over TG01 training with progressively more external data
- 6 training runs: internal-only → +VHRdb → +BASEL → +KlebPhaCol → +GPB → +Tier B
- Metric collection identical to TG03 ablation suite structure
- Leakage verification is the only subtle part, but the holdout split is already locked

**TI10 — Track incremental lift and failure modes by datasource and confidence tier**
- GroupBy aggregation of TI09 ablation results by source and tier
- Failure mode detection: which strains regressed when adding each source?
- Follows ST09 error analysis pattern (documented hypotheses per miss strain)

**TJ01 — One command to regenerate all v1 outputs from raw data**
- Orchestration script that calls existing `run_track_*.py` entry points in dependency order
- Dependency validation (check prerequisites before running each track)
- Follows `run_track_a.py` structural pattern

**TJ02 — Freeze environment specs and seeds for v1 benchmark run**
- Documentation task: inventory all `random_state` parameters, export `pip freeze`
- Verification: run pipeline twice with frozen seeds, compare outputs
- No new algorithms or ML logic

**TP01 — Build digital phagogram visualization for per-strain phage ranking**
- Plotly or Matplotlib visualization of ranked phage list with P(lysis) and confidence bands
- Data inputs are well-defined (TG02 calibrated predictions, TG04 SHAP values)
- Standard charting library usage — no novel algorithmic work

**TP02 — Build panel coverage heatmap across strain diversity**
- Seaborn/Matplotlib heatmap with host phylogroup × phage family axes
- Aggregation: mean P(lysis) per cell from TG02 predictions grouped by ST02 taxonomy
- Hard-to-lyse gaps are visually evident from low-probability cells

**TP03 — Build feature lift visualization from ablation suite results**
- Bar chart from TG03 ablation CSV (already generated and available)
- Standard Matplotlib bar chart with v0 baseline reference line
- Simplest visualization task — no data processing beyond reading the CSV

#### Projected cost impact

Assuming average token usage of ~107K per run (today's observed average):

- **All gpt-5.4 (status quo):** 16 tasks × 107K tokens × ~$17.50/1M blended ≈ **$30**
- **With model selection:** (5 × 107K × $17.50/1M) + (11 × 107K × $5.25/1M) ≈ $9.36 + $6.18 ≈ **$15.50**
- **Estimated savings:** ~48% on implementation runs

This does not include lifecycle (review feedback) runs, which use the same model and add 1–3 rounds per task. With
lifecycle runs, the savings scale proportionally.

#### Implementation approach

Two PRs to minimize risk:

1. **PR 1 (data-only):** Add `model:` field to all 16 pending tasks in `plan.yml`. The existing `load_plan()` function
   ignores unknown YAML fields, so this is a no-op until PR 2 lands. This allows model assignments to be reviewed and
   adjusted independently of the code changes.

2. **PR 2 (code):** Wire the model field through `plan_parser.py` → `orchestrator.py` → issue body → workflow YAML.
   The orchestrator emits a `<!-- model: gpt-5.4-mini -->` HTML comment in the issue body. The workflow extracts it
   with a small Python helper (`parse_model_directive.py`) and passes it to `openai/codex-action@v1`. No default — if
   the model directive is missing, the workflow fails with a clear error message.

#### Open questions for future refinement

1. **Should lifecycle (review feedback) runs use the same model as the original implementation?** Currently planned: yes,
   for consistency. But review feedback is often simpler than initial implementation — a cheaper model might suffice.
2. **Should we track per-task model cost to validate these assignments?** The `ci_token_usage.py` tool already reports
   per-run tokens. After a few tasks run with model selection, we can compare actual mini vs full token usage.
3. **When should a task be upgraded from mini to full?** If a gpt-5.4-mini run fails, the model assignment can be
   changed in plan.yml and the orchestrator re-dispatched. No code change needed.

### 2026-03-22: Label leakage concern — deployment-realistic evaluation needed

#### Problem identified

The TG04 SHAP analysis revealed that the two highest-impact features in the v1 model are derived from training labels,
not from genomic content:

| SHAP rank | Feature | Mean |SHAP| | Source |
|-----------|---------|------------|--------|
| 1 | `legacy_label_breadth_count` | 1.183 | Panel metadata: count of lytic phages per strain |
| 2 | `legacy_receptor_support_count` | 0.572 | TE01: training-fold lysis frequency per receptor variant |

These features are powerful within-panel predictors but are **unavailable for truly novel strains** in a deployment
scenario. A new clinical isolate arrives with a genome assembly — you can derive its defense systems, receptor variants,
LPS type, and phylogenomic embedding, but you cannot know `legacy_label_breadth_count` (you haven't screened it yet) or
`legacy_receptor_support_count` (its specific receptor variant may not appear in the training data).

The model leans heavily on these: `legacy_label_breadth_count` alone has 2x the SHAP impact of the next feature. This means the
holdout metrics (AUC 0.910, top-3 89.2%) are **optimistic for real-world deployment** because the holdout strains are
from the same experimental panel and their receptor variants overlap with training strains.

#### What is genuinely novel genomic signal?

The SHAP ranking after the two panel-artifact features shows real genomic signal:

| SHAP rank | Feature | Mean |SHAP| | Deployment-available? |
|-----------|---------|------------|----------------------|
| 3 | `phage_gc_content` | 0.217 | Yes — from genome |
| 4 | `phage_genome_length_nt` | 0.203 | Yes — from genome |
| 5 | `host_lps_type=R1` | 0.159 | Yes — from genome assembly |
| 6 | `defense_evasion_mean_score` | 0.130 | Partially — rates from training, defense profile from genome |
| 7 | `isolation_host_defense_jaccard_distance` | 0.123 | Yes — computable from genome |

These features have 5–10x smaller SHAP values than the panel artifacts. The model *is* learning genomic signal, but the
panel-memorization features dominate.

#### Why this matters

If we present the 89.2% top-3 hit rate to CDMO partners as "what the model can do for your new clinical isolate," we
are overpromising. The honest number — what the model achieves using only features available at deployment time — will be
lower. That's the number partners need to trust.

This also explains the ablation paradox from the previous note: adding the pairwise compatibility block (which includes
`legacy_receptor_support_count`) shifts the model's attention toward collaborative-filtering signal that is
strong on average but misleading for specific holdout strains.

#### Plan adjustment

Added two acceptance criteria to TG05 (feature-subset sweep):

1. Include a **deployment-realistic arm** that excludes all features derived from training labels
   (`legacy_label_breadth_count`, `legacy_receptor_support_count`) to measure generalization to truly novel strains.
2. Report both **panel-evaluation** and **deployment-realistic** metrics for the winning configuration.

This gives us two numbers to present: the panel metric (what the model does on known strains with full context) and the
deployment metric (what a CDMO partner should expect for a new isolate). Both are valuable — the panel metric validates
the approach, the deployment metric sets honest expectations.

#### No other plan changes needed

The concern does not invalidate Tracks C, D, or E. The genomic features are the right features — they just need to be
evaluated separately from the panel-artifact features. TG05 is the right place to do this, and the task is already
pending.

### 2026-03-22: TG05 results — deployment-realistic model outperforms panel model on ranking

#### TG05 sweep results

The feature-subset sweep (PR #156) evaluated all 10 two-block and three-block combinations of the four new feature
blocks (defense, OMP, phage-genomic, pairwise) with fixed TG01 hyperparameters on the ST03 holdout (65 strains).

**Panel-evaluation results (all arms include v0 baseline + legacy_label_breadth_count):**

| Arm | AUC | Top-3 (all) | Top-3 (susceptible) | Brier |
|-----|-----|-------------|---------------------|-------|
| TG01 all-features reference | 0.9091 | 87.7% | 90.5% | 0.113 |
| defense + phage-genomic | 0.9082 | 89.2% | 92.1% | 0.111 |
| OMP + pairwise | 0.9066 | 89.2% | 92.1% | 0.116 |
| **defense + OMP + phage-genomic (WINNER)** | **0.9108** | 87.7% | 90.5% | **0.110** |
| defense + phage-genomic + pairwise | 0.9082 | 84.6% | 87.3% | 0.113 |

Winner: **defense + OMP + phage-genomic** — best AUC (0.9108), best Brier (0.110), pairwise block excluded. The winner
selection rule required AUC ≥ TG01 all-features (0.9091), then maximized top-3. Only this arm cleared the AUC gate.

**Deployment-realistic result (winner minus legacy_label_breadth_count):**

| | AUC | Top-3 (all) | Top-3 (susceptible) | Brier |
|--|-----|-------------|---------------------|-------|
| Panel model | 0.911 | 87.7% | 90.5% | 0.110 |
| Deployment-realistic | 0.835 | **92.3%** | **95.2%** | 0.158 |

#### Interpretation

1. **Removing `legacy_label_breadth_count` improves ranking.** The deployment-realistic model achieves 92.3% top-3 hit rate —
   higher than any panel-evaluation arm. This is counterintuitive but mechanistically sound: `legacy_label_breadth_count` tells
   the model "this strain is broadly susceptible" which biases it toward recommending the same popular broad-range
   phages. Without that shortcut, the model is forced to use defense subtypes, OMP receptor variants, and phage k-mer
   profiles to make strain-specific picks. Those features produce better *rankings* even though they produce worse
   *pairwise discrimination* (AUC drops from 0.911 to 0.835).

2. **AUC and top-3 measure fundamentally different things.** AUC measures how well the model separates lytic from
   non-lytic pairs across the entire probability range. Top-3 measures whether the correct phages end up in the top 3
   slots per strain. A feature that improves average-case AUC can hurt top-3 by pushing a marginally-higher-scoring
   wrong phage above a correct one. `legacy_label_breadth_count` is exactly this kind of feature — it improves the average but
   dilutes the per-strain signal.

3. **The pairwise block (Track E) was correctly excluded.** No subset containing pairwise cleared the AUC gate while
   improving top-3. The genus-level receptor lookup (TE01) was too coarse — 80% coverage but no within-genus
   specificity. The defense evasion proxy (TE02) showed up at SHAP rank #6 globally, suggesting the signal exists but
   the current encoding doesn't capture it well enough. Worth revisiting in v2 with finer-grained lookups (per-species
   or per-RBP-family), but not worth forcing into v1.

4. **The winner selection rule was too conservative.** Two 2-block arms (defense + phage-genomic, OMP + pairwise) both
   hit 89.2% top-3 but were disqualified because their AUC was 0.001–0.003 below the all-features reference. On 65
   holdout strains, AUC differences this small are noise. The rule prevented the sweep from selecting these
   higher-ranking arms. For future sweeps, the AUC gate should use a tolerance (e.g., AUC ≥ reference − 0.005) rather
   than a strict ≥.

5. **Two numbers for partners.** The locked v1 model gives CDMO partners two honest benchmarks:
   - Panel evaluation (87.7% top-3, AUC 0.911): what the model does on fully characterized strains
   - Novel-strain prediction (92.3% top-3, AUC 0.835): what to expect for a new clinical isolate with only a genome

   The 92.3% clears our 90% target. The 0.835 AUC means per-pair probability estimates are less reliable for novel
   strains, so recommendations should be presented as a ranked shortlist, not as calibrated P(lysis) values.

#### Track P acceptance criteria updated

All three Track P tasks (TP01, TP02, TP03) now require presenting both the panel model and the deployment-realistic
model side by side. Partners need to see both: the panel model validates the approach, the deployment-realistic model
sets honest expectations for their use case.

### 2026-03-23: Label leakage invalidates v1 model — plan restructured

#### Executive summary

Review of the TG04 SHAP results revealed that the v1 model's two strongest features (`legacy_label_breadth_count` with mean
|SHAP| 1.18 and `legacy_receptor_support_count` at 0.57) are derived from training labels, not independent
inputs. The v1 "panel-default" model is not predicting — it is memorizing. The plan has been restructured to delete these
features, retrain from scratch, and report whatever comes out as the honest v1 baseline. The prior dual-arm
panel/deployment framing is abandoned: there is only one model, the leakage-clean one.

#### What was decided

1. **Label-leaked features are a bug, not a variant.** `legacy_label_breadth_count` is literally "how many phages lyse this host"
   repackaged as a feature. `legacy_receptor_support_count` counts training-positive pairs per receptor
   cluster. Both encode the answer. Keeping them as an "optional panel-only arm" would be dishonest.

2. **Track P deleted.** All three presentation artifacts (digital phagogram, coverage heatmap, feature lift
   visualization) were designed around the dual-arm leaked model. The code, tests, and lab notebook have been removed.
   Visualizations will be rebuilt from scratch after the clean model is established.

3. **Track I made a dead end.** All 10 Track I tasks are done, but no Track I output feeds into any downstream track.
   TI09/TI10 count rows but never retrain a model with external data. Track I remains in the plan as completed work but
   nothing depends on it until external data is actually wired into model training.

4. **Track G extended with four new tasks:**
   - TG06: Delete leaked features from ST02, Track E, and Track G code
   - TG07: Retrain, recalibrate, re-run SHAP and ablation on the clean feature set
   - TG08: Re-run Track F evaluation and Track H recommendations, verify Track J end-to-end
   - TG09: Investigate whether non-leaky features can close the ~7.6pp AUC gap

5. **Track J depends only on G now** (removed F, H, I from depends_on to match what TJ01 actually runs).

6. **Track F and H descriptions updated** to note their current metrics are invalidated and will be re-run as part of
   TG08.

#### Why the prior dual-arm framing was wrong

The 2026-03-22 project entry framed two numbers for partners: panel evaluation (AUC 0.911) and novel-strain prediction
(AUC 0.835). The implicit message was "the model works great on known strains and somewhat worse on novel ones." The
actual message should have been: "the model's best feature is the training labels in disguise, and the 0.911 AUC is
inflated by that leakage." The deployment-realistic arm was the honest model all along — it should have been the only
model from the start.

#### What metrics to expect after cleanup

The deployment-realistic numbers from TG05 (top-3 92.3%, AUC 0.835, Brier 0.158) are the current best estimate for the
clean model, but the actual TG07 retrain may produce different numbers since the feature pipeline itself changes (not
just column exclusion at prediction time). TG09 will investigate whether the AUC gap can be partially closed with
non-leaky features.

### 2026-03-24: Clean v1 model locked — `defense + phage_genomic` is the honest baseline

#### Executive summary

Post-merge review of TG06-TG08 found two additional problems: LightGBM nondeterminism causing the sweep winner to flip
across runs, and 5 out of 13 pairwise features being derived from training labels (soft leakage). The v1 winner is now
locked to `defense + phage_genomic` — the 2-block arm that excludes all label-derived pairwise features. LightGBM
determinism will be fixed and the lock file will be treated as a human decision rather than a regenerated output.

#### Current honest v1 numbers

These are from the TG08 Track J end-to-end regeneration (the most realistic run):
- Winner: `defense + phage_genomic`
- Holdout ROC-AUC: ~0.837
- Holdout top-3 hit rate: 90.8%
- Holdout Brier: ~0.160

The 90.8% top-3 meets the 90%+ target. The AUC is lower than the old leaked model (0.911) but honest.

#### Remaining leakage in the codebase

The pairwise block (Track E) still contains label-derived features that are not yet deleted:
- TE02: all 4 `defense_evasion_*` features (collaborative filtering on training lysis rates)
- TE01: `receptor_variant_seen_in_training_positives` (binary flag from training positives)

These are excluded from the v1 model by locking to `defense + phage_genomic`, but the code still exists. A future task
should evaluate whether the clean pairwise features (TE03 distances, TE01 curated lookups) add value individually
without the label-derived ones.

#### Plan updates

- TG09: Fix LightGBM determinism, lock `defense + phage_genomic`, separate sweep from Track J regeneration
- TG10: Re-run downstream verification on the stable 2-block lock
- TG11 (was TG09): Investigate calibration gap — now aware of pairwise soft leakage, should evaluate clean pairwise
  features individually

#### Future: clean up upstream soft leakage in Track E

After TG09-TG11 are done and the v1 baseline is stable, consider a follow-up pass to delete or gate the training-label-
derived features in Track E itself: TE02's `defense_evasion_*` collaborative filtering features and TE01's
`receptor_variant_seen_in_training_positives`. These are currently excluded from v1 by the 2-block lock, but the code
still produces them. Deleting them would make the feature pipeline honest by construction rather than by configuration,
and would prevent future sweep runs from accidentally including them in a winning arm.

#### Future: workflow tooling (Snakemake / Nextflow) — not yet

Considered reimplementing the Track G pipeline in Snakemake for declarative dependencies, automatic skip of up-to-date
steps, and built-in parallelism. Decision: defer. The pipeline is still in flux (TG09-TG11 pending), the pain points
are scientific (leakage, nondeterminism) not operational, and the pipeline is small enough (~5 steps, ~30 min end-to-end,
single machine) that the linear `run_track_g.py` / `run_track_j.py` runners are sufficient. Revisit when: (a) external
data (Track I) is wired in and multiple data variants need training, (b) the pipeline grows beyond ~10 steps with
nontrivial branching, or (c) cluster execution or robust checkpointing becomes necessary.

### 2026-03-24: External-data decision locked for v1

#### Summary

Track K is now closed at the strategy level: the v1 model remains trained on internal data only. TK01 found no
joinable VHRdb rows in the available TI08 artifact, and the fixture-based TK02-TK05 follow-up arms were all neutral on
ROC-AUC, top-3 hit rate, and Brier score relative to the locked internal-only baseline.

#### Decision

- `lyzortx/pipeline/track_g/v1_feature_configuration.json` now records `external_data_lock_task_id: TK06`,
  `locked_training_data_arm: internal_only`, and `locked_external_source_systems: []`.
- No final-model retrain was performed for this task because the promotion condition was not met.
- Future production reruns can still revisit the decision through TK06 once real Track I / Track K manifests exist, but
  the current repo state does not justify changing the v1 release contract.

### 2026-03-24: Track I and Track K completed on empty data — all reopened

#### Executive summary

Post-merge review of TK01-TK05 revealed that all Track K tasks reported zero deltas because Track I never downloaded
any external data. The entire TI03-TI10 chain built plumbing that reads from nonexistent files. Track K inherited the
emptiness and reported "neutral" lift based on zero external rows. TK06 (PR #217) was rejected because it would have
locked an "internal-only" decision based on zero evidence. All 14 tasks (TI03-TI10, TK01-TK06) have been set back to
pending with acceptance criteria that require >0 real rows at every stage.

#### Root cause

1. **No Track I step downloads external data.** The code reads from local paths that were never populated. No HTTP
   requests, no API calls, no `urllib` — the download step was never implemented.
2. **Track K silently tolerated missing TI08 output.** `build_vhrdb_lift_report.py` line 258:
   `cohort_rows = read_csv_rows(...) if path.exists() else []` — silent fallback to empty list.
3. **Agents marked tasks done despite zero results.** Lab notebooks openly acknowledged "0 joinable rows" and
   "validated on a minimal fixture" but tasks were closed as completed anyway.
4. **CI starts with no generated outputs.** The agents ran in GitHub Actions where `lyzortx/generated_outputs/` does not
   exist. Without fail-fast on missing data, every step that depends on generated outputs silently produces nothing.

#### What changed

- New AGENTS.md rules: fail-fast on missing data, substance over plumbing, CI environment note
- TI03-TI06 now require downloading real data from source URLs and producing >0 rows
- TI07-TI10 now require >0 real external rows at each processing stage
- TK01-TK05 now require >0 external rows in the augmented training set
- TI03-TI07 upgraded to gpt-5.4 (external service integration needs research judgment)
- TK06 (PR #217) rejected — cannot synthesize results that don't exist

#### Lessons

- **Zero results is not a finding, it's a failure.** A task that runs to completion on empty inputs and reports zero
  deltas has failed its acceptance criteria, even if the code ran without error.
- **Silent fallback is not graceful degradation.** Code that returns empty results when inputs are missing is hiding a
  bug, not handling an edge case. Raise on missing data.
- **Acceptance criteria must include data volume assertions.** ">0 rows" is the minimum bar. Without it, an agent can
  build correct plumbing that processes nothing and call it done.

### 2026-03-24: V2 plan restructure — external data dead end, mechanistic features forward

#### Executive summary

Deep analysis of Track I and Track K revealed that external data integration is a dead end with the current pipeline:
only VHRdb overlaps with the internal 404×96 panel (23,885 pairs), and that's the paper's own data uploaded to VHRdb by
the original authors (datasource 257). BASEL, KlebPhaCol, GPB, Virus-Host DB, and NCBI all have zero strain overlap.
Track K was deleted. Track I was trimmed to download-only (TI01-TI06). A new Track L (Mechanistic Phage Features)
replaces the deleted label-derived pairwise features with annotation-based features from Pharokka. A label policy fix
(TA11) captures the +3.1pp top-3 improvement from downweighting borderline `matrix_score=0` noise positives.

#### Key findings from VHRdb analysis

1. **VHRdb overlap is circular.** The 23,885 overlapping pairs are the paper's own experiment (datasource 257: "Host
   range of the 96 coliphages from the Antonina Guelin collection on the 403 natural isolates of Escherichia from the
   Bertrand"). VHRdb compressed the paper's 0-4 score to 0-2 for upload.
2. **Perfect agreement on matrix scores 1-4.** Zero off-diagonal entries in the cross-tabulation. VHRdb's 3-level
   scale is a lossless compression of the 0-4 scale for non-zero scores.
3. **1,737 disagreements are all `matrix_score=0, label=1`.** These are pairs where 1-3 replicates out of 8-9 showed
   lysis (noise), the matrix aggregated to 0, but the "any lysis" label policy called them positive. VHRdb correctly
   reports "No infection" for all of them.
4. **No other VHRdb datasource shares our phages.** Only datasource 153 (3 LF82 phages on ECOR) has any panel phage
   overlap — 3 out of 96 phages, across 73 ECOR strains with no panel overlap.

#### What changed in the plan

- **Deleted Track K** — all code, tests, plan entries. Archived notebook to `archive_v1/track_K.md`.
- **Trimmed Track I** to TI01-TI06 (download infrastructure). Deleted TI07-TI10 code (confidence tiers, training
  cohorts, ablations, lift analysis) — these depended on external data being trainable.
- **Added Track L** (Mechanistic Phage Features) — TL01-TL06: set up bioinformatics env, annotate phages with
  Pharokka, build mechanistic RBP-receptor and defense-evasion features, retrain, validate on external strains.
- **Added TA11** (label policy fix) — downweight `matrix_score=0` noise positives based on the VHRdb finding.
- **Track E kept as-is** — Track G depends on it; Track L will eventually supersede it.

#### Pharokka POC results (local, 2026-03-24)

- Pharokka 1.9.1 on LF82_P8: 276 CDS annotated in 2m 43s
- 29 tail genes (including long tail fiber proximal/distal subunits, baseplate components)
- 7 lysis genes (holin, spanins, lysis inhibitors)
- 1 anti-restriction nuclease
- 140 hypothetical proteins (51% unannotated — typical for phage genomes)
- Extrapolation: ~1 hour for all 97 phages at 4 threads

#### Future: External validation dataset for generalized inference pipeline

**Trigger condition:** Revisit when a full interaction matrix (with both positives and negatives) for novel E. coli
strains becomes available. TL09 covers positive-only validation via VHdb; this note is about the stronger full-matrix
validation that would allow AUC and top-3 computation.

**Context:** The v2 plan adds a generalized inference pipeline (TL06-TL08) that accepts arbitrary E. coli genomes and
arbitrary phage FNAs, computes features from sequence, and predicts lysis. The locked v1 model uses only defense subtypes
(79 features from Defense Finder) and phage k-mer SVD (26 features from tetranucleotide frequencies) — both
sequence-derivable, so the pipeline is architecturally sound. TL09 validates on VHdb positive-only pairs, but
full-matrix out-of-distribution validation (with negatives) requires a dataset that doesn't yet exist in public.

**Why existing sources don't provide a full matrix for novel strains:**

- **ECOR strains** are already in the 404-strain training panel (71 of 404). Not novel.
- **VHRdb pair-table overlap** (26,029 rows) is the paper's own data (datasource 257). Novel VHRdb pairs (30,643) failed
  entity resolution. VHRdb strain-level pairs (~500 positives across ~70 hosts) are usable for positive-only validation
  (TL09) but contain no negatives.
- **BASEL** has labeled E. coli interactions but only 4 E. coli host strains. Phages are novel (78 genomes on NCBI,
  MZ501046-MZ501113) but the host count is too small for meaningful metrics.
- **KlebPhaCol** is Klebsiella, not E. coli. Wrong species.
- **GPB** (Gut Phage Biobank) has only 1 E. coli strain (RTGS0219) out of 40 hosts. Phage genomes on CNGB, not NCBI.
- **SNIPR001** (Nature Biotech 2023) has 429 E. coli strains x 162 phages with interaction matrix on GitHub, but strain
  genomes are commercially restricted.

**What a full-matrix validation dataset requires:**

1. **E. coli strains not in our 404-strain panel** — at least 20-30 strains for meaningful ranking metrics.
2. **Full interaction matrix** (lysis AND no-lysis from spot assay or equivalent) — not just positive pairs.
3. **Genome assemblies for the host strains** — so TL07 can run Defense Finder.
4. **Phage genomes (FNA)** — so TL06 can project k-mer features.
5. **Sufficient phage panel size** — at least 10-20 phages per host for top-3 ranking to be meaningful.

**Partial solution found (2026-03-28): Virus-Host DB positive-only validation.**

Virus-Host DB contains ~70 E. coli strains at strain-level resolution (excluding lab strains) with ~900 unique phage
genome accessions and ~500 positive pairs (phage confirmed to lyse host, from literature). Most hosts have NCBI genome
assemblies. This gives a positive-only validation path: download host assemblies + phage FNAs, run generalized inference,
check that known-positive pairs get high predicted P(lysis). Limitations: no negatives, so AUC and top-3 hit rate cannot
be computed. Metrics are limited to recall-on-positives, calibration-on-positives, and rank-of-positives-vs-random. This
is implemented as TL09 in the plan.

**Full interaction matrix validation still requires:**

- Published phage therapy studies with E. coli host-range matrices and deposited genomes (both host and phage).
- Phage biobanks that publish interaction data alongside NCBI accessions (e.g., future GPB or DSMZ releases).
- Direct collaboration with labs running E. coli phage screening panels.
- The SNIPR001 dataset (Nature Biotech 2023) has 429 E. coli strains x 162 phages with interaction matrix on GitHub,
  but strain genomes are commercially restricted.

**What to do when a full-matrix candidate is found:**

- Verify E. coli strain count, phage count, and label quality before committing to ingestion.
- Download host genome assemblies from NCBI, run TL07/TL08, compare predictions against labels.
- Report AUC, top-3 hit rate, and calibration metrics as the honest out-of-distribution benchmark.
- Compare against in-panel holdout metrics to quantify the generalization gap.

#### Future: Structural RBP-receptor prediction via protein folding

**Trigger condition:** Revisit if TL02 (RBP-receptor compatibility from Pharokka annotations) hits the escape hatch —
i.e., PHROG functional annotations are too coarse to map RBPs to specific host receptor targets (FhuA, BtuB, OmpC,
LPS core, etc.).

**The problem:** Pharokka labels phage genes by PHROG family ("tail fiber protein," "tail spike protein") but doesn't
tell you which receptor the RBP binds. That mapping is determined by the 3D structure of the RBP tip domain and its
binding interface, not by sequence homology alone.

**Potential approach — structure-based RBP clustering:**

1. Predict 3D structures for all phage RBPs using AlphaFold2 or ESMFold.
2. Extract receptor-binding tip domains (C-terminal regions of tail fibers / tail spikes).
3. Cluster RBPs by structural similarity of the tip domain. RBPs with similar tip folds likely target the same receptor
   class.
4. Map each structural cluster to a receptor class using known reference structures from the literature (e.g., T4 long
   tail fiber tip → OmpC, T5 pb5 → FhuA).
5. For each phage-host pair, compute a compatibility feature: does this phage's RBP cluster match a receptor present on
   the host (from Track C OMP data)?

**Precedent:** The BASEL phage collection papers (Dunne et al. 2021, Maffei et al. 2025) used structural analysis to
validate receptor assignments for their phage panels. Structure-based receptor grouping is scientifically established.

**Why not now:**

- Heavy computational dependency (AlphaFold or ESMFold, ~200-300 structure predictions).
- Requires curated reference structures to anchor the cluster→receptor mapping.
- Would be a separate track-level effort, not a subtask.
- TL02's escape hatch already handles the annotation-only failure case gracefully.

**What to do if triggered:** Scope a new track (e.g., Track M: Structural RBP-Receptor Prediction) with explicit
acceptance criteria for structure prediction, tip domain extraction, clustering, and receptor assignment. Evaluate
whether the expected lift justifies the computational cost before committing.

#### Future: Depolymerase domain annotation for capsule-specificity matching

**Trigger:** Enrichment analysis (TL02) shows that host capsule/LPS features carry signal for lysis prediction, but
pharokka's generic annotations ("polysaccharide chain length determinant protein", "tail spike protein") are too coarse
to distinguish which capsule types a phage can degrade.

**Context (2026-03-29):** TL01 found 18 "polysaccharide chain length determinant" genes across the panel, but pharokka
does not annotate capsule-type specificity. Dedicated depolymerase tools (DepoScope, DePP, PDP-Miner) only classify
binary depolymerase yes/no — none predict capsule-type targets. The most promising approach is running Pfam/InterPro on
tail spike protein sequences to identify glycosyl hydrolase families, which can be mapped to polysaccharide substrate
classes via literature. Track C has LPS core type (~5 types, well-covered) and Klebsiella capsule type (94% missing), so
the host-side signal may be sparse.

**What to do if triggered:** Run `hmmscan` against Pfam-A on the pharokka `.faa` tail spike sequences (already available
in the per-phage output). Extract glycosyl hydrolase family annotations and map to substrate specificity using CAZy
database cross-references. If enough phages carry classifiable depolymerase domains and enough hosts have capsule type
data, add depolymerase-capsule compatibility as a pairwise feature alongside RBP-receptor features.

#### Future: External interaction matrices as additional training data

**Trigger:** The model architecture is validated on the internal 97-phage panel and generalizes to external phage/host
genomes via the annotation-based feature pipeline (TL05-TL08 complete).

**Context (2026-03-29):** Because Track L features (RBP PHROGs, anti-defense genes, host receptors, defense systems) are
defined by universal sequence databases (PHROGs, Pfam, OMP BLAST clusters, DefenseFinder), any external phage-host
interaction dataset can be encoded into the same feature space. External phage genomes get pharokka-annotated and their
RBPs assigned to the same PHROG families; external host genomes get receptor-typed and defense-system-annotated with the
same pipelines. The external interaction matrix then provides additional (PHROG, receptor) → lysis observations that
strengthen the learned associations.

**What to do if triggered:** Identify external interaction datasets (e.g., from NCBI, PhageScope, published
supplementary tables). Run the Track L annotation + Track C host typing pipelines on the external genomes. Pool the
encoded interactions with internal training data and retrain. Measure lift from the expanded training set.

### 2026-03-30: Track L replan — enrichment holdout leak and external validation failure

#### Executive summary

Post-completion review of Track L identified two independent problems: (1) TL02's enrichment analysis includes ST03
holdout strains in the interaction matrix, leaking test data into TL03/TL04 feature weights; (2) the TL08 genome-only
inference bundle fails external validation on Virus-Host DB (known positives score below random pairs). These are
separate issues — the leak affects in-panel evaluation trustworthiness, the external failure reflects an architectural
gap in the deployable feature set.

#### What changed in the plan

- **TL10 added**: Fix the enrichment holdout leak in `run_enrichment_analysis.py` by filtering out ST03 holdout bacteria
  before calling `compute_enrichment()`. Mechanical fix — the permutation test is sound, only the input selection is
  wrong.
- **TL03/TL04/TL05**: Remain marked done for now. After TL10 lands, they will need re-evaluation (separate tickets) to
  determine whether enrichment features provide any honest lift with holdout-clean weights.

#### Diagnosis summary

Three compounding issues in Track L:

1. **TL02 enrichment circularity (confirmed, 98% confidence)**: `run_enrichment_analysis.py` loads
   `label_set_v1_pairs.csv` (all 369 bacteria) with no holdout filtering. `compute_enrichment()` has no holdout
   parameter. The TL02 acceptance criteria explicitly said "uses the full interaction matrix." Fixed by TL10.

2. **TL08 genome-only bundle is feature-impoverished (confirmed, 80% confidence)**: TL08 trains on only defense subtypes
   (79 features) + phage k-mer SVD (26 features). It drops the entire v0 metadata block (serotype, phylogroup,
   morphotype, ~20 categorical features), OMP receptor variants, UMAP embeddings, and all phage taxonomy. Many of these
   are genome-derivable but were not wired into TL08. The EDL933 round-trip failure (median P(lysis) delta 0.14, 9/96
   ranks matching) is primarily explained by this feature-set mismatch, not by defense annotation lossiness.

3. **No pairwise compatibility signal in deployable model (moderate confidence, 70%)**: Neither defense subtypes nor
   k-mer SVD encode how a specific phage interacts with a specific host surface. The model memorizes panel-specific
   defense-profile × k-mer-profile combinations that don't transfer to novel genomes. Adding genome-derivable
   compatibility features (OMP receptors, enrichment-weighted PHROG × receptor pairs) to the inference bundle is the
   most promising direction, but requires TL10 first to make the enrichment weights honest.

#### What is still sound

- Pharokka annotation pipeline (TL01): correct and biologically reasonable.
- Enrichment module statistics (TL02 implementation): permutation test, BH correction, phage conditioning all sound.
- Generalized inference architecture (TL06–TL08): plumbing is correct, transform persistence works.
- Novel organism projection helpers (TL06, TL07): tested and correct.
- Leakage diagnosis from Track G (TG04–TG12): rigorous and honest.

### 2026-03-30: Track L replan follow-up — tighten downstream acceptance criteria

#### Executive summary

Reviewed the TL03-TL09 PRs, the failed TL03 Codex implement run, and the Track L notebook entries to identify which
mistakes were implementation accidents versus plan-specification failures. The main issue was under-specified acceptance
criteria: several tasks allowed a technically plausible but strategically wrong completion to count as done. The plan
now adds TL11-TL14 to make the next pass fail fast on the specific mistakes we can already anticipate.

#### What changed in the plan

- **TL11 added**: rebuild TL03/TL04 from TL10's holdout-clean enrichment outputs and emit manifests proving which split
  and which excluded bacteria IDs were used.
- **TL12 added**: rerun mechanistic lift with bootstrap confidence intervals and a predeclared lock rule, so tiny noisy
  deltas can no longer justify a new v1 configuration.
- **TL13 added**: rebuild the deployable bundle under an explicit feature-parity audit and self-contained-artifact
  contract; silently dropping training-time feature blocks no longer counts as success.
- **TL14 added**: rerun external validation under a strict cohort contract with a required multi-host round-trip check,
  so "validation succeeded as a script run" is separated from "bundle actually generalized."

#### Mistakes these new tasks are meant to prevent

1. **Leaked-but-plausible mechanistic rebuilds**: TL03/TL04 were scientifically framed correctly, but nothing in their
   original follow-on criteria would have forced the re-evaluation to prove it was using TL10-clean inputs rather than
   stale leaked enrichment CSVs.

2. **Locking on noise**: TL05 proposed a new mechanistic lock from holdout deltas that were already within noise on a
   65-strain holdout. The next task must report uncertainty and apply a stated decision rule before it is allowed to
   promote any arm.

3. **Plumbing without feature-parity honesty**: TL08 proved that the bundle machinery works, but it was allowed to ship
   a genome-only model that omitted many training-time signals without first surfacing that gap as a formal feature
   parity audit.

4. **Validation with too-weak round-trip guarantees**: TL09 named several panel-host examples, but only `EDL933` was
   actually comparable through the saved reference artifact. Future validation tasks must pre-materialize the cohort and
   treat round-trip host count as a gate, not a hope.

5. **Review-thread fixes discovered too late**: several issues were caught only in PR review rather than by the task
   contract itself, including weak negative fixtures (TL04), late cache short-circuiting (TL07), hardcoded panel paths
   in the bundle (TL08), and under-specified parsing/selection semantics (TL09). TL11-TL14 now encode those lessons
   directly as acceptance checks.

#### Additional note from Codex workflow logs

The first TL03 Codex implement run failed before any repo code ran because `conda env create -f environment.yml` was
unsatisfiable on CI (`openjdk` / `fontconfig` solver conflict). That is not a TL03 logic bug, but it is a reminder that
bioinformatics-heavy tasks should explicitly require CI-compatible environment resolution rather than assuming local and
CI solvability stay aligned.
