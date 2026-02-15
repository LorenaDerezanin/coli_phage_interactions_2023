### 2026-02-15: ST0.1b implemented (strict confidence tiers)

#### What we implemented in ST0.1b

1. Added ST0.1b step: `lyzortx/pipeline/steel_thread_v0/steps/st01b_confidence_tiers.py`.
2. Added ST0.1b regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st01b_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st01b_expected_metrics.json`.
4. Extended CI workflow to run both ST0.1 and ST0.1b regression gates.

#### ST0.1b policy (v1)

- `high_conf_pos`: hard label is positive, `score_1_count >= 2`, positive fraction among interpretable observations
  `>= 0.4`, and `score_n_count <= 1`.
- `high_conf_neg`: hard label is negative, `score_0_count >= 7`, and `score_n_count <= 1`.
- `ambiguous`: all remaining pairs.

#### ST0.1b output summary on current internal data

- Total pairs: `35,424`.
- `high_conf_pos`: `4,135`.
- `high_conf_neg`: `24,203`.
- `ambiguous`: `7,086`.
- Strict-slice coverage: `0.799966`.
- Strict positive fraction within strict slice: `0.145917`.

#### Interpretation

1. ST0.1b provides a narrower, more conservative training/evaluation slice while preserving broad coverage (~80%).
2. The strict slice remains class-imbalanced, so downstream modeling will need imbalance-aware training.
3. Positive-side conflict burden remains non-trivial in this first strict policy and should be stress-tested in ST0.2.

### 2026-02-15: ST0.1 diagnostics, CI regression gate, and ST0.1b decision

#### What we implemented

1. Implemented ST0.1 label-policy step: `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`.
2. Added ST0.1 regression checker and baseline: `lyzortx/pipeline/steel_thread_v0/checks/check_st01_regression.py`,
   `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`.
3. Added CI workflow to run the regression gate on push and pull request:
   `.github/workflows/steel-thread-st01-regression.yml`.

#### ST0.1 findings from current internal data

- Raw rows: `318,816`.
- Observed bacteria-phage pairs: `35,424` (full 369 x 96 grid).
- Hard labels:
  - Positive: `9,720` (`27.44%`).
  - Negative: `25,546` (`72.12%`).
  - Unresolved: `158` (`0.45%`).
- Hard-label coverage: `99.554%`.
- Uncertainty flags:
  - Conflicting interpretable observations: `8,917` pairs (`25.17%`).
  - Has uninterpretable (`score='n'`): `4,537` pairs (`12.81%`).
  - High uninterpretable fraction: `1,444` pairs (`4.08%`).

#### Conflict interpretation summary

- Not all conflicts imply pure replicate noise.
- `2,359` conflicting pairs (`6.66%` of all pairs) are clean cross-dilution shifts.
- `6,558` conflicting pairs (`18.51%` of all pairs) include within-dilution replicate disagreement.
- Current positive rule (`any 1`) captures weak positives; `23.21%` of positives are single-hit (`1/9`).

#### Decision and plan impact

We will add **ST0.1b** as a parallel stricter label view with confidence tiers: `high_conf_pos`, `high_conf_neg`, and
`ambiguous`.

Why:

1. Preserve high recall behavior from ST0.1 for candidate generation.
2. Add a higher-trust slice for model debugging and honest early benchmarking.
3. Report metrics on both full-label and high-confidence slices to avoid noise-driven false gains.

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
