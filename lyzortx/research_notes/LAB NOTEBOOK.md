### 2026-02-15: Defining a Meaningful Model and Product-Oriented Benchmarks

Based on a review of the project plan and the available data, we've established a strategic framework for what constitutes a "meaningful model" and how to evaluate its utility for a clinical product that recommends phage cocktails for new *E. coli* strains.

#### 1. Data Sufficiency Assessment

We have a sufficient foundation of raw data (`raw_interactions.csv`, genomic assemblies) to build a predictive model. However, success is contingent on executing the data integrity and feature engineering tracks outlined in `PLAN.md` (Tracks A, C, D, E). The raw data is not yet model-ready, and a significant effort is required to create clean labels and potent predictive features.

#### 2. Defining a Meaningful Model

A "meaningful model" in this context is not a simple classifier. It is a system that produces a **calibrated probability of lysis** for any given phage-bacterium pair.

-   **Input:** The genomic assembly of a new *E. coli* strain.
-   **Output:** A ranked list of all phages in our panel, sorted by their predicted probability of successfully lysing the new strain, e.g., `P(Lysis | new_strain, phage_i)`.

This probabilistic output is essential for the downstream cocktail recommendation algorithm (Track H), which needs to weigh evidence and confidence, not just binary predictions.

#### 3. Proposed Product-Oriented Evaluation Benchmarks

To be considered "very useful" for a clinical product, the model must meet a set of rigorous benchmarks that go beyond standard academic metrics.

-   **Level 1: Foundational Model Performance (Offline)**
    -   **Top-3 Lytic Hit Rate > 95%:** The 3-phage cocktail recommended by the model must contain at least one truly lytic phage for over 95% of new strains.
    -   **Precision @ P(Lysis) > 0.9 must be > 99%:** When the model is highly confident, it must be extremely reliable to build clinical trust.

-   **Level 2: Cocktail Recommendation Utility (Simulation)**
    -   **Simulated 3-Phage Cocktail Coverage > 98%:** An end-to-end simulation must show that the final recommended cocktails are effective against over 98% of strains in the held-out test set.

-   **Level 3: Safety and Robustness (Negative Controls)**
    -   **Sentinel Strain Recovery = 100%:** The model must correctly identify known effective phages for a predefined set of challenging "sentinel" strains, ensuring it doesn't fail on critical edge cases.

#### 4. Rationale for 3-Phage Cocktails

The choice of a 3-phage cocktail is a deliberate trade-off between efficacy and complexity.

-   **Why Not Fewer (1-2 Phages)?**
    -   **Higher Risk of Resistance:** Bacteria can rapidly evolve resistance to a single phage. Using multiple phages targeting different receptors or using different lytic mechanisms makes this much harder.
    -   **Lower Coverage:** A single phage is less likely to be effective against a new, unknown strain.

-   **Why Not More (4+ Phages)?**
    -   **Manufacturing & Regulatory Complexity:** Each phage in a cocktail must be individually produced, characterized, and proven safe and stable. The regulatory burden and cost increase exponentially with each added phage.
    -   **Risk of Antagonistic Interference:** Phages can compete for host cell resources or even block each other's entry, reducing the overall efficacy of the cocktail.
    -   **Diminishing Returns:** The marginal benefit of adding a fourth or fifth phage is often small and does not justify the significant increase in complexity and cost.

A 3-phage cocktail is therefore considered the current industry "sweet spot," providing a robust defense against bacterial resistance while remaining tractable from a manufacturing and regulatory perspective.
---
### 2026-02-15: Research on External Phage-Host Interaction Datasets

#### Summary
To improve model performance, we must expand our training data beyond the internal `raw_interactions.csv`. Research was conducted to identify public databases that contain phage-host interaction data. The findings indicate that while a direct equivalent to our raw interaction matrix is rare, a significant volume of valuable data can be compiled from various sources.

#### Key Data Sources Identified

1.  **Dedicated Phage-Host Databases:** These are the highest-value targets for finding experimentally confirmed interaction pairs.
    *   **Relevant Databases:** `Virus-Host DB`, `ViralHostRangeDB`.
    *   **Content:** Aggregate interaction data from literature, providing "Phage X infects Bacterium Y" pairs. They typically lack the fine-grained dilution/replicate data but are excellent for expanding our core training set.

2.  **Public Sequence Archives (NCBI, CNCB):** These contain the raw data from sequencing experiments and associated metadata.
    *   **NCBI BioSample Database:** A good source for "known positive" interactions. The `isolation_host` field in a phage's BioSample record provides a confirmed host. This is a low-effort way to augment our dataset with thousands of positive examples.
    *   **NCBI Sequence Read Archive (SRA) / China National Center for Bioinformation (CNCB) Genome Warehouse (GWH):** A potential goldmine, but high-effort. These archives contain raw data from high-throughput screening projects. Extracting an interaction matrix would require a significant bioinformatics effort (re-processing raw reads, mapping metadata), making it a long-term strategic goal.

3.  **International Databases (Focus on China):**
    *   **China National Center for Bioinformation (CNCB):** This is a key resource, hosting databases like the **Virus & Host (V&H) database**, which aggregates data from Chinese research and is a priority for exploration.

#### Next Steps & Strategy
The most pragmatic approach is to start with the lowest-effort, highest-value tasks.

1.  **Short-Term:** Systematically query the APIs of **Virus-Host DB** and the **NCBI BioSample database**. The goal is to script a process to download all available phage-host pairs and integrate them as "known positive" in our dataset.
2.  **Mid-Term:** Investigate the curated databases at the CNCB.
3.  **Long-Term:** Scope the effort required to re-process a full screening project from the SRA or GWH.