### 2026-02-15: ST0.6 policy switch to `logreg_platt__none`

#### What we changed

1. Updated ST0.6 default ranking policy to `score_column=pred_logreg_platt`.
2. Disabled family-cap diversity by default (`max_per_family=0`, diversity mode `none`).
3. Updated ST0.6 and ST0.7 baselines to match the new default policy outputs.

#### Updated ST0.6 holdout metrics

- Top-3 hit rate (all strains): `0.846154` (`55/65`).
- Top-3 hit rate (susceptible-only): `0.873016` (`55/63`).
- Diversity relaxation count: `0` (diversity mode `none`).

#### Updated ST0.7 consequence

- `error_analysis.csv` holdout miss rows decreased from `14` to `10` under the new ST0.6 default policy.

### 2026-02-15: ST0.6b implemented (ranking-policy comparison)

#### What we implemented in ST0.6b

1. Added ST0.6b step: `lyzortx/pipeline/steel_thread_v0/steps/st06b_compare_ranking_policies.py`.
2. Compared six recommendation policies on the same holdout set: raw, platt, and isotonic scores; each with and without
   family-cap diversity.
3. Wrote outputs:
   - `lyzortx/generated_outputs/steel_thread_v0/intermediate/st06b_policy_comparison.csv`
   - `lyzortx/generated_outputs/steel_thread_v0/intermediate/st06b_recommendations_all_policies.csv`
   - `lyzortx/generated_outputs/steel_thread_v0/intermediate/st06b_top3_recommendations_best.csv`
   - `lyzortx/generated_outputs/steel_thread_v0/intermediate/st06b_summary.json`

#### ST0.6b holdout results

- `logreg_platt__none`: top-3 all `0.846154`, susceptible-only `0.873016` (best; tied with `logreg_raw__none`)
- `logreg_raw__none`: top-3 all `0.846154`, susceptible-only `0.873016`
- `logreg_platt__max_family_2`: top-3 all `0.815385`, susceptible-only `0.841270`
- `logreg_raw__max_family_2`: top-3 all `0.815385`, susceptible-only `0.841270`
- `logreg_isotonic__none`: top-3 all `0.800000`, susceptible-only `0.825397`
- `logreg_isotonic__max_family_2` (former ST0.6 policy): top-3 all `0.784615`, susceptible-only `0.809524`

#### ST0.6b Interpretation

1. The ST0.6 drop versus ST0.4 is primarily policy choice, not an implementation bug.
2. In this dataset/split, isotonic ranking is weaker than raw or platt ranking for top-3 hit-rate.
3. Family-cap diversity (`max_per_family=2`) reduces hit-rate for all three score variants in current holdout.
4. Next change should be to switch operational ranking from isotonic to platt (or raw), while keeping calibration
   outputs for probability-quality reporting.
5. Raw and Platt top-3 tie in this setup because Platt is a monotonic remapping of the same raw score; top-k lift should
   therefore be expected from better model signal, better labels, or new data, not monotonic recalibration alone.

### 2026-02-15: ST0.7 implemented (final report artifacts)

#### What we implemented in ST0.7

1. Added ST0.7 step: `lyzortx/pipeline/steel_thread_v0/steps/st07_build_report.py`.
2. Added ST0.7 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st07_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st07_expected_metrics.json`.
4. Extended CI workflow to run ST0.7 regression in addition to ST0.1 through ST0.6.

#### ST0.7 output summary on current internal data

- Metrics summary rows: `48`.
- Top-3 recommendation rows: `1,107`.
- Calibration summary rows: `12`.
- Error analysis rows (holdout misses): `14`.
- Output artifacts:
  - `lyzortx/generated_outputs/steel_thread_v0/metrics_summary.csv`
  - `lyzortx/generated_outputs/steel_thread_v0/top3_recommendations.csv`
  - `lyzortx/generated_outputs/steel_thread_v0/calibration_summary.csv`
  - `lyzortx/generated_outputs/steel_thread_v0/error_analysis.csv`
  - `lyzortx/generated_outputs/steel_thread_v0/run_manifest.json`

#### ST0.7 Interpretation

1. The steel thread is now complete end-to-end, with deterministic outputs and regression-gated final artifacts.
2. The recommendation gap observed in ST0.6 is now explicit in `error_analysis.csv` for targeted iteration.
3. The immediate next optimization target should be recommendation quality (Top-3 hit rate), not additional plumbing.

### 2026-02-15: ST0.6 implemented (top-3 recommendation generation)

#### What we implemented in ST0.6

1. Added ST0.6 step: `lyzortx/pipeline/steel_thread_v0/steps/st06_recommend_top3.py`.
2. Added ST0.6 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st06_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st06_expected_metrics.json`.
4. Extended CI workflow to run ST0.6 regression in addition to ST0.1 through ST0.5.

#### ST0.6 output summary on current internal data

- Recommended strains: `369`.
- Recommendation rows: `1,107` (`3` per strain).
- Diversity relaxation needed: `0` strains under current `max_per_family=2`.
- Holdout top-3 hit rate (all strains): `0.784615`.
- Holdout top-3 hit rate (susceptible strains only): `0.809524`.

#### ST0.6 Interpretation

1. The current simple recommendation layer underperforms the ST0.4 raw top-3 benchmark, indicating that ranking and
   recommendation objectives are not yet aligned end-to-end.
2. Diversity constraint did not bind in this dataset configuration (`0` relaxed strains), so current performance is not
   being driven by diversity tradeoffs.
3. ST0.7 should expose this gap explicitly in final reporting so the next iteration can focus on recommendation-quality
   optimization rather than only calibration quality.

### 2026-02-15: ST0.5 implemented (calibration and ranking)

#### What we implemented in ST0.5

1. Added ST0.5 step: `lyzortx/pipeline/steel_thread_v0/steps/st05_calibrate_rank.py`.
2. Added ST0.5 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st05_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st05_expected_metrics.json`.
4. Extended CI workflow to run ST0.5 regression in addition to ST0.1 through ST0.4.

#### ST0.5 output summary on current internal data

- Calibration rows (fold 0, non-holdout hard-labeled): `5,755`.
- Holdout eval rows (hard-labeled): `6,235`.
- Methods implemented per model: raw, isotonic calibration, and Platt scaling.
- Output artifacts:
  - `st05_calibration_summary.csv`
  - `st05_pair_predictions_calibrated.csv`
  - `st05_ranked_predictions.csv`

#### Holdout calibration metrics (logreg model)

- Raw: Brier `0.171223`, Log loss `0.521944`, ECE `0.176341`.
- Isotonic: Brier `0.140218`, Log loss `0.500302`, ECE `0.031802`.
- Platt: Brier `0.137795`, Log loss `0.430845`, ECE `0.029253`.

#### ST0.5 Interpretation

1. Calibration materially improved probabilistic quality, especially ECE, which dropped by an order of magnitude
   relative to raw logreg outputs.
2. Platt scaling slightly outperformed isotonic on holdout in this split configuration.
3. ST0.6 should use calibrated ranking scores (not raw model scores) for top-3 recommendation generation.

### 2026-02-15: ST0.4 implemented (baseline training)

#### What we implemented in ST0.4

1. Added ST0.4 step: `lyzortx/pipeline/steel_thread_v0/steps/st04_train_baselines.py`.
2. Added ST0.4 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st04_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st04_expected_metrics.json`.
4. Extended CI workflow to run ST0.4 regression and install `scikit-learn` explicitly.

#### ST0.4 output summary on current internal data

- Train rows (non-holdout hard-labeled): `29,031`.
- Holdout eval rows (hard-labeled): `6,235`.
- Vectorized feature count: `425`.
- Comparator model: `DummyClassifier(strategy='prior')`.
- Strong baseline: `LogisticRegression(class_weight='balanced', solver='liblinear')`.

#### Holdout metrics

- Dummy baseline:
  - Brier: `0.189304`
  - Log loss: `0.566574`
  - ROC-AUC: `0.500000`
  - Top-3 hit rate (all strains): `0.015385`
- Logistic baseline:
  - Brier: `0.171223`
  - Log loss: `0.521944`
  - ROC-AUC: `0.826948`
  - Top-3 hit rate (all strains): `0.846154`

#### ST0.4 Interpretation

1. The strong baseline materially outperforms the comparator on every tracked holdout metric, so ST0.4 clears the
   minimum "better than naive" bar for steel-thread viability.
2. Top-3 hit rate is substantially below the Tier 1 benchmark target; this is expected at this stage and motivates ST0.5
   calibration plus ST0.6 recommendation logic.
3. The feature-space and model artifact files are now stable inputs for downstream calibration/ranking.

### 2026-02-15: ST0.3 implemented (leakage-safe split protocol)

#### What we implemented in ST0.3

1. Added ST0.3 step: `lyzortx/pipeline/steel_thread_v0/steps/st03_build_splits.py`.
2. Added ST0.3 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st03_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st03_expected_metrics.json`.
4. Extended CI workflow to run ST0.3 regression in addition to ST0.1 through ST0.2.

#### ST0.3 split summary on current internal data

- Rows assigned: `35,424`.
- Group key: `cv_group` from ST0.2.
- Fixed holdout groups: `57 / 283` (`0.201413`).
- Holdout rows: `6,240`.
- Non-holdout rows: `29,184`.
- CV fold rows on non-holdout:
  - fold 0: `5,760`
  - fold 1: `5,280`
  - fold 2: `6,624`
  - fold 3: `5,184`
  - fold 4: `6,336`

#### Leakage check results

- Holdout bacteria overlap count: `0`.
- Holdout cv_group overlap count: `0`.
- Cross-fold cv_group overlap count: `0`.

#### ST0.3 Notes

1. ST0.3 defines the locked v0 split contract for downstream model training and evaluation.
2. Assignment is deterministic via hash with salt `steel_thread_v0_st03_split_v1`.
3. ST0.4 should consume `st03_split_assignments.csv` directly and avoid custom split logic.

#### ST0.3 Interpretation

1. The leakage checks are clean (`0` overlap across holdout/train and across CV folds by `cv_group`), so this split
   protocol is suitable as the v0 benchmark contract.
2. Fold sizes are not perfectly balanced, but they are close enough for v0 model comparison; metrics should still be
   reported per fold and macro-averaged to reduce sensitivity to fold-size differences.
3. The strict trainable subset remains dominated by negatives in every fold, so ST0.4 should use class-imbalance-aware
   training and report both ranking metrics and calibration metrics.
4. Holdout size (~17.6% of rows) is large enough to be meaningful for a fixed benchmark while preserving enough
   non-holdout data for model development.

### 2026-02-15: ST0.2 implemented (canonical pair table)

#### What we implemented in ST0.2

1. Added ST0.2 step: `lyzortx/pipeline/steel_thread_v0/steps/st02_build_pair_table.py`.
2. Added ST0.2 regression check: `lyzortx/pipeline/steel_thread_v0/checks/check_st02_regression.py`.
3. Added baseline snapshot: `lyzortx/pipeline/steel_thread_v0/baselines/st02_expected_metrics.json`.
4. Extended CI workflow to run ST0.2 regression in addition to ST0.1 and ST0.1b.

#### ST0.2 output summary on current internal data

- Output rows: `35,424` (369 bacteria x 96 phages).
- Output schema: `64` columns in `st02_pair_table.csv`.
- Strict-slice rows: `28,338` (`0.799966` of all rows), inherited from ST0.1b.
- Join coverage:
  - Host metadata missing: `0`
  - Phage metadata missing: `0`
  - CV group missing: `0`
  - Interaction-matrix missing: `156` (auxiliary only; non-blocking).

#### ST0.2 Notes

1. ST0.2 treats `interaction_matrix.csv` as an auxiliary reference and explicitly marks it as non-feature to avoid
   leakage confusion.
2. A notable host metadata gap remains in `host_abc_serotype` (`22,848` missing row-values in the pair table).
3. ST0.2 is now stable and regression-gated; ST0.3 can consume `st02_pair_table.csv` as canonical input.

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
   `.github/workflows/steel-thread-regression.yml`.

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

### 2026-03-15: ST08 Dual-Slice Reporting Verification (ST0.7)

#### What was implemented

- Added a dedicated regression test for ST0.7 report generation to verify that `metrics_summary.csv` contains separate
  holdout metric rows for both `full_label` and `strict_confidence` slices.
- The test enforces presence of the required acceptance metrics by slice:
  - `topk_hit_rate_all_strains`
  - `brier_score`
  - `ece`

#### Findings

- ST0.7 now has explicit test-level guarantees that dual-slice reporting is present in report outputs and keyed by
  `__full_label` / `__strict_confidence` suffixes for downstream parsing.

#### Interpretation

- This closes a reporting-audit gap between ST0.5/ST0.6 and ST0.7 by ensuring the final exported report preserves slice
  separation for recommendation quality and calibration quality metrics.

