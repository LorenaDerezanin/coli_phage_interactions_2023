### 2026-03-17: TB04 rescuer phages for narrow-susceptibility strains

#### What we implemented in TB04

1. Added one reproducible TB04 analysis script:
   `lyzortx/research_notes/ad_hoc_analysis_code/rescuer_phages_for_narrow_susceptibility.py`.
2. Reused the TB03 low-susceptibility definition so the rescuer slice stays consistent with prior Track B work:
   resolved strains with `<=3` lytic phages and no missing assays.
3. Configured the script to write generated outputs under
   `lyzortx/generated_outputs/rescuer_phages_for_narrow_susceptibility/`:
   - `narrow_strain_rescuer_summary.csv`
   - `rescuer_phage_summary.csv`
   - `rescuer_phage_group_summary.csv`
   - `tb04_summary.json`
4. Used two operational rescuer modes:
   - `exclusive`: the phage is the only lytic hit for that narrow strain
   - `shared`: the phage is one of `2-3` lytic hits for that narrow strain

#### TB04 output summary

- Resolved narrow-susceptibility strains analyzed: `36`.
- Rescue-mode split: `9` exclusive-rescue strains, `15` shared-rescue strains, and `12` non-rescued narrow strains.
- Rescuer phages: `19 / 96` panel phages (`19.8%`) have at least one lytic hit in the resolved narrow-susceptibility
  slice.
- Top rescuer phages by resolved narrow-strain coverage:
  - `AL505_Ev3`: `5` narrow strains rescued (`13.9%` of the narrow slice), `1` exclusive and `4` shared
  - `NIC06_P2`: `4` narrow strains rescued, with the highest exclusive count (`3`)
  - `536_P9`: `4` narrow strains rescued, all shared
  - `DIJ07_P2`: `4` narrow strains rescued, all shared
  - `LF82_P8`: `4` narrow strains rescued, all shared
- The top five rescuer phages together cover `16 / 36` resolved narrow-susceptibility strains (`44.4%`).
- Exclusive rescue remains strongly Myoviridae-skewed:
  - `8 / 9` exclusive-rescue strains are rescued by Myoviridae phages
  - the only non-Myoviridae exclusive rescuer is podophage `AN24_P4`
- Morphotype/family concentration among rescuer phages:
  - Myoviridae: `17` rescuer phages, `41` narrow-strain rescue events, `8` exclusive rescues
  - Podoviridae: `2` rescuer phages, `4` narrow-strain rescue events, `1` exclusive rescue
  - Siphoviridae: `0` rescuer phages
  - Straboviridae: `11 / 11` panel phages are rescuer phages, contributing `25` narrow-strain rescue events and `7`
    exclusive rescues
- Highest narrow-hit concentration among materially active rescuer phages:
  - `AN24_P4`: `3 / 43` total lysed strains are narrow (`6.98%`)
  - `AL505_Ev3`: `5 / 160` (`3.13%`)
  - `NIC06_P2`: `4 / 170` (`2.35%`)

#### TB04 interpretation

1. Narrow-strain rescue is concentrated in a minority of the panel (`19 / 96` phages), and `12 / 36` resolved narrow
   strains are not rescued at all, so these hard cases are not being solved uniformly by broad host-range phages.
2. Myoviridae, especially `Straboviridae`, still dominate the rescue landscape and provide nearly all exclusive saves,
   which is consistent with the earlier single-lyser signal from the paper gist and TB02.
3. The podophage `AN24_P4` matters despite modest absolute coverage because it has the highest narrow-hit concentration
   and supplies the only non-Myoviridae exclusive rescue in the resolved slice. That makes it a targeted exception,
   not noise.
4. The top rescuers combine broad-rescue specialists (`AL505_Ev3`, `NIC06_P2`) with shared-support phages
   (`536_P9`, `DIJ07_P2`, `LF82_P8`), suggesting that narrow-susceptibility coverage is partly driven by a small
   backbone set plus a few strain-specific add-ons rather than one universally dominant rescuer.
5. Immediate modeling implication: phage feature work should keep morphotype/family signals, but it also needs enough
   phage-specific capacity to preserve exceptions like `AN24_P4` and the `NIC06_P2` exclusive-rescue pattern instead of
   collapsing everything into a broad Myoviridae prior.

### 2026-03-17: TB03 hard-to-lyse strains by host traits

#### What we implemented in TB03

1. Added one reproducible TB03 analysis script:
   `lyzortx/research_notes/ad_hoc_analysis_code/hard_to_lyse_host_traits.py`.
2. Configured the script to write generated outputs under
   `lyzortx/generated_outputs/hard_to_lyse_host_traits/`:
   - `hard_to_lyse_strain_summary.csv`
   - `host_trait_low_susceptibility_summary.csv`
   - `tb03_summary.json`
3. Used `<=3` lytic phages as the low-susceptibility threshold to stay aligned with the prior narrow-susceptibility
   slice from TB02.
4. Used derived `O-type:H-type` serotype labels for the main serotype analysis because `ABC_serotype` is missing for
   `251 / 402` strains (`62.4%`) and would mostly measure metadata completeness instead of biology.

#### TB03 output summary

- Interaction-matrix strains analyzed: `402`.
- Zero-lysis strains: `12 / 402` (`2.99%`).
- Low-susceptibility strains (`<=3` lytic phages): `36 / 401` resolved strains (`8.98%`), with `S1-84` remaining
  ambiguous because missing assays do not rule in or rule out the threshold.
- Zero-lysis strains:
  `B156`, `B253`, `DEC2a`, `E_albertiiCIP107988T`, `FN-B4`, `FN-B7`, `H1-002-0060-C-T`, `H1-007-0015-D-G`,
  `NILS22`, `NILS24`, `ROAR205`, `ROAR220`.
- Field-level stratification of lytic-phage counts was significant for all three requested host metadata fields:
  - Serotype (`O:H`): Kruskal-Wallis `p = 1.44e-03`
  - Phylogroup: Kruskal-Wallis `p = 1.24e-09`
  - ST: Kruskal-Wallis `p = 1.71e-05`
- No individual trait value met the current multiple-testing cutoff (`q <= 0.1`).
- Strongest nominal enrichments among testable groups (`n >= 4`) were:
  - Phylogroup `Clade V`: `3 / 7` low-susceptibility (`42.9%`), odds ratio `8.20`, `q = 0.218`
  - Phylogroup `E. fergusonii`: `2 / 5` low-susceptibility (`40.0%`), odds ratio `7.10`, `q = 0.264`
  - ST `58`: `5 / 17` low-susceptibility (`29.4%`), odds ratio `4.74`, `q = 0.296`
- Broad-susceptibility counterexample:
  - Phylogroup `B2`: only `6 / 126` low-susceptibility (`4.8%`), median `27` lytic phages, odds ratio `0.41`

#### TB03 interpretation

1. Hard-to-lyse behavior is not random noise in the matrix; it tracks host background strongly at the field level for
   serotype, phylogroup, and ST.
2. The clearest nominal concentration is in `Clade V` (`42.9%` low susceptibility versus an `8.98%` resolved-panel
   baseline), but none of the tested trait values yet survive multiple-testing correction.
3. `ST58` is still the strongest recurring ST signal in the currently testable set, but the ST landscape is fragmented
   across many small sequence types, so the category-level evidence remains weak after correction.
4. Serotype still matters at the field level, but the effect is diffuse across many rare `O:H` categories rather than a
   single dominant high-risk serotype. That means serotype is better treated as part of a multifeature host context than
   as a standalone rule.
5. Immediate next-step implication for modeling: host background features should keep explicit phylogroup and ST
   encodings, and mechanistic follow-up should prioritize the `Clade V` and `ST58` subsets first, with
   `E. fergusonii` as a secondary nominal candidate that still needs more support.

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

### 2026-03-16: ST09 holdout miss failure hypotheses

#### What we analyzed

1. Re-ran ST0.1 through ST0.7 and confirmed `lyzortx/generated_outputs/steel_thread_v0/error_analysis.csv` contains `10`
   current holdout misses under the active `logreg_platt__none` recommendation policy.
2. Ran ST0.6b and confirmed none of the `10` missed strains become hits under the six compared policy variants
   (`raw/platt/isotonic`, with and without family cap), so these are not simple monotonic-calibration or family-cap
   policy issues.
3. Reviewed miss context from `error_analysis.csv`, `st05_pair_predictions_calibrated.csv`,
   `st06_top3_recommendations.csv`, `st06b_recommendations_all_policies.csv`, and `st02_pair_table.csv`.

#### Error buckets

1. **No-susceptibility / abstention failure:** `FN-B4`, `NILS24`. These strains have zero labeled positive holdout
   phages, so the current always-return-top-3 policy is guaranteed to emit false positives.
2. **Single-weak-positive strains buried under broad-host-range priors:** `ECOR-06`, `H1-002-0060-C-T`. Each strain has
   exactly one positive holdout phage, and that positive sits below a dense block of higher-scored `Straboviridae`
   negatives.
3. **Within-family ordering misses among crowded `Straboviridae` candidates:** `ECOR-14`, `NILS41`, `NILS70`, `ROAR205`.
   At least one true positive is already near the top, but top-3 truncation inside a dense same-family score band still
   excludes it.
4. **Cross-family blind spot / `Straboviridae` collapse:** `ECOR-69`, `NILS53`. The recommender top-3 remains entirely
   `Straboviridae`, while most true positives for these strains live in `Other` or `Autographiviridae`.

#### Quantitative notes

- Miss strains are much narrower than hit strains: mean holdout positive count `5.8` vs `27.5` for holdout hits.
- `4/10` miss strains have `host_n_infections = 0`; none of the `55` holdout-hit strains do.
- Current ST0.6 recommendations are `Straboviridae` for every holdout strain, not just the missed ones.
- Among true-positive pairs from the `10` missed strains, family counts are `Other = 40`, `Autographiviridae = 9`,
  `Straboviridae = 9`.

#### Strain-level hypotheses and next steps

| Strain            | Bucket                         | Failure hypothesis                                                                                                                                                                                                                                            | Actionable next step                                                                                                                                                                            |
| ----------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ECOR-06`         | Single-weak-positive           | Only one holdout positive (`AL505_Ev3`) exists, and it is buried under a broad `Straboviridae` block of slightly higher-scored negatives. This looks like narrow-susceptibility recall failure rather than a recommendation-policy bug.                       | Add a narrow-susceptibility audit slice (`n_true_positive_phages_holdout <= 3`) and test whether reweighting these strains or adding a recall-oriented reranker lifts single-positive recovery. |
| `ECOR-14`         | Within-family ordering         | Multiple true positives exist, but the best one lands just outside top-3 behind same-family negatives. The model appears to know the right family but not the right phage within that family.                                                                 | Add a second-stage within-family reranker using per-phage precision / empirical host-range features and check whether `LF73_P1` and related positives move into top-3.                          |
| `ECOR-69`         | Cross-family blind spot        | This strain has many positives, but the top ranks are dominated by `Straboviridae` negatives while most true positives are `Other`. That pattern suggests family-level prior collapse is overpowering host-specific compatibility.                            | Run a targeted ablation that weakens phage-family identity features, then measure whether non-`Straboviridae` positives recover on this strain and on the full miss bucket.                     |
| `FN-B4`           | No-susceptibility / abstention | Holdout has zero positives, so the model should have abstained instead of forcing three phages. This is a product-contract gap, not just a ranking gap.                                                                                                       | Prototype a no-recommend / low-confidence threshold using max score and score-margin features, then evaluate false-abstain vs false-positive tradeoffs on holdout.                              |
| `H1-002-0060-C-T` | Single-weak-positive           | The only positive (`DIJ07_P1`) is a low-support singleton far below a `Straboviridae` block. With `host_n_infections = 0` and `host_n_defense_systems = 10`, the current feature set likely misses a rare host-compatibility signal.                          | Prioritize receptor/defense feature work for zero-infection, high-defense strains and test whether `DIJ07_P1` retrieval improves after those host features are added.                           |
| `NILS24`          | No-susceptibility / abstention | Like `FN-B4`, this strain has zero holdout positives, so any fixed top-3 output is necessarily wrong. Its zero prior infections and high defense burden make it a plausible genuinely hard-negative strain.                                                   | Include `NILS24` in the abstention-threshold benchmark set and require any future recommendation layer to permit an explicit no-match outcome.                                                  |
| `NILS41`          | Within-family ordering         | This strain is broadly susceptible, but positives sit inside a large, nearly saturated score band where many `Straboviridae` phages receive similarly high scores. The failure is phage selection inside the right region, not gross family misspecification. | Audit score ties / near-ties around ranks 1-10 and test a tie-aware reranker that uses within-family host-range evidence instead of pure score ordering.                                        |
| `NILS53`          | Cross-family blind spot        | Despite many positives, especially outside `Straboviridae`, the model still pushes a `Straboviridae`-only top-3. Its very high defense burden (`15`) suggests missing host-surface / defense context may be hiding the compatible non-`Straboviridae` phages. | Prioritize host receptor + defense features, then re-evaluate whether `Other` / `Autographiviridae` positives move upward for this strain and the rest of the family-collapse bucket.           |
| `NILS70`          | Within-family ordering         | This is a narrow near-miss: at least one true positive is very close to the cutoff, but top-3 truncation still favors nearby negatives. This looks like ranking-resolution failure around the decision boundary.                                              | Measure score margins for ranks 3-6 on holdout and test whether a boundary-aware reranker or expanded candidate set before reranking improves top-3 hit rate.                                   |
| `ROAR205`         | Within-family ordering         | The top region already contains the correct family and a true positive sits just beyond the cutoff, but sparse prior infection history suggests weak host evidence for choosing the correct member.                                                           | Add a host-neighbor retrieval diagnostic for zero-infection strains and test whether nearest-neighbor host evidence can rerank `BCH953_P4` into the recommended set.                            |

#### ST09 interpretation

1. The remaining `10` misses are primarily model/data failures, not recommendation-policy failures; ST0.6b does not
   rescue any of them.
2. The most important structural issue is family collapse toward `Straboviridae`, which suppresses many true positives
   from `Other` and `Autographiviridae`.
3. The second issue is lack of abstention: two misses are guaranteed errors because the current interface always emits
   three phages even when no labeled susceptible phage exists.
4. The next technically highest-value iteration is not more calibration tuning; it is adding host-compatibility signal
   plus an abstention/no-match mechanism, then re-checking this exact 10-strain bucket.

### 2026-03-17: TI03 VHRdb source-fidelity fields

#### What was implemented

- Updated `ST0.8` VHRdb ingest so ingested rows now carry raw `global_response` and `datasource_response` columns
  directly, instead of only preserving those values under `source_*response` aliases.
- Kept `label_hard_any_lysis` derived from the normalized global response for downstream compatibility, while preserving
  the original mixed-case source values in the exported ingest table.
- Added a file-boundary regression test for `main()` to verify that the emitted
  `st08_vhrdb_ingested_pairs.csv` preserves raw response strings and populates `source_datasource_id`,
  `source_native_record_id`, and `source_disagreement_flag`.

#### Findings

- Before this change, the ingest step already retained raw VHRdb response strings in memory, but only under
  `source_global_response` / `source_datasource_response`; the exported ingest schema did not expose the raw values
  under the original column names required by the task acceptance criteria.
- The source metadata columns were already populated for VHRdb rows, so the main implementation gap was output-schema
  fidelity rather than missing provenance logic.

#### Interpretation

- `ST0.8` now preserves both downstream usability and source auditability: the normalized label remains available for
  modeling, while the raw VHRdb response fields survive unchanged in the ingested table for provenance checks and later
  harmonization work.
