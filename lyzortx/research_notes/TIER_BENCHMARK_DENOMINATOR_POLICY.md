# Tier 1 vs Tier 2 Benchmark Denominator and Metric Policy

## Purpose

This note locks the denominator and metric contract for benchmark reporting so that Tier 1 and Tier 2 results are
comparable across model versions.

Primary references:

- Plan execution driver: `lyzortx/research_notes/PLAN.md`
- Cohort construction logic: `lyzortx/pipeline/track_a/steps/build_track_a_foundation.py`
- Integrity guardrails for expected cohort sizes: `lyzortx/pipeline/track_a/checks/check_track_a_integrity.py`

## Locked Cohort Denominators

### Canonical host cohorts

Host cohorts are defined by canonicalized bacteria IDs produced by the Track A foundation step.

1. `raw369`
   - Definition: bacteria present in `data/interactions/raw/raw_interactions.csv`
   - Locked host denominator: **369 strains**
2. `matrix402`
   - Definition: bacteria present in `data/interactions/interaction_matrix.csv`
   - Locked host denominator: **402 strains**
3. `features404`
   - Definition: bacteria present in `data/metadata/370+host_cross_validation_groups_1e-4.csv`
   - Clarification: the filename's `370+` prefix is legacy naming; the canonicalized union represented
     by this file is 404 strains.
   - Locked host denominator: **404 strains**

These three denominators are asserted in the Track A integrity checker (`raw369=369`, `matrix402=402`,
`features404=404`).

### Phage denominator

- **Fixed-panel denominator:** `phage96` = 96 canonical phages (union of canonical phage names from raw interactions,
  interaction matrix, and phage metadata as produced by Track A foundation logic).
- Tier 1 uses this fixed `phage96` panel only.
- Tier 2 may use an expanded panel, but every report must explicitly state the panel size and source composition.

## Tier Policy (What Counts in Each Benchmark Tier)

### Tier 1 (Current-panel feasible)

- Host denominator policy:
  - Primary benchmark denominator: `raw369` (default)
  - Auxiliary reporting: `matrix402` and/or `features404`, clearly labeled as auxiliary
- Phage denominator policy: fixed `phage96` panel
- Data provenance policy: internal repository data only (no external datasource expansion)

### Tier 2 (North-star)

- Host denominator policy:
  - Must include a declared external evaluation cohort and report its exact host count (`N_hosts_external`)
  - Must also report internal comparator denominator (`raw369`) for continuity
- Phage denominator policy:
  - Expanded panel permitted and expected
  - Must report exact panel denominator (`N_phages_tier2`) and expansion source manifest
- Data provenance policy: external integration is allowed, with source-aware leakage checks required

## Metric Definitions (Locked)

For all metrics below, report both slices when available:

- **Full-label slice:** all labeled pairs included by the benchmark protocol
- **Strict-confidence slice:** only pairs meeting strict-confidence criteria from ST0.1b policy artifacts

### 1) Top-3 Lytic Hit Rate (all strains)

- Unit denominator: strains (`N_strains_eval`)
- Per-strain success indicator:
  - `Hit_i = 1` if at least one of top-3 recommended phages for strain `i` is truly lytic in evaluation labels
  - `Hit_i = 0` otherwise
- Metric:
  - `Top3_HitRate_all = (sum_i Hit_i) / N_strains_eval`
- Tier alignment:
  - Tier 1 target applies on fixed 96-phage panel
  - Tier 2 target applies on declared expanded panel denominator

### 2) Top-3 Lytic Hit Rate (susceptible-only)

- Unit denominator: susceptible strains only (`N_strains_susceptible`)
- Susceptible strain definition: strain has at least one known lytic phage in the evaluation denominator/panel
- Metric:
  - `Top3_HitRate_susc = (sum_{i in susceptible} Hit_i) / N_strains_susceptible`

### 3) Precision at High Confidence

- Unit denominator: high-confidence positive predictions (`N_pred_high_conf`)
- High-confidence thresholding must be declared in the report (score cutoff and calibration method)
- Metric:
  - `Precision_high_conf = TP_high_conf / N_pred_high_conf`
- Zero-support rule:
  - If `N_pred_high_conf = 0`, report `Precision_high_conf = NA` and `Support_high_conf = 0` (do not coerce to 0 or 1).
- Mandatory companion quantity:
  - `Support_high_conf = N_pred_high_conf` (or equivalent strain-level support count, explicitly named)

### 4) Calibration Quality Gates

- **Brier score** (probability quality):
  - `Brier = mean((p_i - y_i)^2)` over evaluation pairs
- **ECE** (Expected Calibration Error):
  - Bin predictions into 10 equal-width bins over [0, 1]; report weighted absolute confidence-accuracy gap across bins
  - If a report intentionally uses a different binning scheme, it must declare the exact scheme and justify the override
- Reporting policy:
  - Always report Brier and ECE together
  - Always report the exact denominator used for each (pair count and slice)

## Mandatory Reporting Header (Tier 1 and Tier 2)

Every benchmark table/report must include:

- `tier`: `tier1` or `tier2`
- `host_cohort_name`: one of `raw369`, `matrix402`, `features404`, or declared external cohort tag
- `host_denominator_n`
- `phage_denominator_n`
- `label_slice`: `full_label` or `strict_confidence`
- `split_protocol_id`
- `metric_name`
- `metric_value`
- `support_n` (when metric-specific support differs from cohort denominator)

## Reproducibility Checks

The denominator contracts are considered valid only when both checks pass:

1. Track A foundation artifacts regenerate without cohort-count drift.
2. Track A integrity check confirms expected host cohort sizes
   (`raw369=369`, `matrix402=402`, `features404=404`) and fixed phage panel size
   (`phage96=96`).

If any denominator drifts, Tier 1/Tier 2 benchmark comparisons are invalid until the drift is explained and
documented.
