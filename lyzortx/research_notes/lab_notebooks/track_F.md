### 2026-03-22: TF01 ST03 v1 benchmark protocol and bootstrap CIs
#### Executive summary
Locked the ST0.3 grouped host split as the canonical v1 benchmark protocol by surfacing the protocol ID in the split
artifacts and carrying it through the TG02 benchmark summary. Added 1000-strain bootstrap confidence intervals for the
v1 benchmark's dual-slice metrics on the isotonic LightGBM outputs: ROC-AUC, Brier score, ECE, and top-3 hit rate.
The benchmark summary now records both full-label and strict-confidence results against the same ST0.3 contract.

#### Interpretation
- The split contract is now explicit rather than implied by the salt string alone, which makes downstream benchmark
  reporting easier to audit and compare across model revisions.
- Bootstrapping at the holdout-strain level keeps the confidence intervals aligned with the evaluation denominator used
  by the recommendation metric, instead of treating pairs as independent samples.
- The strict-confidence slice remains a materially smaller and harder evaluation set, so the dual-slice reporting is
  still necessary to avoid overstating benchmark performance.
