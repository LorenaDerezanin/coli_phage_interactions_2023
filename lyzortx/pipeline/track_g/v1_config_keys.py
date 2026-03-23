"""Canonical key names for the v1 feature configuration contract.

The v1 feature configuration is the locked schema shared between Track G
(producer) and Track P (consumer).  These constants define the single source
of truth for every key that appears in v1_feature_configuration.json and the
``final_feature_lock`` section of the TG05 summary.

Import these instead of using string literals so that typos and renames
become import errors or grep-visible refactors.
"""

# Top-level wrapper
LOCKED_V1_FEATURE_CONFIGURATION = "locked_v1_feature_configuration"

# Scalars inside the lock
SELECTION_POLICY = "selection_policy"
WINNER_ARM_ID = "winner_arm_id"
WINNER_LABEL = "winner_label"
WINNER_SUBSET_BLOCKS = "winner_subset_blocks"

# Metric sections
PANEL_DEFAULT = "panel_default"
DEPLOYMENT_REALISTIC_SENSITIVITY = "deployment_realistic_sensitivity"
TG01_ALL_FEATURES_REFERENCE = "tg01_all_features_reference"

# Metric field names (shared across metric sections)
HOLDOUT_ROC_AUC = "holdout_roc_auc"
HOLDOUT_BRIER_SCORE = "holdout_brier_score"
HOLDOUT_TOP3_HIT_RATE_ALL_STRAINS = "holdout_top3_hit_rate_all_strains"
HOLDOUT_TOP3_HIT_RATE_SUSCEPTIBLE_ONLY = "holdout_top3_hit_rate_susceptible_only"

# Deployment-specific
EXCLUDED_LABEL_DERIVED_COLUMNS = "excluded_label_derived_columns"

# Reviewed label-derived columns list
LABEL_DERIVED_COLUMNS_REVIEWED = "label_derived_columns_reviewed"
