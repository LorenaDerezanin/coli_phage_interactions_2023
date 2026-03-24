from lyzortx.pipeline.track_g.steps.investigate_non_leaky_candidate_features import (
    build_candidate_arms,
    build_locked_baseline_arm,
    compute_gap_recovery_fraction,
    summarize_candidate_row,
)
from lyzortx.pipeline.track_g.steps.train_v1_binary_classifier import FeatureSpace


def test_build_locked_baseline_arm_uses_locked_subset_blocks() -> None:
    feature_space = FeatureSpace(
        categorical_columns=("host_pathotype", "host_surface_lps_core_type"),
        numeric_columns=(
            "host_mouse_killed_10",
            "host_defense_subtype_abi_a",
            "host_receptor_variant_btub_01",
            "phage_gc_content",
            "lookup_available",
        ),
        track_c_additional_columns=(
            "host_defense_subtype_abi_a",
            "host_receptor_variant_btub_01",
            "host_surface_lps_core_type",
        ),
        track_d_columns=("phage_gc_content",),
        track_e_columns=("lookup_available",),
    )

    arm = build_locked_baseline_arm(
        feature_space,
        {
            "winner_arm_id": "subset_defense__phage_genomic",
            "winner_label": "defense + phage-genomic",
            "winner_subset_blocks": ["defense", "phage_genomic"],
        },
    )

    assert arm.arm_id == "subset_defense__phage_genomic"
    assert arm.candidate_column is None
    assert arm.locked_subset_blocks == ("defense", "phage_genomic")
    assert "host_defense_subtype_abi_a" in arm.numeric_columns
    assert "phage_gc_content" in arm.numeric_columns
    assert "host_receptor_variant_btub_01" not in arm.numeric_columns


def test_build_candidate_arms_adds_each_clean_feature_on_top_of_locked_baseline() -> None:
    feature_space = FeatureSpace(
        categorical_columns=("host_pathotype",),
        numeric_columns=(
            "host_mouse_killed_10",
            "host_defense_subtype_abi_a",
            "phage_gc_content",
            "lookup_available",
            "target_receptor_present",
            "protein_target_present",
            "surface_target_present",
            "receptor_cluster_matches",
            "isolation_host_umap_euclidean_distance",
            "isolation_host_defense_jaccard_distance",
        ),
        track_c_additional_columns=("host_defense_subtype_abi_a",),
        track_d_columns=("phage_gc_content",),
        track_e_columns=(
            "lookup_available",
            "target_receptor_present",
            "protein_target_present",
            "surface_target_present",
            "receptor_cluster_matches",
            "isolation_host_umap_euclidean_distance",
            "isolation_host_defense_jaccard_distance",
        ),
    )

    arms = build_candidate_arms(
        feature_space,
        {
            "winner_arm_id": "subset_defense__phage_genomic",
            "winner_label": "defense + phage-genomic",
            "winner_subset_blocks": ["defense", "phage_genomic"],
        },
    )

    assert len(arms) == 8
    assert arms[0].candidate_column is None
    assert arms[1].candidate_column == "lookup_available"
    assert arms[-1].candidate_column == "isolation_host_defense_jaccard_distance"
    assert "lookup_available" in arms[1].numeric_columns
    assert "phage_gc_content" in arms[1].numeric_columns


def test_compute_gap_recovery_fraction_uses_locked_and_leaked_auc() -> None:
    assert compute_gap_recovery_fraction(0.874, 0.8372, 0.910766) == 0.500231


def test_summarize_candidate_row_marks_acceptance_when_auc_clears_half_gap_and_top3_holds() -> None:
    feature_space = FeatureSpace(
        categorical_columns=("host_pathotype",),
        numeric_columns=(
            "host_mouse_killed_10",
            "host_defense_subtype_abi_a",
            "phage_gc_content",
            "lookup_available",
            "target_receptor_present",
            "protein_target_present",
            "surface_target_present",
            "receptor_cluster_matches",
            "isolation_host_umap_euclidean_distance",
            "isolation_host_defense_jaccard_distance",
        ),
        track_c_additional_columns=("host_defense_subtype_abi_a",),
        track_d_columns=("phage_gc_content",),
        track_e_columns=(
            "lookup_available",
            "target_receptor_present",
            "protein_target_present",
            "surface_target_present",
            "receptor_cluster_matches",
            "isolation_host_umap_euclidean_distance",
            "isolation_host_defense_jaccard_distance",
        ),
    )
    candidate_arm = build_candidate_arms(
        feature_space,
        {
            "winner_arm_id": "subset_defense__phage_genomic",
            "winner_label": "defense + phage-genomic",
            "winner_subset_blocks": ["defense", "phage_genomic"],
        },
    )[1]

    row = summarize_candidate_row(
        candidate_arm,
        holdout_binary_metrics={"roc_auc": 0.8745, "brier_score": 0.15},
        holdout_top3_metrics={
            "top3_hit_rate_all_strains": 0.907692,
            "top3_hit_rate_susceptible_only": 0.952381,
        },
        locked_baseline_auc=0.8372,
        locked_baseline_top3=0.907692,
        leaked_reference_auc=0.910766,
    )

    assert row["candidate_source"] == "TE01 curated lookup"
    assert row["gap_recovery_fraction_vs_locked_v1"] == 0.507028
    assert row["top3_non_degrading_vs_locked_v1"] is True
    assert row["recovers_gt_50pct_auc_gap_without_top3_degradation"] is True
