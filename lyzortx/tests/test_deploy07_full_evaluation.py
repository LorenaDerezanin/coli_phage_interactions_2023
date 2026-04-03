from lyzortx.pipeline.deployment_paired_features.run_deploy07_full_evaluation import (
    PHAGE_BASELINE_FAMILY_COUNT_COLUMN,
    build_baseline_phage_rows_from_continuous,
    build_baseline_defense_rows_from_counts,
    select_winning_arm_id_from_auc_ci,
    validate_no_duplicate_block_columns,
    validate_schema_columns,
)


def test_validate_schema_columns_reports_extra_columns_but_requires_schema_columns() -> None:
    schema = {
        "columns": [
            {"name": "bacteria", "dtype": "string"},
            {"name": "feature_a", "dtype": "float64"},
        ]
    }
    rows = [{"bacteria": "host1", "feature_a": 1.0, "extra_feature": 2.0}]

    summary = validate_schema_columns(
        block_id="host_block",
        key_column="bacteria",
        schema=schema,
        rows=rows,
    )

    assert summary["missing_columns"] == []
    assert summary["extra_columns"] == ["extra_feature"]


def test_validate_no_duplicate_block_columns_rejects_duplicate_non_key_columns() -> None:
    block_a = type("Block", (), {})()
    block_a.block_id = "a"
    block_a.key_column = "bacteria"
    block_a.schema = {
        "columns": [
            {"name": "bacteria", "dtype": "string"},
            {"name": "shared_feature", "dtype": "float64"},
        ]
    }
    block_a.rows = [{"bacteria": "host1", "shared_feature": 1.0}]
    block_a.categorical_columns = ()
    block_a.numeric_columns = ("shared_feature",)

    block_b = type("Block", (), {})()
    block_b.block_id = "b"
    block_b.key_column = "bacteria"
    block_b.schema = {
        "columns": [
            {"name": "bacteria", "dtype": "string"},
            {"name": "shared_feature", "dtype": "float64"},
        ]
    }
    block_b.rows = [{"bacteria": "host1", "shared_feature": 2.0}]
    block_b.categorical_columns = ()
    block_b.numeric_columns = ("shared_feature",)

    try:
        validate_no_duplicate_block_columns((block_a, block_b))
    except ValueError as exc:
        assert "shared_feature" in str(exc)
    else:
        raise AssertionError("Expected duplicate column validation to fail")


def test_build_baseline_phage_rows_from_continuous_binarizes_family_scores() -> None:
    rows = [
        {
            "phage": "P1",
            "family_11_percent_identity": 88.0,
            "family_22_percent_identity": 0.0,
            "tl17_rbp_reference_hit_count": 3,
        }
    ]

    converted_rows, numeric_columns = build_baseline_phage_rows_from_continuous(
        rows,
        family_score_columns=("family_11_percent_identity", "family_22_percent_identity"),
        hit_count_column="tl17_rbp_reference_hit_count",
    )

    assert converted_rows == [
        {
            "phage": "P1",
            "family_11_percent_identity": 1,
            "family_22_percent_identity": 0,
            "tl17_rbp_reference_hit_count": 3,
            PHAGE_BASELINE_FAMILY_COUNT_COLUMN: 1,
        }
    ]
    assert numeric_columns == (
        "family_11_percent_identity",
        "family_22_percent_identity",
        "tl17_rbp_reference_hit_count",
        PHAGE_BASELINE_FAMILY_COUNT_COLUMN,
    )


def test_build_baseline_defense_rows_from_counts_matches_track_c_summary_logic() -> None:
    rows = build_baseline_defense_rows_from_counts(
        [{"bacteria": "host1", "AbiD": 2, "CAS_Type_I-E": 0, "RM_Type_IV": 1}],
        subtype_columns=("AbiD", "CAS_Type_I-E", "RM_Type_IV"),
    )

    assert rows == [
        {
            "bacteria": "host1",
            "host_defense_subtype_abi_d": 1,
            "host_defense_subtype_cas_type_i_e": 0,
            "host_defense_subtype_rm_type_iv": 1,
            "host_defense_diversity": 2,
            "host_defense_has_crispr": 0,
            "host_defense_abi_burden": 1,
        }
    ]


def test_select_winning_arm_id_from_auc_ci_only_promotes_when_ci_excludes_zero() -> None:
    baseline = {
        "arm_id": "baseline",
        "auc_delta_ci_low_vs_baseline": 0.0,
    }
    deployment_not_locked = {
        "arm_id": "deployment",
        "auc_delta_ci_low_vs_baseline": 0.0,
    }
    deployment_locked = {
        "arm_id": "deployment",
        "auc_delta_ci_low_vs_baseline": 0.002,
    }

    assert (
        select_winning_arm_id_from_auc_ci(
            (baseline, deployment_not_locked),
            baseline_arm_id="baseline",
            deployment_arm_id="deployment",
        )
        == "baseline"
    )
    assert (
        select_winning_arm_id_from_auc_ci(
            (baseline, deployment_locked),
            baseline_arm_id="baseline",
            deployment_arm_id="deployment",
        )
        == "deployment"
    )
