import json
from pathlib import Path

from lyzortx.pipeline.track_l.steps import build_tl17_phage_compatibility_preprocessor as tl17_step
from lyzortx.pipeline.track_l.steps.deployable_tl17_runtime import Tl17FamilyRuntime


def test_build_candidate_audit_rows_marks_tl17_as_chosen() -> None:
    rows = tl17_step.build_candidate_audit_rows()

    chosen_rows = [row for row in rows if row["chosen_for_tl17"] == 1]
    assert len(chosen_rows) == 1
    assert chosen_rows[0]["candidate_block_id"] == tl17_step.TL17_BLOCK_ID


def test_select_surface_hosts_uses_validation_subset_overlap(tmp_path: Path) -> None:
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    for host in ("EDL933", "LF82", "not_in_predictions"):
        (fasta_dir / f"{host}.fasta").write_text(f">{host}\nATGC\n", encoding="utf-8")

    selected = tl17_step.select_surface_hosts(fasta_dir, {"EDL933", "LF82", "other"})

    assert selected == ("EDL933", "LF82")


def test_build_surface_delta_rows_summarizes_prediction_changes(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    baseline_path.write_text(
        (
            "pair_id,bacteria,phage,split_holdout,split_cv5_fold,label_hard_any_lysis,pred_lightgbm_isotonic,"
            "rank_lightgbm_isotonic\n"
            "B1__P1,EDL933,P1,train_non_holdout,0,1,0.2,2\n"
            "B1__P2,EDL933,P2,train_non_holdout,0,0,0.4,1\n"
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        (
            "pair_id,bacteria,phage,split_holdout,split_cv5_fold,label_hard_any_lysis,pred_lightgbm_isotonic,"
            "rank_lightgbm_isotonic\n"
            "B1__P1,EDL933,P1,train_non_holdout,0,1,0.5,1\n"
            "B1__P2,EDL933,P2,train_non_holdout,0,0,0.3,2\n"
        ),
        encoding="utf-8",
    )

    delta_rows, summary_rows = tl17_step.build_surface_delta_rows(
        baseline_predictions_path=baseline_path,
        candidate_predictions_path=candidate_path,
        surface_hosts=("EDL933",),
    )

    assert len(delta_rows) == 2
    assert summary_rows == [
        {
            "bacteria": "EDL933",
            "changed_prediction_count": 2,
            "median_abs_probability_delta": 0.2,
            "max_abs_probability_delta": 0.3,
            "identical_rank_count": 0,
        }
    ]


def test_build_validation_summary_rows_uses_family_score_columns_only() -> None:
    summary_rows = tl17_step.build_validation_summary_rows(
        projected_feature_rows=[
            {
                "phage": "P1",
                "tl17_phage_rbp_family_11_percent_identity": 88.0,
                "tl17_phage_rbp_family_22_percent_identity": 0.0,
                tl17_step.SUMMARY_HIT_COUNT_COLUMN: 1,
            },
            {
                "phage": "P2",
                "tl17_phage_rbp_family_11_percent_identity": 0.0,
                "tl17_phage_rbp_family_22_percent_identity": 45.0,
                tl17_step.SUMMARY_HIT_COUNT_COLUMN: 1,
            },
        ],
        family_score_columns=[
            "tl17_phage_rbp_family_11_percent_identity",
            "tl17_phage_rbp_family_22_percent_identity",
        ],
        surface_summary_rows=[{"changed_prediction_count": 3}],
        selected_surface_hosts=("EDL933",),
    )

    assert summary_rows == [
        {"metric": "projected_panel_phage_count", "value": 2},
        {"metric": "tl17_family_feature_count", "value": 2},
        {"metric": "nonzero_family_feature_count", "value": 2},
        {"metric": "panel_phages_with_any_tl17_feature", "value": 2},
        {"metric": "surface_probe_host_count", "value": 1},
        {"metric": "surface_probe_changed_prediction_count", "value": 3},
    ]


def test_write_schema_manifest_records_percent_identity_columns(tmp_path: Path) -> None:
    family_rows = [
        Tl17FamilyRuntime(
            family_id="RBP_PHROG_11",
            column_name="tl17_phage_rbp_family_11_percent_identity",
            supporting_phage_count=2,
            supporting_reference_count=2,
        )
    ]

    schema_path = tl17_step.write_schema_manifest(family_rows, tmp_path / "schema_manifest.json")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["columns"] == [
        {"name": "phage", "dtype": "string"},
        {"name": "tl17_phage_rbp_family_11_percent_identity", "dtype": "float64"},
        {"name": tl17_step.SUMMARY_HIT_COUNT_COLUMN, "dtype": "int64"},
    ]
    assert schema["dropped_legacy_columns"] == ["tl17_rbp_family_count"]
