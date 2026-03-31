from pathlib import Path

from lyzortx.pipeline.track_l.steps import build_tl17_phage_compatibility_preprocessor as tl17_step


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
