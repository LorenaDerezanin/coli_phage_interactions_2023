import csv
from pathlib import Path

import joblib
import pandas as pd
import pytest

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import write_csv
from lyzortx.pipeline.steel_thread_v0.steps import (
    st01_label_policy,
    st01b_confidence_tiers,
    st02_build_pair_table,
    st03_build_splits,
)
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import build_genome_kmer_feature_block
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle as tl08_bundle
from lyzortx.pipeline.track_l.steps import generalized_inference as tl08_infer
from lyzortx.pipeline.track_l.steps.novel_organism_feature_projection import project_novel_host

LOCKED_LIGHTGBM_PARAMS = {
    "learning_rate": 0.05,
    "min_child_samples": 25,
    "n_estimators": 300,
    "num_leaves": 31,
}


def _read_semicolon_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _build_panel_foundation(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "steel"
    st01_label_policy.main(["--output-dir", str(output_dir)])
    st01b_confidence_tiers.main(
        [
            "--st01-pair-audit-path",
            str(output_dir / "st01_pair_label_audit.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    st02_build_pair_table.main(
        [
            "--st01b-pair-audit-path",
            str(output_dir / "st01b_pair_confidence_audit.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    st03_build_splits.main(
        [
            "--st02-pair-table-path",
            str(output_dir / "st02_pair_table.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    return output_dir / "st02_pair_table.csv", output_dir / "st03_split_assignments.csv"


def _build_phage_kmer_outputs(tmp_path: Path) -> tuple[Path, Path]:
    panel_path = Path("data/genomics/phages/guelin_collection.csv")
    fna_dir = Path("data/genomics/phages/FNA")
    output_dir = tmp_path / "track_d"
    panel_phages = read_panel_phages(panel_path, expected_panel_count=96)
    build_genome_kmer_feature_block(
        panel_phages=panel_phages,
        fna_dir=fna_dir,
        output_dir=output_dir,
        metadata_path=panel_path,
        embedding_dim=24,
    )
    return output_dir / "phage_genome_kmer_features.csv", output_dir / "phage_genome_kmer_svd.joblib"


def test_build_training_rows_merges_host_and_phage_blocks() -> None:
    rows = tl08_bundle.build_training_rows(
        st02_rows=[
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "label_hard_any_lysis": "1",
                "training_weight_v3": "1.0",
            }
        ],
        split_rows=[
            {
                "pair_id": "B1__P1",
                "split_holdout": "train_non_holdout",
                "split_cv5_fold": "0",
                "is_hard_trainable": "1",
            }
        ],
        host_rows=[{"bacteria": "B1", "host_defense_subtype_a": 1, "host_defense_diversity": 1}],
        phage_rows=[{"phage": "P1", "phage_gc_content": "0.5", "phage_genome_tetra_svd_00": "0.1"}],
    )

    assert rows == [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "label_hard_any_lysis": "1",
            "training_weight_v3": "1.0",
            "split_holdout": "train_non_holdout",
            "split_cv5_fold": "0",
            "is_hard_trainable": "1",
            "host_defense_subtype_a": 1,
            "host_defense_diversity": 1,
            "phage_gc_content": "0.5",
            "phage_genome_tetra_svd_00": "0.1",
        }
    ]


def test_infer_reproduces_locked_panel_predictions_for_panel_host(tmp_path: Path, monkeypatch) -> None:
    st02_path, st03_path = _build_panel_foundation(tmp_path)
    phage_feature_path, phage_svd_path = _build_phage_kmer_outputs(tmp_path)
    defense_subtypes_path = Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv")
    bundle_output_dir = tmp_path / "bundle"

    build_result = tl08_bundle.build_model_bundle(
        st02_pair_table_path=st02_path,
        st03_split_assignments_path=st03_path,
        defense_subtypes_path=defense_subtypes_path,
        phage_kmer_feature_path=phage_feature_path,
        phage_kmer_svd_path=phage_svd_path,
        output_dir=bundle_output_dir,
        lightgbm_params=LOCKED_LIGHTGBM_PARAMS,
        random_state=42,
        calibration_fold=0,
    )

    reference_predictions = pd.read_csv(build_result["panel_predictions_path"])
    bacteria = "001-023"
    bacteria_predictions = reference_predictions[reference_predictions["bacteria"] == bacteria].copy()
    assert not bacteria_predictions.empty

    defense_rows = _read_semicolon_rows(defense_subtypes_path)
    defense_row = next(row for row in defense_rows if row["bacteria"] == bacteria)
    mask_path = bundle_output_dir / tl08_bundle.DEFENSE_MASK_FILENAME
    mask = joblib.load(mask_path)

    def fake_run_novel_host_defense_finder(
        assembly_path: Path,
        *,
        output_dir: Path,
        column_mask_path: Path,
        panel_defense_subtypes_path: Path,
        models_dir: Path,
        workers: int,
        force_model_update: bool,
        force_run: bool,
        preserve_raw: bool,
    ) -> dict[str, object]:
        single_row_path = output_dir / "raw_defense.csv"
        write_csv(single_row_path, list(defense_row.keys()), [defense_row])
        projected = project_novel_host(single_row_path, column_mask_path)
        ordered_row = {"bacteria": projected["bacteria"]}
        for column in mask["ordered_feature_columns"]:
            ordered_row[column] = projected[column]
        write_csv(
            output_dir / "novel_host_defense_features.csv",
            ["bacteria", *mask["ordered_feature_columns"]],
            [ordered_row],
        )
        return {"outputs": {"projected_feature_csv": str(output_dir / "novel_host_defense_features.csv")}}

    monkeypatch.setattr(
        tl08_infer.run_novel_host_defense_finder,
        "run_novel_host_defense_finder",
        fake_run_novel_host_defense_finder,
    )

    host_genome_path = tmp_path / "panel_host.fna"
    host_genome_path.write_text(">host\nATGCATGCATGCATGC\n", encoding="utf-8")
    phage_paths = [Path("data/genomics/phages/FNA") / f"{phage}.fna" for phage in bacteria_predictions["phage"]]
    inferred = tl08_infer.infer(host_genome_path, phage_paths, build_result["bundle_path"])

    expected = bacteria_predictions[["phage", "pred_lightgbm_isotonic", "rank_lightgbm_isotonic"]].copy()
    expected = (
        expected.rename(
            columns={
                "pred_lightgbm_isotonic": "p_lysis",
                "rank_lightgbm_isotonic": "rank",
            }
        )
        .sort_values(["rank", "phage"])
        .reset_index(drop=True)
    )
    observed = inferred.sort_values(["rank", "phage"]).reset_index(drop=True)

    assert list(observed["phage"]) == list(expected["phage"])
    assert list(observed["rank"]) == list(expected["rank"])
    assert observed["p_lysis"].tolist() == pytest.approx(expected["p_lysis"].tolist(), abs=1e-6)
