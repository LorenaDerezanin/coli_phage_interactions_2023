import csv
from pathlib import Path

import joblib

from lyzortx.pipeline.track_l.steps import run_novel_host_defense_finder


def test_build_defense_subtype_count_row_counts_known_and_unmatched_systems() -> None:
    source_subtype_columns = ["AbiD", "CAS_Class1-Subtype-I-E", "RM_Type_I"]
    system_rows = [
        {"sys_id": "sys_1", "type": "AbiD", "subtype": "", "activity": "defense"},
        {"sys_id": "sys_2", "type": "Cas", "subtype": "CAS_Class1-Subtype-I-E", "activity": "defense"},
        {"sys_id": "sys_3", "type": "RM", "subtype": "RM_Type_I", "activity": "defense"},
        {"sys_id": "sys_4", "type": "NovelSystem", "subtype": "NovelSubtype", "activity": "defense"},
    ]

    subtype_row, matched, unmatched = run_novel_host_defense_finder.build_defense_subtype_count_row(
        bacteria_id="novel_host",
        system_rows=system_rows,
        source_subtype_columns=source_subtype_columns,
    )

    assert subtype_row == {
        "bacteria": "novel_host",
        "AbiD": 1,
        "CAS_Class1-Subtype-I-E": 1,
        "RM_Type_I": 1,
    }
    assert matched == {"AbiD": 1, "CAS_Class1-Subtype-I-E": 1, "RM_Type_I": 1}
    assert unmatched == {"NovelSubtype": 1}


def test_resolve_defense_mask_rebuilds_from_panel_csv_when_joblib_missing(tmp_path: Path) -> None:
    rebuilt_path, mask_status = run_novel_host_defense_finder.resolve_defense_mask(
        column_mask_path=tmp_path / "missing_mask.joblib",
        panel_defense_subtypes_path=Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"),
        output_dir=tmp_path,
    )

    mask = joblib.load(rebuilt_path)

    assert mask_status == "rebuilt_from_panel_subtypes"
    assert rebuilt_path == tmp_path / "defense_subtype_column_mask.joblib"
    assert rebuilt_path.exists()
    assert len(mask["ordered_feature_columns"]) > 0
    assert "CAS_Class1-Subtype-I-E" in mask["source_subtype_columns"]


def test_run_novel_host_defense_finder_projects_expected_feature_columns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assembly_path = tmp_path / "mg1655.fna"
    assembly_path.write_text(">chromosome\nATGCGTATGCGTATGCGTATGCGT\n", encoding="utf-8")
    output_dir = tmp_path / "runner_output"

    monkeypatch.setattr(
        run_novel_host_defense_finder,
        "ensure_defense_finder_models",
        lambda *, models_dir, force_update: "existing",
    )

    def fake_run_defense_finder_on_assembly(
        assembly_path: Path,
        *,
        output_dir: Path,
        models_dir: Path,
        workers: int,
        preserve_raw: bool,
        force_run: bool,
    ) -> tuple[Path, dict[str, object]]:
        systems_path = output_dir / f"{assembly_path.stem}_defense_finder_systems.tsv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with systems_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sys_id", "type", "subtype", "activity", "sys_beg", "sys_end"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sys_id": "sys_1",
                    "type": "Cas",
                    "subtype": "CAS_Class1-Subtype-I-E",
                    "activity": "defense",
                    "sys_beg": "prot_1",
                    "sys_end": "prot_4",
                }
            )
            writer.writerow(
                {
                    "sys_id": "sys_2",
                    "type": "RM",
                    "subtype": "RM_Type_I",
                    "activity": "defense",
                    "sys_beg": "prot_6",
                    "sys_end": "prot_9",
                }
            )
            writer.writerow(
                {
                    "sys_id": "sys_3",
                    "type": "Novel",
                    "subtype": "NovelSubtype",
                    "activity": "defense",
                    "sys_beg": "prot_11",
                    "sys_end": "prot_12",
                }
            )
        return (
            systems_path,
            {
                "protein_fasta_path": str(output_dir / f"{assembly_path.stem}.prt"),
                "replicon_count": 1,
                "genome_nt_count": 24,
                "predicted_cds_count": 12,
                "gene_finder_modes": ["meta"],
                "used_cached_systems": False,
            },
        )

    monkeypatch.setattr(
        run_novel_host_defense_finder,
        "run_defense_finder_on_assembly",
        fake_run_defense_finder_on_assembly,
    )

    manifest = run_novel_host_defense_finder.run_novel_host_defense_finder(
        assembly_path,
        bacteria_id="mg1655",
        output_dir=output_dir,
        column_mask_path=tmp_path / "missing_mask.joblib",
        panel_defense_subtypes_path=Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"),
        models_dir=tmp_path / "models",
        workers=0,
        force_model_update=False,
        model_install_mode=run_novel_host_defense_finder.MODEL_INSTALL_MODE_ENSURE,
        force_run=False,
        preserve_raw=False,
    )

    projected_path = output_dir / "novel_host_defense_features.csv"
    with projected_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        projected_rows = list(reader)

    mask = joblib.load(output_dir / "defense_subtype_column_mask.joblib")
    projected_row = projected_rows[0]

    assert manifest["counts"]["detected_system_count"] == 3
    assert manifest["counts"]["matched_training_subtype_system_count"] == 2
    assert manifest["counts"]["unmatched_detected_system_count"] == 1
    assert reader.fieldnames == ["bacteria", *mask["ordered_feature_columns"]]
    assert projected_row["bacteria"] == "mg1655"
    assert float(projected_row["host_defense_subtype_cas_class1_subtype_i_e"]) == 1.0
    assert float(projected_row["host_defense_subtype_rm_type_i"]) == 1.0
    assert float(projected_row["host_defense_has_crispr"]) == 1.0
    assert float(projected_row["host_defense_diversity"]) == 2.0


def test_run_defense_finder_on_assembly_skips_pyrodigal_when_cached(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assembly_path = tmp_path / "mg1655.fna"
    assembly_path.write_text(">chromosome\nATGCGTATGCGTATGCGTATGCGT\n", encoding="utf-8")
    output_dir = tmp_path / "runner_output"
    output_dir.mkdir()
    systems_path = output_dir / "mg1655_defense_finder_systems.tsv"
    systems_path.write_text(
        "sys_id\ttype\tsubtype\tactivity\nsys_1\tCas\tCAS_Class1-Subtype-I-E\tdefense\n",
        encoding="utf-8",
    )

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("predict_proteins_with_pyrodigal should not run when cached systems TSV exists")

    monkeypatch.setattr(
        run_novel_host_defense_finder,
        "predict_proteins_with_pyrodigal",
        fail_if_called,
    )

    resolved_systems_path, protein_metadata = run_novel_host_defense_finder.run_defense_finder_on_assembly(
        assembly_path,
        output_dir=output_dir,
        models_dir=tmp_path / "models",
        workers=0,
        preserve_raw=False,
        force_run=False,
    )

    assert resolved_systems_path == systems_path
    assert protein_metadata == {
        "protein_fasta_path": str(output_dir / "mg1655.prt"),
        "replicon_count": None,
        "genome_nt_count": None,
        "predicted_cds_count": None,
        "gene_finder_modes": [],
        "used_cached_systems": True,
    }


def test_validate_pinned_defense_finder_models_rejects_source_checkout_shape(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "defense-finder-models").mkdir(parents=True)
    (models_dir / "CasFinder").mkdir(parents=True)

    try:
        run_novel_host_defense_finder.validate_pinned_defense_finder_models(models_dir)
    except FileNotFoundError as exc:
        assert "source checkout" in str(exc)
    else:
        raise AssertionError("Expected validate_pinned_defense_finder_models to reject missing metadata")


def test_resolve_defense_finder_model_status_forbid_rejects_force_update(tmp_path: Path) -> None:
    try:
        run_novel_host_defense_finder.resolve_defense_finder_model_status(
            models_dir=tmp_path / "models",
            force_update=True,
            model_install_mode=run_novel_host_defense_finder.MODEL_INSTALL_MODE_FORBID,
        )
    except ValueError as exc:
        assert "force_update cannot be used" in str(exc)
    else:
        raise AssertionError("Expected force_update to be rejected when model-install-mode=forbid")
