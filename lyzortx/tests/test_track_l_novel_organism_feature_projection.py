import csv
from pathlib import Path

import joblib
import pytest

from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import (
    build_defense_column_mask,
    build_defense_feature_rows,
)
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import build_genome_kmer_feature_block
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages
from lyzortx.pipeline.track_l.steps.novel_organism_feature_projection import (
    project_novel_host,
    project_novel_phage,
)


def _write_single_row_csv(path: Path, row: dict[str, str], *, delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def test_project_novel_phage_round_trip_matches_panel_embedding(tmp_path: Path) -> None:
    panel_path = Path("data/genomics/phages/guelin_collection.csv")
    fna_dir = Path("data/genomics/phages/FNA")
    output_dir = tmp_path / "phage"
    panel_phages = read_panel_phages(panel_path, expected_panel_count=96)
    manifest = build_genome_kmer_feature_block(
        panel_phages=panel_phages,
        fna_dir=fna_dir,
        output_dir=output_dir,
        metadata_path=panel_path,
        embedding_dim=24,
    )

    phage = "409_P1"
    expected_rows = list(csv.DictReader((output_dir / "phage_genome_kmer_features.csv").open(encoding="utf-8")))
    expected_row = next(row for row in expected_rows if row["phage"] == phage)
    projected = project_novel_phage(fna_dir / f"{phage}.fna", output_dir / "phage_genome_kmer_svd.joblib")

    assert manifest["counts"]["embedding_dim_effective"] == 24
    assert projected["phage"] == phage
    assert projected["phage_gc_content"] == pytest.approx(float(expected_row["phage_gc_content"]))
    assert projected["phage_genome_length_nt"] == int(expected_row["phage_genome_length_nt"])
    for index in range(24):
        key = f"phage_genome_tetra_svd_{index:02d}"
        assert projected[key] == pytest.approx(float(expected_row[key]))


def test_project_novel_host_round_trip_matches_panel_features(tmp_path: Path) -> None:
    defense_path = Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv")
    output_dir = tmp_path / "host"
    rows = list(csv.DictReader(defense_path.open(encoding="utf-8"), delimiter=";"))
    feature_rows, _, manifest = build_defense_feature_rows(rows)
    mask = build_defense_column_mask(rows)
    mask_path = output_dir / "defense_subtype_column_mask.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mask, mask_path)

    bacteria = "001-023"
    expected_row = next(row for row in feature_rows if row["bacteria"] == bacteria)
    single_row_path = output_dir / "novel_defense_finder_output.csv"
    source_row = next(row for row in rows if row["bacteria"] == bacteria)
    _write_single_row_csv(single_row_path, source_row, delimiter=";")

    projected = project_novel_host(single_row_path, mask_path)

    assert manifest["retained_subtype_count"] == 79
    assert projected["bacteria"] == bacteria
    for column in mask["ordered_feature_columns"]:
        assert projected[column] == pytest.approx(float(expected_row[column]))
