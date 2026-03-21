import csv
import json
from pathlib import Path

from lyzortx.pipeline.track_d import run_track_d
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages
from lyzortx.pipeline.track_d.steps.build_phage_genome_kmer_features import (
    build_genome_kmer_feature_block,
    compute_kmer_frequency_vector,
)


def _write_panel_metadata(path: Path, phages: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phage"], delimiter=";")
        writer.writeheader()
        for phage in phages:
            writer.writerow({"phage": phage})


def test_compute_kmer_frequency_vector_skips_ambiguous_windows() -> None:
    out = compute_kmer_frequency_vector("AAAANAAA", k=4)

    assert out.shape == (256,)
    assert out.sum() == 1.0
    assert out.max() == 1.0


def test_build_genome_kmer_feature_block_writes_joinable_panel_csv_and_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "panel.csv"
    fna_dir = tmp_path / "FNA"
    output_dir = tmp_path / "out"
    fna_dir.mkdir(parents=True)

    _write_panel_metadata(metadata_path, ["P1", "P2"])
    (fna_dir / "P1.fna").write_text(">p1\nAAAACCCCGGGGTTTTAAAACCCC\n", encoding="utf-8")
    (fna_dir / "P2.fna").write_text(">p2\nATGCATGCATGCATGCATGCATGC\n", encoding="utf-8")
    (fna_dir / "EXTRA.fna").write_text(">extra\nGGGGCCCCAAAATTTTGGGGCCCC\n", encoding="utf-8")

    panel_phages = read_panel_phages(metadata_path, expected_panel_count=2)
    manifest = build_genome_kmer_feature_block(
        panel_phages=panel_phages,
        fna_dir=fna_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        embedding_dim=4,
    )

    features_path = output_dir / "phage_genome_kmer_features.csv"
    metadata_csv_path = output_dir / "phage_genome_kmer_feature_metadata.csv"
    manifest_path = output_dir / "manifest.json"

    rows = list(csv.DictReader(features_path.open(encoding="utf-8")))
    metadata_rows = list(csv.DictReader(metadata_csv_path.open(encoding="utf-8")))
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [row["phage"] for row in rows] == ["P1", "P2"]
    assert len(rows[0]) == 1 + 4 + 2
    assert rows[0]["phage_genome_length_nt"] == "24"
    assert float(rows[0]["phage_gc_content"]) == 0.5
    assert len(metadata_rows) == 6
    assert manifest["counts"]["discovered_genome_count"] == 3
    assert manifest["counts"]["output_row_count"] == 2
    assert manifest["counts"]["non_panel_genome_count"] == 1
    assert manifest_json["non_panel_genomes"] == ["EXTRA"]


def test_run_track_d_dispatches_genome_kmer_step(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        run_track_d.build_phage_protein_sets,
        "main",
        lambda argv: calls.append("protein-sets"),
    )
    monkeypatch.setattr(
        run_track_d.build_phage_genome_kmer_features,
        "main",
        lambda argv: calls.append("genome-kmers"),
    )

    run_track_d.main(["--step", "genome-kmers"])
    assert calls == ["genome-kmers"]

    calls.clear()
    run_track_d.main(["--step", "all"])
    assert calls == ["protein-sets", "genome-kmers"]
