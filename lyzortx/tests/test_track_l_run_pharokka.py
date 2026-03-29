"""Tests for Track L pharokka runner logic (pure functions only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyzortx.pipeline.track_l.run_track_l import cache_key_tsvs
from lyzortx.pipeline.track_l.steps.run_pharokka import (
    discover_fna_files,
    verify_annotations,
)


def test_discover_fna_files_finds_all_and_ignores_non_fna(tmp_path: Path) -> None:
    for name in ("phage_A.fna", "phage_B.fna", "phage_C.fna"):
        (tmp_path / name).write_text(">seq\nATCG\n", encoding="utf-8")
    (tmp_path / "desktop.ini").write_text("", encoding="utf-8")

    files = discover_fna_files(tmp_path)
    assert len(files) == 3
    assert all(f.suffix == ".fna" for f in files)
    assert [f.stem for f in files] == ["phage_A", "phage_B", "phage_C"]


def test_verify_annotations_counts_cds(tmp_path: Path) -> None:
    phage_dir = tmp_path / "phage_A"
    phage_dir.mkdir()
    tsv = phage_dir / "phage_A_cds_final_merged_output.tsv"
    tsv.write_text("gene\tstart\tstop\nCDS_1\t1\t100\nCDS_2\t200\t400\n", encoding="utf-8")
    assert verify_annotations(phage_dir, "phage_A") == 2


def test_verify_annotations_raises_on_zero_cds(tmp_path: Path) -> None:
    phage_dir = tmp_path / "phage_Z"
    phage_dir.mkdir()
    tsv = phage_dir / "phage_Z_cds_final_merged_output.tsv"
    tsv.write_text("gene\tstart\tstop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Zero annotated CDS"):
        verify_annotations(phage_dir, "phage_Z")


def test_cache_key_tsvs_copies_only_key_files(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    cached = tmp_path / "cached"
    for name in ("phageA", "phageB"):
        d = annotations / name
        d.mkdir(parents=True)
        (d / f"{name}_cds_final_merged_output.tsv").write_text("data\n", encoding="utf-8")
        (d / f"{name}_cds_functions.tsv").write_text("funcs\n", encoding="utf-8")
        (d / f"{name}.gff").write_text("gff\n", encoding="utf-8")

    copied = cache_key_tsvs(annotations, cached)
    assert copied == 4
    assert (cached / "phageA_cds_final_merged_output.tsv").exists()
    assert (cached / "phageB_cds_functions.tsv").exists()
    assert not (cached / "phageA.gff").exists()
