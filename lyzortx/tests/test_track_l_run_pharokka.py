"""Tests for Track L pharokka runner logic (pure functions only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyzortx.pipeline.track_l.steps.run_pharokka import (
    discover_fna_files,
    verify_annotations,
)


def test_discover_fna_files_finds_all(tmp_path: Path) -> None:
    for name in ("phage_A.fna", "phage_B.fna", "phage_C.fna"):
        (tmp_path / name).write_text(">seq\nATCG\n", encoding="utf-8")
    # Add a non-fna file that should be ignored
    (tmp_path / "desktop.ini").write_text("", encoding="utf-8")

    files = discover_fna_files(tmp_path)
    assert len(files) == 3
    assert all(f.suffix == ".fna" for f in files)
    # Check sorted order
    assert [f.stem for f in files] == ["phage_A", "phage_B", "phage_C"]


def test_discover_fna_files_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_fna_files(tmp_path / "nonexistent")


def test_discover_fna_files_raises_on_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .fna files"):
        discover_fna_files(empty)


def test_verify_annotations_counts_cds(tmp_path: Path) -> None:
    phage_dir = tmp_path / "phage_A"
    phage_dir.mkdir()
    tsv = phage_dir / "phage_A_cds_final_merged_output.tsv"
    tsv.write_text("gene\tstart\tstop\nCDS_1\t1\t100\nCDS_2\t200\t400\n", encoding="utf-8")
    assert verify_annotations(phage_dir, "phage_A") == 2


def test_verify_annotations_raises_on_missing_file(tmp_path: Path) -> None:
    phage_dir = tmp_path / "phage_X"
    phage_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing merged output"):
        verify_annotations(phage_dir, "phage_X")


def test_verify_annotations_raises_on_zero_cds(tmp_path: Path) -> None:
    phage_dir = tmp_path / "phage_Z"
    phage_dir.mkdir()
    tsv = phage_dir / "phage_Z_cds_final_merged_output.tsv"
    tsv.write_text("gene\tstart\tstop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Zero annotated CDS"):
        verify_annotations(phage_dir, "phage_Z")
