from __future__ import annotations

import io
import csv
import shutil
import zipfile
from pathlib import Path

import pytest

from lyzortx.pipeline.deployment_paired_features import download_picard_assemblies as picard_download


def _write_fasta_zip(zip_path: Path, fasta_names: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for fasta_name in fasta_names:
            archive.writestr(fasta_name, f">{Path(fasta_name).stem}\nATGC\n")


def test_load_st02_bacteria_ids_reads_unique_values(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_interactions.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bacteria", "phage"], delimiter=";")
        writer.writeheader()
        writer.writerow({"bacteria": "B2", "phage": "P1"})
        writer.writerow({"bacteria": "A1", "phage": "P2"})
        writer.writerow({"bacteria": "B2", "phage": "P3"})

    assert picard_download.load_st02_bacteria_ids(raw_path) == ["A1", "B2"]


def test_download_zip_file_uses_timeout(monkeypatch, tmp_path: Path) -> None:
    observed_timeout = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(download_url, timeout):
        observed_timeout["download_url"] = download_url
        observed_timeout["timeout"] = timeout
        return FakeResponse(b"payload")

    monkeypatch.setattr(picard_download.urllib.request, "urlopen", fake_urlopen)

    destination_path = tmp_path / "download.zip"
    picard_download._download_zip_file("https://example.invalid/picard.zip", destination_path)

    assert observed_timeout == {
        "download_url": "https://example.invalid/picard.zip",
        "timeout": picard_download.DOWNLOAD_TIMEOUT_SECONDS,
    }
    assert destination_path.read_bytes() == b"payload"


def test_download_picard_assemblies_extracts_and_validates(monkeypatch, tmp_path: Path) -> None:
    assembly_dir = tmp_path / "assemblies"
    zip_path = tmp_path / "picard.zip"
    _write_fasta_zip(zip_path, ["A1.fna", "B2.fna", "C3.fna"])

    monkeypatch.setattr(
        picard_download, "load_st02_bacteria_ids", lambda raw_interactions_path=None: ["A1", "B2", "C3"]
    )
    monkeypatch.setattr(
        picard_download,
        "_download_zip_file",
        lambda download_url, destination_path: shutil.copyfile(zip_path, destination_path),
    )

    observed = picard_download.download_picard_assemblies(
        assembly_dir=assembly_dir,
        raw_interactions_path=tmp_path / "unused.csv",
        expected_assembly_count=3,
    )

    assert observed == assembly_dir
    assert sorted(path.name for path in assembly_dir.glob("*.fna")) == ["A1.fna", "B2.fna", "C3.fna"]


def test_download_picard_assemblies_skips_when_complete(monkeypatch, tmp_path: Path) -> None:
    assembly_dir = tmp_path / "assemblies"
    assembly_dir.mkdir()
    for name in ["A1.fna", "B2.fna"]:
        (assembly_dir / name).write_text(f">{Path(name).stem}\nATGC\n", encoding="utf-8")

    monkeypatch.setattr(picard_download, "load_st02_bacteria_ids", lambda raw_interactions_path=None: ["A1", "B2"])
    monkeypatch.setattr(
        picard_download,
        "_download_zip_file",
        lambda *args, **kwargs: pytest.fail("download should be skipped when 2 FASTA files are present"),
    )

    observed = picard_download.download_picard_assemblies(
        assembly_dir=assembly_dir,
        raw_interactions_path=tmp_path / "unused.csv",
        expected_assembly_count=2,
    )

    assert observed == assembly_dir


def test_download_picard_assemblies_raises_for_missing_st02_ids(monkeypatch, tmp_path: Path) -> None:
    assembly_dir = tmp_path / "assemblies"
    zip_path = tmp_path / "picard.zip"
    _write_fasta_zip(zip_path, ["A1.fna", "B2.fna", "C3.fna"])

    monkeypatch.setattr(
        picard_download, "load_st02_bacteria_ids", lambda raw_interactions_path=None: ["A1", "B2", "D4"]
    )
    monkeypatch.setattr(
        picard_download,
        "_download_zip_file",
        lambda download_url, destination_path: shutil.copyfile(zip_path, destination_path),
    )

    with pytest.raises(ValueError, match="D4"):
        picard_download.download_picard_assemblies(
            assembly_dir=assembly_dir,
            raw_interactions_path=tmp_path / "unused.csv",
            expected_assembly_count=3,
        )
