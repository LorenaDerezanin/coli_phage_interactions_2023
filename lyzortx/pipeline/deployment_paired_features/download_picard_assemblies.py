#!/usr/bin/env python3
"""Download and validate the Picard collection host assemblies used by DEPLOY."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from lyzortx.log_config import setup_logging

LOGGER = logging.getLogger(__name__)

FIGSHARE_DOWNLOAD_URL = "https://ndownloader.figshare.com/articles/25941691/versions/1"
DEFAULT_ASSEMBLY_DIR = Path("lyzortx/data/assemblies/picard")
DEFAULT_RAW_INTERACTIONS_PATH = Path("data/interactions/raw/raw_interactions.csv")
EXPECTED_ASSEMBLY_COUNT = 403
FASTA_SUFFIXES = (".fa", ".faa", ".fasta", ".fna", ".ffn", ".frn")
SCRATCH_DOWNLOAD_DIR = Path(".scratch/deployment_paired_features")
DOWNLOAD_TIMEOUT_SECONDS = 300


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", type=Path, default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--raw-interactions-path", type=Path, default=DEFAULT_RAW_INTERACTIONS_PATH)
    parser.add_argument("--download-url", default=FIGSHARE_DOWNLOAD_URL)
    parser.add_argument("--expected-assembly-count", type=int, default=EXPECTED_ASSEMBLY_COUNT)
    return parser.parse_args(argv)


def _is_fasta_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in FASTA_SUFFIXES


def list_fasta_files(assembly_dir: Path) -> list[Path]:
    if not assembly_dir.exists():
        return []
    return sorted(path for path in assembly_dir.rglob("*") if _is_fasta_path(path))


def load_st02_bacteria_ids(raw_interactions_path: Path = DEFAULT_RAW_INTERACTIONS_PATH) -> list[str]:
    if not raw_interactions_path.exists():
        raise FileNotFoundError(f"Raw interactions file not found: {raw_interactions_path}")

    with raw_interactions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {raw_interactions_path}")
        if "bacteria" not in reader.fieldnames:
            raise ValueError(f"Missing 'bacteria' column in {raw_interactions_path}")

        bacteria_ids: set[str] = set()
        for row in reader:
            bacteria = (row.get("bacteria") or "").strip()
            if bacteria:
                bacteria_ids.add(bacteria)

    if not bacteria_ids:
        raise ValueError(f"No bacteria IDs found in {raw_interactions_path}")

    return sorted(bacteria_ids)


def _download_zip_file(download_url: str, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(download_url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        destination_path.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)


def _safe_extract_zip(zip_path: Path, destination_dir: Path) -> None:
    destination_root = destination_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError(f"Zip archive was empty: {zip_path}")
        for member in members:
            if member.is_dir():
                continue
            target_path = (destination_dir / member.filename).resolve()
            if not target_path.is_relative_to(destination_root):
                raise ValueError(f"Unsafe path in archive {zip_path}: {member.filename}")
        archive.extractall(destination_dir)


def _remove_existing_fastas(assembly_dir: Path) -> None:
    for fasta_path in list_fasta_files(assembly_dir):
        fasta_path.unlink()


def validate_picard_assemblies(
    assembly_dir: Path,
    st02_bacteria_ids: Sequence[str],
    *,
    expected_assembly_count: int = EXPECTED_ASSEMBLY_COUNT,
) -> None:
    fasta_paths = list_fasta_files(assembly_dir)
    fasta_stems = {path.stem for path in fasta_paths}
    missing_ids = sorted(set(st02_bacteria_ids) - fasta_stems)
    if missing_ids:
        raise ValueError("Missing Picard assemblies for ST02 bacteria: " + ", ".join(missing_ids))

    if len(fasta_paths) != expected_assembly_count:
        raise ValueError(f"Expected {expected_assembly_count} FASTA files in {assembly_dir}, found {len(fasta_paths)}.")


def download_picard_assemblies(
    assembly_dir: Path = DEFAULT_ASSEMBLY_DIR,
    *,
    raw_interactions_path: Path = DEFAULT_RAW_INTERACTIONS_PATH,
    download_url: str = FIGSHARE_DOWNLOAD_URL,
    expected_assembly_count: int = EXPECTED_ASSEMBLY_COUNT,
) -> Path:
    """Ensure the Picard collection assemblies are present and match the ST02 bacteria set."""

    assembly_dir.mkdir(parents=True, exist_ok=True)
    st02_bacteria_ids = load_st02_bacteria_ids(raw_interactions_path)
    fasta_paths = list_fasta_files(assembly_dir)

    if len(fasta_paths) != expected_assembly_count:
        LOGGER.info("Starting Picard assembly download from %s into %s", download_url, assembly_dir)
        _remove_existing_fastas(assembly_dir)
        SCRATCH_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=SCRATCH_DOWNLOAD_DIR, suffix=".zip", delete=False) as handle:
            zip_path = Path(handle.name)
        try:
            _download_zip_file(download_url, zip_path)
            LOGGER.info("Finished downloading Picard assembly archive to %s", zip_path)
            LOGGER.info("Starting extraction of Picard assemblies into %s", assembly_dir)
            _safe_extract_zip(zip_path, assembly_dir)
            LOGGER.info("Finished extraction of Picard assemblies into %s", assembly_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()

    validate_picard_assemblies(
        assembly_dir,
        st02_bacteria_ids,
        expected_assembly_count=expected_assembly_count,
    )
    return assembly_dir


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    download_picard_assemblies(
        assembly_dir=args.assembly_dir,
        raw_interactions_path=args.raw_interactions_path,
        download_url=args.download_url,
        expected_assembly_count=args.expected_assembly_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
