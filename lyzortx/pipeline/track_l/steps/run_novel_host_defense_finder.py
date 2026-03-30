#!/usr/bin/env python3
"""Run Defense Finder on a novel E. coli assembly and project the host-defense vector."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import joblib
import pyrodigal

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import (
    DEFENSE_SUBTYPE_MASK_NAME,
    build_defense_column_mask,
)
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import MIN_SINGLE_GENOME_TRAINING_NT, read_fasta_records
from lyzortx.pipeline.track_l.steps.novel_organism_feature_projection import (
    EXPECTED_DEFENSE_FEATURE_KEY,
    EXPECTED_DEFENSE_SOURCE_KEY,
    project_novel_host,
)

logger = logging.getLogger(__name__)

DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH = Path(
    "data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"
)
DEFAULT_COLUMN_MASK_PATH = (
    Path("lyzortx/generated_outputs/track_c/v1_host_feature_pair_table") / DEFENSE_SUBTYPE_MASK_NAME
)
DEFAULT_MODELS_DIR = Path(".scratch/defense_finder_models")
DEFAULT_OUTPUT_ROOT = Path("lyzortx/generated_outputs/track_l/novel_host_defense_finder")
RAW_SUBTYPE_COUNTS_FILENAME = "defense_finder_subtype_counts.csv"
PROJECTED_FEATURES_FILENAME = "novel_host_defense_features.csv"
MANIFEST_FILENAME = "novel_host_defense_manifest.json"
SYSTEMS_SUFFIX = "_defense_finder_systems.tsv"
DEFENSE_FINDER_COLUMNS: Tuple[str, ...] = ("sys_id", "type", "subtype", "activity")
PROTEIN_FASTA_SUFFIX = ".prt"
PINNED_MODEL_REQUIREMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("mdmparis", "defense-finder-models", "2.0.2"),
    ("macsy-models", "CasFinder", "3.1.0"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assembly_path",
        type=Path,
        help="Genome assembly FASTA for one novel E. coli strain.",
    )
    parser.add_argument(
        "--bacteria-id",
        type=str,
        default=None,
        help="Identifier to store in the output row. Defaults to the assembly stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Per-strain output directory. Defaults to lyzortx/generated_outputs/track_l/novel_host_defense_finder/<stem>/.",
    )
    parser.add_argument(
        "--column-mask-path",
        type=Path,
        default=DEFAULT_COLUMN_MASK_PATH,
        help="TL06 defense subtype mask joblib. Rebuilt from the panel subtype CSV if missing.",
    )
    parser.add_argument(
        "--panel-defense-subtypes-path",
        type=Path,
        default=DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
        help="Panel defense subtype CSV used to rebuild the TL06 mask when the joblib is absent.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory holding Defense Finder models. Missing models are downloaded here automatically.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers passed to defense-finder run. Default 0 lets Defense Finder use all available cores.",
    )
    parser.add_argument(
        "--force-model-update",
        action="store_true",
        help="Re-download Defense Finder models even if they already exist.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run Defense Finder even when the systems TSV already exists in the output directory.",
    )
    parser.add_argument(
        "--preserve-raw",
        action="store_true",
        help="Preserve raw MacSyFinder outputs produced by Defense Finder.",
    )
    return parser.parse_args(argv)


def _tool_bin(name: str) -> Path:
    candidate = Path(sys.executable).resolve().parent / name
    if not candidate.exists():
        raise FileNotFoundError(f"Expected tool {name!r} next to {sys.executable}, but {candidate} does not exist.")
    return candidate


def _tool_env() -> Dict[str, str]:
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run_command(command: Sequence[str], *, env: Mapping[str, str], description: str) -> None:
    logger.info("Starting %s", description)
    start_time = datetime.now(timezone.utc)
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=dict(env))
    elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    if result.returncode != 0:
        logger.error(
            "%s failed (exit %d, %.1fs)\nstdout: %s\nstderr: %s",
            description,
            result.returncode,
            elapsed_seconds,
            result.stdout[-4000:] if result.stdout else "",
            result.stderr[-4000:] if result.stderr else "",
        )
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    logger.info("Completed %s in %.1fs", description, elapsed_seconds)


def _read_panel_defense_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Panel defense subtype CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [{key: value for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No defense subtype rows found in {path}")
    return rows


def resolve_defense_mask(
    *,
    column_mask_path: Path,
    panel_defense_subtypes_path: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    if column_mask_path.exists():
        return column_mask_path, "existing_joblib"

    rows = _read_panel_defense_rows(panel_defense_subtypes_path)
    mask = build_defense_column_mask(rows)
    rebuilt_path = output_dir / DEFENSE_SUBTYPE_MASK_NAME
    ensure_directory(rebuilt_path.parent)
    joblib.dump(mask, rebuilt_path)
    return rebuilt_path, "rebuilt_from_panel_subtypes"


def _read_installed_model_version(models_dir: Path, package_name: str) -> str | None:
    metadata_path = models_dir / package_name / "metadata.yml"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("vers:"):
                return line.split(":", maxsplit=1)[1].strip()
    return None


def ensure_defense_finder_models(
    *,
    models_dir: Path,
    force_update: bool,
) -> str:
    ensure_directory(models_dir)
    tool_env = _tool_env()
    install_required = force_update
    for _, package_name, expected_version in PINNED_MODEL_REQUIREMENTS:
        if _read_installed_model_version(models_dir, package_name) != expected_version:
            install_required = True
            break

    if not install_required:
        logger.info("Defense Finder models already pinned correctly in %s", models_dir)
        return "existing_pinned"

    for org, package_name, expected_version in PINNED_MODEL_REQUIREMENTS:
        requirement = f"{package_name}=={expected_version}"
        command = [
            str(_tool_bin("macsydata")),
            "install",
            "--models-dir",
            str(models_dir),
            "--org",
            org,
            "--force",
            requirement,
        ]
        _run_command(command, env=tool_env, description=f"Model install {requirement} from {org}")
    return "installed_pinned"


def _read_defense_finder_system_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Defense Finder systems TSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing_columns = [column for column in DEFENSE_FINDER_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{path} is missing required Defense Finder columns: {', '.join(missing_columns)}")
        rows = [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]
    return rows


def resolve_detected_subtype(
    system_row: Mapping[str, str],
    *,
    allowed_subtypes: set[str],
) -> str | None:
    type_name = str(system_row.get("type", "")).strip()
    subtype_name = str(system_row.get("subtype", "")).strip()
    candidates = [
        subtype_name,
        type_name,
        f"{type_name}_{subtype_name}" if type_name and subtype_name else "",
        f"{type_name}-{subtype_name}" if type_name and subtype_name else "",
    ]
    for candidate in candidates:
        if candidate and candidate in allowed_subtypes:
            return candidate
    return None


def build_defense_subtype_count_row(
    *,
    bacteria_id: str,
    system_rows: Sequence[Mapping[str, str]],
    source_subtype_columns: Sequence[str],
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    allowed_subtypes = set(source_subtype_columns)
    subtype_row: dict[str, object] = {"bacteria": bacteria_id}
    subtype_row.update({column: 0 for column in source_subtype_columns})
    matched = Counter()
    unmatched = Counter()

    for system_row in system_rows:
        subtype_name = resolve_detected_subtype(system_row, allowed_subtypes=allowed_subtypes)
        if subtype_name is None:
            unmatched_name = str(system_row.get("subtype") or system_row.get("type") or "UNKNOWN").strip() or "UNKNOWN"
            unmatched[unmatched_name] += 1
            continue
        subtype_row[subtype_name] = int(subtype_row[subtype_name]) + 1
        matched[subtype_name] += 1

    return subtype_row, dict(sorted(matched.items())), dict(sorted(unmatched.items()))


def _write_single_row_csv(path: Path, row: Mapping[str, object], *, delimiter: str) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def predict_proteins_with_pyrodigal(assembly_path: Path, *, protein_fasta_path: Path) -> dict[str, object]:
    records = read_fasta_records(assembly_path, protein=False)
    total_cds = 0
    total_nt = 0
    gene_finder_modes: set[str] = set()

    ensure_directory(protein_fasta_path.parent)
    with protein_fasta_path.open("w", encoding="utf-8") as handle:
        for record in records:
            sequence_bytes = record.sequence.encode("ascii")
            if len(sequence_bytes) < MIN_SINGLE_GENOME_TRAINING_NT:
                gene_finder = pyrodigal.GeneFinder(meta=True)
                genes = gene_finder.find_genes(sequence_bytes)
                gene_finder_modes.add("meta")
            else:
                gene_finder = pyrodigal.GeneFinder(meta=False)
                gene_finder.train(sequence_bytes)
                genes = gene_finder.find_genes(sequence_bytes)
                gene_finder_modes.add("single")
            genes.write_translations(handle, record.identifier)
            total_cds += len(genes)
            total_nt += len(record.sequence)

    if total_cds <= 0:
        raise ValueError(f"Pyrodigal did not predict any CDS for {assembly_path}")

    return {
        "protein_fasta_path": str(protein_fasta_path),
        "replicon_count": len(records),
        "genome_nt_count": total_nt,
        "predicted_cds_count": total_cds,
        "gene_finder_modes": sorted(gene_finder_modes),
        "used_cached_systems": False,
    }


def _cached_protein_metadata(protein_fasta_path: Path) -> dict[str, object]:
    return {
        "protein_fasta_path": str(protein_fasta_path),
        "replicon_count": None,
        "genome_nt_count": None,
        "predicted_cds_count": None,
        "gene_finder_modes": [],
        "used_cached_systems": True,
    }


def run_defense_finder_on_assembly(
    assembly_path: Path,
    *,
    output_dir: Path,
    models_dir: Path,
    workers: int,
    preserve_raw: bool,
    force_run: bool,
) -> tuple[Path, dict[str, object]]:
    systems_path = output_dir / f"{assembly_path.stem}{SYSTEMS_SUFFIX}"
    ensure_directory(output_dir)
    protein_fasta_path = output_dir / f"{assembly_path.stem}{PROTEIN_FASTA_SUFFIX}"
    if systems_path.exists() and not force_run:
        logger.info("Skipping Defense Finder and Pyrodigal because %s already exists", systems_path)
        return systems_path, _cached_protein_metadata(protein_fasta_path)

    logger.info("Starting Pyrodigal gene prediction for %s", assembly_path.name)
    pyrodigal_start = datetime.now(timezone.utc)
    protein_metadata = predict_proteins_with_pyrodigal(assembly_path, protein_fasta_path=protein_fasta_path)
    pyrodigal_elapsed = (datetime.now(timezone.utc) - pyrodigal_start).total_seconds()
    logger.info(
        "Completed Pyrodigal gene prediction for %s in %.1fs (%d CDS across %d replicons)",
        assembly_path.name,
        pyrodigal_elapsed,
        protein_metadata["predicted_cds_count"],
        protein_metadata["replicon_count"],
    )

    tool_env = _tool_env()
    command = [
        str(_tool_bin("defense-finder")),
        "run",
        str(protein_fasta_path),
        "--out-dir",
        str(output_dir),
        "--models-dir",
        str(models_dir),
        "--workers",
        str(workers),
        "--db-type",
        "gembase",
    ]
    if preserve_raw:
        command.append("--preserve-raw")
    _run_command(command, env=tool_env, description=f"Defense Finder on {assembly_path.name}")
    if not systems_path.exists():
        raise FileNotFoundError(f"Defense Finder completed but {systems_path} was not created.")
    return systems_path, protein_metadata


def _ordered_feature_row(
    projected_row: Mapping[str, object], ordered_feature_columns: Sequence[str]
) -> dict[str, object]:
    ordered_row: dict[str, object] = {"bacteria": projected_row["bacteria"]}
    for column in ordered_feature_columns:
        ordered_row[column] = projected_row[column]
    return ordered_row


def run_novel_host_defense_finder(
    assembly_path: Path,
    *,
    bacteria_id: str | None = None,
    output_dir: Path,
    column_mask_path: Path,
    panel_defense_subtypes_path: Path,
    models_dir: Path,
    workers: int,
    force_model_update: bool,
    force_run: bool,
    preserve_raw: bool,
) -> dict[str, object]:
    if not assembly_path.exists():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_path}")

    resolved_bacteria_id = bacteria_id or assembly_path.stem
    ensure_directory(output_dir)
    model_status = ensure_defense_finder_models(models_dir=models_dir, force_update=force_model_update)
    resolved_mask_path, mask_status = resolve_defense_mask(
        column_mask_path=column_mask_path,
        panel_defense_subtypes_path=panel_defense_subtypes_path,
        output_dir=output_dir,
    )

    systems_path, protein_metadata = run_defense_finder_on_assembly(
        assembly_path,
        output_dir=output_dir,
        models_dir=models_dir,
        workers=workers,
        preserve_raw=preserve_raw,
        force_run=force_run,
    )

    mask = joblib.load(resolved_mask_path)
    source_subtype_columns = list(mask[EXPECTED_DEFENSE_SOURCE_KEY])
    ordered_feature_columns = list(mask[EXPECTED_DEFENSE_FEATURE_KEY])
    system_rows = _read_defense_finder_system_rows(systems_path)
    subtype_row, matched_detected_subtypes, unmatched_detected_subtypes = build_defense_subtype_count_row(
        bacteria_id=resolved_bacteria_id,
        system_rows=system_rows,
        source_subtype_columns=source_subtype_columns,
    )

    raw_subtype_counts_path = output_dir / RAW_SUBTYPE_COUNTS_FILENAME
    _write_single_row_csv(raw_subtype_counts_path, subtype_row, delimiter=";")

    projected_row = project_novel_host(raw_subtype_counts_path, resolved_mask_path)
    projected_feature_path = output_dir / PROJECTED_FEATURES_FILENAME
    write_csv(
        projected_feature_path,
        ["bacteria", *ordered_feature_columns],
        [_ordered_feature_row(projected_row, ordered_feature_columns)],
    )

    manifest = {
        "step_name": "run_novel_host_defense_finder",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assembly_path": str(assembly_path),
            "requested_column_mask_path": str(column_mask_path),
            "resolved_column_mask_path": str(resolved_mask_path),
            "panel_defense_subtypes_path": str(panel_defense_subtypes_path),
            "models_dir": str(models_dir),
        },
        "outputs": {
            "predicted_protein_fasta": protein_metadata["protein_fasta_path"],
            "systems_tsv": str(systems_path),
            "raw_subtype_counts_csv": str(raw_subtype_counts_path),
            "projected_feature_csv": str(projected_feature_path),
        },
        "counts": {
            "assembly_replicon_count": protein_metadata["replicon_count"],
            "assembly_nt_count": protein_metadata["genome_nt_count"],
            "predicted_cds_count": protein_metadata["predicted_cds_count"],
            "detected_system_count": len(system_rows),
            "matched_training_subtype_system_count": sum(matched_detected_subtypes.values()),
            "matched_training_subtype_count": len(matched_detected_subtypes),
            "unmatched_detected_system_count": sum(unmatched_detected_subtypes.values()),
            "projected_feature_count": len(ordered_feature_columns),
            "projected_nonzero_feature_count": sum(
                int(float(projected_row[column]) > 0.0) for column in ordered_feature_columns
            ),
        },
        "matched_detected_subtypes": matched_detected_subtypes,
        "unmatched_detected_subtypes": unmatched_detected_subtypes,
        "provenance": {
            "gene_finder_modes": protein_metadata["gene_finder_modes"],
            "used_cached_systems": protein_metadata["used_cached_systems"],
            "model_status": model_status,
            "mask_status": mask_status,
            "workers": workers,
            "preserve_raw": preserve_raw,
        },
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / (args.bacteria_id or args.assembly_path.stem))
    manifest = run_novel_host_defense_finder(
        args.assembly_path,
        bacteria_id=args.bacteria_id,
        output_dir=output_dir,
        column_mask_path=args.column_mask_path,
        panel_defense_subtypes_path=args.panel_defense_subtypes_path,
        models_dir=args.models_dir,
        workers=args.workers,
        force_model_update=args.force_model_update,
        force_run=args.force_run,
        preserve_raw=args.preserve_raw,
    )
    logger.info(
        "Novel host defense projection complete: %d systems detected, %d projected features at %s",
        manifest["counts"]["detected_system_count"],
        manifest["counts"]["projected_feature_count"],
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
