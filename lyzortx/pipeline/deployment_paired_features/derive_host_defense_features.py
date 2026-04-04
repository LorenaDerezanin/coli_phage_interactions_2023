#!/usr/bin/env python3
"""DEPLOY02: derive host-defense gene-count features from raw assemblies."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import build_defense_column_mask
from lyzortx.pipeline.track_l.steps.run_novel_host_defense_finder import (
    DEFAULT_MODELS_DIR,
    MODEL_INSTALL_MODE_ENSURE,
    build_defense_subtype_count_row,
    resolve_defense_finder_model_status,
    run_defense_finder_on_assembly,
    _read_defense_finder_system_rows,
    _read_panel_defense_rows,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH = Path(
    "data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv"
)
DEFAULT_VALIDATION_FASTAS_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_defense")
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
VALIDATION_COUNTS_FILENAME = "validation_host_defense_gene_counts.csv"
VALIDATION_REPORT_FILENAME = "validation_disagreement_report.json"
PER_HOST_COUNTS_FILENAME = "host_defense_gene_counts.csv"
VALIDATION_HOSTS: tuple[str, ...] = ("55989", "EDL933", "LF82")
STRING_DTYPE = "string"
INTEGER_DTYPE = "int64"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly_path", nargs="?", type=Path, help="Assembly FASTA for one host strain.")
    parser.add_argument("--bacteria-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-defense-subtypes-path", type=Path, default=DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force-model-update", action="store_true")
    parser.add_argument(
        "--model-install-mode",
        choices=(MODEL_INSTALL_MODE_ENSURE, "forbid"),
        default=MODEL_INSTALL_MODE_ENSURE,
        help="Whether to install pinned release models when missing or require a preinstalled pinned models directory.",
    )
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--preserve-raw", action="store_true")
    parser.add_argument(
        "--run-validation-subset",
        action="store_true",
        help="Run the committed validation hosts (55989, EDL933, LF82) instead of a single assembly.",
    )
    parser.add_argument("--validation-fastas-dir", type=Path, default=DEFAULT_VALIDATION_FASTAS_DIR)
    return parser.parse_args(argv)


def _parse_nonnegative_int(value: object) -> int:
    normalized = str(value).strip()
    if normalized in {"", "0", "0.0"}:
        return 0
    parsed = float(normalized)
    if parsed < 0:
        raise ValueError(f"Expected a non-negative defense count, found {value!r}")
    if not parsed.is_integer():
        raise ValueError(f"Expected an integer-valued defense count, found {value!r}")
    return int(parsed)


def build_host_defense_schema(
    panel_rows: Sequence[Mapping[str, str]],
    *,
    min_present_count: int = 5,
    max_present_count: int = 395,
) -> dict[str, Any]:
    mask = build_defense_column_mask(
        panel_rows,
        min_present_count=min_present_count,
        max_present_count=max_present_count,
    )
    retained_subtype_columns = list(mask["retained_subtype_columns"])
    columns = [{"name": "bacteria", "dtype": STRING_DTYPE}]
    columns.extend({"name": column, "dtype": INTEGER_DTYPE} for column in retained_subtype_columns)
    return {
        "feature_block": "host_defense",
        "key_column": "bacteria",
        "column_count": len(columns),
        "columns": columns,
        "retained_subtype_columns": retained_subtype_columns,
        "retained_subtype_count": len(retained_subtype_columns),
        "source_subtype_count": len(mask["source_subtype_columns"]),
        "source_subtype_columns": list(mask["source_subtype_columns"]),
        "derived_columns_dropped": [
            "host_defense_has_crispr",
            "host_defense_diversity",
            "host_defense_abi_burden",
        ],
        "support_counts": dict(mask["support_counts"]),
    }


def _column_names_from_schema(schema: Mapping[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _retained_subtype_columns(schema: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in schema["retained_subtype_columns"]]


def _write_single_row_csv(path: Path, row: Mapping[str, object], *, delimiter: str = ",") -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def derive_host_defense_features(
    assembly_path: Path,
    *,
    bacteria_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    panel_defense_subtypes_path: Path = DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    workers: int = 0,
    force_model_update: bool = False,
    model_install_mode: str = MODEL_INSTALL_MODE_ENSURE,
    force_run: bool = False,
    preserve_raw: bool = False,
) -> dict[str, Any]:
    if not assembly_path.exists():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_path}")

    resolved_bacteria_id = bacteria_id or assembly_path.stem
    panel_rows = _read_panel_defense_rows(panel_defense_subtypes_path)
    schema = build_host_defense_schema(panel_rows)
    subtype_columns = _retained_subtype_columns(schema)

    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)

    model_status = resolve_defense_finder_model_status(
        models_dir=models_dir,
        force_update=force_model_update,
        model_install_mode=model_install_mode,
    )
    systems_path, protein_metadata = run_defense_finder_on_assembly(
        assembly_path,
        output_dir=output_dir,
        models_dir=models_dir,
        workers=workers,
        preserve_raw=preserve_raw,
        force_run=force_run,
    )
    system_rows = _read_defense_finder_system_rows(systems_path)
    feature_row, matched_detected_subtypes, unmatched_detected_subtypes = build_defense_subtype_count_row(
        bacteria_id=resolved_bacteria_id,
        system_rows=system_rows,
        source_subtype_columns=subtype_columns,
    )

    counts_output_path = output_dir / PER_HOST_COUNTS_FILENAME
    _write_single_row_csv(counts_output_path, feature_row)

    manifest = {
        "step_name": "derive_host_defense_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assembly_path": str(assembly_path),
            "panel_defense_subtypes_path": str(panel_defense_subtypes_path),
            "models_dir": str(models_dir),
            "schema_manifest_path": str(output_dir / SCHEMA_MANIFEST_FILENAME),
        },
        "outputs": {
            "systems_tsv": str(systems_path),
            "feature_counts_csv": str(counts_output_path),
        },
        "counts": {
            "retained_subtype_count": len(subtype_columns),
            "detected_system_count": len(system_rows),
            "matched_panel_subtype_system_count": sum(matched_detected_subtypes.values()),
            "matched_panel_subtype_count": len(matched_detected_subtypes),
            "unmatched_detected_system_count": sum(unmatched_detected_subtypes.values()),
            "nonzero_feature_count": sum(int(feature_row[column] > 0) for column in subtype_columns),
            "predicted_cds_count": protein_metadata["predicted_cds_count"],
            "assembly_replicon_count": protein_metadata["replicon_count"],
            "assembly_nt_count": protein_metadata["genome_nt_count"],
        },
        "matched_detected_subtypes": matched_detected_subtypes,
        "unmatched_detected_subtypes": unmatched_detected_subtypes,
        "provenance": {
            "model_status": model_status,
            "model_install_mode": model_install_mode,
            "used_cached_systems": protein_metadata["used_cached_systems"],
            "gene_finder_modes": protein_metadata["gene_finder_modes"],
            "workers": workers,
            "preserve_raw": preserve_raw,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return {
        "schema": schema,
        "feature_row": feature_row,
        "manifest": manifest,
    }


def compare_host_defense_to_panel(
    derived_row: Mapping[str, object],
    panel_row: Mapping[str, str],
    *,
    subtype_columns: Sequence[str],
) -> dict[str, Any]:
    systems_gained: list[dict[str, int | str]] = []
    systems_lost: list[dict[str, int | str]] = []
    count_changes: list[dict[str, int | str]] = []
    exact_match_count = 0

    for subtype in subtype_columns:
        derived_count = _parse_nonnegative_int(derived_row.get(subtype, 0))
        panel_count = _parse_nonnegative_int(panel_row.get(subtype, 0))
        if derived_count == panel_count:
            exact_match_count += 1
            continue
        if panel_count == 0 and derived_count > 0:
            systems_gained.append({"subtype": subtype, "panel_count": panel_count, "derived_count": derived_count})
            continue
        if panel_count > 0 and derived_count == 0:
            systems_lost.append({"subtype": subtype, "panel_count": panel_count, "derived_count": derived_count})
            continue
        count_changes.append(
            {
                "subtype": subtype,
                "panel_count": panel_count,
                "derived_count": derived_count,
                "delta": derived_count - panel_count,
            }
        )

    return {
        "bacteria": str(derived_row["bacteria"]),
        "systems_gained": systems_gained,
        "systems_lost": systems_lost,
        "count_changes": count_changes,
        "exact_match_subtype_count": exact_match_count,
        "disagreement_subtype_count": len(systems_gained) + len(systems_lost) + len(count_changes),
    }


def run_validation_subset(
    *,
    validation_fastas_dir: Path = DEFAULT_VALIDATION_FASTAS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    panel_defense_subtypes_path: Path = DEFAULT_PANEL_DEFENSE_SUBTYPES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    workers: int = 0,
    force_model_update: bool = False,
    model_install_mode: str = MODEL_INSTALL_MODE_ENSURE,
    force_run: bool = False,
    preserve_raw: bool = False,
) -> dict[str, Any]:
    panel_rows = _read_panel_defense_rows(panel_defense_subtypes_path)
    panel_by_bacteria = {row["bacteria"]: row for row in panel_rows}
    schema = build_host_defense_schema(panel_rows)
    subtype_columns = _retained_subtype_columns(schema)

    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)

    derived_rows: list[dict[str, object]] = []
    reports: list[dict[str, Any]] = []

    for host in VALIDATION_HOSTS:
        assembly_path = validation_fastas_dir / f"{host}.fasta"
        if not assembly_path.exists():
            raise FileNotFoundError(f"Validation FASTA not found: {assembly_path}")
        if host not in panel_by_bacteria:
            raise ValueError(f"Validation host {host!r} missing from panel defense annotations")
        host_result = derive_host_defense_features(
            assembly_path,
            bacteria_id=host,
            output_dir=output_dir / host,
            panel_defense_subtypes_path=panel_defense_subtypes_path,
            models_dir=models_dir,
            workers=workers,
            force_model_update=force_model_update,
            model_install_mode=model_install_mode,
            force_run=force_run,
            preserve_raw=preserve_raw,
        )
        derived_row = dict(host_result["feature_row"])
        derived_rows.append(derived_row)
        reports.append(
            {
                **compare_host_defense_to_panel(
                    derived_row,
                    panel_by_bacteria[host],
                    subtype_columns=subtype_columns,
                ),
                "matched_detected_subtypes": dict(host_result["manifest"]["matched_detected_subtypes"]),
                "unmatched_detected_subtypes": dict(host_result["manifest"]["unmatched_detected_subtypes"]),
            }
        )

    write_csv(output_dir / VALIDATION_COUNTS_FILENAME, _column_names_from_schema(schema), derived_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_defense_subtypes_path": str(panel_defense_subtypes_path),
        "schema_manifest_path": str(output_dir / SCHEMA_MANIFEST_FILENAME),
        "validation_fastas_dir": str(validation_fastas_dir),
        "comparison_scope": {
            "retained_subtype_count": len(subtype_columns),
            "retained_subtype_columns": subtype_columns,
            "derived_columns_dropped": list(schema["derived_columns_dropped"]),
        },
        "host_reports": reports,
        "average_disagreement_systems_per_host": round(
            sum(report["disagreement_subtype_count"] for report in reports) / len(reports),
            3,
        ),
        "gate_threshold_average_systems": 3.0,
    }
    write_json(output_dir / VALIDATION_REPORT_FILENAME, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    if args.run_validation_subset:
        summary = run_validation_subset(
            validation_fastas_dir=args.validation_fastas_dir,
            output_dir=args.output_dir,
            panel_defense_subtypes_path=args.panel_defense_subtypes_path,
            models_dir=args.models_dir,
            workers=args.workers,
            force_model_update=args.force_model_update,
            model_install_mode=args.model_install_mode,
            force_run=args.force_run,
            preserve_raw=args.preserve_raw,
        )
        LOGGER.info(
            "Validation subset complete: average disagreement %.3f systems/host",
            summary["average_disagreement_systems_per_host"],
        )
        return 0

    if args.assembly_path is None:
        raise SystemExit("assembly_path is required unless --run-validation-subset is set")

    result = derive_host_defense_features(
        args.assembly_path,
        bacteria_id=args.bacteria_id,
        output_dir=args.output_dir,
        panel_defense_subtypes_path=args.panel_defense_subtypes_path,
        models_dir=args.models_dir,
        workers=args.workers,
        force_model_update=args.force_model_update,
        model_install_mode=args.model_install_mode,
        force_run=args.force_run,
        preserve_raw=args.preserve_raw,
    )
    LOGGER.info(
        "Derived host-defense features for %s with %d nonzero subtypes",
        result["feature_row"]["bacteria"],
        result["manifest"]["counts"]["nonzero_feature_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
