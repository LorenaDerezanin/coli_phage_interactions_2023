#!/usr/bin/env python3
"""DEPLOY04: derive categorical host-typing features from raw assemblies."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_l.steps import build_host_typing_projector as tl16

LOGGER = logging.getLogger(__name__)

DEFAULT_VALIDATION_FASTAS_DIR = Path("data/genomics/bacteria/validation_subset/fastas")
DEFAULT_PANEL_METADATA_PATH = tl16.DEFAULT_PANEL_METADATA_PATH
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_typing")
SCHEMA_MANIFEST_FILENAME = "schema_manifest.json"
PER_HOST_FEATURES_FILENAME = "host_typing_features.csv"
VALIDATION_FEATURES_FILENAME = "validation_host_typing_features.csv"
VALIDATION_REPORT_FILENAME = "validation_report.json"
STRING_DTYPE = "string"
VALIDATION_HOSTS: tuple[str, ...] = ("55989", "EDL933", "LF82")

HOST_TYPING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("bacteria", STRING_DTYPE),
    ("host_clermont_phylo", STRING_DTYPE),
    ("host_st_warwick", STRING_DTYPE),
    ("host_o_type", STRING_DTYPE),
    ("host_h_type", STRING_DTYPE),
    ("host_serotype", STRING_DTYPE),
)


@dataclass(frozen=True)
class HostTypingRuntimeOutputs:
    phylogroup_report_path: Path
    serotype_output_path: Path
    mlst_output_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly_path", nargs="?", type=Path, help="Assembly FASTA for one host strain.")
    parser.add_argument("--bacteria-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--picard-metadata-path", type=Path, default=DEFAULT_PANEL_METADATA_PATH)
    parser.add_argument(
        "--run-validation-subset",
        action="store_true",
        help="Run the committed validation hosts (55989, EDL933, LF82) instead of a single assembly.",
    )
    parser.add_argument("--validation-fastas-dir", type=Path, default=DEFAULT_VALIDATION_FASTAS_DIR)
    return parser.parse_args(argv)


def build_host_typing_schema() -> dict[str, Any]:
    columns = [{"name": name, "dtype": dtype} for name, dtype in HOST_TYPING_COLUMNS]
    return {
        "feature_block": "host_typing",
        "key_column": "bacteria",
        "column_count": len(columns),
        "columns": columns,
        "categorical_columns": [name for name, _ in HOST_TYPING_COLUMNS if name != "bacteria"],
        "caller_envs": {
            "phylogroup": tl16.PHYLOGROUP_ENV_NAME,
            "serotype": tl16.SEROTYPE_ENV_NAME,
            "sequence_type": tl16.SEQUENCE_TYPE_ENV_NAME,
        },
    }


def _column_names_from_schema(schema: Mapping[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema["columns"]]


def _write_single_row_csv(path: Path, row: Mapping[str, object], *, delimiter: str = ",") -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)


def build_host_typing_feature_row(
    *,
    bacteria: str,
    phylogroup_call: Mapping[str, str],
    serotype_call: Mapping[str, str],
    mlst_call: Mapping[str, str],
) -> dict[str, object]:
    o_type = tl16.normalize_text(serotype_call.get("o_type", ""))
    h_type = tl16.normalize_text(serotype_call.get("h_type", ""))
    return {
        "bacteria": bacteria,
        "host_clermont_phylo": tl16.normalize_text(phylogroup_call.get("phylogroup", "")),
        "host_st_warwick": tl16.normalize_text(mlst_call.get("st_warwick", "")),
        "host_o_type": o_type,
        "host_h_type": h_type,
        "host_serotype": tl16.derive_serotype(o_type, h_type),
    }


def compare_host_typing_to_panel(
    derived_row: Mapping[str, object],
    panel_row: Mapping[str, str],
) -> dict[str, Any]:
    panel_phylogroup = tl16.normalize_text(panel_row.get("Clermont_Phylo", ""))
    panel_o_type = tl16.normalize_text(panel_row.get("O-type", ""))
    panel_h_type = tl16.normalize_text(panel_row.get("H-type", ""))
    panel_st_warwick = tl16.normalize_text(panel_row.get("ST_Warwick", ""))
    panel_serotype = tl16.derive_serotype(panel_o_type, panel_h_type)
    derived_phylogroup = str(derived_row["host_clermont_phylo"])
    derived_o_type = str(derived_row["host_o_type"])
    derived_h_type = str(derived_row["host_h_type"])
    derived_st_warwick = str(derived_row["host_st_warwick"])
    derived_serotype = str(derived_row["host_serotype"])
    field_matches = {
        "phylogroup": derived_phylogroup == panel_phylogroup,
        "o_type": derived_o_type == panel_o_type,
        "h_type": derived_h_type == panel_h_type,
        "st_warwick": derived_st_warwick == panel_st_warwick,
        "serotype": derived_serotype == panel_serotype,
    }
    resolved_field_count = sum(
        1
        for value in (
            panel_row.get("Clermont_Phylo"),
            panel_row.get("O-type"),
            panel_row.get("H-type"),
            panel_row.get("ST_Warwick"),
        )
        if tl16.normalize_text(value)
    )
    return {
        "bacteria": str(derived_row["bacteria"]),
        "panel_values": {
            "phylogroup": panel_phylogroup,
            "o_type": panel_o_type,
            "h_type": panel_h_type,
            "st_warwick": panel_st_warwick,
            "serotype": panel_serotype,
        },
        "derived_values": {
            "phylogroup": derived_phylogroup,
            "o_type": derived_o_type,
            "h_type": derived_h_type,
            "st_warwick": derived_st_warwick,
            "serotype": derived_serotype,
        },
        "field_matches": field_matches,
        "exact_match_field_count": sum(int(match) for match in field_matches.values()),
        "resolved_field_count": resolved_field_count,
    }


def _validate_host_metadata_row(panel_metadata: Mapping[str, Mapping[str, str]], bacteria: str) -> Mapping[str, str]:
    if bacteria not in panel_metadata:
        raise ValueError(f"Validation host {bacteria!r} missing from Picard metadata")
    return panel_metadata[bacteria]


def derive_host_typing_features(
    assembly_path: Path,
    *,
    bacteria_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    picard_metadata_path: Path = DEFAULT_PANEL_METADATA_PATH,
) -> dict[str, Any]:
    if not assembly_path.exists():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_path}")

    resolved_bacteria_id = bacteria_id or assembly_path.stem
    panel_metadata = tl16.load_panel_metadata(picard_metadata_path)
    schema = build_host_typing_schema()
    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)

    phylogroup_report_path = tl16.run_phylogroup_caller(
        bacteria=resolved_bacteria_id,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )
    serotype_output_path, _ = tl16.run_serotype_caller(
        bacteria=resolved_bacteria_id,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )
    mlst_output_path = tl16.run_sequence_type_caller(
        bacteria=resolved_bacteria_id,
        assembly_path=assembly_path,
        output_dir=output_dir,
        force=False,
    )

    outputs = HostTypingRuntimeOutputs(
        phylogroup_report_path=phylogroup_report_path,
        serotype_output_path=serotype_output_path,
        mlst_output_path=mlst_output_path,
    )
    feature_row = build_host_typing_feature_row(
        bacteria=resolved_bacteria_id,
        phylogroup_call=tl16.parse_phylogroup_report(outputs.phylogroup_report_path),
        serotype_call=tl16.parse_ectyper_output(outputs.serotype_output_path),
        mlst_call=tl16.parse_mlst_legacy_output(outputs.mlst_output_path),
    )

    feature_csv_path = output_dir / PER_HOST_FEATURES_FILENAME
    _write_single_row_csv(feature_csv_path, feature_row)

    panel_row = _validate_host_metadata_row(panel_metadata, resolved_bacteria_id)
    comparison = compare_host_typing_to_panel(feature_row, panel_row)

    manifest = {
        "step_name": "derive_host_typing_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assembly_path": str(assembly_path),
            "picard_metadata_path": str(picard_metadata_path),
        },
        "outputs": {
            "phylogroup_report_path": str(outputs.phylogroup_report_path),
            "serotype_output_path": str(outputs.serotype_output_path),
            "mlst_output_path": str(outputs.mlst_output_path),
            "feature_csv_path": str(feature_csv_path),
        },
        "comparison": comparison,
    }
    write_json(output_dir / "manifest.json", manifest)
    return {
        "schema": schema,
        "feature_row": feature_row,
        "comparison": comparison,
        "manifest": manifest,
    }


def run_validation_subset(
    *,
    validation_fastas_dir: Path = DEFAULT_VALIDATION_FASTAS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    picard_metadata_path: Path = DEFAULT_PANEL_METADATA_PATH,
) -> dict[str, Any]:
    panel_metadata = tl16.load_panel_metadata(picard_metadata_path)
    schema = build_host_typing_schema()
    ensure_directory(output_dir)
    write_json(output_dir / SCHEMA_MANIFEST_FILENAME, schema)

    derived_rows: list[dict[str, object]] = []
    host_reports: list[dict[str, Any]] = []
    per_field_exact_match_counts = {"phylogroup": 0, "o_type": 0, "h_type": 0, "st_warwick": 0, "serotype": 0}

    for host in VALIDATION_HOSTS:
        assembly_path = validation_fastas_dir / f"{host}.fasta"
        if not assembly_path.exists():
            raise FileNotFoundError(f"Validation FASTA not found: {assembly_path}")
        panel_row = _validate_host_metadata_row(panel_metadata, host)
        host_result = derive_host_typing_features(
            assembly_path,
            bacteria_id=host,
            output_dir=output_dir / host,
            picard_metadata_path=picard_metadata_path,
        )
        derived_row = dict(host_result["feature_row"])
        derived_rows.append(derived_row)
        comparison = compare_host_typing_to_panel(derived_row, panel_row)
        host_reports.append(comparison)
        for field_name, matched in comparison["field_matches"].items():
            per_field_exact_match_counts[field_name] += int(bool(matched))

    write_csv(output_dir / VALIDATION_FEATURES_FILENAME, _column_names_from_schema(schema), derived_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "picard_metadata_path": str(picard_metadata_path),
        "validation_fastas_dir": str(validation_fastas_dir),
        "schema_manifest_path": str(output_dir / SCHEMA_MANIFEST_FILENAME),
        "comparison_scope": {
            "fields": ["phylogroup", "o_type", "h_type", "st_warwick", "serotype"],
            "categorical_columns": list(schema["categorical_columns"]),
        },
        "host_reports": host_reports,
        "field_exact_match_counts": per_field_exact_match_counts,
        "host_count": len(host_reports),
    }
    write_json(output_dir / VALIDATION_REPORT_FILENAME, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    if args.run_validation_subset:
        run_validation_subset(
            validation_fastas_dir=args.validation_fastas_dir,
            output_dir=args.output_dir,
            picard_metadata_path=args.picard_metadata_path,
        )
        return 0
    if args.assembly_path is None:
        raise ValueError("assembly_path is required unless --run-validation-subset is set")
    derive_host_typing_features(
        args.assembly_path,
        bacteria_id=args.bacteria_id,
        output_dir=args.output_dir,
        picard_metadata_path=args.picard_metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
