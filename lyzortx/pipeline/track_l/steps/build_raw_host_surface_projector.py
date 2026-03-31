#!/usr/bin/env python3
"""TL15: Build a raw-host surface projector for deployable compatibility features."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import re
import sys
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_c.steps.build_receptor_surface_feature_block import (
    build_feature_rows as build_track_c_feature_rows,
    read_delimited_rows as read_track_c_rows,
)
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_fasta_records
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import sha256_file
from lyzortx.pipeline.track_l.steps.run_novel_host_defense_finder import (
    _run_command,
    _tool_bin,
    _tool_env,
    predict_proteins_with_pyrodigal,
)
from lyzortx.pipeline.track_l.steps.validate_vhdb_generalized_inference import (
    _download_binary,
    _download_text,
    _normalize_host_name,
    AssemblyRecord,
    choose_best_assembly,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/raw_host_surface_projector")
DEFAULT_PANEL_METADATA_PATH = Path("data/genomics/bacteria/picard_collection.csv")
DEFAULT_LPS_CORE_PATH = Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_host.txt")
DEFAULT_RECEPTOR_CLUSTER_PATH = Path(
    "data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"
)
DEFAULT_ASSEMBLY_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
DEFAULT_VERSION = "v1"
MMSEQS_FORMAT_COLUMNS = ("query", "target", "pident", "qcov", "tcov", "bits", "evalue")
MMSEQS_FORMAT_OUTPUT = ",".join(MMSEQS_FORMAT_COLUMNS)

FAMILY_DETECTION_MIN_PIDENT = 0.45
FAMILY_DETECTION_MIN_COVERAGE = 0.70
VARIANT_ASSIGNMENT_MIN_PIDENT = 0.90
VARIANT_ASSIGNMENT_MIN_COVERAGE = 0.90
AMBIGUOUS_BITS_DELTA = 1e-6

PANEL_COLLECTION_HOST = "Host"
ASSEMBLY_CATALOG_COLUMNS = (
    "bacteria",
    "gembase",
    "assembly_match_status",
    "match_reason",
    "assembly_accession",
    "assembly_level",
    "refseq_category",
    "assembly_ftp_path",
    "organism_name",
    "infraspecific_name",
    "isolate",
)
PROJECTION_COLUMNS = (
    "bacteria",
    "assembly_accession",
    "host_lps_core_present",
    "host_lps_core_type",
    "host_lps_core_call_status",
    "host_lps_core_best_reference_label",
    "host_lps_core_best_reference_pident",
    "host_lps_core_best_reference_qcov",
    "host_lps_core_best_reference_tcov",
    "host_receptor_btub_present",
    "host_receptor_btub_variant",
    "host_receptor_btub_call_status",
    "host_receptor_btub_best_reference_label",
    "host_receptor_btub_best_reference_pident",
    "host_receptor_btub_best_reference_qcov",
    "host_receptor_btub_best_reference_tcov",
    "host_receptor_fadL_present",
    "host_receptor_fadL_variant",
    "host_receptor_fadL_call_status",
    "host_receptor_fadL_best_reference_label",
    "host_receptor_fadL_best_reference_pident",
    "host_receptor_fadL_best_reference_qcov",
    "host_receptor_fadL_best_reference_tcov",
    "host_receptor_fhua_present",
    "host_receptor_fhua_variant",
    "host_receptor_fhua_call_status",
    "host_receptor_fhua_best_reference_label",
    "host_receptor_fhua_best_reference_pident",
    "host_receptor_fhua_best_reference_qcov",
    "host_receptor_fhua_best_reference_tcov",
    "host_receptor_lamB_present",
    "host_receptor_lamB_variant",
    "host_receptor_lamB_call_status",
    "host_receptor_lamB_best_reference_label",
    "host_receptor_lamB_best_reference_pident",
    "host_receptor_lamB_best_reference_qcov",
    "host_receptor_lamB_best_reference_tcov",
    "host_receptor_lptD_present",
    "host_receptor_lptD_variant",
    "host_receptor_lptD_call_status",
    "host_receptor_lptD_best_reference_label",
    "host_receptor_lptD_best_reference_pident",
    "host_receptor_lptD_best_reference_qcov",
    "host_receptor_lptD_best_reference_tcov",
    "host_receptor_nfrA_present",
    "host_receptor_nfrA_variant",
    "host_receptor_nfrA_call_status",
    "host_receptor_nfrA_best_reference_label",
    "host_receptor_nfrA_best_reference_pident",
    "host_receptor_nfrA_best_reference_qcov",
    "host_receptor_nfrA_best_reference_tcov",
    "host_receptor_ompA_present",
    "host_receptor_ompA_variant",
    "host_receptor_ompA_call_status",
    "host_receptor_ompA_best_reference_label",
    "host_receptor_ompA_best_reference_pident",
    "host_receptor_ompA_best_reference_qcov",
    "host_receptor_ompA_best_reference_tcov",
    "host_receptor_ompC_present",
    "host_receptor_ompC_variant",
    "host_receptor_ompC_call_status",
    "host_receptor_ompC_best_reference_label",
    "host_receptor_ompC_best_reference_pident",
    "host_receptor_ompC_best_reference_qcov",
    "host_receptor_ompC_best_reference_tcov",
    "host_receptor_ompF_present",
    "host_receptor_ompF_variant",
    "host_receptor_ompF_call_status",
    "host_receptor_ompF_best_reference_label",
    "host_receptor_ompF_best_reference_pident",
    "host_receptor_ompF_best_reference_qcov",
    "host_receptor_ompF_best_reference_tcov",
    "host_receptor_tolC_present",
    "host_receptor_tolC_variant",
    "host_receptor_tolC_call_status",
    "host_receptor_tolC_best_reference_label",
    "host_receptor_tolC_best_reference_pident",
    "host_receptor_tolC_best_reference_qcov",
    "host_receptor_tolC_best_reference_tcov",
    "host_receptor_tsx_present",
    "host_receptor_tsx_variant",
    "host_receptor_tsx_call_status",
    "host_receptor_tsx_best_reference_label",
    "host_receptor_tsx_best_reference_pident",
    "host_receptor_tsx_best_reference_qcov",
    "host_receptor_tsx_best_reference_tcov",
    "host_receptor_yncD_present",
    "host_receptor_yncD_variant",
    "host_receptor_yncD_call_status",
    "host_receptor_yncD_best_reference_label",
    "host_receptor_yncD_best_reference_pident",
    "host_receptor_yncD_best_reference_qcov",
    "host_receptor_yncD_best_reference_tcov",
)
AGREEMENT_COLUMNS = (
    "feature_family",
    "training_columns",
    "status",
    "panel_hosts_with_assemblies",
    "callable_count",
    "not_callable_count",
    "exact_match_count",
    "agreement_rate_on_callable",
)
SUPPORT_TABLE_COLUMNS = ("feature_family", "training_columns", "support_status", "projection_method", "rationale")
MISMATCH_COLUMNS = (
    "bacteria",
    "assembly_accession",
    "feature_family",
    "expected_present",
    "expected_label",
    "projected_present",
    "projected_label",
    "call_status",
    "best_reference_label",
)
REFERENCE_INDEX_COLUMNS = (
    "feature_family",
    "reference_label",
    "source_bacteria",
    "protein_identifier",
    "reference_fasta",
    "sha256",
)


@dataclass(frozen=True)
class SurfaceFamilyConfig:
    family_key: str
    display_name: str
    seed_gene_name: str
    output_present_column: str
    output_label_column: str
    output_status_column: str
    output_best_label_column: str
    output_best_pident_column: str
    output_best_qcov_column: str
    output_best_tcov_column: str
    support_status: str
    projection_method: str
    rationale: str


SUPPORTED_SURFACE_FAMILIES: tuple[SurfaceFamilyConfig, ...] = (
    SurfaceFamilyConfig(
        family_key="LPS_CORE",
        display_name="lps_core",
        seed_gene_name="waaL",
        output_present_column="host_lps_core_present",
        output_label_column="host_lps_core_type",
        output_status_column="host_lps_core_call_status",
        output_best_label_column="host_lps_core_best_reference_label",
        output_best_pident_column="host_lps_core_best_reference_pident",
        output_best_qcov_column="host_lps_core_best_reference_qcov",
        output_best_tcov_column="host_lps_core_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs seed search + waaL type representative matching.",
        rationale=(
            "The original Track C value comes from a curated waaL typing table; TL15 approximates it by matching the "
            "assembly's best waaL homolog to saved type representatives."
        ),
    ),
    SurfaceFamilyConfig(
        family_key="BTUB",
        display_name="receptor_btub",
        seed_gene_name="btuB",
        output_present_column="host_receptor_btub_present",
        output_label_column="host_receptor_btub_variant",
        output_status_column="host_receptor_btub_call_status",
        output_best_label_column="host_receptor_btub_best_reference_label",
        output_best_pident_column="host_receptor_btub_best_reference_pident",
        output_best_qcov_column="host_receptor_btub_best_reference_qcov",
        output_best_tcov_column="host_receptor_btub_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale=(
            "The original Track C variant label is a BLAST cluster ID; TL15 approximates it by matching the best "
            "BtuB homolog against saved panel-derived variant representatives."
        ),
    ),
    SurfaceFamilyConfig(
        family_key="FADL",
        display_name="receptor_fadL",
        seed_gene_name="fadL",
        output_present_column="host_receptor_fadL_present",
        output_label_column="host_receptor_fadL_variant",
        output_status_column="host_receptor_fadL_call_status",
        output_best_label_column="host_receptor_fadL_best_reference_label",
        output_best_pident_column="host_receptor_fadL_best_reference_pident",
        output_best_qcov_column="host_receptor_fadL_best_reference_qcov",
        output_best_tcov_column="host_receptor_fadL_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C FadL cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="FHUA",
        display_name="receptor_fhua",
        seed_gene_name="fhuA",
        output_present_column="host_receptor_fhua_present",
        output_label_column="host_receptor_fhua_variant",
        output_status_column="host_receptor_fhua_call_status",
        output_best_label_column="host_receptor_fhua_best_reference_label",
        output_best_pident_column="host_receptor_fhua_best_reference_pident",
        output_best_qcov_column="host_receptor_fhua_best_reference_qcov",
        output_best_tcov_column="host_receptor_fhua_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C FhuA cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="LAMB",
        display_name="receptor_lamB",
        seed_gene_name="lamB",
        output_present_column="host_receptor_lamB_present",
        output_label_column="host_receptor_lamB_variant",
        output_status_column="host_receptor_lamB_call_status",
        output_best_label_column="host_receptor_lamB_best_reference_label",
        output_best_pident_column="host_receptor_lamB_best_reference_pident",
        output_best_qcov_column="host_receptor_lamB_best_reference_qcov",
        output_best_tcov_column="host_receptor_lamB_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C LamB cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="LPTD",
        display_name="receptor_lptD",
        seed_gene_name="lptD",
        output_present_column="host_receptor_lptD_present",
        output_label_column="host_receptor_lptD_variant",
        output_status_column="host_receptor_lptD_call_status",
        output_best_label_column="host_receptor_lptD_best_reference_label",
        output_best_pident_column="host_receptor_lptD_best_reference_pident",
        output_best_qcov_column="host_receptor_lptD_best_reference_qcov",
        output_best_tcov_column="host_receptor_lptD_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C LptD cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="NFRA",
        display_name="receptor_nfrA",
        seed_gene_name="nfrA",
        output_present_column="host_receptor_nfrA_present",
        output_label_column="host_receptor_nfrA_variant",
        output_status_column="host_receptor_nfrA_call_status",
        output_best_label_column="host_receptor_nfrA_best_reference_label",
        output_best_pident_column="host_receptor_nfrA_best_reference_pident",
        output_best_qcov_column="host_receptor_nfrA_best_reference_qcov",
        output_best_tcov_column="host_receptor_nfrA_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C NfrA cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="OMPA",
        display_name="receptor_ompA",
        seed_gene_name="ompA",
        output_present_column="host_receptor_ompA_present",
        output_label_column="host_receptor_ompA_variant",
        output_status_column="host_receptor_ompA_call_status",
        output_best_label_column="host_receptor_ompA_best_reference_label",
        output_best_pident_column="host_receptor_ompA_best_reference_pident",
        output_best_qcov_column="host_receptor_ompA_best_reference_qcov",
        output_best_tcov_column="host_receptor_ompA_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C OmpA cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="OMPC",
        display_name="receptor_ompC",
        seed_gene_name="ompC",
        output_present_column="host_receptor_ompC_present",
        output_label_column="host_receptor_ompC_variant",
        output_status_column="host_receptor_ompC_call_status",
        output_best_label_column="host_receptor_ompC_best_reference_label",
        output_best_pident_column="host_receptor_ompC_best_reference_pident",
        output_best_qcov_column="host_receptor_ompC_best_reference_qcov",
        output_best_tcov_column="host_receptor_ompC_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C OmpC cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="OMPF",
        display_name="receptor_ompF",
        seed_gene_name="ompF",
        output_present_column="host_receptor_ompF_present",
        output_label_column="host_receptor_ompF_variant",
        output_status_column="host_receptor_ompF_call_status",
        output_best_label_column="host_receptor_ompF_best_reference_label",
        output_best_pident_column="host_receptor_ompF_best_reference_pident",
        output_best_qcov_column="host_receptor_ompF_best_reference_qcov",
        output_best_tcov_column="host_receptor_ompF_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C OmpF cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="TOLC",
        display_name="receptor_tolC",
        seed_gene_name="tolC",
        output_present_column="host_receptor_tolC_present",
        output_label_column="host_receptor_tolC_variant",
        output_status_column="host_receptor_tolC_call_status",
        output_best_label_column="host_receptor_tolC_best_reference_label",
        output_best_pident_column="host_receptor_tolC_best_reference_pident",
        output_best_qcov_column="host_receptor_tolC_best_reference_qcov",
        output_best_tcov_column="host_receptor_tolC_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C TolC cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="TSX",
        display_name="receptor_tsx",
        seed_gene_name="tsx",
        output_present_column="host_receptor_tsx_present",
        output_label_column="host_receptor_tsx_variant",
        output_status_column="host_receptor_tsx_call_status",
        output_best_label_column="host_receptor_tsx_best_reference_label",
        output_best_pident_column="host_receptor_tsx_best_reference_pident",
        output_best_qcov_column="host_receptor_tsx_best_reference_qcov",
        output_best_tcov_column="host_receptor_tsx_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C Tsx cluster is approximated with a saved family-specific representative bundle.",
    ),
    SurfaceFamilyConfig(
        family_key="YNCD",
        display_name="receptor_yncD",
        seed_gene_name="yncD",
        output_present_column="host_receptor_yncD_present",
        output_label_column="host_receptor_yncD_variant",
        output_status_column="host_receptor_yncD_call_status",
        output_best_label_column="host_receptor_yncD_best_reference_label",
        output_best_pident_column="host_receptor_yncD_best_reference_pident",
        output_best_qcov_column="host_receptor_yncD_best_reference_qcov",
        output_best_tcov_column="host_receptor_yncD_best_reference_tcov",
        support_status="approximated",
        projection_method="Pyrodigal protein calls + mmseqs family seed detection + saved Track C variant representatives.",
        rationale="The original Track C YncD cluster is approximated with a saved family-specific representative bundle.",
    ),
)

UNSUPPORTED_SUPPORT_ROWS: tuple[dict[str, str], ...] = (
    {
        "feature_family": "o_antigen",
        "training_columns": "host_o_antigen_present;host_o_antigen_type",
        "support_status": "unsupported",
        "projection_method": "",
        "rationale": (
            "The repository contains historical O-typing outputs, but TL15 does not yet ship a runtime ECTyper-style "
            "caller or a saved serotype reference contract for raw assemblies."
        ),
    },
    {
        "feature_family": "k_antigen",
        "training_columns": "host_k_antigen_present;host_k_antigen_type;host_k_antigen_type_source",
        "support_status": "unsupported",
        "projection_method": "",
        "rationale": (
            "Typed K-antigen calls depend on external capsule typing workflows that are not packaged as a runtime "
            "projector in this repository."
        ),
    },
    {
        "feature_family": "k_antigen_proxy_capsule",
        "training_columns": (
            "host_k_antigen_proxy_present;host_capsule_abc_present;host_capsule_groupiv_e_present;"
            "host_capsule_groupiv_e_stricte_present;host_capsule_groupiv_s;host_capsule_wzy_stricte_present"
        ),
        "support_status": "unsupported",
        "projection_method": "",
        "rationale": (
            "The capsule proxy block was copied from historical host metadata and has no saved raw-assembly caller or "
            "reference asset in the current codebase."
        ),
    },
    {
        "feature_family": "receptor_tonB",
        "training_columns": "host_receptor_tonB_present;host_receptor_tonB_variant",
        "support_status": "unsupported",
        "projection_method": "",
        "rationale": (
            "Track C already marks TonB as missing because the repository has no source table analogous to the other "
            "receptor clusters, so TL15 leaves it unsupported."
        ),
    },
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-metadata-path", type=Path, default=DEFAULT_PANEL_METADATA_PATH)
    parser.add_argument("--lps-core-path", type=Path, default=DEFAULT_LPS_CORE_PATH)
    parser.add_argument("--receptor-clusters-path", type=Path, default=DEFAULT_RECEPTOR_CLUSTER_PATH)
    parser.add_argument("--assembly-summary-url", default=DEFAULT_ASSEMBLY_SUMMARY_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def _index_unique(rows: Sequence[Mapping[str, str]], key: str, *, path: Path) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if value in indexed:
            raise ValueError(f"Duplicate {key} value {value!r} in {path}")
        indexed[value] = dict(row)
    return indexed


def _normalize_fraction(value: str) -> float:
    parsed = float(value)
    return parsed / 100.0 if parsed > 1.0 else parsed


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir))


def _write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def _read_fasta_dict(path: Path) -> Dict[str, str]:
    return {record.identifier: record.sequence for record in read_fasta_records(path, protein=True)}


def _host_collection_rows(panel_metadata_path: Path) -> list[dict[str, str]]:
    rows = read_track_c_rows(panel_metadata_path, ";")
    host_rows = [row for row in rows if row.get("Collection", "") == PANEL_COLLECTION_HOST]
    if not host_rows:
        raise ValueError(f"No Collection=Host rows found in {panel_metadata_path}")
    return host_rows


def build_expected_track_c_rows(
    *,
    panel_metadata_path: Path,
    lps_core_path: Path,
    receptor_clusters_path: Path,
) -> list[dict[str, object]]:
    host_rows = _host_collection_rows(panel_metadata_path)
    metadata_index = _index_unique(read_track_c_rows(panel_metadata_path, ";"), "bacteria", path=panel_metadata_path)
    lps_key = "bacteria"
    lps_rows = read_track_c_rows(lps_core_path, "\t")
    if lps_rows and "Strain" in lps_rows[0]:
        lps_key = "Strain"
    lps_index = _index_unique(lps_rows, lps_key, path=lps_core_path)
    receptor_index = _index_unique(
        read_track_c_rows(receptor_clusters_path, "\t"),
        "bacteria",
        path=receptor_clusters_path,
    )
    bacteria = sorted(
        row["bacteria"] for row in host_rows if row["bacteria"] in lps_index and row["bacteria"] in receptor_index
    )
    if not bacteria:
        raise ValueError("TL15 found zero host-collection strains with both LPS and receptor annotations.")
    return build_track_c_feature_rows(bacteria, metadata_index, lps_index, receptor_index)


def _candidate_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in value.lower().replace("_", " ").replace("-", " ").split():
            normalized = "".join(ch for ch in token if ch.isalnum())
            if normalized:
                tokens.add(normalized)
    return tokens


def _assembly_matches_host(bacteria: str, record: Mapping[str, str]) -> bool:
    normalized = _normalize_host_name(bacteria)
    if not normalized:
        return False
    haystack = " ".join(
        (
            str(record.get("organism_name", "")),
            str(record.get("infraspecific_name", "")),
            str(record.get("isolate", "")),
        )
    )
    normalized_haystack = _normalize_host_name(haystack)
    if any(char.isalpha() for char in bacteria):
        return normalized in normalized_haystack
    return normalized in _candidate_tokens(
        str(record.get("organism_name", "")),
        str(record.get("infraspecific_name", "")),
        str(record.get("isolate", "")),
    )


def resolve_panel_host_assemblies(
    host_rows: Sequence[Mapping[str, str]],
    *,
    assembly_summary_path: Path,
) -> list[dict[str, str]]:
    host_by_normalized = {
        _normalize_host_name(str(row["bacteria"]).strip()): str(row["bacteria"]).strip() for row in host_rows
    }
    letter_hosts = {
        normalized: bacteria
        for normalized, bacteria in host_by_normalized.items()
        if any(char.isalpha() for char in normalized)
    }
    numeric_hosts = {
        normalized: bacteria
        for normalized, bacteria in host_by_normalized.items()
        if normalized and not any(char.isalpha() for char in normalized)
    }
    letter_pattern = re.compile("|".join(sorted((re.escape(value) for value in letter_hosts), key=len, reverse=True)))
    matches_by_bacteria: dict[str, list[AssemblyRecord]] = defaultdict(list)

    header: list[str] | None = None
    with assembly_summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith("#assembly_accession"):
                header = raw_line[1:].rstrip("\n").split("\t")
                continue
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            if header is None:
                raise ValueError("Assembly summary header was not found.")
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            if row.get("ftp_path", "") in {"", "na"}:
                continue
            record = AssemblyRecord(
                assembly_accession=row["assembly_accession"],
                taxid=row["taxid"],
                organism_name=row["organism_name"],
                infraspecific_name=row["infraspecific_name"],
                isolate=row["isolate"],
                version_status=row["version_status"],
                assembly_level=row["assembly_level"],
                refseq_category=row["refseq_category"],
                ftp_path=row["ftp_path"],
            )
            matched_hosts: set[str] = set()
            normalized_haystack = _normalize_host_name(
                " ".join((record.organism_name, record.infraspecific_name, record.isolate))
            )
            if letter_hosts:
                for matched_token in letter_pattern.findall(normalized_haystack):
                    bacteria = letter_hosts.get(matched_token)
                    if bacteria:
                        matched_hosts.add(bacteria)
            if numeric_hosts:
                candidate_tokens = _candidate_tokens(record.organism_name, record.infraspecific_name, record.isolate)
                for normalized, bacteria in numeric_hosts.items():
                    if normalized in candidate_tokens:
                        matched_hosts.add(bacteria)
            for bacteria in matched_hosts:
                matches_by_bacteria[bacteria].append(record)

    catalog_rows: list[dict[str, str]] = []
    for host_row in sorted(host_rows, key=lambda row: row["bacteria"]):
        bacteria = str(host_row["bacteria"]).strip()
        matches = matches_by_bacteria.get(bacteria, [])
        if not matches:
            catalog_rows.append(
                {
                    "bacteria": bacteria,
                    "gembase": str(host_row.get("Gembase", "")).strip(),
                    "assembly_match_status": "unmatched",
                    "match_reason": "no_refseq_assembly_match",
                    "assembly_accession": "",
                    "assembly_level": "",
                    "refseq_category": "",
                    "assembly_ftp_path": "",
                    "organism_name": "",
                    "infraspecific_name": "",
                    "isolate": "",
                }
            )
            continue
        best = choose_best_assembly(matches)
        catalog_rows.append(
            {
                "bacteria": bacteria,
                "gembase": str(host_row.get("Gembase", "")).strip(),
                "assembly_match_status": "matched",
                "match_reason": "normalized_name_match",
                "assembly_accession": best.assembly_accession,
                "assembly_level": best.assembly_level,
                "refseq_category": best.refseq_category,
                "assembly_ftp_path": best.ftp_path,
                "organism_name": best.organism_name,
                "infraspecific_name": best.infraspecific_name,
                "isolate": best.isolate,
            }
        )
    return catalog_rows


def ensure_assembly_summary(url: str, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    if output_path.exists():
        return output_path
    LOGGER.info("Starting TL15 assembly summary download from %s", url)
    output_path.write_text(_download_text(url), encoding="utf-8")
    LOGGER.info("Completed TL15 assembly summary download -> %s", output_path)
    return output_path


def download_panel_assemblies(
    catalog_rows: Sequence[Mapping[str, str]],
    *,
    assembly_root: Path,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    for row in catalog_rows:
        if row["assembly_match_status"] != "matched":
            continue
        bacteria = row["bacteria"]
        accession = row["assembly_accession"]
        output_path = assembly_root / bacteria / f"{accession}.fna"
        if output_path.exists():
            output_paths[bacteria] = output_path
            continue
        matched_url = str(row.get("assembly_ftp_path", "")).strip()
        if not matched_url:
            raise ValueError(f"Matched host {bacteria} is missing assembly_ftp_path in the TL15 catalog.")
        ftp_basename = Path(matched_url).name
        fasta_url = f"{matched_url.replace('ftp://', 'https://', 1)}/{ftp_basename}_genomic.fna.gz"
        ensure_directory(output_path.parent)
        LOGGER.info("Starting TL15 host assembly download for %s", bacteria)
        fasta_text = gzip.decompress(_download_binary(fasta_url)).decode("utf-8")
        output_path.write_text(fasta_text, encoding="utf-8")
        LOGGER.info("Completed TL15 host assembly download for %s -> %s", bacteria, output_path)
        output_paths[bacteria] = output_path
    return output_paths


def fetch_seed_sequences(
    *,
    seed_dir: Path,
    version: str,
) -> tuple[Path, list[dict[str, str]]]:
    ensure_directory(seed_dir)
    seed_records: list[tuple[str, str]] = []
    seed_metadata: list[dict[str, str]] = []
    for family in SUPPORTED_SURFACE_FAMILIES:
        seed_path = seed_dir / f"{family.family_key.lower()}_{version}.faa"
        if not seed_path.exists():
            query = urllib.parse.quote_plus(f"(gene_exact:{family.seed_gene_name}) AND (organism_id:83333)")
            url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=fasta&size=1"
            fasta_text = _download_text(url)
            if not fasta_text.startswith(">"):
                raise ValueError(f"UniProt seed fetch for {family.family_key} did not return FASTA content.")
            seed_path.write_text(fasta_text, encoding="utf-8")
        seed_sequences = _read_fasta_dict(seed_path)
        if len(seed_sequences) != 1:
            raise ValueError(f"Expected exactly one seed sequence in {seed_path}, found {len(seed_sequences)}")
        seed_identifier, sequence = next(iter(seed_sequences.items()))
        combined_identifier = f"{family.family_key}|{family.seed_gene_name}|{seed_identifier}"
        seed_records.append((combined_identifier, sequence))
        seed_metadata.append(
            {
                "feature_family": family.display_name,
                "seed_fasta": str(seed_path.name),
                "seed_identifier": combined_identifier,
            }
        )
    combined_path = seed_dir / f"surface_family_seeds_{version}.faa"
    _write_fasta(combined_path, seed_records)
    return combined_path, seed_metadata


def predict_host_proteins(assembly_path: Path, *, proteins_path: Path) -> dict[str, object]:
    if proteins_path.exists():
        records = read_fasta_records(proteins_path, protein=True)
        if not records:
            raise ValueError(f"Cached protein FASTA at {proteins_path} was empty.")
        return {"protein_count": len(records), "protein_fasta_path": str(proteins_path), "source": "cached"}
    LOGGER.info("Starting TL15 protein prediction for %s", assembly_path.name)
    metadata = predict_proteins_with_pyrodigal(assembly_path, protein_fasta_path=proteins_path)
    LOGGER.info("Completed TL15 protein prediction for %s", assembly_path.name)
    return metadata


def run_mmseqs_easy_search(
    query_fasta: Path,
    target_fasta: Path,
    *,
    output_tsv: Path,
    tmp_dir: Path,
    threads: int,
) -> Path:
    ensure_directory(output_tsv.parent)
    ensure_directory(tmp_dir)
    command = [
        str(_tool_bin("mmseqs")),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(tmp_dir),
        "--format-mode",
        "4",
        "--format-output",
        MMSEQS_FORMAT_OUTPUT,
        "--threads",
        str(threads),
        "-v",
        "1",
    ]
    _run_command(
        command,
        env=_tool_env(),
        description=f"mmseqs easy-search {query_fasta.name} vs {target_fasta.name}",
    )
    return output_tsv


def read_mmseqs_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected mmseqs result TSV at {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [{key: value.strip() for key, value in row.items()} for row in reader]


def _sort_hit(row: Mapping[str, str]) -> tuple[float, float, float]:
    return (
        float(row["bits"]),
        _normalize_fraction(row["pident"]),
        min(_normalize_fraction(row["qcov"]), _normalize_fraction(row["tcov"])),
    )


def best_seed_hits_by_family(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        family_key = str(row["query"]).split("|", maxsplit=1)[0]
        current = best.get(family_key)
        if current is None or _sort_hit(row) > _sort_hit(current):
            best[family_key] = dict(row)
    return best


def build_reference_assets(
    *,
    expected_rows: Sequence[Mapping[str, object]],
    catalog_rows: Sequence[Mapping[str, str]],
    assembly_paths: Mapping[str, Path],
    seed_fasta_path: Path,
    runtime_dir: Path,
    proteins_dir: Path,
    threads: int,
    version: str,
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    expected_by_bacteria = {str(row["bacteria"]): dict(row) for row in expected_rows}
    seed_hit_dir = runtime_dir / "seed_hits"
    reference_sequences: dict[str, list[tuple[str, str]]] = {
        family.family_key: [] for family in SUPPORTED_SURFACE_FAMILIES
    }
    reference_index_rows: list[dict[str, str]] = []
    for catalog_row in catalog_rows:
        bacteria = str(catalog_row["bacteria"])
        if bacteria not in assembly_paths:
            continue
        proteins_path = proteins_dir / f"{bacteria}.faa"
        predict_host_proteins(assembly_paths[bacteria], proteins_path=proteins_path)
        seed_hit_path = seed_hit_dir / f"{bacteria}_seed_hits.tsv"
        run_mmseqs_easy_search(
            seed_fasta_path,
            proteins_path,
            output_tsv=seed_hit_path,
            tmp_dir=seed_hit_dir / f"{bacteria}_tmp",
            threads=threads,
        )
        best_hits = best_seed_hits_by_family(read_mmseqs_rows(seed_hit_path))
        protein_sequences = _read_fasta_dict(proteins_path)
        expected_row = expected_by_bacteria[bacteria]
        for family in SUPPORTED_SURFACE_FAMILIES:
            label = str(expected_row.get(family.output_label_column, "")).strip()
            if not label:
                continue
            hit = best_hits.get(family.family_key)
            if hit is None:
                continue
            if _normalize_fraction(hit["pident"]) < FAMILY_DETECTION_MIN_PIDENT:
                continue
            if min(_normalize_fraction(hit["qcov"]), _normalize_fraction(hit["tcov"])) < FAMILY_DETECTION_MIN_COVERAGE:
                continue
            protein_identifier = hit["target"]
            if protein_identifier not in protein_sequences:
                raise KeyError(f"Protein {protein_identifier!r} missing from predicted FASTA {proteins_path}")
            sequence = protein_sequences[protein_identifier]
            reference_id = f"{family.family_key}|{label}|{bacteria}"
            if any(
                existing_id.split("|", maxsplit=2)[1] == label
                for existing_id, _ in reference_sequences[family.family_key]
            ):
                continue
            reference_sequences[family.family_key].append((reference_id, sequence))
    family_reference_paths: dict[str, Path] = {}
    for family in SUPPORTED_SURFACE_FAMILIES:
        family_path = runtime_dir / f"{family.family_key.lower()}_references_{version}.faa"
        records = reference_sequences[family.family_key]
        if records:
            _write_fasta(family_path, records)
            family_reference_paths[family.family_key] = family_path
            for record_id, _sequence in records:
                _, label, bacteria = record_id.split("|", maxsplit=2)
                reference_index_rows.append(
                    {
                        "feature_family": family.display_name,
                        "reference_label": label,
                        "source_bacteria": bacteria,
                        "protein_identifier": record_id,
                        "reference_fasta": family_path.name,
                        "sha256": sha256_file(family_path),
                    }
                )
    if not family_reference_paths:
        raise ValueError("TL15 did not recover any runtime reference sequences from assembly-backed panel hosts.")
    reference_index_path = runtime_dir / f"surface_reference_index_{version}.csv"
    write_csv(reference_index_path, REFERENCE_INDEX_COLUMNS, reference_index_rows)
    return family_reference_paths, reference_index_rows


def _family_hits_for_candidate(
    rep_rows: Sequence[Mapping[str, str]],
    *,
    candidate_protein: str,
    family_key: str,
) -> list[dict[str, str]]:
    out = []
    for row in rep_rows:
        if row["query"] != candidate_protein:
            continue
        if str(row["target"]).split("|", maxsplit=1)[0] != family_key:
            continue
        out.append(dict(row))
    out.sort(key=_sort_hit, reverse=True)
    return out


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def project_host_surface_row(
    *,
    bacteria: str,
    assembly_accession: str,
    assembly_path: Path,
    seed_fasta_path: Path,
    family_reference_paths: Mapping[str, Path],
    proteins_dir: Path,
    projection_dir: Path,
    threads: int,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in PROJECTION_COLUMNS}
    row["bacteria"] = bacteria
    row["assembly_accession"] = assembly_accession
    proteins_path = proteins_dir / f"{bacteria}.faa"
    predict_host_proteins(assembly_path, proteins_path=proteins_path)
    seed_hit_path = projection_dir / f"{bacteria}_seed_hits.tsv"
    run_mmseqs_easy_search(
        seed_fasta_path,
        proteins_path,
        output_tsv=seed_hit_path,
        tmp_dir=projection_dir / f"{bacteria}_seed_tmp",
        threads=threads,
    )
    seed_hits = best_seed_hits_by_family(read_mmseqs_rows(seed_hit_path))
    all_reference_fasta = projection_dir / f"{bacteria}_all_references.faa"
    all_reference_records: list[tuple[str, str]] = []
    for family in SUPPORTED_SURFACE_FAMILIES:
        family_path = family_reference_paths.get(family.family_key)
        if family_path and family_path.exists():
            all_reference_records.extend(_read_fasta_dict(family_path).items())
    if all_reference_records:
        _write_fasta(all_reference_fasta, all_reference_records)
        representative_hits_path = projection_dir / f"{bacteria}_reference_hits.tsv"
        run_mmseqs_easy_search(
            proteins_path,
            all_reference_fasta,
            output_tsv=representative_hits_path,
            tmp_dir=projection_dir / f"{bacteria}_reference_tmp",
            threads=threads,
        )
        representative_rows = read_mmseqs_rows(representative_hits_path)
    else:
        representative_rows = []

    for family in SUPPORTED_SURFACE_FAMILIES:
        seed_hit = seed_hits.get(family.family_key)
        if seed_hit is None:
            row[family.output_present_column] = 0
            row[family.output_label_column] = ""
            row[family.output_status_column] = "called_absent"
            continue
        if (
            _normalize_fraction(seed_hit["pident"]) < FAMILY_DETECTION_MIN_PIDENT
            or min(
                _normalize_fraction(seed_hit["qcov"]),
                _normalize_fraction(seed_hit["tcov"]),
            )
            < FAMILY_DETECTION_MIN_COVERAGE
        ):
            row[family.output_present_column] = 0
            row[family.output_label_column] = ""
            row[family.output_status_column] = "called_absent"
            continue
        family_reference_path = family_reference_paths.get(family.family_key)
        if family_reference_path is None:
            row[family.output_present_column] = ""
            row[family.output_label_column] = ""
            row[family.output_status_column] = "family_detected_reference_missing"
            continue
        candidate_hits = _family_hits_for_candidate(
            representative_rows,
            candidate_protein=seed_hit["target"],
            family_key=family.family_key,
        )
        if not candidate_hits:
            row[family.output_present_column] = ""
            row[family.output_label_column] = ""
            row[family.output_status_column] = "family_detected_variant_unresolved"
            continue
        best = candidate_hits[0]
        best_pident = _normalize_fraction(best["pident"])
        best_qcov = _normalize_fraction(best["qcov"])
        best_tcov = _normalize_fraction(best["tcov"])
        label = str(best["target"]).split("|", maxsplit=2)[1]
        row[family.output_best_label_column] = label
        row[family.output_best_pident_column] = _format_metric(best_pident)
        row[family.output_best_qcov_column] = _format_metric(best_qcov)
        row[family.output_best_tcov_column] = _format_metric(best_tcov)
        if best_pident < VARIANT_ASSIGNMENT_MIN_PIDENT or min(best_qcov, best_tcov) < VARIANT_ASSIGNMENT_MIN_COVERAGE:
            row[family.output_present_column] = ""
            row[family.output_label_column] = ""
            row[family.output_status_column] = "family_detected_variant_unresolved"
            continue
        if len(candidate_hits) > 1:
            second = candidate_hits[1]
            second_label = str(second["target"]).split("|", maxsplit=2)[1]
            if second_label != label and abs(float(second["bits"]) - float(best["bits"])) <= AMBIGUOUS_BITS_DELTA:
                row[family.output_present_column] = ""
                row[family.output_label_column] = ""
                row[family.output_status_column] = "family_detected_variant_ambiguous"
                continue
        row[family.output_present_column] = 1
        row[family.output_label_column] = label
        row[family.output_status_column] = "called_present"
    return row


def summarize_projection_agreement(
    *,
    projected_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    family_reference_paths: Mapping[str, Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    expected_by_bacteria = {str(row["bacteria"]): dict(row) for row in expected_rows}
    agreement_rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, str]] = []
    for family in SUPPORTED_SURFACE_FAMILIES:
        callable_count = 0
        exact_match_count = 0
        not_callable_count = 0
        for projected_row in projected_rows:
            bacteria = str(projected_row["bacteria"])
            expected_row = expected_by_bacteria[bacteria]
            expected_present = expected_row.get(family.output_present_column, "")
            expected_label = str(expected_row.get(family.output_label_column, "")).strip()
            projected_present = projected_row.get(family.output_present_column, "")
            projected_label = str(projected_row.get(family.output_label_column, "")).strip()
            call_status = str(projected_row.get(family.output_status_column, "")).strip()
            if call_status not in {"called_absent", "called_present"}:
                not_callable_count += 1
                mismatch_rows.append(
                    {
                        "bacteria": bacteria,
                        "assembly_accession": str(projected_row.get("assembly_accession", "")),
                        "feature_family": family.display_name,
                        "expected_present": str(expected_present),
                        "expected_label": expected_label,
                        "projected_present": str(projected_present),
                        "projected_label": projected_label,
                        "call_status": call_status,
                        "best_reference_label": str(projected_row.get(family.output_best_label_column, "")),
                    }
                )
                continue
            callable_count += 1
            if str(expected_present) == str(projected_present) and expected_label == projected_label:
                exact_match_count += 1
            else:
                mismatch_rows.append(
                    {
                        "bacteria": bacteria,
                        "assembly_accession": str(projected_row.get("assembly_accession", "")),
                        "feature_family": family.display_name,
                        "expected_present": str(expected_present),
                        "expected_label": expected_label,
                        "projected_present": str(projected_present),
                        "projected_label": projected_label,
                        "call_status": call_status,
                        "best_reference_label": str(projected_row.get(family.output_best_label_column, "")),
                    }
                )
        support_status = family.support_status if family.family_key in family_reference_paths else "unsupported"
        support_rows.append(
            {
                "feature_family": family.display_name,
                "training_columns": f"{family.output_present_column};{family.output_label_column}",
                "support_status": support_status,
                "projection_method": family.projection_method if support_status != "unsupported" else "",
                "rationale": (
                    family.rationale
                    if support_status != "unsupported"
                    else "No assembly-backed representative sequence was recovered for this family, so TL15 leaves it unsupported."
                ),
            }
        )
        agreement_rows.append(
            {
                "feature_family": family.display_name,
                "training_columns": f"{family.output_present_column};{family.output_label_column}",
                "status": support_status,
                "panel_hosts_with_assemblies": len(projected_rows),
                "callable_count": callable_count,
                "not_callable_count": not_callable_count,
                "exact_match_count": exact_match_count,
                "agreement_rate_on_callable": round(exact_match_count / callable_count, 6) if callable_count else "",
            }
        )
    support_rows.extend(UNSUPPORTED_SUPPORT_ROWS)
    return agreement_rows, mismatch_rows, support_rows


def build_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    assembly_summary_path: Path,
    assembly_catalog_path: Path,
    projection_path: Path,
    agreement_path: Path,
    mismatch_path: Path,
    support_path: Path,
    runtime_dir: Path,
    family_reference_paths: Mapping[str, Path],
    seed_fasta_path: Path,
    seed_metadata: Sequence[Mapping[str, str]],
    catalog_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    matched_hosts = [row for row in catalog_rows if row["assembly_match_status"] == "matched"]
    return {
        "task_id": "TL15",
        "format_version": f"tl15_raw_host_surface_projector_{args.version}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "panel_metadata_path": str(args.panel_metadata_path),
            "lps_core_path": str(args.lps_core_path),
            "receptor_clusters_path": str(args.receptor_clusters_path),
            "assembly_summary_url": args.assembly_summary_url,
            "assembly_summary_local_path": _relative_to_output(assembly_summary_path, output_dir),
        },
        "runtime_assets": {
            "directory": _relative_to_output(runtime_dir, output_dir),
            "seed_fasta": _relative_to_output(seed_fasta_path, output_dir),
            "family_reference_fastas": {
                family_key: _relative_to_output(path, output_dir) for family_key, path in family_reference_paths.items()
            },
            "seed_metadata": list(seed_metadata),
        },
        "outputs": {
            "assembly_catalog_csv": _relative_to_output(assembly_catalog_path, output_dir),
            "projection_csv": _relative_to_output(projection_path, output_dir),
            "agreement_csv": _relative_to_output(agreement_path, output_dir),
            "mismatch_csv": _relative_to_output(mismatch_path, output_dir),
            "support_table_csv": _relative_to_output(support_path, output_dir),
            "projection_sha256": sha256_file(projection_path),
            "agreement_sha256": sha256_file(agreement_path),
            "mismatch_sha256": sha256_file(mismatch_path),
            "support_table_sha256": sha256_file(support_path),
        },
        "summary": {
            "host_collection_count": len(catalog_rows),
            "matched_assembly_count": len(matched_hosts),
            "unmatched_assembly_count": len(catalog_rows) - len(matched_hosts),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    runtime_dir = output_dir / "runtime_assets"
    assembly_root = raw_dir / "assemblies"
    proteins_dir = raw_dir / "predicted_proteins"
    projections_dir = raw_dir / "projection_hits"
    seed_dir = runtime_dir / "seed_sequences"
    ensure_directory(output_dir)
    ensure_directory(raw_dir)
    ensure_directory(runtime_dir)

    assembly_summary_path = ensure_assembly_summary(args.assembly_summary_url, raw_dir / "assembly_summary_refseq.txt")
    host_rows = _host_collection_rows(args.panel_metadata_path)
    expected_rows = build_expected_track_c_rows(
        panel_metadata_path=args.panel_metadata_path,
        lps_core_path=args.lps_core_path,
        receptor_clusters_path=args.receptor_clusters_path,
    )
    catalog_rows = resolve_panel_host_assemblies(host_rows, assembly_summary_path=assembly_summary_path)
    assembly_catalog_path = output_dir / f"panel_host_assembly_catalog_{args.version}.csv"
    write_csv(assembly_catalog_path, ASSEMBLY_CATALOG_COLUMNS, catalog_rows)
    matched_rows = [row for row in catalog_rows if row["assembly_match_status"] == "matched"]
    if not matched_rows:
        raise ValueError("TL15 found zero assembly-backed panel hosts, so no agreement analysis is possible.")

    seed_fasta_path, seed_metadata = fetch_seed_sequences(seed_dir=seed_dir, version=args.version)
    assembly_paths = download_panel_assemblies(catalog_rows, assembly_root=assembly_root)
    family_reference_paths, _ = build_reference_assets(
        expected_rows=expected_rows,
        catalog_rows=catalog_rows,
        assembly_paths=assembly_paths,
        seed_fasta_path=seed_fasta_path,
        runtime_dir=runtime_dir,
        proteins_dir=proteins_dir,
        threads=args.threads,
        version=args.version,
    )

    projected_rows = [
        project_host_surface_row(
            bacteria=row["bacteria"],
            assembly_accession=row["assembly_accession"],
            assembly_path=assembly_paths[row["bacteria"]],
            seed_fasta_path=seed_fasta_path,
            family_reference_paths=family_reference_paths,
            proteins_dir=proteins_dir,
            projection_dir=projections_dir,
            threads=args.threads,
        )
        for row in matched_rows
        if row["bacteria"] in assembly_paths
    ]
    if not projected_rows:
        raise ValueError("TL15 projected zero panel hosts after downloading assemblies.")

    projection_path = output_dir / f"projected_host_surface_features_{args.version}.csv"
    write_csv(projection_path, PROJECTION_COLUMNS, projected_rows)

    expected_subset = [
        row for row in expected_rows if str(row["bacteria"]) in {item["bacteria"] for item in projected_rows}
    ]
    agreement_rows, mismatch_rows, support_rows = summarize_projection_agreement(
        projected_rows=projected_rows,
        expected_rows=expected_subset,
        family_reference_paths=family_reference_paths,
    )
    agreement_path = output_dir / f"projected_host_surface_agreement_{args.version}.csv"
    mismatch_path = output_dir / f"projected_host_surface_mismatches_{args.version}.csv"
    support_path = output_dir / f"training_host_surface_feature_support_{args.version}.csv"
    write_csv(agreement_path, AGREEMENT_COLUMNS, agreement_rows)
    write_csv(mismatch_path, MISMATCH_COLUMNS, mismatch_rows)
    write_csv(support_path, SUPPORT_TABLE_COLUMNS, support_rows)

    manifest = build_manifest(
        args=args,
        output_dir=output_dir,
        assembly_summary_path=assembly_summary_path,
        assembly_catalog_path=assembly_catalog_path,
        projection_path=projection_path,
        agreement_path=agreement_path,
        mismatch_path=mismatch_path,
        support_path=support_path,
        runtime_dir=runtime_dir,
        family_reference_paths=family_reference_paths,
        seed_fasta_path=seed_fasta_path,
        seed_metadata=seed_metadata,
        catalog_rows=catalog_rows,
    )
    manifest_path = output_dir / f"raw_host_surface_projector_manifest_{args.version}.json"
    write_json(manifest_path, manifest)
    LOGGER.info("Completed TL15 raw-host surface projector build -> %s", output_dir)


if __name__ == "__main__":
    main()
