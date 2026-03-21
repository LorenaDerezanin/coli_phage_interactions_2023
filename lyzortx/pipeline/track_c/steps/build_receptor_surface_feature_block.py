#!/usr/bin/env python3
"""Build Track C receptor and surface host features with per-column provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

RECEPTOR_CLUSTER_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("BTUB", "host_receptor_btub"),
    ("FADL", "host_receptor_fadL"),
    ("FHUA", "host_receptor_fhua"),
    ("LAMB", "host_receptor_lamB"),
    ("LPTD", "host_receptor_lptD"),
    ("NFRA", "host_receptor_nfrA"),
    ("OMPA", "host_receptor_ompA"),
    ("OMPC", "host_receptor_ompC"),
    ("OMPF", "host_receptor_ompF"),
    ("TOLC", "host_receptor_tolC"),
    ("TSX", "host_receptor_tsx"),
    ("YNCD", "host_receptor_yncD"),
)

CAPSULE_LOCUS_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Capsule_ABC", "host_capsule_abc_present"),
    ("Capsule_GroupIV_e", "host_capsule_groupiv_e_present"),
    ("Capsule_GroupIV_e_stricte", "host_capsule_groupiv_e_stricte_present"),
    ("Capsule_GroupIV_s", "host_capsule_groupiv_s"),
    ("Capsule_Wzy_stricte", "host_capsule_wzy_stricte_present"),
)

FEATURE_COLUMNS: Tuple[str, ...] = (
    "bacteria",
    "host_o_antigen_present",
    "host_o_antigen_type",
    "host_k_antigen_present",
    "host_k_antigen_type",
    "host_k_antigen_type_source",
    "host_k_antigen_proxy_present",
    "host_lps_core_present",
    "host_lps_core_type",
    "host_capsule_abc_present",
    "host_capsule_groupiv_e_present",
    "host_capsule_groupiv_e_stricte_present",
    "host_capsule_groupiv_s",
    "host_capsule_wzy_stricte_present",
    "host_receptor_btub_present",
    "host_receptor_btub_variant",
    "host_receptor_fadL_present",
    "host_receptor_fadL_variant",
    "host_receptor_fhua_present",
    "host_receptor_fhua_variant",
    "host_receptor_lamB_present",
    "host_receptor_lamB_variant",
    "host_receptor_lptD_present",
    "host_receptor_lptD_variant",
    "host_receptor_nfrA_present",
    "host_receptor_nfrA_variant",
    "host_receptor_ompA_present",
    "host_receptor_ompA_variant",
    "host_receptor_ompC_present",
    "host_receptor_ompC_variant",
    "host_receptor_ompF_present",
    "host_receptor_ompF_variant",
    "host_receptor_tolC_present",
    "host_receptor_tolC_variant",
    "host_receptor_tonB_present",
    "host_receptor_tonB_variant",
    "host_receptor_tsx_present",
    "host_receptor_tsx_variant",
    "host_receptor_yncD_present",
    "host_receptor_yncD_variant",
)

FEATURE_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "host_o_antigen_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["O-type"],
        "transform": "1 when O-type is non-empty and not '-', else 0.",
        "note": "Copied from curated host metadata.",
    },
    "host_o_antigen_type": {
        "group": "surface_antigen",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["O-type"],
        "transform": "Copy O-type category; normalize '-' to empty string.",
        "note": "O-antigen type is available for the 369-host genomic subset.",
    },
    "host_k_antigen_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Klebs_capsule_type", "ABC_serotype"],
        "transform": "1 when a typed capsule/K annotation is available from either source, else 0.",
        "note": "Treats Klebsiella Kaptive calls and ABC serotypes as typed K/capsule annotations.",
    },
    "host_k_antigen_type": {
        "group": "surface_antigen",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Klebs_capsule_type", "ABC_serotype"],
        "transform": "Prefer Klebs_capsule_type when present; otherwise carry ABC_serotype verbatim except Unknown/-.",
        "note": "ABC serotype nomenclature is mixed in the source file and is preserved as-is.",
    },
    "host_k_antigen_type_source": {
        "group": "surface_antigen",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Klebs_capsule_type", "ABC_serotype"],
        "transform": "Emit the source column used for host_k_antigen_type.",
        "note": "Blank when no typed K/capsule annotation is available.",
    },
    "host_k_antigen_proxy_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": [
            "Klebs_capsule_type",
            "ABC_serotype",
            "Capsule_ABC",
            "Capsule_GroupIV_e",
            "Capsule_GroupIV_e_stricte",
            "Capsule_GroupIV_s",
            "Capsule_Wzy_stricte",
        ],
        "transform": "1 when any typed K annotation or capsule-locus proxy is present, else 0.",
        "note": "Designed to keep capsule-associated signal even when a typed K label is missing.",
    },
    "host_lps_core_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt"],
        "source_columns": ["LPS_type"],
        "transform": "1 when LPS core type is non-empty, else 0.",
        "note": "The 369-row host subset is defined by overlap between interaction hosts and this LPS file.",
    },
    "host_lps_core_type": {
        "group": "surface_antigen",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt"],
        "source_columns": ["LPS_type"],
        "transform": "Copy curated LPS core type directly from the waaL-based table.",
        "note": "Includes values such as R1-R4, K12, and No_waaL when present in the source.",
    },
    "host_capsule_abc_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Capsule_ABC"],
        "transform": "Cast the source float-like flag to integer 0/1.",
        "note": "Capsule-related proxy copied from curated host metadata.",
    },
    "host_capsule_groupiv_e_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Capsule_GroupIV_e"],
        "transform": "Cast the source float-like flag to integer 0/1.",
        "note": "Capsule-related proxy copied from curated host metadata.",
    },
    "host_capsule_groupiv_e_stricte_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Capsule_GroupIV_e_stricte"],
        "transform": "Cast the source float-like flag to integer 0/1.",
        "note": "Capsule-related proxy copied from curated host metadata.",
    },
    "host_capsule_groupiv_s": {
        "group": "surface_antigen",
        "type": "integer",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Capsule_GroupIV_s"],
        "transform": "Cast the source float-like value to an integer after validating it is integral.",
        "note": (
            "Capsule-related proxy copied from curated host metadata. The source currently includes "
            "integer-valued levels rather than a pure binary presence flag."
        ),
    },
    "host_capsule_wzy_stricte_present": {
        "group": "surface_antigen",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/picard_collection.csv"],
        "source_columns": ["Capsule_Wzy_stricte"],
        "transform": "Cast the source float-like flag to integer 0/1.",
        "note": "Capsule-related proxy copied from curated host metadata.",
    },
    "host_receptor_btub_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["BTUB"],
        "transform": "1 when a BtuB receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_btub_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["BTUB"],
        "transform": "Copy the BtuB cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_fadL_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["FADL"],
        "transform": "1 when a FadL receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_fadL_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["FADL"],
        "transform": "Copy the FadL cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_fhua_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["FHUA"],
        "transform": "1 when an FhuA receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_fhua_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["FHUA"],
        "transform": "Copy the FhuA cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_lamB_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["LAMB"],
        "transform": "1 when a LamB receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_lamB_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["LAMB"],
        "transform": "Copy the LamB cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_lptD_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["LPTD"],
        "transform": "1 when an LptD receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_lptD_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["LPTD"],
        "transform": "Copy the LptD cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_nfrA_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["NFRA"],
        "transform": "1 when an NfrA receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_nfrA_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["NFRA"],
        "transform": "Copy the NfrA cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_ompA_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPA"],
        "transform": "1 when an OmpA receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_ompA_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPA"],
        "transform": "Copy the OmpA cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_ompC_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPC"],
        "transform": "1 when an OmpC receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_ompC_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPC"],
        "transform": "Copy the OmpC cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_ompF_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPF"],
        "transform": "1 when an OmpF receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters. OmpF is a known T-even phage receptor.",
    },
    "host_receptor_ompF_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["OMPF"],
        "transform": "Copy the OmpF cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_tolC_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["TOLC"],
        "transform": "1 when a TolC receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_tolC_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["TOLC"],
        "transform": "Copy the TolC cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_tonB_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": [],
        "source_columns": [],
        "transform": "No direct source available in-repo; emit missing values for explicit downstream backfill.",
        "note": (
            "The repository contains no TonB locus table analogous to the OMP cluster source. "
            "Consumers should treat empty values as NA, not as 0."
        ),
    },
    "host_receptor_tonB_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": [],
        "source_columns": [],
        "transform": "No direct source available in-repo; emit missing values for explicit downstream backfill.",
        "note": (
            "The repository contains no TonB locus table analogous to the OMP cluster source. "
            "Consumers should treat empty values as NA, not as 0."
        ),
    },
    "host_receptor_tsx_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["TSX"],
        "transform": "1 when a Tsx receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_tsx_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["TSX"],
        "transform": "Copy the Tsx cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
    "host_receptor_yncD_present": {
        "group": "receptor",
        "type": "binary",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["YNCD"],
        "transform": "1 when a YncD receptor cluster assignment is present, else 0.",
        "note": "Cluster assignments are 99% identity BLAST clusters.",
    },
    "host_receptor_yncD_variant": {
        "group": "receptor",
        "type": "categorical",
        "source_paths": ["data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"],
        "source_columns": ["YNCD"],
        "transform": "Copy the YncD cluster identifier verbatim.",
        "note": "Blank when no cluster assignment is available.",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interaction-matrix-path",
        type=Path,
        default=Path("data/interactions/interaction_matrix.csv"),
        help="Semicolon-delimited interaction matrix used to define the host panel.",
    )
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Semicolon-delimited curated host metadata table.",
    )
    parser.add_argument(
        "--lps-core-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_core_lps/LPS_type_waaL_370.txt"),
        help="Tab-delimited LPS core type table.",
    )
    parser.add_argument(
        "--receptor-clusters-path",
        type=Path,
        default=Path("data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv"),
        help="Tab-delimited receptor cluster assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_c/receptor_surface_feature_block"),
        help="Directory for generated Track C artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and the manifest.",
    )
    parser.add_argument(
        "--expected-host-count",
        type=int,
        default=369,
        help="Expected number of hosts in the final matrix.",
    )
    return parser.parse_args(argv)


def read_delimited_rows(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def _normalize_category(value: str) -> str:
    normalized = value.strip()
    return "" if normalized in {"", "-", "Unknown"} else normalized


def _parse_integral_value(value: str) -> int:
    normalized = value.strip()
    if normalized == "":
        return 0
    parsed = float(normalized)
    if not parsed.is_integer():
        raise ValueError(f"Expected an integer-like value, got {value!r}")
    return int(parsed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_unique(rows: Sequence[Mapping[str, str]], key: str, *, path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            continue
        if value in out:
            raise ValueError(f"Duplicate {key} value {value!r} in {path}")
        out[value] = dict(row)
    return out


def target_host_set(
    interaction_rows: Sequence[Mapping[str, str]],
    lps_rows: Sequence[Mapping[str, str]],
    receptor_rows: Sequence[Mapping[str, str]],
) -> List[str]:
    interaction_hosts = {row["bacteria"] for row in interaction_rows if row.get("bacteria", "")}
    lps_hosts = {row["bacteria"] for row in lps_rows if row.get("bacteria", "")}
    receptor_hosts = {row["bacteria"] for row in receptor_rows if row.get("bacteria", "")}
    return sorted(interaction_hosts & lps_hosts & receptor_hosts)


def resolve_k_antigen_type(host_row: Mapping[str, str]) -> Tuple[str, str]:
    klebsiella_type = _normalize_category(host_row.get("Klebs_capsule_type", ""))
    if klebsiella_type:
        return klebsiella_type, "Klebs_capsule_type"

    abc_serotype = _normalize_category(host_row.get("ABC_serotype", ""))
    if abc_serotype:
        return abc_serotype, "ABC_serotype"

    return "", ""


def build_feature_rows(
    hosts: Sequence[str],
    host_metadata_by_bacteria: Mapping[str, Mapping[str, str]],
    lps_by_bacteria: Mapping[str, Mapping[str, str]],
    receptor_by_bacteria: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []

    for bacteria in hosts:
        if bacteria not in host_metadata_by_bacteria:
            raise KeyError(f"Missing host metadata for {bacteria}")
        if bacteria not in lps_by_bacteria:
            raise KeyError(f"Missing LPS core annotation for {bacteria}")
        if bacteria not in receptor_by_bacteria:
            raise KeyError(f"Missing receptor cluster annotation for {bacteria}")

        host_row = host_metadata_by_bacteria[bacteria]
        lps_row = lps_by_bacteria[bacteria]
        receptor_row = receptor_by_bacteria[bacteria]

        o_antigen_type = _normalize_category(host_row.get("O-type", ""))
        k_antigen_type, k_antigen_source = resolve_k_antigen_type(host_row)
        lps_core_type = _normalize_category(lps_row.get("LPS_type", ""))

        row: Dict[str, object] = {
            "bacteria": bacteria,
            "host_o_antigen_present": 1 if o_antigen_type else 0,
            "host_o_antigen_type": o_antigen_type,
            "host_k_antigen_present": 1 if k_antigen_type else 0,
            "host_k_antigen_type": k_antigen_type,
            "host_k_antigen_type_source": k_antigen_source,
            "host_lps_core_present": 1 if lps_core_type else 0,
            "host_lps_core_type": lps_core_type,
        }

        capsule_proxy_present = 1 if k_antigen_type else 0
        for source_column, output_column in CAPSULE_LOCUS_COLUMNS:
            flag = _parse_integral_value(host_row.get(source_column, ""))
            row[output_column] = flag
            if flag > 0:
                capsule_proxy_present = 1
        row["host_k_antigen_proxy_present"] = capsule_proxy_present

        for source_column, receptor_prefix in RECEPTOR_CLUSTER_COLUMNS:
            variant = _normalize_category(receptor_row.get(source_column, ""))
            row[f"{receptor_prefix}_present"] = 1 if variant else 0
            row[f"{receptor_prefix}_variant"] = variant

        row["host_receptor_tonB_present"] = ""
        row["host_receptor_tonB_variant"] = ""

        out.append(row)

    return out


def column_missingness(rows: Sequence[Mapping[str, object]], columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    row_count = len(rows)
    out: Dict[str, Dict[str, float]] = {}
    for column in columns:
        missing_count = sum(1 for row in rows if row.get(column, "") == "")
        out[column] = {
            "missing_count": int(missing_count),
            "missing_rate": round(missing_count / row_count, 6) if row_count else 1.0,
        }
    return out


def build_column_metadata(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    missingness = column_missingness(rows, FEATURE_COLUMNS[1:])
    metadata_rows: List[Dict[str, object]] = []
    for column in FEATURE_COLUMNS[1:]:
        definition = FEATURE_DEFINITIONS[column]
        metadata_rows.append(
            {
                "column_name": column,
                "feature_group": definition["group"],
                "data_type": definition["type"],
                "source_paths": json.dumps(definition["source_paths"]),
                "source_columns": json.dumps(definition["source_columns"]),
                "transform": definition["transform"],
                "provenance_note": definition["note"],
                "missing_count": missingness[column]["missing_count"],
                "missing_rate": missingness[column]["missing_rate"],
            }
        )
    return metadata_rows


def build_manifest(
    *,
    version: str,
    rows: Sequence[Mapping[str, object]],
    interaction_matrix_path: Path,
    host_metadata_path: Path,
    lps_core_path: Path,
    receptor_clusters_path: Path,
    matrix_output_path: Path,
    metadata_output_path: Path,
) -> Dict[str, object]:
    return {
        "version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_count": len(rows),
        "schema": {
            "matrix_columns": list(FEATURE_COLUMNS),
            "feature_definitions": FEATURE_DEFINITIONS,
        },
        "missingness": column_missingness(rows, FEATURE_COLUMNS[1:]),
        "inputs": {
            "interaction_matrix": {"path": str(interaction_matrix_path), "sha256": _sha256(interaction_matrix_path)},
            "host_metadata": {"path": str(host_metadata_path), "sha256": _sha256(host_metadata_path)},
            "lps_core": {"path": str(lps_core_path), "sha256": _sha256(lps_core_path)},
            "receptor_clusters": {"path": str(receptor_clusters_path), "sha256": _sha256(receptor_clusters_path)},
        },
        "outputs": {
            "matrix_csv": str(matrix_output_path),
            "column_metadata_csv": str(metadata_output_path),
        },
        "host_set_definition": {
            "rule": "interaction_hosts ∩ lps_core_hosts ∩ receptor_cluster_hosts",
            "note": "This resolves to the 369-host genomic subset requested by the task.",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    interaction_rows = read_delimited_rows(args.interaction_matrix_path, delimiter=";")
    host_metadata_rows = read_delimited_rows(args.host_metadata_path, delimiter=";")
    lps_rows = read_delimited_rows(args.lps_core_path, delimiter="\t")
    receptor_rows = read_delimited_rows(args.receptor_clusters_path, delimiter="\t")

    hosts = target_host_set(interaction_rows, lps_rows, receptor_rows)
    if len(hosts) != args.expected_host_count:
        raise ValueError(
            f"Expected {args.expected_host_count} hosts after joining interaction/LPS/receptor sources, found {len(hosts)}"
        )

    rows = build_feature_rows(
        hosts=hosts,
        host_metadata_by_bacteria=_index_unique(host_metadata_rows, "bacteria", path=args.host_metadata_path),
        lps_by_bacteria=_index_unique(lps_rows, "bacteria", path=args.lps_core_path),
        receptor_by_bacteria=_index_unique(receptor_rows, "bacteria", path=args.receptor_clusters_path),
    )

    ensure_directory(args.output_dir)
    matrix_output_path = args.output_dir / f"host_receptor_surface_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"host_receptor_surface_feature_metadata_{args.version}.csv"
    manifest_output_path = args.output_dir / f"host_receptor_surface_feature_manifest_{args.version}.json"

    write_csv(matrix_output_path, FEATURE_COLUMNS, rows)
    write_csv(
        metadata_output_path,
        (
            "column_name",
            "feature_group",
            "data_type",
            "source_paths",
            "source_columns",
            "transform",
            "provenance_note",
            "missing_count",
            "missing_rate",
        ),
        build_column_metadata(rows),
    )
    write_json(
        manifest_output_path,
        build_manifest(
            version=args.version,
            rows=rows,
            interaction_matrix_path=args.interaction_matrix_path,
            host_metadata_path=args.host_metadata_path,
            lps_core_path=args.lps_core_path,
            receptor_clusters_path=args.receptor_clusters_path,
            matrix_output_path=matrix_output_path,
            metadata_output_path=metadata_output_path,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
