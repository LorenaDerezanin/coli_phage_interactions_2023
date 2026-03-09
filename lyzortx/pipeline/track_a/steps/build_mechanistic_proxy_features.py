#!/usr/bin/env python3
"""Build mechanistic host/phage proxy features from internal metadata.

The output is intentionally versioned and accompanied by a manifest that captures
schema, provenance, missingness, and confidence metadata for downstream auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

HOST_SURFACE_COLUMNS: Sequence[str] = (
    "LPS_type",
    "O-type",
    "H-type",
    "Capsule_ABC",
    "Capsule_GroupIV_e",
    "Capsule_GroupIV_e_stricte",
    "Capsule_GroupIV_s",
    "Capsule_Wzy_stricte",
    "ABC_serotype",
)

KNOWN_DEPOLYMERASE_HINTS: Sequence[str] = (
    "depolymerase",
    "podovir",
    "autographivir",
    "k1",
    "k5",
    "capsule",
)

HOST_CONFIDENCE_FIELDS: Sequence[str] = ("host_receptor_confidence", "host_defense_confidence")
PHAGE_CONFIDENCE_FIELDS: Sequence[str] = ("phage_rbp_confidence", "phage_depolymerase_confidence")

FEATURE_DEFINITIONS: Dict[str, Dict[str, Dict[str, object]]] = {
    "host": {
        "host_receptor_surface_fields_observed": {
            "description": "Count of non-empty host surface structure fields used as receptor proxies.",
            "source_columns": list(HOST_SURFACE_COLUMNS),
        },
        "host_receptor_proxy_available": {
            "description": "Binary indicator for whether at least one receptor-related host surface field is known.",
            "source_columns": list(HOST_SURFACE_COLUMNS),
        },
        "host_receptor_surface_proxy_score": {
            "description": "Fraction of receptor-related host surface fields that are observed.",
            "source_columns": list(HOST_SURFACE_COLUMNS),
        },
        "host_defense_system_count": {
            "description": "Reported number of defense systems in host metadata.",
            "source_columns": ["n_defense_systems"],
        },
        "host_defense_per_infection_proxy": {
            "description": "Defense-system count divided by infections count when both are present and denominator > 0.",
            "source_columns": ["n_defense_systems", "n_infections"],
        },
        "host_receptor_confidence": {
            "description": "Heuristic confidence score for receptor proxy completeness based on observed surface fields.",
            "source_columns": list(HOST_SURFACE_COLUMNS),
        },
        "host_defense_confidence": {
            "description": "Heuristic confidence score for defense proxies based on availability of defense count metadata.",
            "source_columns": ["n_defense_systems"],
        },
    },
    "phage": {
        "phage_rbp_tail_associated_proxy": {
            "description": "Binary proxy for presence of tail-associated receptor-binding machinery metadata.",
            "source_columns": ["Morphotype", "Family", "Genus"],
        },
        "phage_depolymerase_proxy": {
            "description": "Binary proxy based on keyword hints related to depolymerase/capsule tropism in taxonomy fields.",
            "source_columns": ["Morphotype", "Family", "Subfamily", "Genus", "Species"],
        },
        "phage_domain_complexity_proxy": {
            "description": "Coarse genome-size bin (1/2/3) used as a proxy for protein-domain complexity.",
            "source_columns": ["Genome_size"],
        },
        "phage_rbp_confidence": {
            "description": "Heuristic confidence score for RBP proxy based on tail-associated metadata presence.",
            "source_columns": ["Morphotype", "Family", "Genus"],
        },
        "phage_depolymerase_confidence": {
            "description": "Heuristic confidence score for depolymerase proxy based on keyword evidence.",
            "source_columns": ["Morphotype", "Family", "Subfamily", "Genus", "Species"],
        },
    },
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Host metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--phage-metadata-path",
        type=Path,
        default=Path("data/genomics/phages/guelin_collection.csv"),
        help="Phage metadata CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_a/mechanistic_proxy_features"),
        help="Directory for mechanistic proxy feature artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output file names and manifest.",
    )
    return parser.parse_args(argv)


def _read_semicolon_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def _safe_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _any_present(row: Mapping[str, str], columns: Iterable[str]) -> bool:
    return any((row.get(col) or "") != "" for col in columns)


def _to_missing_or_int(value: Optional[float]) -> object:
    if value is None:
        return ""
    return int(value)


def build_host_proxy_rows(host_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in host_rows:
        bacteria = row.get("bacteria", "")
        if not bacteria:
            continue

        known_surface_count = sum(1 for col in HOST_SURFACE_COLUMNS if (row.get(col) or "") != "")
        n_defense = _safe_float(row.get("n_defense_systems", "") or "")
        n_infections = _safe_float(row.get("n_infections", "") or "")

        receptor_conf = 0.85 if known_surface_count >= 3 else (0.65 if known_surface_count > 0 else 0.2)
        defense_conf = 0.9 if n_defense is not None else 0.25

        defense_per_infection: object = ""
        if n_defense is not None and n_infections is not None and n_infections > 0:
            defense_per_infection = round(n_defense / n_infections, 6)

        out.append(
            {
                "bacteria": bacteria,
                "host_receptor_surface_fields_observed": known_surface_count,
                "host_receptor_proxy_available": 1 if known_surface_count > 0 else 0,
                "host_receptor_surface_proxy_score": round(known_surface_count / len(HOST_SURFACE_COLUMNS), 6),
                "host_defense_system_count": _to_missing_or_int(n_defense),
                "host_defense_per_infection_proxy": defense_per_infection,
                "host_receptor_confidence": round(receptor_conf, 3),
                "host_defense_confidence": round(defense_conf, 3),
            }
        )
    return sorted(out, key=lambda row: str(row["bacteria"]))


def _row_tokens(row: Mapping[str, str], fields: Sequence[str]) -> str:
    return " ".join((row.get(f, "") or "").lower() for f in fields)


def build_phage_proxy_rows(phage_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in phage_rows:
        phage = row.get("phage", "")
        if not phage:
            continue

        genome_size = _safe_float(row.get("Genome_size", "") or "")
        tokenized = _row_tokens(row, ["Morphotype", "Family", "Subfamily", "Genus", "Species"])

        has_tail = 1 if _any_present(row, ["Morphotype", "Family", "Genus"]) else 0
        depolymerase_proxy = 1 if any(hint in tokenized for hint in KNOWN_DEPOLYMERASE_HINTS) else 0
        domain_complexity: object = ""
        if genome_size is not None:
            domain_complexity = 1 if genome_size < 45000 else (2 if genome_size < 100000 else 3)

        rbp_conf = 0.75 if has_tail else 0.3
        dep_conf = 0.65 if depolymerase_proxy else 0.35

        out.append(
            {
                "phage": phage,
                "phage_rbp_tail_associated_proxy": has_tail,
                "phage_depolymerase_proxy": depolymerase_proxy,
                "phage_domain_complexity_proxy": domain_complexity,
                "phage_rbp_confidence": round(rbp_conf, 3),
                "phage_depolymerase_confidence": round(dep_conf, 3),
            }
        )
    return sorted(out, key=lambda row: str(row["phage"]))


def _missingness(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {col: {"missing_count": 0, "missing_rate": 1.0} for col in columns}

    missing = Counter()
    for row in rows:
        for col in columns:
            if row.get(col, "") == "":
                missing[col] += 1

    n = len(rows)
    return {
        col: {
            "missing_count": int(missing[col]),
            "missing_rate": round(missing[col] / n, 6),
        }
        for col in columns
    }


def _confidence_summary(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for field in fields:
        values: List[float] = []
        for row in rows:
            raw = row.get(field, "")
            if isinstance(raw, float):
                values.append(raw)
            elif isinstance(raw, int):
                values.append(float(raw))
        if values:
            summary[field] = {
                "min": round(min(values), 6),
                "mean": round(mean(values), 6),
                "max": round(max(values), 6),
            }
    return summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    version: str,
    host_rows: Sequence[Mapping[str, object]],
    phage_rows: Sequence[Mapping[str, object]],
    host_input_path: Path,
    phage_input_path: Path,
    host_output_path: Path,
    phage_output_path: Path,
) -> Dict[str, object]:
    host_columns = list(host_rows[0].keys()) if host_rows else []
    phage_columns = list(phage_rows[0].keys()) if phage_rows else []

    return {
        "step_name": "build_mechanistic_proxy_features",
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "host_metadata_path": str(host_input_path),
            "host_metadata_sha256": _sha256(host_input_path),
            "phage_metadata_path": str(phage_input_path),
            "phage_metadata_sha256": _sha256(phage_input_path),
        },
        "artifacts": {
            "host_features_csv": str(host_output_path),
            "phage_features_csv": str(phage_output_path),
        },
        "schema": {
            "host_columns": host_columns,
            "phage_columns": phage_columns,
            "confidence_fields": {
                "host": list(HOST_CONFIDENCE_FIELDS),
                "phage": list(PHAGE_CONFIDENCE_FIELDS),
            },
            "feature_definitions": FEATURE_DEFINITIONS,
        },
        "missingness": {
            "host": _missingness(host_rows, host_columns),
            "phage": _missingness(phage_rows, phage_columns),
        },
        "confidence_summary": {
            "host": _confidence_summary(host_rows, HOST_CONFIDENCE_FIELDS),
            "phage": _confidence_summary(phage_rows, PHAGE_CONFIDENCE_FIELDS),
        },
        "counts": {
            "n_hosts": len(host_rows),
            "n_phages": len(phage_rows),
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    host_input = _read_semicolon_csv(args.host_metadata_path)
    phage_input = _read_semicolon_csv(args.phage_metadata_path)

    host_rows = build_host_proxy_rows(host_input)
    phage_rows = build_phage_proxy_rows(phage_input)

    host_output = args.output_dir / f"host_mechanistic_proxy_features_{args.version}.csv"
    phage_output = args.output_dir / f"phage_mechanistic_proxy_features_{args.version}.csv"
    manifest_output = args.output_dir / f"mechanistic_proxy_feature_manifest_{args.version}.json"

    write_csv(host_output, list(host_rows[0].keys()) if host_rows else [], host_rows)
    write_csv(phage_output, list(phage_rows[0].keys()) if phage_rows else [], phage_rows)

    manifest = build_manifest(
        version=args.version,
        host_rows=host_rows,
        phage_rows=phage_rows,
        host_input_path=args.host_metadata_path,
        phage_input_path=args.phage_metadata_path,
        host_output_path=host_output,
        phage_output_path=phage_output,
    )
    write_json(manifest_output, manifest)

    print(json.dumps({"host_rows": len(host_rows), "phage_rows": len(phage_rows), "manifest": str(manifest_output)}))


if __name__ == "__main__":
    main()
