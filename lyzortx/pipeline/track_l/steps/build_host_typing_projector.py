#!/usr/bin/env python3
"""TL16: Build a genome-derived host typing projector for deployable bundle parity."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

OUTPUT_FEATURES_FILENAME = "tl16_host_typing_projected_features.csv"
OUTPUT_SCHEMA_FILENAME = "tl16_host_typing_schema.json"
OUTPUT_STATUS_FILENAME = "tl16_metadata_projection_status.csv"
OUTPUT_VALIDATION_SUMMARY_FILENAME = "tl16_validation_summary.csv"
OUTPUT_VALIDATION_DETAILS_FILENAME = "tl16_validation_details.csv"
OUTPUT_MANIFEST_FILENAME = "tl16_manifest.json"

MISSING_TOKENS = {"", "-", "Unknown", "unknown", "nan", "NaN", "None", "none"}
K_TYPE_PATTERN = re.compile(r"^K[0-9A-Za-z._-]+$")

PROJECTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "bacteria",
    "source_gembase",
    "host_clermont_phylo",
    "host_clermont_phylo_status",
    "host_st_warwick",
    "host_st_warwick_status",
    "host_o_type",
    "host_h_type",
    "host_serotype",
    "host_serotype_status",
    "host_abc_serotype",
    "host_abc_serotype_status",
    "host_surface_klebsiella_capsule_type",
    "host_surface_klebsiella_capsule_type_status",
    "host_capsule_call_status",
    "host_capsule_best_match_locus",
    "host_capsule_best_match_type",
    "host_capsule_match_confidence",
    "host_capsule_problem_flags",
    "host_capsule_coverage_pct",
    "host_capsule_identity_pct",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Semicolon-delimited panel host metadata CSV.",
    )
    parser.add_argument(
        "--phylogroup-path",
        type=Path,
        default=Path("data/genomics/bacteria/isolation_strains/Phylogroup/host_phylogroup.tsv"),
        help="Tab-delimited Clermont phylogroup calls keyed by Gembase assembly ID.",
    )
    parser.add_argument(
        "--sequence-type-path",
        type=Path,
        default=Path("data/genomics/bacteria/isolation_strains/ST/host_ST.tsv"),
        help="Tab-delimited mlst calls keyed by Gembase assembly ID.",
    )
    parser.add_argument(
        "--serotype-path",
        type=Path,
        default=Path("data/genomics/bacteria/o_type/output.tsv"),
        help="Tab-delimited ECTyper serotype calls keyed by bacteria ID.",
    )
    parser.add_argument(
        "--capsule-high-hit-path",
        type=Path,
        default=Path("data/genomics/bacteria/capsules/klebsiella_capsules/kaptive_results_high_hits_cured.txt"),
        help="Tab-delimited curated high-hit Kaptive capsule calls keyed by bacteria ID.",
    )
    parser.add_argument(
        "--capsule-all-results-path",
        type=Path,
        default=Path("data/genomics/bacteria/capsules/klebsiella_capsules/kaptive_results_table_colocoli.txt"),
        help="Tab-delimited full Kaptive results table keyed by bacteria ID in the Assembly column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lyzortx/generated_outputs/track_l/host_typing_projector"),
        help="Directory for TL16 generated outputs.",
    )
    return parser.parse_args(argv)


def _normalize_category(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in MISSING_TOKENS:
        return ""
    return text


def _normalize_sequence_type(value: object) -> str:
    text = _normalize_category(value)
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _normalize_percent(value: object) -> str:
    text = _normalize_category(value)
    if not text:
        return ""
    if text.endswith("%"):
        text = text[:-1]
    try:
        return f"{float(text):.3f}"
    except ValueError:
        return ""


def derive_serotype(o_type: str, h_type: str) -> str:
    if o_type and h_type:
        return f"{o_type}:{h_type}"
    if o_type:
        return o_type
    if h_type:
        return h_type
    return ""


def _split_ambiguous_tokens(value: str) -> set[str]:
    return {token.strip() for token in value.split("/") if token.strip()}


def _is_loose_match(*, family: str, source_value: str, projected_value: str) -> bool:
    if not source_value or not projected_value:
        return False
    if source_value == projected_value:
        return True
    if family == "serotype" and ":" in source_value and ":" in projected_value:
        source_o, source_h = source_value.split(":", maxsplit=1)
        projected_o, projected_h = projected_value.split(":", maxsplit=1)
        return source_h == projected_h and source_o in _split_ambiguous_tokens(projected_o)
    return source_value in _split_ambiguous_tokens(projected_value)


def _normalize_capsule_proxy(value: object) -> str:
    text = _normalize_category(value)
    if not text:
        return ""
    if text.lower().startswith("unknown"):
        return ""
    if not K_TYPE_PATTERN.match(text):
        return ""
    return text


def _require_columns(frame: pd.DataFrame, *, path: Path, required: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame


def load_panel_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Panel metadata CSV not found: {path}")
    frame = pd.read_csv(path, sep=";").fillna("")
    frame = _require_columns(
        frame,
        path=path,
        required=(
            "bacteria",
            "Gembase",
            "Clermont_Phylo",
            "ST_Warwick",
            "O-type",
            "H-type",
            "ABC_serotype",
            "Klebs_capsule_type",
        ),
    )
    if frame.empty:
        raise ValueError(f"Panel metadata CSV was empty at {path}")
    if frame["bacteria"].duplicated().any():
        duplicate = frame.loc[frame["bacteria"].duplicated(), "bacteria"].iloc[0]
        raise ValueError(f"Duplicate bacteria ID {duplicate!r} in {path}")
    return frame


def load_phylogroup_calls(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phylogroup TSV not found: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gembase", "genes", "presence_absence", "allele", "phylogroup", "mash"],
    ).fillna("")
    if frame.empty:
        raise ValueError(f"Phylogroup TSV was empty at {path}")
    return frame[["gembase", "phylogroup"]].drop_duplicates()


def load_sequence_type_calls(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sequence-type TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t").fillna("")
    frame = _require_columns(frame, path=path, required=("FILE", "ST"))
    if frame.empty:
        raise ValueError(f"Sequence-type TSV was empty at {path}")
    return frame[["FILE", "ST"]].drop_duplicates()


def load_serotype_calls(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Serotype TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t").fillna("")
    frame = _require_columns(frame, path=path, required=("Name", "O-type", "H-type", "Serotype"))
    if frame.empty:
        raise ValueError(f"Serotype TSV was empty at {path}")
    return frame[["Name", "O-type", "H-type", "Serotype"]].drop_duplicates()


def load_capsule_high_hits(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Capsule high-hit TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t").fillna("")
    frame = _require_columns(frame, path=path, required=("bacteria", "Klebs_capsule_type"))
    if frame.empty:
        raise ValueError(f"Capsule high-hit TSV was empty at {path}")
    return frame[["bacteria", "Klebs_capsule_type"]].drop_duplicates()


def load_capsule_calls(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Capsule results TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t").fillna("")
    frame = _require_columns(
        frame,
        path=path,
        required=(
            "Assembly",
            "Best match locus",
            "Best match type",
            "Match confidence",
            "Problems",
            "Coverage",
            "Identity",
        ),
    )
    if frame.empty:
        raise ValueError(f"Capsule results TSV was empty at {path}")
    return frame[
        ["Assembly", "Best match locus", "Best match type", "Match confidence", "Problems", "Coverage", "Identity"]
    ].drop_duplicates()


def determine_capsule_call_status(
    *,
    high_hit_type: str,
    proxy_abc_serotype: str,
    best_match_type: str,
    match_confidence: str,
) -> str:
    if high_hit_type:
        return "high_hit_typed"
    if proxy_abc_serotype:
        return "typed_proxy"
    if best_match_type or match_confidence:
        return "attempted_no_stable_type"
    return "not_callable"


def build_projected_feature_rows(
    *,
    panel_metadata: pd.DataFrame,
    phylogroup_calls: pd.DataFrame,
    sequence_type_calls: pd.DataFrame,
    serotype_calls: pd.DataFrame,
    capsule_high_hits: pd.DataFrame,
    capsule_calls: pd.DataFrame,
) -> list[dict[str, object]]:
    merged = panel_metadata.merge(
        phylogroup_calls.rename(columns={"gembase": "Gembase", "phylogroup": "projected_phylogroup"}),
        on="Gembase",
        how="left",
    )
    merged = merged.merge(
        sequence_type_calls.rename(columns={"FILE": "Gembase", "ST": "projected_st"}),
        on="Gembase",
        how="left",
    )
    merged = merged.merge(
        serotype_calls.rename(
            columns={
                "Name": "bacteria",
                "O-type": "projected_o_type",
                "H-type": "projected_h_type",
                "Serotype": "projected_serotype",
            }
        ),
        on="bacteria",
        how="left",
    )
    merged = merged.merge(
        capsule_high_hits.rename(columns={"Klebs_capsule_type": "projected_klebsiella_capsule_type"}),
        on="bacteria",
        how="left",
    )
    merged = merged.merge(
        capsule_calls.rename(
            columns={
                "Assembly": "bacteria",
                "Best match locus": "capsule_best_match_locus",
                "Best match type": "capsule_best_match_type",
                "Match confidence": "capsule_match_confidence",
                "Problems": "capsule_problems",
                "Coverage": "capsule_coverage",
                "Identity": "capsule_identity",
            }
        ),
        on="bacteria",
        how="left",
    )
    if merged.empty:
        raise ValueError("Projected typing merge unexpectedly produced zero rows.")

    feature_rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        phylogroup = _normalize_category(row.get("projected_phylogroup", ""))
        sequence_type = _normalize_sequence_type(row.get("projected_st", ""))
        o_type = _normalize_category(row.get("projected_o_type", ""))
        h_type = _normalize_category(row.get("projected_h_type", ""))
        serotype = _normalize_category(row.get("projected_serotype", "")) or derive_serotype(o_type, h_type)
        high_hit_capsule = _normalize_capsule_proxy(row.get("projected_klebsiella_capsule_type", ""))
        best_match_type = _normalize_category(row.get("capsule_best_match_type", ""))
        proxy_abc_serotype = high_hit_capsule or _normalize_capsule_proxy(best_match_type)
        match_confidence = _normalize_category(row.get("capsule_match_confidence", ""))

        feature_rows.append(
            {
                "bacteria": row["bacteria"],
                "source_gembase": row["Gembase"],
                "host_clermont_phylo": phylogroup,
                "host_clermont_phylo_status": "called" if phylogroup else "not_callable",
                "host_st_warwick": sequence_type,
                "host_st_warwick_status": "called" if sequence_type else "not_callable",
                "host_o_type": o_type,
                "host_h_type": h_type,
                "host_serotype": serotype,
                "host_serotype_status": "called" if serotype else "not_callable",
                "host_abc_serotype": proxy_abc_serotype,
                "host_abc_serotype_status": "typed_proxy" if proxy_abc_serotype else "not_callable",
                "host_surface_klebsiella_capsule_type": high_hit_capsule,
                "host_surface_klebsiella_capsule_type_status": "high_hit_typed" if high_hit_capsule else "not_callable",
                "host_capsule_call_status": determine_capsule_call_status(
                    high_hit_type=high_hit_capsule,
                    proxy_abc_serotype=proxy_abc_serotype,
                    best_match_type=best_match_type,
                    match_confidence=match_confidence,
                ),
                "host_capsule_best_match_locus": _normalize_category(row.get("capsule_best_match_locus", "")),
                "host_capsule_best_match_type": best_match_type,
                "host_capsule_match_confidence": match_confidence,
                "host_capsule_problem_flags": _normalize_category(row.get("capsule_problems", "")),
                "host_capsule_coverage_pct": _normalize_percent(row.get("capsule_coverage", "")),
                "host_capsule_identity_pct": _normalize_percent(row.get("capsule_identity", "")),
            }
        )

    feature_rows = sorted(feature_rows, key=lambda item: str(item["bacteria"]))
    if [column for column in feature_rows[0].keys()] != list(PROJECTED_FEATURE_COLUMNS):
        raise ValueError("Projected feature row schema drifted from PROJECTED_FEATURE_COLUMNS.")
    return feature_rows


def build_metadata_projection_status_rows() -> list[dict[str, object]]:
    rows = [
        {
            "legacy_column": "Clermont_Phylo",
            "projected_column": "host_clermont_phylo",
            "projection_status": "reproduced_exactly",
            "rationale": "Direct Clermont phylogroup call from assembly-derived typing output.",
        },
        {
            "legacy_column": "ST_Warwick",
            "projected_column": "host_st_warwick",
            "projection_status": "reproduced_exactly",
            "rationale": "Direct mlst Achtman ST call from assembly-derived typing output.",
        },
        {
            "legacy_column": "O-type",
            "projected_column": "host_o_type",
            "projection_status": "reproduced_exactly",
            "rationale": "Direct ECTyper O-antigen call from assembly-derived serotyping output.",
        },
        {
            "legacy_column": "H-type",
            "projected_column": "host_h_type",
            "projection_status": "reproduced_exactly",
            "rationale": "Direct ECTyper H-antigen call from assembly-derived serotyping output.",
        },
        {
            "legacy_column": "ABC_serotype",
            "projected_column": "host_abc_serotype",
            "projection_status": "reproduced_as_proxy",
            "rationale": "Approximated from Kaptive best-match K type when a stable K-type call exists; non-K ABC labels remain unsupported.",
        },
        {
            "legacy_column": "Klebs_capsule_type",
            "projected_column": "host_surface_klebsiella_capsule_type",
            "projection_status": "reproduced_exactly_when_callable",
            "rationale": "Copied from the curated high-hit Kaptive call table on the subset with stable capsule calls.",
        },
        {
            "legacy_column": "Capsule_ABC",
            "projected_column": "host_capsule_call_status",
            "projection_status": "family_proxy_only",
            "rationale": "The repo does not contain the original ABC-capsule caller output, so only a generic capsule-callability proxy is available.",
        },
        {
            "legacy_column": "Capsule_GroupIV_e",
            "projected_column": "host_capsule_call_status",
            "projection_status": "family_proxy_only",
            "rationale": "Exact Group IV capsule-family replication is unsupported with the committed assets; capsule callability is retained as a proxy only.",
        },
        {
            "legacy_column": "Capsule_GroupIV_e_stricte",
            "projected_column": "host_capsule_call_status",
            "projection_status": "family_proxy_only",
            "rationale": "Exact strict Group IV capsule-family replication is unsupported with the committed assets; capsule callability is retained as a proxy only.",
        },
        {
            "legacy_column": "Capsule_GroupIV_s",
            "projected_column": "host_capsule_call_status",
            "projection_status": "family_proxy_only",
            "rationale": "Exact Group IV-s level replication is unsupported with the committed assets; capsule callability is retained as a proxy only.",
        },
        {
            "legacy_column": "Capsule_Wzy_stricte",
            "projected_column": "host_capsule_call_status",
            "projection_status": "family_proxy_only",
            "rationale": "Exact Wzy-stricte replication is unsupported with the committed assets; capsule callability is retained as a proxy only.",
        },
        {
            "legacy_column": "Host",
            "projected_column": "",
            "projection_status": "non_derivable_metadata",
            "rationale": "Collection/source metadata, not a genome-derived typing feature.",
        },
        {
            "legacy_column": "Origin",
            "projected_column": "",
            "projection_status": "non_derivable_metadata",
            "rationale": "Collection/source metadata, not a genome-derived typing feature.",
        },
        {
            "legacy_column": "Pathotype",
            "projected_column": "",
            "projection_status": "non_derivable_metadata",
            "rationale": "Phenotype/curation metadata, not a genome-derived typing feature in the committed runtime assets.",
        },
        {
            "legacy_column": "Collection",
            "projected_column": "",
            "projection_status": "non_derivable_metadata",
            "rationale": "Collection membership is provenance metadata, not something projected from a host assembly.",
        },
        {
            "legacy_column": "Mouse_killed_10",
            "projected_column": "",
            "projection_status": "non_derivable_metadata",
            "rationale": "Wet-lab virulence assay metadata, not derivable from a raw assembly alone.",
        },
        {
            "legacy_column": "LPS_type",
            "projected_column": "",
            "projection_status": "handled_by_tl15_not_tl16",
            "rationale": "Surface/LPS projection belongs to the TL15 host-surface projector rather than the TL16 typing projector.",
        },
    ]
    return rows


def _summary_row(
    *,
    family: str,
    legacy_column: str,
    projected_column: str,
    projected_status_column: str | None,
    panel_metadata: pd.DataFrame,
    projected_frame: pd.DataFrame,
    note: str,
    source_transform: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if source_transform is None:
        source_values = panel_metadata[legacy_column].map(_normalize_category)
    else:
        source_values = source_transform(panel_metadata)

    projected_values = projected_frame[projected_column].map(_normalize_category)
    if projected_status_column is None:
        callable_mask = pd.Series([False] * len(projected_frame), index=projected_frame.index)
    else:
        callable_mask = projected_frame[projected_status_column].astype(str).ne("not_callable")
    comparable_mask = source_values.ne("") & projected_values.ne("")
    exact_matches = (source_values == projected_values) & comparable_mask
    loose_matches = (
        pd.Series(
            [
                _is_loose_match(family=family, source_value=source_value, projected_value=projected_value)
                for source_value, projected_value in zip(source_values, projected_values, strict=True)
            ],
            index=projected_frame.index,
        )
        & comparable_mask
    )

    validation_status = "unsupported"
    if int(comparable_mask.sum()) > 0:
        validation_status = "clean" if bool(exact_matches[comparable_mask].all()) else "noisy_or_partial"

    details = [
        {
            "bacteria": bacteria,
            "feature_family": family,
            "legacy_column": legacy_column,
            "projected_column": projected_column,
            "source_value": source_value,
            "projected_value": projected_value,
            "projected_status": projected_status if projected_status_column else "unsupported",
            "comparable": int(comparable),
            "exact_match": int(match),
            "loose_match": int(loose_match),
        }
        for bacteria, source_value, projected_value, projected_status, comparable, match, loose_match in zip(
            panel_metadata["bacteria"],
            source_values,
            projected_values,
            projected_frame[projected_status_column]
            if projected_status_column
            else ["unsupported"] * len(projected_frame),
            comparable_mask,
            exact_matches,
            loose_matches,
            strict=True,
        )
    ]
    summary = {
        "feature_family": family,
        "legacy_column": legacy_column,
        "projected_column": projected_column,
        "validation_status": validation_status,
        "panel_host_count": len(panel_metadata),
        "source_nonempty_count": int(source_values.ne("").sum()),
        "callable_host_count": int(callable_mask.sum()),
        "comparable_host_count": int(comparable_mask.sum()),
        "exact_match_count": int(exact_matches.sum()),
        "exact_match_rate": f"{float(exact_matches.sum() / comparable_mask.sum()):.6f}"
        if comparable_mask.any()
        else "",
        "loose_match_count": int(loose_matches.sum()),
        "loose_match_rate": f"{float(loose_matches.sum() / comparable_mask.sum()):.6f}"
        if comparable_mask.any()
        else "",
        "note": note,
    }
    return summary, details


def build_validation_rows(
    *,
    panel_metadata: pd.DataFrame,
    projected_feature_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    projected_frame = pd.DataFrame(projected_feature_rows)
    panel_frame = panel_metadata.copy()
    joined = panel_frame.merge(projected_frame, on="bacteria", how="left", validate="one_to_one")
    if joined.empty:
        raise ValueError("Validation join unexpectedly produced zero rows.")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    comparisons = [
        (
            "phylogroup",
            "Clermont_Phylo",
            "host_clermont_phylo",
            "host_clermont_phylo_status",
            "Clermont calls are only available for the Gembase-keyed assembly subset.",
            None,
        ),
        (
            "sequence_type",
            "ST_Warwick",
            "host_st_warwick",
            "host_st_warwick_status",
            "MLST calls are only available for the Gembase-keyed assembly subset.",
            lambda frame: frame["ST_Warwick"].map(_normalize_sequence_type),
        ),
        (
            "o_type",
            "O-type",
            "host_o_type",
            "host_serotype_status",
            "ECTyper panel calls cover nearly the full panel and should reproduce the old O-type field directly.",
            None,
        ),
        (
            "h_type",
            "H-type",
            "host_h_type",
            "host_serotype_status",
            "ECTyper panel calls cover nearly the full panel and should reproduce the old H-type field directly.",
            None,
        ),
        (
            "serotype",
            "Serotype",
            "host_serotype",
            "host_serotype_status",
            "Serotype is derived from the projected O/H calls and compared against the old O-type/H-type combination.",
            lambda frame: (
                frame["O-type"]
                .map(_normalize_category)
                .combine(frame["H-type"].map(_normalize_category), derive_serotype)
            ),
        ),
        (
            "klebsiella_capsule_type",
            "Klebs_capsule_type",
            "host_surface_klebsiella_capsule_type",
            "host_surface_klebsiella_capsule_type_status",
            "Exact comparison is only possible on the sparse high-hit Kaptive subset.",
            None,
        ),
        (
            "abc_serotype_proxy",
            "ABC_serotype",
            "host_abc_serotype",
            "host_abc_serotype_status",
            "This is an intentionally noisy Kaptive-derived proxy, not an exact replica of the heterogeneous ABC serotype field.",
            None,
        ),
    ]

    for family, legacy_column, projected_column, status_column, note, source_transform in comparisons:
        summary, details = _summary_row(
            family=family,
            legacy_column=legacy_column,
            projected_column=projected_column,
            projected_status_column=status_column,
            panel_metadata=joined,
            projected_frame=joined,
            note=note,
            source_transform=source_transform,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    unsupported_family = {
        "feature_family": "legacy_capsule_binary_flags",
        "legacy_column": "Capsule_ABC|Capsule_GroupIV_e|Capsule_GroupIV_e_stricte|Capsule_GroupIV_s|Capsule_Wzy_stricte",
        "projected_column": "host_capsule_call_status",
        "validation_status": "unsupported",
        "panel_host_count": len(joined),
        "source_nonempty_count": int(
            joined[
                [
                    "Capsule_ABC",
                    "Capsule_GroupIV_e",
                    "Capsule_GroupIV_e_stricte",
                    "Capsule_GroupIV_s",
                    "Capsule_Wzy_stricte",
                ]
            ]
            .apply(lambda column: column.map(_normalize_category))
            .ne("")
            .any(axis=1)
            .sum()
        ),
        "callable_host_count": int(joined["host_capsule_call_status"].astype(str).ne("not_callable").sum()),
        "comparable_host_count": 0,
        "exact_match_count": 0,
        "exact_match_rate": "",
        "loose_match_count": 0,
        "loose_match_rate": "",
        "note": "Exact legacy capsule-binary replication is unsupported with the committed assets; TL16 only preserves capsule callability/type proxies.",
    }
    summary_rows.append(unsupported_family)
    return summary_rows, detail_rows


def build_schema_payload() -> dict[str, object]:
    return {
        "task_id": "TL16",
        "join_key": "bacteria",
        "ordered_feature_columns": list(PROJECTED_FEATURE_COLUMNS),
        "feature_families": {
            "phylogroup": ["host_clermont_phylo", "host_clermont_phylo_status"],
            "sequence_type": ["host_st_warwick", "host_st_warwick_status"],
            "serotype": ["host_o_type", "host_h_type", "host_serotype", "host_serotype_status"],
            "capsule": [
                "host_abc_serotype",
                "host_abc_serotype_status",
                "host_surface_klebsiella_capsule_type",
                "host_surface_klebsiella_capsule_type_status",
                "host_capsule_call_status",
                "host_capsule_best_match_locus",
                "host_capsule_best_match_type",
                "host_capsule_match_confidence",
                "host_capsule_problem_flags",
                "host_capsule_coverage_pct",
                "host_capsule_identity_pct",
            ],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    LOGGER.info("Starting TL16 host typing projector build")

    panel_metadata = load_panel_metadata(args.panel_metadata_path)
    phylogroup_calls = load_phylogroup_calls(args.phylogroup_path)
    sequence_type_calls = load_sequence_type_calls(args.sequence_type_path)
    serotype_calls = load_serotype_calls(args.serotype_path)
    capsule_high_hits = load_capsule_high_hits(args.capsule_high_hit_path)
    capsule_calls = load_capsule_calls(args.capsule_all_results_path)

    feature_rows = build_projected_feature_rows(
        panel_metadata=panel_metadata,
        phylogroup_calls=phylogroup_calls,
        sequence_type_calls=sequence_type_calls,
        serotype_calls=serotype_calls,
        capsule_high_hits=capsule_high_hits,
        capsule_calls=capsule_calls,
    )
    status_rows = build_metadata_projection_status_rows()
    validation_summary_rows, validation_detail_rows = build_validation_rows(
        panel_metadata=panel_metadata,
        projected_feature_rows=feature_rows,
    )
    schema_payload = build_schema_payload()

    ensure_directory(args.output_dir)
    write_csv(args.output_dir / OUTPUT_FEATURES_FILENAME, PROJECTED_FEATURE_COLUMNS, feature_rows)
    write_csv(
        args.output_dir / OUTPUT_STATUS_FILENAME,
        ["legacy_column", "projected_column", "projection_status", "rationale"],
        status_rows,
    )
    write_csv(
        args.output_dir / OUTPUT_VALIDATION_SUMMARY_FILENAME,
        [
            "feature_family",
            "legacy_column",
            "projected_column",
            "validation_status",
            "panel_host_count",
            "source_nonempty_count",
            "callable_host_count",
            "comparable_host_count",
            "exact_match_count",
            "exact_match_rate",
            "loose_match_count",
            "loose_match_rate",
            "note",
        ],
        validation_summary_rows,
    )
    write_csv(
        args.output_dir / OUTPUT_VALIDATION_DETAILS_FILENAME,
        [
            "bacteria",
            "feature_family",
            "legacy_column",
            "projected_column",
            "source_value",
            "projected_value",
            "projected_status",
            "comparable",
            "exact_match",
            "loose_match",
        ],
        validation_detail_rows,
    )
    write_json(args.output_dir / OUTPUT_SCHEMA_FILENAME, schema_payload)
    write_json(
        args.output_dir / OUTPUT_MANIFEST_FILENAME,
        {
            "task_id": "TL16",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_paths": {
                "panel_metadata_path": str(args.panel_metadata_path),
                "phylogroup_path": str(args.phylogroup_path),
                "sequence_type_path": str(args.sequence_type_path),
                "serotype_path": str(args.serotype_path),
                "capsule_high_hit_path": str(args.capsule_high_hit_path),
                "capsule_all_results_path": str(args.capsule_all_results_path),
            },
            "counts": {
                "panel_host_count": len(panel_metadata),
                "phylogroup_call_count": int(
                    sum(row["host_clermont_phylo_status"] == "called" for row in feature_rows)
                ),
                "sequence_type_call_count": int(sum(row["host_st_warwick_status"] == "called" for row in feature_rows)),
                "serotype_call_count": int(sum(row["host_serotype_status"] == "called" for row in feature_rows)),
                "capsule_high_hit_type_count": int(
                    sum(row["host_surface_klebsiella_capsule_type_status"] == "high_hit_typed" for row in feature_rows)
                ),
                "capsule_any_call_count": int(
                    sum(row["host_capsule_call_status"] != "not_callable" for row in feature_rows)
                ),
            },
            "outputs": {
                "projected_features_csv": str(args.output_dir / OUTPUT_FEATURES_FILENAME),
                "schema_json": str(args.output_dir / OUTPUT_SCHEMA_FILENAME),
                "projection_status_csv": str(args.output_dir / OUTPUT_STATUS_FILENAME),
                "validation_summary_csv": str(args.output_dir / OUTPUT_VALIDATION_SUMMARY_FILENAME),
                "validation_details_csv": str(args.output_dir / OUTPUT_VALIDATION_DETAILS_FILENAME),
            },
        },
    )
    LOGGER.info("Completed TL16 host typing projector build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
