"""Helpers for deployable TL17 RBP-profile runtime projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from lyzortx.pipeline.track_l.steps.parse_annotations import classify_rbp_genes, parse_merged_tsv

NO_PHROG = "No_PHROG"
TL17_DIRECT_BLOCK_ID = "tl17_rbp_phrog_profiles"


@dataclass(frozen=True)
class Tl17ProfileRuntime:
    profile_id: str
    direct_column: str
    member_features: tuple[str, ...]


@dataclass(frozen=True)
class Tl17SummaryRuntime:
    profile_count_column: str
    gene_count_column: str
    unique_phrog_count_column: str


def extract_rbp_runtime_inputs(annotation_tsv_path: Path) -> tuple[set[str], int]:
    records = classify_rbp_genes(parse_merged_tsv(annotation_tsv_path))
    features = {f"RBP_PHROG_{record.phrog}" for record in records if record.phrog and record.phrog != NO_PHROG}
    return features, len(records)


def build_profile_presence(
    present_features: set[str],
    profiles: Sequence[Tl17ProfileRuntime],
) -> dict[str, int]:
    presence: dict[str, int] = {}
    for profile in profiles:
        has_member = any(member in present_features for member in profile.member_features)
        presence[profile.profile_id] = 1 if has_member else 0
    return presence


def build_direct_feature_values(
    *,
    present_features: set[str],
    rbp_gene_count: int,
    profile_presence: Mapping[str, int],
    profiles: Sequence[Tl17ProfileRuntime],
    summary: Tl17SummaryRuntime,
) -> dict[str, int]:
    values = {profile.direct_column: int(profile_presence.get(profile.profile_id, 0)) for profile in profiles}
    values[summary.profile_count_column] = int(sum(values[profile.direct_column] for profile in profiles))
    values[summary.gene_count_column] = int(rbp_gene_count)
    values[summary.unique_phrog_count_column] = int(len(present_features))
    return values


def build_tl17_runtime_payload(
    *,
    profile_rows: Sequence[Mapping[str, object]],
    summary_columns: Mapping[str, str],
) -> dict[str, object]:
    profiles = []
    for row in profile_rows:
        members = tuple(token.strip() for token in str(row["member_features"]).split("|") if token.strip())
        profiles.append(
            {
                "profile_id": str(row["profile_id"]),
                "direct_column": str(row["direct_column"]),
                "member_features": list(members),
            }
        )
    return {
        "block_id": TL17_DIRECT_BLOCK_ID,
        "profiles": profiles,
        "summary_columns": {
            "profile_count_column": str(summary_columns["profile_count_column"]),
            "gene_count_column": str(summary_columns["gene_count_column"]),
            "unique_phrog_count_column": str(summary_columns["unique_phrog_count_column"]),
        },
    }


def parse_tl17_runtime_payload(payload: Mapping[str, object]) -> tuple[list[Tl17ProfileRuntime], Tl17SummaryRuntime]:
    profiles = [
        Tl17ProfileRuntime(
            profile_id=str(row["profile_id"]),
            direct_column=str(row["direct_column"]),
            member_features=tuple(str(value) for value in row["member_features"]),
        )
        for row in payload.get("profiles", [])
    ]
    summary_payload = dict(payload.get("summary_columns", {}))
    summary = Tl17SummaryRuntime(
        profile_count_column=str(summary_payload["profile_count_column"]),
        gene_count_column=str(summary_payload["gene_count_column"]),
        unique_phrog_count_column=str(summary_payload["unique_phrog_count_column"]),
    )
    return profiles, summary
