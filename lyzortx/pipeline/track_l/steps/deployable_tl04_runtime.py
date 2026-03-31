"""Helpers for deployable TL04 anti-defense/defense runtime projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from lyzortx.pipeline.track_c.steps.build_v1_host_feature_pair_table import slugify_token
from lyzortx.pipeline.track_l.steps.parse_annotations import classify_anti_defense_genes, parse_merged_tsv

NO_PHROG = "No_PHROG"
TL04_DIRECT_BLOCK_ID = "tl04_antidef_defense_evasion"


@dataclass(frozen=True)
class Tl04ProfileRuntime:
    profile_id: str
    direct_column: str
    member_features: tuple[str, ...]


@dataclass(frozen=True)
class Tl04AssociationRuntime:
    profile_id: str
    host_feature: str
    host_feature_column: str
    pairwise_column: str
    weight: float


def build_defense_host_feature_lookup(defense_mask: Mapping[str, object]) -> dict[str, str]:
    retained_source_columns = tuple(str(value) for value in defense_mask["retained_subtype_columns"])
    retained_feature_columns = tuple(str(value) for value in defense_mask["retained_feature_columns"])
    if len(retained_source_columns) != len(retained_feature_columns):
        raise ValueError("Defense mask has mismatched retained subtype and feature columns.")
    lookup = {}
    for source_column, feature_column in zip(retained_source_columns, retained_feature_columns, strict=True):
        lookup[f"defense_{source_column}"] = feature_column
    return lookup


def extract_antidef_feature_names(annotation_tsv_path: Path) -> set[str]:
    records = parse_merged_tsv(annotation_tsv_path)
    features = {
        f"ANTIDEF_PHROG_{record.phrog}"
        for record in classify_anti_defense_genes(records)
        if record.phrog and record.phrog != NO_PHROG
    }
    return features


def build_profile_presence(
    present_features: set[str],
    profiles: Sequence[Tl04ProfileRuntime],
) -> dict[str, int]:
    presence: dict[str, int] = {}
    for profile in profiles:
        has_member = any(member in present_features for member in profile.member_features)
        presence[profile.profile_id] = 1 if has_member else 0
    return presence


def build_direct_feature_values(
    profile_presence: Mapping[str, int],
    profiles: Sequence[Tl04ProfileRuntime],
) -> dict[str, int]:
    return {profile.direct_column: int(profile_presence.get(profile.profile_id, 0)) for profile in profiles}


def build_pairwise_feature_values(
    *,
    host_row: Mapping[str, object],
    profile_presence: Mapping[str, int],
    associations: Sequence[Tl04AssociationRuntime],
) -> dict[str, float]:
    pairwise_values: dict[str, float] = {}
    for association in associations:
        host_has_feature = int(host_row.get(association.host_feature_column, 0) or 0) > 0
        phage_has_profile = int(profile_presence.get(association.profile_id, 0)) > 0
        pairwise_values[association.pairwise_column] = (
            association.weight if host_has_feature and phage_has_profile else 0.0
        )
    return pairwise_values


def build_tl04_runtime_payload(
    *,
    profile_rows: Sequence[Mapping[str, object]],
    metadata_rows: Sequence[Mapping[str, object]],
    defense_mask: Mapping[str, object],
) -> dict[str, object]:
    defense_lookup = build_defense_host_feature_lookup(defense_mask)
    profiles = []
    associations = []
    for row in profile_rows:
        members = tuple(token.strip() for token in str(row["member_features"]).split("|") if token.strip())
        profiles.append(
            {
                "profile_id": str(row["profile_id"]),
                "direct_column": str(row["direct_column"]),
                "member_features": list(members),
            }
        )
    for row in metadata_rows:
        if str(row["block_type"]) != "pairwise_defense_evasion":
            continue
        host_feature = str(row["host_feature"])
        host_feature_column = defense_lookup.get(host_feature)
        if host_feature_column is None:
            fallback = f"host_defense_subtype_{slugify_token(host_feature.removeprefix('defense_'))}"
            host_feature_column = fallback
        associations.append(
            {
                "profile_id": str(row["profile_id"]),
                "host_feature": host_feature,
                "host_feature_column": host_feature_column,
                "pairwise_column": str(row["column_name"]),
                "weight": float(row["weight"]),
            }
        )
    return {
        "block_id": TL04_DIRECT_BLOCK_ID,
        "profiles": profiles,
        "associations": associations,
        "defense_host_feature_lookup": defense_lookup,
    }


def parse_tl04_runtime_payload(
    payload: Mapping[str, object],
) -> tuple[list[Tl04ProfileRuntime], list[Tl04AssociationRuntime]]:
    profiles = [
        Tl04ProfileRuntime(
            profile_id=str(row["profile_id"]),
            direct_column=str(row["direct_column"]),
            member_features=tuple(str(value) for value in row["member_features"]),
        )
        for row in payload.get("profiles", [])
    ]
    associations = [
        Tl04AssociationRuntime(
            profile_id=str(row["profile_id"]),
            host_feature=str(row["host_feature"]),
            host_feature_column=str(row["host_feature_column"]),
            pairwise_column=str(row["pairwise_column"]),
            weight=float(row["weight"]),
        )
        for row in payload.get("associations", [])
    ]
    return profiles, associations
