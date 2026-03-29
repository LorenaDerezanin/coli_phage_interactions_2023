"""Shared utilities for Track L mechanistic feature builders."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

LABEL_REQUIRED_COLUMNS = ("bacteria", "phage")
SIGNIFICANT_TRUE_VALUES = frozenset({"true", "1"})


@dataclass(frozen=True)
class CollapsedProfile:
    """One unique phage-carrier profile shared by one or more source features."""

    profile_id: str
    member_features: tuple[str, ...]
    representative_feature: str
    carrier_count: int
    direct_column: str


@dataclass(frozen=True)
class CollapsedAssociation:
    """One unique collapsed phage-profile x host-feature association."""

    profile_id: str
    member_features: tuple[str, ...]
    host_feature: str
    weight: float
    pairwise_column: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_delimited_rows(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = [
            {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()} for row in reader
        ]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def require_columns(rows: Sequence[Mapping[str, str]], path: Path, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")


def normalize_token(token: str) -> str:
    return token.lower().replace("|", "_").replace("-", "_")


def build_pair_rows(label_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    require_columns(label_rows, Path("<label_rows>"), LABEL_REQUIRED_COLUMNS)
    pair_rows: List[Dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in label_rows:
        bacteria = row["bacteria"]
        phage = row["phage"]
        if not bacteria or not phage:
            raise ValueError("Encountered empty bacteria or phage in label rows")
        pair_key = (bacteria, phage)
        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate pair in label rows: {bacteria!r}, {phage!r}")
        seen_pairs.add(pair_key)
        pair_rows.append({"pair_id": f"{bacteria}__{phage}", "bacteria": bacteria, "phage": phage})
    pair_rows.sort(key=lambda row: (row["bacteria"], row["phage"]))
    return pair_rows


def collapse_duplicate_profiles(
    feature_names: Sequence[str],
    matrix: np.ndarray,
    *,
    direct_column_prefix: str,
) -> tuple[np.ndarray, List[CollapsedProfile], Dict[str, str]]:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match matrix column count")

    grouped: Dict[tuple[int, ...], List[str]] = defaultdict(list)
    for column_index, feature_name in enumerate(feature_names):
        grouped[tuple(int(value) for value in matrix[:, column_index].tolist())].append(feature_name)

    profiles: List[CollapsedProfile] = []
    collapsed_columns: List[np.ndarray] = []
    feature_to_profile_id: Dict[str, str] = {}
    feature_to_index = {feature_name: index for index, feature_name in enumerate(feature_names)}

    grouped_feature_sets = sorted(
        (tuple(sorted(features)) for features in grouped.values()), key=lambda members: members[0]
    )
    for profile_index, members in enumerate(grouped_feature_sets, start=1):
        profile_id = f"profile_{profile_index:03d}"
        representative_feature = members[0]
        direct_column = f"{direct_column_prefix}_{profile_id}_present"
        vector = matrix[:, feature_to_index[representative_feature]]
        collapsed_columns.append(vector.astype(np.int8))
        profiles.append(
            CollapsedProfile(
                profile_id=profile_id,
                member_features=members,
                representative_feature=representative_feature,
                carrier_count=int(vector.sum()),
                direct_column=direct_column,
            )
        )
        for feature_name in members:
            feature_to_profile_id[feature_name] = profile_id

    collapsed_matrix = (
        np.column_stack(collapsed_columns) if collapsed_columns else np.zeros((matrix.shape[0], 0), dtype=np.int8)
    )
    return collapsed_matrix, profiles, feature_to_profile_id


def collapse_significant_associations(
    enrichment_rows: Sequence[Mapping[str, str]],
    feature_to_profile_id: Mapping[str, str],
    profile_by_id: Mapping[str, CollapsedProfile],
    *,
    pairwise_column_prefix: str,
    missing_profile_label: str,
) -> List[CollapsedAssociation]:
    association_rows: Dict[tuple[str, str], float] = {}

    for row in enrichment_rows:
        significant = str(row["significant"]).strip().lower()
        if significant not in SIGNIFICANT_TRUE_VALUES:
            continue

        phage_feature = row["phage_feature"]
        profile_id = feature_to_profile_id.get(phage_feature)
        if profile_id is None:
            raise ValueError(f"Enrichment feature {phage_feature!r} not found in {missing_profile_label}")

        host_feature = row["host_feature"]
        weight = float(row["lysis_rate_diff"])
        key = (profile_id, host_feature)
        existing_weight = association_rows.get(key)
        if existing_weight is None:
            association_rows[key] = weight
            continue

        if abs(existing_weight - weight) > 1e-9:
            raise ValueError(
                f"Collapsed association mismatch for {profile_id!r} x {host_feature!r}: {existing_weight} vs {weight}"
            )

    associations: List[CollapsedAssociation] = []
    for profile_id, host_feature in sorted(association_rows.keys()):
        members = profile_by_id[profile_id].member_features
        host_token = normalize_token(host_feature)
        associations.append(
            CollapsedAssociation(
                profile_id=profile_id,
                member_features=members,
                host_feature=host_feature,
                weight=association_rows[(profile_id, host_feature)],
                pairwise_column=f"{pairwise_column_prefix}_{profile_id}_x_{host_token}_weight",
            )
        )
    return associations


def build_presence_index(
    entities: Sequence[str],
    feature_names: Sequence[str],
    matrix: np.ndarray,
) -> Dict[str, set[str]]:
    if matrix.shape != (len(entities), len(feature_names)):
        raise ValueError("Presence-index matrix shape does not match provided entities/features")

    presence: Dict[str, set[str]] = {entity: set() for entity in entities}
    row_indices, col_indices = np.nonzero(np.asarray(matrix, dtype=np.int8))
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist(), strict=True):
        presence[entities[row_index]].add(feature_names[col_index])
    return presence


def build_binary_value_index(
    entities: Sequence[str],
    feature_names: Sequence[str],
    matrix: np.ndarray,
) -> Dict[str, Dict[str, int]]:
    if matrix.shape != (len(entities), len(feature_names)):
        raise ValueError("Binary-value matrix shape does not match provided entities/features")

    row_values = np.asarray(matrix, dtype=np.int8).tolist()
    return {
        entity: dict(zip(feature_names, row, strict=True)) for entity, row in zip(entities, row_values, strict=True)
    }
