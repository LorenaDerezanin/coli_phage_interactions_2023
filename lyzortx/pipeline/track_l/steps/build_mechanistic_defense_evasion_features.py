#!/usr/bin/env python3
"""TL04: Build mechanistic defense-evasion features from annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_a.steps import build_track_a_foundation
from lyzortx.pipeline.track_l.steps.run_enrichment_analysis import (
    CACHED_ANNOTATIONS_DIR,
    DEFENSE_SUBTYPES_PATH,
    LABEL_SET_V1_PATH,
    load_defense_host_matrix,
    load_pharokka_phrog_matrices,
    main as run_tl02_enrichment,
)

logger = logging.getLogger(__name__)

DEFAULT_ANTIDEF_ENRICHMENT_PATH = Path(
    "lyzortx/generated_outputs/track_l/enrichment/enrichment_antidef_phrog_x_defense_subtype.csv"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/mechanistic_defense_evasion_features")
ENRICHMENT_REQUIRED_COLUMNS = (
    "phage_feature",
    "host_feature",
    "lysis_rate_diff",
    "significant",
)
LABEL_REQUIRED_COLUMNS = ("bacteria", "phage")
EXPERIMENTAL_STATUS = "experimental_candidate"


@dataclass(frozen=True)
class CollapsedProfile:
    """One unique phage-carrier profile shared by one or more anti-defense PHROGs."""

    profile_id: str
    member_features: tuple[str, ...]
    representative_feature: str
    carrier_count: int
    direct_column: str


@dataclass(frozen=True)
class CollapsedAssociation:
    """One unique collapsed anti-defense profile x host-defense association."""

    profile_id: str
    member_features: tuple[str, ...]
    host_feature: str
    weight: float
    pairwise_column: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-path",
        type=Path,
        default=LABEL_SET_V1_PATH,
        help="Track A label_set_v1_pairs.csv used to define the panel pair universe.",
    )
    parser.add_argument(
        "--cached-annotations-dir",
        type=Path,
        default=CACHED_ANNOTATIONS_DIR,
        help="Flat directory of cached Pharokka merged TSVs.",
    )
    parser.add_argument(
        "--defense-path",
        type=Path,
        default=DEFENSE_SUBTYPES_PATH,
        help="DefenseFinder subtype CSV used in TL02.",
    )
    parser.add_argument(
        "--antidef-enrichment-path",
        type=Path,
        default=DEFAULT_ANTIDEF_ENRICHMENT_PATH,
        help="TL02 anti-defense x defense-subtype enrichment CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated TL04 outputs.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output filenames and the manifest.",
    )
    return parser.parse_args(argv)


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
        direct_column = f"tl04_phage_antidef_{profile_id}_present"
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
) -> List[CollapsedAssociation]:
    association_rows: Dict[tuple[str, str], Dict[str, object]] = {}

    for row in enrichment_rows:
        significant = str(row["significant"]).strip().lower()
        if significant not in {"true", "1"}:
            continue

        phage_feature = row["phage_feature"]
        profile_id = feature_to_profile_id.get(phage_feature)
        if profile_id is None:
            raise ValueError(f"Enrichment feature {phage_feature!r} not found in collapsed anti-defense profiles")

        host_feature = row["host_feature"]
        weight = float(row["lysis_rate_diff"])
        key = (profile_id, host_feature)
        existing = association_rows.get(key)
        if existing is None:
            association_rows[key] = {"weight": weight, "source_features": {phage_feature}}
            continue

        if abs(float(existing["weight"]) - weight) > 1e-9:
            raise ValueError(
                f"Collapsed association mismatch for {profile_id!r} x {host_feature!r}: "
                f"{existing['weight']} vs {weight}"
            )
        existing["source_features"].add(phage_feature)

    associations: List[CollapsedAssociation] = []
    for profile_id, host_feature in sorted(association_rows.keys()):
        members = profile_by_id[profile_id].member_features
        host_token = normalize_token(host_feature)
        associations.append(
            CollapsedAssociation(
                profile_id=profile_id,
                member_features=members,
                host_feature=host_feature,
                weight=float(association_rows[(profile_id, host_feature)]["weight"]),
                pairwise_column=f"tl04_pair_{profile_id}_x_{host_token}_weight",
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
    presence: Dict[str, set[str]] = {}
    for row_index, entity in enumerate(entities):
        active = {
            feature_names[column_index] for column_index, value in enumerate(matrix[row_index]) if int(value) == 1
        }
        presence[entity] = active
    return presence


def build_binary_value_index(
    entities: Sequence[str],
    feature_names: Sequence[str],
    matrix: np.ndarray,
) -> Dict[str, Dict[str, int]]:
    if matrix.shape != (len(entities), len(feature_names)):
        raise ValueError("Binary-value matrix shape does not match provided entities/features")
    values: Dict[str, Dict[str, int]] = {}
    for row_index, entity in enumerate(entities):
        values[entity] = {
            feature_names[column_index]: int(matrix[row_index, column_index])
            for column_index in range(len(feature_names))
        }
    return values


def build_feature_rows(
    pair_rows: Sequence[Mapping[str, str]],
    phage_to_profile_presence: Mapping[str, Mapping[str, int]],
    bacteria_to_defense_features: Mapping[str, set[str]],
    profiles: Sequence[CollapsedProfile],
    associations: Sequence[CollapsedAssociation],
) -> tuple[List[Dict[str, object]], List[str]]:
    direct_columns = [profile.direct_column for profile in profiles]
    pairwise_columns = [association.pairwise_column for association in associations]
    output_columns = ["pair_id", "bacteria", "phage", *direct_columns, *pairwise_columns]

    feature_rows: List[Dict[str, object]] = []
    for pair_row in pair_rows:
        bacteria = pair_row["bacteria"]
        phage = pair_row["phage"]
        phage_profiles = phage_to_profile_presence.get(phage)
        host_features = bacteria_to_defense_features.get(bacteria)
        if phage_profiles is None:
            raise ValueError(f"Missing collapsed anti-defense profile row for phage {phage!r}")
        if host_features is None:
            raise ValueError(f"Missing defense feature row for bacteria {bacteria!r}")

        row: Dict[str, object] = {
            "pair_id": pair_row["pair_id"],
            "bacteria": bacteria,
            "phage": phage,
        }
        for profile in profiles:
            row[profile.direct_column] = phage_profiles[profile.profile_id]
        for association in associations:
            phage_has_profile = phage_profiles[association.profile_id]
            host_has_feature = association.host_feature in host_features
            row[association.pairwise_column] = (
                round(association.weight, 4) if phage_has_profile and host_has_feature else 0.0
            )
        feature_rows.append(row)

    return feature_rows, output_columns


def build_feature_metadata_rows(
    profiles: Sequence[CollapsedProfile],
    associations: Sequence[CollapsedAssociation],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for profile in profiles:
        rows.append(
            {
                "column_name": profile.direct_column,
                "block_type": "phage_antidef_profile",
                "experimental_status": EXPERIMENTAL_STATUS,
                "profile_id": profile.profile_id,
                "host_feature": "",
                "member_features": "|".join(profile.member_features),
                "weight": "",
                "transform": "1 when the phage carries this collapsed anti-defense PHROG profile, else 0.",
            }
        )
    for association in associations:
        rows.append(
            {
                "column_name": association.pairwise_column,
                "block_type": "pairwise_defense_evasion",
                "experimental_status": EXPERIMENTAL_STATUS,
                "profile_id": association.profile_id,
                "host_feature": association.host_feature,
                "member_features": "|".join(association.member_features),
                "weight": association.weight,
                "transform": (
                    "lysis_rate_diff weight when the phage carries this collapsed anti-defense PHROG profile and the "
                    "host carries the enriched defense subtype, else 0."
                ),
            }
        )
    return rows


def build_profile_metadata_rows(profiles: Sequence[CollapsedProfile]) -> List[Dict[str, object]]:
    return [
        {
            "profile_id": profile.profile_id,
            "representative_feature": profile.representative_feature,
            "member_features": "|".join(profile.member_features),
            "member_count": len(profile.member_features),
            "carrier_count": profile.carrier_count,
            "direct_column": profile.direct_column,
            "experimental_status": EXPERIMENTAL_STATUS,
        }
        for profile in profiles
    ]


def ensure_default_label_path(label_path: Path) -> None:
    if label_path.exists():
        return
    if label_path != LABEL_SET_V1_PATH:
        raise FileNotFoundError(f"Missing label input: {label_path}")
    logger.info("Track A labels missing at %s; rebuilding Track A foundation", label_path)
    build_track_a_foundation.main([])
    if not label_path.exists():
        raise FileNotFoundError(f"Track A rebuild did not produce expected label file: {label_path}")


def ensure_default_tl02_output(label_path: Path, antidef_enrichment_path: Path) -> None:
    if antidef_enrichment_path.exists():
        return
    if label_path != LABEL_SET_V1_PATH or antidef_enrichment_path != DEFAULT_ANTIDEF_ENRICHMENT_PATH:
        raise FileNotFoundError(f"Missing TL02 enrichment input: {antidef_enrichment_path}")
    ensure_default_label_path(label_path)
    logger.info("TL02 anti-defense enrichment output missing; running Track L enrichment analysis")
    run_tl02_enrichment([])
    if not antidef_enrichment_path.exists():
        raise FileNotFoundError(f"TL02 rebuild did not produce expected enrichment input: {antidef_enrichment_path}")


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    start_time = datetime.now(timezone.utc)
    logger.info(
        "TL04 mechanistic defense-evasion feature build starting at %s", start_time.isoformat(timespec="seconds")
    )

    ensure_default_label_path(args.label_path)
    ensure_default_tl02_output(args.label_path, args.antidef_enrichment_path)

    label_rows = read_delimited_rows(args.label_path)
    pair_rows = build_pair_rows(label_rows)
    bacteria = sorted({row["bacteria"] for row in pair_rows})
    phages = sorted({row["phage"] for row in pair_rows})

    _, _, anti_def_matrix, anti_def_phrog_names = load_pharokka_phrog_matrices(args.cached_annotations_dir, phages)
    anti_def_feature_names = [f"ANTIDEF_PHROG_{phrog}" for phrog in anti_def_phrog_names]
    if not anti_def_feature_names:
        raise ValueError("No anti-defense PHROGs met the panel support threshold; TL04 cannot build pairwise features")
    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(
        anti_def_feature_names,
        anti_def_matrix,
    )
    profile_by_id = {profile.profile_id: profile for profile in profiles}

    defense_matrix, defense_feature_names = load_defense_host_matrix(args.defense_path, bacteria)
    bacteria_to_defense_features = build_presence_index(bacteria, defense_feature_names, defense_matrix)

    enrichment_rows = read_delimited_rows(args.antidef_enrichment_path)
    require_columns(enrichment_rows, args.antidef_enrichment_path, ENRICHMENT_REQUIRED_COLUMNS)
    associations = collapse_significant_associations(enrichment_rows, feature_to_profile_id, profile_by_id)
    if not associations:
        raise ValueError("No significant collapsed anti-defense x defense associations found; TL04 would be empty")

    phage_to_profile_presence = build_binary_value_index(
        phages,
        [profile.profile_id for profile in profiles],
        collapsed_matrix,
    )
    feature_rows, feature_columns = build_feature_rows(
        pair_rows=pair_rows,
        phage_to_profile_presence=phage_to_profile_presence,
        bacteria_to_defense_features=bacteria_to_defense_features,
        profiles=profiles,
        associations=associations,
    )

    metadata_rows = build_feature_metadata_rows(profiles, associations)
    profile_metadata_rows = build_profile_metadata_rows(profiles)

    feature_output_path = args.output_dir / f"mechanistic_defense_evasion_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"mechanistic_defense_evasion_feature_metadata_{args.version}.csv"
    profile_output_path = args.output_dir / f"mechanistic_defense_evasion_profile_metadata_{args.version}.csv"
    manifest_output_path = args.output_dir / f"mechanistic_defense_evasion_manifest_{args.version}.json"

    write_csv(feature_output_path, feature_columns, feature_rows)
    write_csv(
        metadata_output_path,
        [
            "column_name",
            "block_type",
            "experimental_status",
            "profile_id",
            "host_feature",
            "member_features",
            "weight",
            "transform",
        ],
        metadata_rows,
    )
    write_csv(
        profile_output_path,
        [
            "profile_id",
            "representative_feature",
            "member_features",
            "member_count",
            "carrier_count",
            "direct_column",
            "experimental_status",
        ],
        profile_metadata_rows,
    )

    manifest = {
        "step_name": "build_mechanistic_defense_evasion_features",
        "task_id": "TL04",
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experimental_status": EXPERIMENTAL_STATUS,
        "pair_count": len(feature_rows),
        "distinct_bacteria_count": len(bacteria),
        "distinct_phage_count": len(phages),
        "original_antidef_phrog_count": len(anti_def_feature_names),
        "collapsed_profile_count": len(profiles),
        "pairwise_association_count": len(associations),
        "feature_block_column_count": len(feature_columns) - 3,
        "inputs": {
            "label_set_v1_pairs": {"path": str(args.label_path), "sha256": _sha256(args.label_path)},
            "cached_annotations_dir": str(args.cached_annotations_dir),
            "defense_subtypes": {"path": str(args.defense_path), "sha256": _sha256(args.defense_path)},
            "antidef_enrichment": {
                "path": str(args.antidef_enrichment_path),
                "sha256": _sha256(args.antidef_enrichment_path),
            },
        },
        "outputs": {
            "feature_csv": str(feature_output_path),
            "feature_metadata_csv": str(metadata_output_path),
            "profile_metadata_csv": str(profile_output_path),
        },
        "notes": [
            "Experimental candidate block for TL05; not a confirmed mechanistic signal.",
            "Built from Pharokka anti-defense PHROG annotations and TL02 enrichment results only.",
            "Generic methyltransferase annotations may inflate this feature set.",
        ],
    }
    write_json(manifest_output_path, manifest)

    end_time = datetime.now(timezone.utc)
    logger.info(
        "TL04 mechanistic defense-evasion feature build completed at %s (elapsed: %s)",
        end_time.isoformat(timespec="seconds"),
        end_time - start_time,
    )
    print(f"Wrote TL04 mechanistic defense-evasion features to {feature_output_path}")
    print(f"- Pair rows: {len(feature_rows)}")
    print(f"- Collapsed anti-defense PHROG profiles: {len(profiles)}")
    print(f"- Collapsed weighted pairwise features: {len(associations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
