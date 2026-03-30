#!/usr/bin/env python3
"""TL04: Build mechanistic defense-evasion features from annotations."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.track_a.steps import build_track_a_foundation
from lyzortx.pipeline.track_l.steps._mechanistic_builder_common import (
    _sha256,
    CollapsedAssociation,
    CollapsedProfile,
    build_binary_value_index,
    build_pair_rows,
    build_presence_index,
    collapse_duplicate_profiles as collapse_duplicate_profiles_common,
    collapse_significant_associations as collapse_significant_associations_common,
    load_holdout_bacteria_ids,
    load_json,
    read_delimited_rows,
    require_columns,
)
from lyzortx.pipeline.track_l.steps.run_enrichment_analysis import (
    CACHED_ANNOTATIONS_DIR,
    DEFENSE_SUBTYPES_PATH,
    LABEL_SET_V1_PATH,
    ST03_SPLIT_ASSIGNMENTS_PATH,
    load_defense_host_matrix,
    load_pharokka_phrog_matrices,
    main as run_tl02_enrichment,
)

logger = logging.getLogger(__name__)

DEFAULT_ANTIDEF_ENRICHMENT_PATH = Path(
    "lyzortx/generated_outputs/track_l/enrichment/enrichment_antidef_phrog_x_defense_subtype.csv"
)
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/mechanistic_defense_evasion_features")
TL02_MANIFEST_FILENAME = "manifest.json"
ENRICHMENT_REQUIRED_COLUMNS = (
    "phage_feature",
    "host_feature",
    "lysis_rate_diff",
    "significant",
)
EXPERIMENTAL_STATUS = "experimental_candidate"


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
        "--st03-split-assignments-path",
        type=Path,
        default=ST03_SPLIT_ASSIGNMENTS_PATH,
        help="ST0.3 split assignments used to derive the holdout bacteria exclusion list.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag embedded in output filenames and the manifest.",
    )
    return parser.parse_args(argv)


def collapse_duplicate_profiles(
    feature_names: Sequence[str],
    matrix: np.ndarray,
) -> tuple[np.ndarray, List[CollapsedProfile], Dict[str, str]]:
    return collapse_duplicate_profiles_common(
        feature_names,
        matrix,
        direct_column_prefix="tl04_phage_antidef",
    )


def collapse_significant_associations(
    enrichment_rows: Sequence[Mapping[str, str]],
    feature_to_profile_id: Mapping[str, str],
    profile_by_id: Mapping[str, CollapsedProfile],
) -> List[CollapsedAssociation]:
    return collapse_significant_associations_common(
        enrichment_rows,
        feature_to_profile_id,
        profile_by_id,
        pairwise_column_prefix="tl04_pair",
        missing_profile_label="collapsed anti-defense profiles",
    )


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


def summarize_nonzero_rows(
    feature_rows: Sequence[Mapping[str, object]], feature_columns: Sequence[str]
) -> Dict[str, int]:
    mechanistic_columns = feature_columns[3:]
    nonzero_row_count = sum(any(float(row[column]) != 0.0 for column in mechanistic_columns) for row in feature_rows)
    pairwise_nonzero_row_count = sum(
        any(column.startswith("tl04_pair_") and float(row[column]) != 0.0 for column in mechanistic_columns)
        for row in feature_rows
    )
    return {
        "nonzero_row_count": nonzero_row_count,
        "pairwise_nonzero_row_count": pairwise_nonzero_row_count,
    }


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
    run_tl02_enrichment(None)
    if not antidef_enrichment_path.exists():
        raise FileNotFoundError(f"TL02 rebuild did not produce expected enrichment input: {antidef_enrichment_path}")


def load_tl02_holdout_clean_provenance(
    antidef_enrichment_path: Path,
    split_assignments_path: Path,
) -> dict[str, object]:
    holdout_bacteria_ids = load_holdout_bacteria_ids(split_assignments_path)
    manifest_path = antidef_enrichment_path.parent / TL02_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing TL02 manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    holdout_section = manifest.get("holdout_exclusion")
    if not isinstance(holdout_section, dict):
        raise ValueError(f"TL02 manifest missing holdout_exclusion section: {manifest_path}")

    manifest_holdout_ids = holdout_section.get("excluded_holdout_bacteria_ids")
    if sorted(str(value) for value in manifest_holdout_ids or []) != holdout_bacteria_ids:
        raise ValueError("TL02 manifest holdout bacteria IDs do not match the ST03 split assignments.")

    split_manifest = holdout_section.get("split_assignments")
    if not isinstance(split_manifest, dict):
        raise ValueError(f"TL02 manifest missing split_assignments entry: {manifest_path}")
    if split_manifest.get("path") != str(split_assignments_path):
        raise ValueError("TL02 manifest split assignments path does not match the TL03/TL04 rebuild input.")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"TL02 manifest missing outputs section: {manifest_path}")

    entry = outputs.get("antidef_phrog_x_defense_subtype")
    if not isinstance(entry, dict):
        raise ValueError(f"TL02 manifest missing output entry for antidef_phrog_x_defense_subtype: {manifest_path}")
    if entry.get("path") != str(antidef_enrichment_path):
        raise ValueError("TL02 manifest output path mismatch for antidef_phrog_x_defense_subtype.")
    if entry.get("sha256") != _sha256(antidef_enrichment_path):
        raise ValueError("TL02 manifest hash mismatch for antidef_phrog_x_defense_subtype.")

    return {
        "manifest_path": manifest_path,
        "split_assignments_path": split_assignments_path,
        "split_assignments_sha256": _sha256(split_assignments_path),
        "holdout_bacteria_ids": holdout_bacteria_ids,
        "enrichment_inputs": {
            "antidef_phrog_x_defense_subtype": {
                "path": str(antidef_enrichment_path),
                "sha256": _sha256(antidef_enrichment_path),
            }
        },
        "enrichment_manifest_sha256": _sha256(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    start_time = datetime.now(timezone.utc)
    logger.info(
        "TL04 mechanistic defense-evasion feature build starting at %s", start_time.isoformat(timespec="seconds")
    )

    ensure_default_tl02_output(args.label_path, args.antidef_enrichment_path)
    provenance = load_tl02_holdout_clean_provenance(
        args.antidef_enrichment_path,
        args.st03_split_assignments_path,
    )

    label_rows = read_delimited_rows(args.label_path)
    all_pair_rows = build_pair_rows(label_rows)
    pair_rows = [row for row in all_pair_rows if row["bacteria"] not in set(provenance["holdout_bacteria_ids"])]
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
    nonzero_summary = summarize_nonzero_rows(feature_rows, feature_columns)

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
        "nonzero_row_summary": nonzero_summary,
        "provenance": {
            "tl02_manifest_path": str(provenance["manifest_path"]),
            "tl02_manifest_sha256": provenance["enrichment_manifest_sha256"],
            "split_assignments": {
                "path": str(provenance["split_assignments_path"]),
                "sha256": provenance["split_assignments_sha256"],
            },
            "excluded_holdout_bacteria_ids": provenance["holdout_bacteria_ids"],
            "excluded_holdout_bacteria_count": len(provenance["holdout_bacteria_ids"]),
            "enrichment_inputs": provenance["enrichment_inputs"],
        },
        "holdout_exclusion": {
            "excluded_pair_rows": len(all_pair_rows) - len(feature_rows),
        },
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
            "feature_csv_sha256": _sha256(feature_output_path),
            "feature_metadata_csv_sha256": _sha256(metadata_output_path),
            "profile_metadata_csv_sha256": _sha256(profile_output_path),
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
