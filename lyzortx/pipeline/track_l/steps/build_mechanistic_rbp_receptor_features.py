#!/usr/bin/env python3
"""TL03: Build mechanistic RBP-receptor compatibility features from annotations."""

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
from lyzortx.pipeline.track_l.steps.parse_annotations import classify_rbp_genes, parse_merged_tsv
from lyzortx.pipeline.track_l.steps.run_enrichment_analysis import (
    CACHED_ANNOTATIONS_DIR,
    LABEL_SET_V1_PATH,
    LPS_PRIMARY_PATH,
    LPS_SUPPLEMENTAL_PATH,
    OMP_CLUSTERS_PATH,
    ST03_SPLIT_ASSIGNMENTS_PATH,
    load_lps_host_matrix,
    load_omp_receptor_host_matrix,
    load_pharokka_phrog_matrices,
    main as run_tl02_enrichment,
)

logger = logging.getLogger(__name__)

DEFAULT_OMP_ENRICHMENT_PATH = Path(
    "lyzortx/generated_outputs/track_l/enrichment/enrichment_rbp_phrog_x_omp_receptor.csv"
)
DEFAULT_LPS_ENRICHMENT_PATH = Path("lyzortx/generated_outputs/track_l/enrichment/enrichment_rbp_phrog_x_lps_core.csv")
DEFAULT_RBP_LIST_PATH = Path("data/genomics/phages/RBP/RBP_list.csv")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/mechanistic_rbp_receptor_features")
ENRICHMENT_REQUIRED_COLUMNS = (
    "phage_feature",
    "host_feature",
    "lysis_rate_diff",
    "significant",
)
TL02_MANIFEST_FILENAME = "manifest.json"


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
        "--omp-path",
        type=Path,
        default=OMP_CLUSTERS_PATH,
        help="OMP receptor cluster TSV used in TL02.",
    )
    parser.add_argument(
        "--lps-primary-path",
        type=Path,
        default=LPS_PRIMARY_PATH,
        help="Primary LPS core type TSV used in TL02.",
    )
    parser.add_argument(
        "--lps-supplemental-path",
        type=Path,
        default=LPS_SUPPLEMENTAL_PATH,
        help="Supplemental LPS core type TSV used in TL02.",
    )
    parser.add_argument(
        "--omp-enrichment-path",
        type=Path,
        default=DEFAULT_OMP_ENRICHMENT_PATH,
        help="TL02 OMP enrichment CSV.",
    )
    parser.add_argument(
        "--lps-enrichment-path",
        type=Path,
        default=DEFAULT_LPS_ENRICHMENT_PATH,
        help="TL02 LPS enrichment CSV.",
    )
    parser.add_argument(
        "--rbp-list-path",
        type=Path,
        default=DEFAULT_RBP_LIST_PATH,
        help="Curated RBP_list.csv for sanity-check comparisons.",
    )
    parser.add_argument(
        "--st03-split-assignments-path",
        type=Path,
        default=ST03_SPLIT_ASSIGNMENTS_PATH,
        help="ST0.3 split assignments used to derive the holdout bacteria exclusion list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated TL03 outputs.",
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
        direct_column_prefix="tl03_phage_rbp",
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
        pairwise_column_prefix="tl03_pair",
        missing_profile_label="collapsed PHROG profiles",
    )


def build_feature_rows(
    pair_rows: Sequence[Mapping[str, str]],
    phage_to_profile_presence: Mapping[str, Mapping[str, int]],
    bacteria_to_host_features: Mapping[str, set[str]],
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
        host_features = bacteria_to_host_features.get(bacteria)
        if phage_profiles is None:
            raise ValueError(f"Missing collapsed PHROG profile row for phage {phage!r}")
        if host_features is None:
            raise ValueError(f"Missing host feature row for bacteria {bacteria!r}")

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


def load_curated_rbp_summary(path: Path, panel_phages: Sequence[str]) -> Dict[str, Dict[str, object]]:
    rows = read_delimited_rows(path, delimiter=";")
    require_columns(rows, path, ("phage", "RBP", "type"))
    panel_set = set(panel_phages)
    summary: Dict[str, Dict[str, object]] = {
        phage: {"curated_rbp_count": 0, "curated_types": set(), "curated_has_rbp": 0} for phage in panel_phages
    }

    for row in rows:
        phage = row["phage"]
        if phage not in panel_set:
            continue
        rbp_value = row["RBP"]
        rbp_type = row["type"].lower()
        if rbp_value in {"", "NA"} or rbp_type in {"", "na"}:
            continue
        summary[phage]["curated_rbp_count"] = int(summary[phage]["curated_rbp_count"]) + 1
        summary[phage]["curated_types"].add(rbp_type)
        summary[phage]["curated_has_rbp"] = 1

    return summary


def load_pharokka_rbp_gene_summary(
    cached_annotations_dir: Path,
    panel_phages: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    panel_set = set(panel_phages)
    summary: Dict[str, Dict[str, object]] = {
        phage: {"pharokka_rbp_gene_count": 0, "pharokka_has_rbp": 0} for phage in panel_phages
    }

    tsv_paths = sorted(cached_annotations_dir.glob("*_cds_final_merged_output.tsv"))
    if not tsv_paths:
        raise FileNotFoundError(f"No cached merged TSVs found in {cached_annotations_dir}")

    for tsv_path in tsv_paths:
        phage = tsv_path.name.removesuffix("_cds_final_merged_output.tsv")
        if phage not in panel_set:
            continue
        rbp_genes = classify_rbp_genes(parse_merged_tsv(tsv_path))
        summary[phage]["pharokka_rbp_gene_count"] = len(rbp_genes)
        summary[phage]["pharokka_has_rbp"] = int(len(rbp_genes) > 0)

    return summary


def build_sanity_check_rows(
    panel_phages: Sequence[str],
    curated_summary: Mapping[str, Mapping[str, object]],
    pharokka_summary: Mapping[str, Mapping[str, object]],
    phage_to_profile_presence: Mapping[str, Mapping[str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phage in sorted(panel_phages):
        curated = curated_summary.get(phage, {"curated_rbp_count": 0, "curated_types": set(), "curated_has_rbp": 0})
        pharokka = pharokka_summary.get(phage, {"pharokka_rbp_gene_count": 0, "pharokka_has_rbp": 0})
        collapsed_profile_count = int(sum(phage_to_profile_presence[phage].values()))
        curated_has_rbp = int(curated["curated_has_rbp"])
        pharokka_has_rbp = int(pharokka["pharokka_has_rbp"])
        rows.append(
            {
                "phage": phage,
                "curated_rbp_count": int(curated["curated_rbp_count"]),
                "curated_type_set": "|".join(sorted(str(value) for value in curated["curated_types"])),
                "curated_has_rbp": curated_has_rbp,
                "pharokka_rbp_gene_count": int(pharokka["pharokka_rbp_gene_count"]),
                "pharokka_has_rbp": pharokka_has_rbp,
                "collapsed_profile_count": collapsed_profile_count,
                "agreement_has_rbp": int(curated_has_rbp == pharokka_has_rbp),
            }
        )
    return rows


def build_feature_metadata_rows(
    profiles: Sequence[CollapsedProfile],
    associations: Sequence[CollapsedAssociation],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for profile in profiles:
        rows.append(
            {
                "column_name": profile.direct_column,
                "block_type": "phage_profile",
                "profile_id": profile.profile_id,
                "host_feature": "",
                "member_features": "|".join(profile.member_features),
                "weight": "",
                "transform": "1 when the phage carries this collapsed RBP PHROG profile, else 0.",
            }
        )
    for association in associations:
        rows.append(
            {
                "column_name": association.pairwise_column,
                "block_type": "pairwise_compatibility",
                "profile_id": association.profile_id,
                "host_feature": association.host_feature,
                "member_features": "|".join(association.member_features),
                "weight": association.weight,
                "transform": (
                    "lysis_rate_diff weight when the phage carries this collapsed RBP PHROG profile and the host "
                    "carries the enriched OMP/LPS feature, else 0."
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
        }
        for profile in profiles
    ]


def summarize_nonzero_rows(
    feature_rows: Sequence[Mapping[str, object]], feature_columns: Sequence[str]
) -> Dict[str, int]:
    mechanistic_columns = feature_columns[3:]
    nonzero_row_count = sum(any(float(row[column]) != 0.0 for column in mechanistic_columns) for row in feature_rows)
    pairwise_nonzero_row_count = sum(
        any(column.startswith("tl03_pair_") and float(row[column]) != 0.0 for column in mechanistic_columns)
        for row in feature_rows
    )
    return {
        "nonzero_row_count": nonzero_row_count,
        "pairwise_nonzero_row_count": pairwise_nonzero_row_count,
    }


def summarize_sanity_check(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    return {
        "phage_count": len(rows),
        "agreement_has_rbp_count": sum(int(row["agreement_has_rbp"]) for row in rows),
        "curated_has_rbp_count": sum(int(row["curated_has_rbp"]) for row in rows),
        "pharokka_has_rbp_count": sum(int(row["pharokka_has_rbp"]) for row in rows),
        "both_has_rbp_count": sum(int(row["curated_has_rbp"]) and int(row["pharokka_has_rbp"]) for row in rows),
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


def ensure_default_tl02_outputs(
    label_path: Path,
    omp_enrichment_path: Path,
    lps_enrichment_path: Path,
) -> None:
    if omp_enrichment_path.exists() and lps_enrichment_path.exists():
        return
    if (
        label_path != LABEL_SET_V1_PATH
        or omp_enrichment_path != DEFAULT_OMP_ENRICHMENT_PATH
        or lps_enrichment_path != DEFAULT_LPS_ENRICHMENT_PATH
    ):
        missing = [str(path) for path in (omp_enrichment_path, lps_enrichment_path) if not path.exists()]
        raise FileNotFoundError("Missing TL02 enrichment input(s): " + ", ".join(missing))
    ensure_default_label_path(label_path)
    logger.info("TL02 enrichment outputs missing; running Track L enrichment analysis")
    run_tl02_enrichment(None)
    missing = [str(path) for path in (omp_enrichment_path, lps_enrichment_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("TL02 rebuild did not produce expected enrichment input(s): " + ", ".join(missing))


def load_tl02_holdout_clean_provenance(
    omp_enrichment_path: Path,
    lps_enrichment_path: Path,
    split_assignments_path: Path,
) -> dict[str, object]:
    holdout_bacteria_ids = load_holdout_bacteria_ids(split_assignments_path)
    manifest_path = omp_enrichment_path.parent / TL02_MANIFEST_FILENAME
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

    expected_outputs = {
        "rbp_phrog_x_omp_receptor": omp_enrichment_path,
        "rbp_phrog_x_lps_core": lps_enrichment_path,
    }
    for key, expected_path in expected_outputs.items():
        entry = outputs.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"TL02 manifest missing output entry for {key}: {manifest_path}")
        if entry.get("path") != str(expected_path):
            raise ValueError(f"TL02 manifest output path mismatch for {key}: {expected_path}")
        if entry.get("sha256") != _sha256(expected_path):
            raise ValueError(f"TL02 manifest hash mismatch for {key}: {expected_path}")

    return {
        "manifest_path": manifest_path,
        "split_assignments_path": split_assignments_path,
        "split_assignments_sha256": _sha256(split_assignments_path),
        "holdout_bacteria_ids": holdout_bacteria_ids,
        "enrichment_inputs": {
            "rbp_phrog_x_omp_receptor": {
                "path": str(omp_enrichment_path),
                "sha256": _sha256(omp_enrichment_path),
            },
            "rbp_phrog_x_lps_core": {
                "path": str(lps_enrichment_path),
                "sha256": _sha256(lps_enrichment_path),
            },
        },
        "enrichment_manifest_sha256": _sha256(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)

    start_time = datetime.now(timezone.utc)
    logger.info("TL03 mechanistic RBP-receptor feature build starting at %s", start_time.isoformat(timespec="seconds"))

    ensure_default_label_path(args.label_path)
    ensure_default_tl02_outputs(args.label_path, args.omp_enrichment_path, args.lps_enrichment_path)
    provenance = load_tl02_holdout_clean_provenance(
        args.omp_enrichment_path,
        args.lps_enrichment_path,
        args.st03_split_assignments_path,
    )

    label_rows = read_delimited_rows(args.label_path)
    all_pair_rows = build_pair_rows(label_rows)
    pair_rows = [row for row in all_pair_rows if row["bacteria"] not in set(provenance["holdout_bacteria_ids"])]
    bacteria = sorted({row["bacteria"] for row in pair_rows})
    phages = sorted({row["phage"] for row in pair_rows})

    rbp_matrix, rbp_phrog_names, _, _ = load_pharokka_phrog_matrices(args.cached_annotations_dir, phages)
    rbp_feature_names = [f"RBP_PHROG_{phrog}" for phrog in rbp_phrog_names]
    collapsed_matrix, profiles, feature_to_profile_id = collapse_duplicate_profiles(rbp_feature_names, rbp_matrix)
    profile_by_id = {profile.profile_id: profile for profile in profiles}

    omp_matrix, omp_feature_names = load_omp_receptor_host_matrix(args.omp_path, bacteria)
    lps_matrix, lps_feature_names = load_lps_host_matrix(args.lps_primary_path, args.lps_supplemental_path, bacteria)
    bacteria_to_omp_features = build_presence_index(bacteria, omp_feature_names, omp_matrix)
    bacteria_to_lps_features = build_presence_index(bacteria, lps_feature_names, lps_matrix)
    bacteria_to_host_features = {
        bacteria_name: bacteria_to_omp_features[bacteria_name] | bacteria_to_lps_features[bacteria_name]
        for bacteria_name in bacteria
    }

    omp_enrichment_rows = read_delimited_rows(args.omp_enrichment_path)
    lps_enrichment_rows = read_delimited_rows(args.lps_enrichment_path)
    require_columns(omp_enrichment_rows, args.omp_enrichment_path, ENRICHMENT_REQUIRED_COLUMNS)
    require_columns(lps_enrichment_rows, args.lps_enrichment_path, ENRICHMENT_REQUIRED_COLUMNS)
    associations = collapse_significant_associations(
        [*omp_enrichment_rows, *lps_enrichment_rows],
        feature_to_profile_id,
        profile_by_id,
    )

    phage_to_profile_presence = build_binary_value_index(
        phages,
        [profile.profile_id for profile in profiles],
        collapsed_matrix,
    )
    feature_rows, feature_columns = build_feature_rows(
        pair_rows=pair_rows,
        phage_to_profile_presence=phage_to_profile_presence,
        bacteria_to_host_features=bacteria_to_host_features,
        profiles=profiles,
        associations=associations,
    )

    curated_summary = load_curated_rbp_summary(args.rbp_list_path, phages)
    pharokka_summary = load_pharokka_rbp_gene_summary(args.cached_annotations_dir, phages)
    sanity_check_rows = build_sanity_check_rows(phages, curated_summary, pharokka_summary, phage_to_profile_presence)
    sanity_summary = summarize_sanity_check(sanity_check_rows)
    nonzero_summary = summarize_nonzero_rows(feature_rows, feature_columns)

    metadata_rows = build_feature_metadata_rows(profiles, associations)
    profile_metadata_rows = build_profile_metadata_rows(profiles)

    feature_output_path = args.output_dir / f"mechanistic_rbp_receptor_features_{args.version}.csv"
    metadata_output_path = args.output_dir / f"mechanistic_rbp_receptor_feature_metadata_{args.version}.csv"
    profile_output_path = args.output_dir / f"mechanistic_rbp_profile_metadata_{args.version}.csv"
    sanity_output_path = args.output_dir / f"mechanistic_rbp_sanity_check_{args.version}.csv"
    manifest_output_path = args.output_dir / f"mechanistic_rbp_receptor_manifest_{args.version}.json"

    write_csv(feature_output_path, feature_columns, feature_rows)
    write_csv(
        metadata_output_path,
        ["column_name", "block_type", "profile_id", "host_feature", "member_features", "weight", "transform"],
        metadata_rows,
    )
    write_csv(
        profile_output_path,
        ["profile_id", "representative_feature", "member_features", "member_count", "carrier_count", "direct_column"],
        profile_metadata_rows,
    )
    write_csv(
        sanity_output_path,
        [
            "phage",
            "curated_rbp_count",
            "curated_type_set",
            "curated_has_rbp",
            "pharokka_rbp_gene_count",
            "pharokka_has_rbp",
            "collapsed_profile_count",
            "agreement_has_rbp",
        ],
        sanity_check_rows,
    )

    manifest = {
        "step_name": "build_mechanistic_rbp_receptor_features",
        "task_id": "TL03",
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_count": len(feature_rows),
        "distinct_bacteria_count": len(bacteria),
        "distinct_phage_count": len(phages),
        "original_rbp_phrog_count": len(rbp_feature_names),
        "collapsed_profile_count": len(profiles),
        "pairwise_association_count": len(associations),
        "feature_block_column_count": len(feature_columns) - 3,
        "nonzero_row_summary": nonzero_summary,
        "sanity_check_summary": sanity_summary,
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
            "omp_clusters": {"path": str(args.omp_path), "sha256": _sha256(args.omp_path)},
            "lps_primary": {"path": str(args.lps_primary_path), "sha256": _sha256(args.lps_primary_path)},
            "lps_supplemental": {
                "path": str(args.lps_supplemental_path),
                "sha256": _sha256(args.lps_supplemental_path),
            },
            "omp_enrichment": {"path": str(args.omp_enrichment_path), "sha256": _sha256(args.omp_enrichment_path)},
            "lps_enrichment": {"path": str(args.lps_enrichment_path), "sha256": _sha256(args.lps_enrichment_path)},
            "curated_rbp_list": {"path": str(args.rbp_list_path), "sha256": _sha256(args.rbp_list_path)},
        },
        "outputs": {
            "feature_csv": str(feature_output_path),
            "feature_metadata_csv": str(metadata_output_path),
            "profile_metadata_csv": str(profile_output_path),
            "sanity_check_csv": str(sanity_output_path),
            "feature_csv_sha256": _sha256(feature_output_path),
            "feature_metadata_csv_sha256": _sha256(metadata_output_path),
            "profile_metadata_csv_sha256": _sha256(profile_output_path),
            "sanity_check_csv_sha256": _sha256(sanity_output_path),
        },
    }
    write_json(manifest_output_path, manifest)

    end_time = datetime.now(timezone.utc)
    logger.info(
        "TL03 mechanistic RBP-receptor feature build completed at %s (elapsed: %s)",
        end_time.isoformat(timespec="seconds"),
        end_time - start_time,
    )
    print(f"Wrote TL03 mechanistic features to {feature_output_path}")
    print(f"- Pair rows: {len(feature_rows)}")
    print(f"- Collapsed RBP PHROG profiles: {len(profiles)}")
    print(f"- Collapsed weighted pairwise features: {len(associations)}")
    print(f"- Sanity-check agreement on any-RBP presence: {sanity_summary['agreement_has_rbp_count']} / {len(phages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
