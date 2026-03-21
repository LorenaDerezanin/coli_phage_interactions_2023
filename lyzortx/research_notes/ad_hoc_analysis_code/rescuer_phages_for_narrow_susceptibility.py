#!/usr/bin/env python3
"""TB04: characterize rescuer phages for narrow-susceptibility strains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json  # noqa: E402
from lyzortx.research_notes.ad_hoc_analysis_code.hard_to_lyse_host_traits import (  # noqa: E402
    LOW_SUSCEPTIBILITY_THRESHOLD,
    build_per_strain_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interaction-matrix-path",
        type=Path,
        default=Path("data/interactions/interaction_matrix.csv"),
        help="Interaction matrix CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--host-metadata-path",
        type=Path,
        default=Path("data/genomics/bacteria/picard_collection.csv"),
        help="Host metadata CSV used to reuse the TB03 susceptibility logic.",
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
        default=Path("lyzortx/generated_outputs/rescuer_phages_for_narrow_susceptibility"),
        help="Output directory.",
    )
    parser.add_argument(
        "--low-susceptibility-threshold",
        type=int,
        default=LOW_SUSCEPTIBILITY_THRESHOLD,
        help="Maximum number of lytic phages for the narrow-susceptibility slice.",
    )
    return parser.parse_args()


def clean_metadata_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "Missing"
    text = str(value).strip()
    return text if text else "Missing"


def binary_lysis_matrix(interaction_matrix: pd.DataFrame) -> pd.DataFrame:
    return interaction_matrix.gt(0).where(interaction_matrix.notna(), False)


def resolved_narrow_susceptibility_strains(
    strain_summary: pd.DataFrame,
) -> pd.DataFrame:
    mask = strain_summary["is_low_susceptibility"].eq(True)
    return strain_summary.loc[mask].copy().sort_values(["known_lytic_phages", "bacteria"]).reset_index(drop=True)


def classify_rescue_mode(n_lytic_phages: int) -> str:
    if n_lytic_phages == 0:
        return "non_rescued"
    return "exclusive" if n_lytic_phages == 1 else "shared"


def count_rescue_modes(narrow_strain_summary: pd.DataFrame) -> dict[str, int]:
    return {
        "exclusive": int((narrow_strain_summary["rescue_mode"] == "exclusive").sum()),
        "shared": int((narrow_strain_summary["rescue_mode"] == "shared").sum()),
        "non_rescued": int((narrow_strain_summary["rescue_mode"] == "non_rescued").sum()),
    }


def join_sorted(values: Sequence[str]) -> str:
    return ",".join(sorted(values))


def build_narrow_strain_rescuer_summary(
    interaction_matrix: pd.DataFrame,
    strain_summary: pd.DataFrame,
    phage_metadata: pd.DataFrame,
) -> pd.DataFrame:
    binary_matrix = binary_lysis_matrix(interaction_matrix)
    phage_metadata_by_name = phage_metadata.set_index("phage")
    narrow_strains = resolved_narrow_susceptibility_strains(strain_summary)

    rows: list[dict[str, object]] = []
    for row in narrow_strains.itertuples(index=False):
        lytic_phages = [phage for phage, is_lytic in binary_matrix.loc[row.bacteria].items() if bool(is_lytic)]
        morphotypes = [clean_metadata_value(phage_metadata_by_name.at[phage, "Morphotype"]) for phage in lytic_phages]
        families = [clean_metadata_value(phage_metadata_by_name.at[phage, "Family"]) for phage in lytic_phages]
        rescue_mode = classify_rescue_mode(int(row.known_lytic_phages))
        rows.append(
            {
                "bacteria": row.bacteria,
                "known_lytic_phages": int(row.known_lytic_phages),
                "rescue_mode": rescue_mode,
                "rescuer_phages": join_sorted(lytic_phages),
                "rescuer_morphotypes": join_sorted(morphotypes),
                "rescuer_families": join_sorted(families),
                "unique_rescuer_phage": lytic_phages[0] if rescue_mode == "exclusive" else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "bacteria",
                "known_lytic_phages",
                "rescue_mode",
                "rescuer_phages",
                "rescuer_morphotypes",
                "rescuer_families",
                "unique_rescuer_phage",
            ]
        )

    return pd.DataFrame(rows).sort_values(["known_lytic_phages", "bacteria"]).reset_index(drop=True)


def build_rescuer_phage_summary(
    narrow_strain_summary: pd.DataFrame,
    interaction_matrix: pd.DataFrame,
    phage_metadata: pd.DataFrame,
) -> pd.DataFrame:
    binary_matrix = binary_lysis_matrix(interaction_matrix)
    phage_metadata_by_name = phage_metadata.set_index("phage")
    narrow_strain_count = int(len(narrow_strain_summary))
    phage_to_rescued_strains: dict[str, list[dict[str, object]]] = {}
    for row in narrow_strain_summary.to_dict(orient="records"):
        for phage in str(row["rescuer_phages"]).split(","):
            if phage:
                phage_to_rescued_strains.setdefault(phage, []).append(row)

    rows: list[dict[str, object]] = []

    for phage in interaction_matrix.columns:
        rescued_strains = pd.DataFrame(
            phage_to_rescued_strains.get(phage, []),
            columns=narrow_strain_summary.columns,
        )
        exclusive_rows = rescued_strains[rescued_strains["rescue_mode"] == "exclusive"]
        shared_rows = rescued_strains[rescued_strains["rescue_mode"] == "shared"]
        total_strains_lysed = int(binary_matrix[phage].sum())
        narrow_count = int(len(rescued_strains))
        rows.append(
            {
                "phage": phage,
                "morphotype": clean_metadata_value(phage_metadata_by_name.at[phage, "Morphotype"]),
                "family": clean_metadata_value(phage_metadata_by_name.at[phage, "Family"]),
                "total_strains_lysed": total_strains_lysed,
                "narrow_strains_rescued": narrow_count,
                "exclusive_rescue_count": int(len(exclusive_rows)),
                "shared_rescue_count": int(len(shared_rows)),
                "fraction_of_all_lysed_strains_that_are_narrow": (
                    narrow_count / total_strains_lysed if total_strains_lysed else None
                ),
                "fraction_of_resolved_narrow_strains_covered": (
                    narrow_count / narrow_strain_count if narrow_strain_count else None
                ),
                "rescued_strains": join_sorted(rescued_strains["bacteria"].tolist()),
                "exclusive_rescued_strains": join_sorted(exclusive_rows["bacteria"].tolist()),
                "shared_rescued_strains": join_sorted(shared_rows["bacteria"].tolist()),
                "is_rescuer_phage": narrow_count > 0,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "exclusive_rescue_count",
            "narrow_strains_rescued",
            "fraction_of_all_lysed_strains_that_are_narrow",
            "phage",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_rescuer_group_summary(
    rescuer_phage_summary: pd.DataFrame,
    group_field: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_value, group in rescuer_phage_summary.groupby(group_field, dropna=False):
        rescuer_group = group[group["is_rescuer_phage"]]
        rows.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "n_total_phages": int(len(group)),
                "n_rescuer_phages": int(len(rescuer_group)),
                "rescuer_phage_rate": (len(rescuer_group) / len(group)) if len(group) else None,
                "narrow_strains_rescued": int(rescuer_group["narrow_strains_rescued"].sum()),
                "exclusive_rescue_count": int(rescuer_group["exclusive_rescue_count"].sum()),
                "shared_rescue_count": int(rescuer_group["shared_rescue_count"].sum()),
                "median_narrow_strains_rescued_per_rescuer_phage": (
                    float(rescuer_group["narrow_strains_rescued"].median()) if len(rescuer_group) else None
                ),
                "median_fraction_of_all_lysed_strains_that_are_narrow": (
                    float(rescuer_group["fraction_of_all_lysed_strains_that_are_narrow"].median())
                    if len(rescuer_group)
                    else None
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -row["exclusive_rescue_count"],
            -row["narrow_strains_rescued"],
            row["group_value"],
        ),
    )


def top_rows(
    rescuer_phage_summary: pd.DataFrame,
    sort_columns: Sequence[str],
    limit: int,
) -> list[dict[str, object]]:
    subset = rescuer_phage_summary[rescuer_phage_summary["is_rescuer_phage"]].sort_values(
        list(sort_columns) + ["phage"],
        ascending=[False] * len(sort_columns) + [True],
    )
    return subset.head(limit)[
        [
            "phage",
            "morphotype",
            "family",
            "narrow_strains_rescued",
            "exclusive_rescue_count",
            "shared_rescue_count",
            "fraction_of_all_lysed_strains_that_are_narrow",
        ]
    ].to_dict(orient="records")


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    interaction_matrix = pd.read_csv(args.interaction_matrix_path, sep=";").set_index("bacteria")
    host_metadata = pd.read_csv(args.host_metadata_path, sep=";")
    phage_metadata = pd.read_csv(args.phage_metadata_path, sep=";")

    strain_summary = build_per_strain_summary(
        interaction_matrix=interaction_matrix,
        host_metadata=host_metadata,
        low_threshold=args.low_susceptibility_threshold,
    )
    narrow_strain_summary = build_narrow_strain_rescuer_summary(
        interaction_matrix=interaction_matrix,
        strain_summary=strain_summary,
        phage_metadata=phage_metadata,
    )
    rescuer_phage_summary = build_rescuer_phage_summary(
        narrow_strain_summary=narrow_strain_summary,
        interaction_matrix=interaction_matrix,
        phage_metadata=phage_metadata,
    )
    group_summary_rows = [
        *build_rescuer_group_summary(rescuer_phage_summary, "morphotype"),
        *build_rescuer_group_summary(rescuer_phage_summary, "family"),
    ]

    narrow_output = args.output_dir / "narrow_strain_rescuer_summary.csv"
    phage_output = args.output_dir / "rescuer_phage_summary.csv"
    group_output = args.output_dir / "rescuer_phage_group_summary.csv"
    manifest_output = args.output_dir / "tb04_summary.json"

    write_csv(
        narrow_output,
        fieldnames=[
            "bacteria",
            "known_lytic_phages",
            "rescue_mode",
            "rescuer_phages",
            "rescuer_morphotypes",
            "rescuer_families",
            "unique_rescuer_phage",
        ],
        rows=narrow_strain_summary.to_dict(orient="records"),
    )
    write_csv(
        phage_output,
        fieldnames=[
            "phage",
            "morphotype",
            "family",
            "total_strains_lysed",
            "narrow_strains_rescued",
            "exclusive_rescue_count",
            "shared_rescue_count",
            "fraction_of_all_lysed_strains_that_are_narrow",
            "fraction_of_resolved_narrow_strains_covered",
            "rescued_strains",
            "exclusive_rescued_strains",
            "shared_rescued_strains",
            "is_rescuer_phage",
        ],
        rows=rescuer_phage_summary.to_dict(orient="records"),
    )
    write_csv(
        group_output,
        fieldnames=[
            "group_field",
            "group_value",
            "n_total_phages",
            "n_rescuer_phages",
            "rescuer_phage_rate",
            "narrow_strains_rescued",
            "exclusive_rescue_count",
            "shared_rescue_count",
            "median_narrow_strains_rescued_per_rescuer_phage",
            "median_fraction_of_all_lysed_strains_that_are_narrow",
        ],
        rows=group_summary_rows,
    )

    rescue_mode_counts = count_rescue_modes(narrow_strain_summary)
    manifest = {
        "analysis_id": "TB04",
        "interaction_matrix_path": str(args.interaction_matrix_path),
        "host_metadata_path": str(args.host_metadata_path),
        "phage_metadata_path": str(args.phage_metadata_path),
        "low_susceptibility_threshold": args.low_susceptibility_threshold,
        "n_resolved_narrow_susceptibility_strains": int(len(narrow_strain_summary)),
        "n_exclusive_rescue_strains": rescue_mode_counts["exclusive"],
        "n_shared_rescue_strains": rescue_mode_counts["shared"],
        "n_non_rescued_narrow_strains": rescue_mode_counts["non_rescued"],
        "n_rescuer_phages": int(rescuer_phage_summary["is_rescuer_phage"].sum()),
        "n_exclusive_rescuer_phages": int(rescuer_phage_summary["exclusive_rescue_count"].gt(0).sum()),
        "top_rescuer_phages_by_narrow_strains": top_rows(
            rescuer_phage_summary,
            sort_columns=("narrow_strains_rescued", "exclusive_rescue_count"),
            limit=10,
        ),
        "top_exclusive_rescuer_phages": top_rows(
            rescuer_phage_summary,
            sort_columns=("exclusive_rescue_count", "narrow_strains_rescued"),
            limit=10,
        ),
        "top_narrow_enriched_rescuer_phages": top_rows(
            rescuer_phage_summary,
            sort_columns=("fraction_of_all_lysed_strains_that_are_narrow", "narrow_strains_rescued"),
            limit=10,
        ),
        "notes": [
            "Resolved narrow-susceptibility strains reuse the TB03 operational definition: <= threshold lytic phages and no missing assays.",
            "Rescue mode is non_rescued for zero-lysis strains, exclusive for single-lyser strains, and shared for strains with two or three lytic phages.",
            "A rescuer phage is any phage with at least one lytic hit in the resolved narrow-susceptibility slice.",
        ],
    }
    write_json(manifest_output, manifest)

    print(
        f"TB04 summary: narrow_strains={manifest['n_resolved_narrow_susceptibility_strains']}, "
        f"exclusive={manifest['n_exclusive_rescue_strains']}, rescuer_phages={manifest['n_rescuer_phages']}"
    )
    for row in manifest["top_rescuer_phages_by_narrow_strains"][:5]:
        print(
            f"{row['phage']}: narrow={row['narrow_strains_rescued']}, "
            f"exclusive={row['exclusive_rescue_count']}, "
            f"narrow_hit_share={row['fraction_of_all_lysed_strains_that_are_narrow']:.3f}"
        )


if __name__ == "__main__":
    main()
