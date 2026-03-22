"""Unit tests for TF01/ST0.3c fixed split protocol helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lyzortx.pipeline.steel_thread_v0.steps.st03c_build_fixed_split_protocol import build_fixed_split_assignments
from lyzortx.pipeline.steel_thread_v0.steps.st03c_build_fixed_split_protocol import phage_clade_key


def test_phage_clade_key_uses_taxonomy_fallbacks() -> None:
    assert phage_clade_key({"phage_family": "Other", "phage_subfamily": "Ounavirinae"}) == "Ounavirinae"
    assert phage_clade_key({"phage_family": "Autographiviridae", "phage_subfamily": "Other"}) == "Autographiviridae"
    assert phage_clade_key({"phage_family": "", "phage_subfamily": "", "phage_genus": ""}) == "__MISSING_PHAGE_CLADE__"


def test_build_fixed_split_assignments_is_deterministic_and_leakage_free() -> None:
    rows = [
        {
            "pair_id": "B1__P1",
            "bacteria": "B1",
            "phage": "P1",
            "cv_group": "G1",
            "phage_family": "Other",
            "phage_subfamily": "CladeA",
            "phage_genus": "GenusA",
            "phage_old_family": "",
            "phage_old_genus": "",
        },
        {
            "pair_id": "B2__P1",
            "bacteria": "B2",
            "phage": "P1",
            "cv_group": "G2",
            "phage_family": "Other",
            "phage_subfamily": "CladeA",
            "phage_genus": "GenusA",
            "phage_old_family": "",
            "phage_old_genus": "",
        },
        {
            "pair_id": "B3__P2",
            "bacteria": "B3",
            "phage": "P2",
            "cv_group": "G3",
            "phage_family": "FamB",
            "phage_subfamily": "Other",
            "phage_genus": "GenusB",
            "phage_old_family": "",
            "phage_old_genus": "",
        },
    ]

    first_assignments, first_protocol, first_audit = build_fixed_split_assignments(
        rows,
        host_holdout_fraction=0.34,
        phage_clade_holdout_fraction=0.34,
        host_split_salt="salt-host",
        phage_split_salt="salt-phage",
    )
    second_assignments, second_protocol, second_audit = build_fixed_split_assignments(
        rows,
        host_holdout_fraction=0.34,
        phage_clade_holdout_fraction=0.34,
        host_split_salt="salt-host",
        phage_split_salt="salt-phage",
    )

    assert first_assignments == second_assignments
    assert first_protocol == second_protocol
    assert first_audit == second_audit
    assert first_audit["leakage_checks"]["host_cluster_holdout_cv_group_overlap_count"] == 0
    assert first_audit["leakage_checks"]["phage_clade_holdout_clade_overlap_count"] == 0
    assert {row["split_protocol_id"] for row in first_assignments} == {"tf01_fixed_split_protocol_v1"}
    assert any(row["split_host_cluster_holdout"] == "holdout_test" for row in first_assignments)
    assert any(row["split_phage_clade_holdout"] == "holdout_test" for row in first_assignments)


def test_main_writes_versioned_outputs(tmp_path: Path) -> None:
    from lyzortx.pipeline.steel_thread_v0.steps.st03c_build_fixed_split_protocol import main

    st02_path = tmp_path / "st02_pair_table.csv"
    output_dir = tmp_path / "out"
    with st02_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "bacteria",
                "phage",
                "cv_group",
                "phage_family",
                "phage_subfamily",
                "phage_genus",
                "phage_old_family",
                "phage_old_genus",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "B1__P1",
                "bacteria": "B1",
                "phage": "P1",
                "cv_group": "G1",
                "phage_family": "Other",
                "phage_subfamily": "CladeA",
                "phage_genus": "GenusA",
                "phage_old_family": "",
                "phage_old_genus": "",
            }
        )
        writer.writerow(
            {
                "pair_id": "B2__P2",
                "bacteria": "B2",
                "phage": "P2",
                "cv_group": "G2",
                "phage_family": "FamB",
                "phage_subfamily": "Other",
                "phage_genus": "GenusB",
                "phage_old_family": "",
                "phage_old_genus": "",
            }
        )

    main(
        [
            "--st02-pair-table-path",
            str(st02_path),
            "--output-dir",
            str(output_dir),
            "--host-holdout-fraction",
            "0.5",
            "--phage-clade-holdout-fraction",
            "0.5",
            "--host-split-salt",
            "salt-host",
            "--phage-split-salt",
            "salt-phage",
        ]
    )

    assignments_path = output_dir / "st03c_fixed_split_protocol_v1_assignments.csv"
    protocol_path = output_dir / "st03c_fixed_split_protocol_v1_protocol.json"
    audit_path = output_dir / "st03c_fixed_split_protocol_v1_audit.json"

    assert assignments_path.exists()
    assert protocol_path.exists()
    assert audit_path.exists()

    with assignments_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["split_protocol_id"] == "tf01_fixed_split_protocol_v1"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert protocol["split_type"] == "leave_cluster_out_host_plus_phage_clade_holdout"
    assert audit["leakage_checks"]["host_cluster_holdout_cv_group_overlap_count"] == 0
    assert audit["leakage_checks"]["phage_clade_holdout_clade_overlap_count"] == 0
