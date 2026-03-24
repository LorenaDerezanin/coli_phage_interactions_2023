"""Unit tests for TI07 external-label confidence tiering."""

from __future__ import annotations

import csv
import json

import pytest

from lyzortx.pipeline.track_i.steps.build_external_label_confidence_tiers import (
    ExternalConfidenceConfig,
    apply_external_confidence_policy,
    main,
)


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_apply_external_confidence_policy_assigns_expected_tiers() -> None:
    config = ExternalConfidenceConfig()
    source_registry = {
        "vhrdb": {
            "source_id": "vhrdb",
            "source_type": "curated_database",
            "confidence_tier": "A",
            "confidence_basis": "direct_experimental_screening",
            "host_resolution": "strain",
            "notes": "",
        },
        "virus_host_db": {
            "source_id": "virus_host_db",
            "source_type": "metadata_knowledgebase",
            "confidence_tier": "B",
            "confidence_basis": "metadata_inferred_without_uniform_wet_lab_assay",
            "host_resolution": "species_or_higher_taxonomy",
            "notes": "",
        },
        "ncbi_virus_biosample": {
            "source_id": "ncbi_virus_biosample",
            "source_type": "metadata_repository",
            "confidence_tier": "B",
            "confidence_basis": "submitter_metadata_with_variable_validation",
            "host_resolution": "species_or_higher_taxonomy",
            "notes": "",
        },
    }
    rows = [
        {
            "pair_id": "b1__p1",
            "source_system": "vhrdb",
            "source_disagreement_flag": "0",
            "source_qc_flag": "",
            "source_resolution_status": "",
        },
        {
            "pair_id": "b2__p2",
            "source_system": "virus_host_db",
            "source_disagreement_flag": "0",
            "source_qc_flag": "ok",
            "source_resolution_status": "resolved|resolved_via_alias",
        },
        {
            "pair_id": "b3__p3",
            "source_system": "ncbi_virus_biosample",
            "source_disagreement_flag": "0",
            "source_qc_flag": "host_conflict",
            "source_resolution_status": "resolved",
        },
    ]

    output_rows = apply_external_confidence_policy(rows, source_registry_rows=source_registry, config=config)

    assert output_rows[0]["external_label_confidence_tier"] == "high"
    assert output_rows[0]["external_label_training_weight"] == 1.0
    assert output_rows[1]["external_label_confidence_tier"] == "medium"
    assert output_rows[1]["external_label_training_weight"] == 0.5
    assert output_rows[2]["external_label_confidence_tier"] == "exclude"
    assert output_rows[2]["external_label_include_in_training"] == 0


def test_apply_external_confidence_policy_downgrades_for_unresolved_entities() -> None:
    config = ExternalConfidenceConfig()
    source_registry = {
        "virus_host_db": {
            "source_id": "virus_host_db",
            "source_type": "metadata_knowledgebase",
            "confidence_tier": "B",
            "confidence_basis": "metadata_inferred_without_uniform_wet_lab_assay",
            "host_resolution": "species_or_higher_taxonomy",
            "notes": "",
        }
    }
    rows = [
        {
            "pair_id": "b1__p1",
            "source_system": "virus_host_db",
            "source_disagreement_flag": "0",
            "source_qc_flag": "ok",
            "source_resolution_status": "resolved|unresolved",
        }
    ]

    output_rows = apply_external_confidence_policy(rows, source_registry_rows=source_registry, config=config)

    assert output_rows[0]["external_label_confidence_tier"] == "low"
    assert output_rows[0]["external_label_confidence_reason"] == (
        "base_curated_metadata_association|unresolved_entity_mapping"
    )


def test_main_emits_external_confidence_outputs(tmp_path) -> None:
    source_registry = tmp_path / "source_registry.csv"
    _write_csv(
        source_registry,
        [
            "source_id",
            "source_type",
            "confidence_tier",
            "confidence_basis",
            "host_resolution",
            "notes",
        ],
        [
            {
                "source_id": "vhrdb",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "direct_experimental_screening",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "basel",
                "source_type": "publication_dataset",
                "confidence_tier": "A",
                "confidence_basis": "direct_experimental_screening",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "klebphacol",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "curated_experimental_records",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "gpb",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "assay_backed_records_from_bank_workflows",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "virus_host_db",
                "source_type": "metadata_knowledgebase",
                "confidence_tier": "B",
                "confidence_basis": "metadata_inferred_without_uniform_wet_lab_assay",
                "host_resolution": "species_or_higher_taxonomy",
                "notes": "",
            },
            {
                "source_id": "ncbi_virus_biosample",
                "source_type": "metadata_repository",
                "confidence_tier": "B",
                "confidence_basis": "submitter_metadata_with_variable_validation",
                "host_resolution": "species_or_higher_taxonomy",
                "notes": "",
            },
        ],
    )
    tier_a_ingest = tmp_path / "st08_tier_a_ingested_pairs.csv"
    _write_csv(
        tier_a_ingest,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "source_disagreement_flag",
        ],
        [
            {
                "pair_id": "b1__p1",
                "bacteria": "b1",
                "phage": "p1",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "vhrdb",
                "source_disagreement_flag": "0",
            },
            {
                "pair_id": "b1__p2",
                "bacteria": "b1",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "basel",
                "source_disagreement_flag": "0",
            },
            {
                "pair_id": "b1__p3",
                "bacteria": "b1",
                "phage": "p3",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "klebphacol",
                "source_disagreement_flag": "1",
            },
            {
                "pair_id": "b1__p4",
                "bacteria": "b1",
                "phage": "p4",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "A",
                "source_system": "gpb",
                "source_disagreement_flag": "0",
            },
        ],
    )
    tier_b_ingest = tmp_path / "ti06_weak_label_ingested_pairs.csv"
    _write_csv(
        tier_b_ingest,
        [
            "pair_id",
            "bacteria",
            "phage",
            "label_hard_any_lysis",
            "label_strict_confidence_tier",
            "source_system",
            "source_disagreement_flag",
            "source_qc_flag",
            "source_resolution_status",
        ],
        [
            {
                "pair_id": "b2__p2",
                "bacteria": "b2",
                "phage": "p2",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "",
                "source_system": "virus_host_db",
                "source_disagreement_flag": "0",
                "source_qc_flag": "ok",
                "source_resolution_status": "resolved",
            },
            {
                "pair_id": "b3__p3",
                "bacteria": "b3",
                "phage": "p3",
                "label_hard_any_lysis": "1",
                "label_strict_confidence_tier": "",
                "source_system": "ncbi_virus_biosample",
                "source_disagreement_flag": "0",
                "source_qc_flag": "biosample_missing",
                "source_resolution_status": "resolved",
            },
        ],
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--source-registry-path",
            str(source_registry),
            "--tier-a-ingest-path",
            str(tier_a_ingest),
            "--tier-b-ingest-path",
            str(tier_b_ingest),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti07_external_label_confidence_pairs.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["external_label_confidence_tier"] for row in rows} == {"high", "medium", "exclude"}
    assert {row["confidence_tier"] for row in rows} == {"high", "medium", "exclude"}
    assert {row["training_weight"] for row in rows} == {"1.0", "0.5", "0.0"}
    assert {row["include_in_training"] for row in rows} == {"1", "0"}

    with (output_dir / "ti07_external_label_confidence_summary.csv").open("r", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert {row["slice_type"] for row in summary_rows} == {
        "confidence_tier",
        "include_in_training",
        "source_and_tier",
    }

    policy = json.loads((output_dir / "ti07_external_label_confidence_policy.json").read_text(encoding="utf-8"))
    assert policy["policy_name"] == "track_i_external_label_confidence_tiers"
    assert policy["policy_version"] == "v2"

    manifest = json.loads((output_dir / "ti07_external_label_confidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_name"] == "build_external_label_confidence_tiers"
    assert manifest["source_row_counts"] == {
        "vhrdb": 1,
        "basel": 1,
        "klebphacol": 1,
        "gpb": 1,
        "virus_host_db": 1,
        "ncbi_virus_biosample": 1,
    }


def test_main_raises_when_expected_source_has_zero_rows(tmp_path) -> None:
    source_registry = tmp_path / "source_registry.csv"
    _write_csv(
        source_registry,
        [
            "source_id",
            "source_type",
            "confidence_tier",
            "confidence_basis",
            "host_resolution",
            "notes",
        ],
        [
            {
                "source_id": "vhrdb",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "direct_experimental_screening",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "basel",
                "source_type": "publication_dataset",
                "confidence_tier": "A",
                "confidence_basis": "direct_experimental_screening",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "klebphacol",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "curated_experimental_records",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "gpb",
                "source_type": "curated_database",
                "confidence_tier": "A",
                "confidence_basis": "assay_backed_records_from_bank_workflows",
                "host_resolution": "strain",
                "notes": "",
            },
            {
                "source_id": "virus_host_db",
                "source_type": "metadata_knowledgebase",
                "confidence_tier": "B",
                "confidence_basis": "metadata_inferred_without_uniform_wet_lab_assay",
                "host_resolution": "species_or_higher_taxonomy",
                "notes": "",
            },
            {
                "source_id": "ncbi_virus_biosample",
                "source_type": "metadata_repository",
                "confidence_tier": "B",
                "confidence_basis": "submitter_metadata_with_variable_validation",
                "host_resolution": "species_or_higher_taxonomy",
                "notes": "",
            },
        ],
    )
    tier_a_ingest = tmp_path / "ti05_tier_a_harmonized_pairs.csv"
    _write_csv(
        tier_a_ingest,
        [
            "pair_id",
            "source_system",
        ],
        [
            {
                "pair_id": "b1__p1",
                "source_system": "vhrdb",
            }
        ],
    )
    tier_b_ingest = tmp_path / "ti06_weak_label_ingested_pairs.csv"
    _write_csv(
        tier_b_ingest,
        [
            "pair_id",
            "source_system",
        ],
        [
            {
                "pair_id": "b2__p2",
                "source_system": "virus_host_db",
            },
            {
                "pair_id": "b3__p3",
                "source_system": "ncbi_virus_biosample",
            },
        ],
    )

    with pytest.raises(ValueError, match="basel, gpb, klebphacol"):
        main(
            [
                "--source-registry-path",
                str(source_registry),
                "--tier-a-ingest-path",
                str(tier_a_ingest),
                "--tier-b-ingest-path",
                str(tier_b_ingest),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
