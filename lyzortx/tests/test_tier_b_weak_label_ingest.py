"""Unit tests for TI06 Tier B weak-label ingestion."""

from __future__ import annotations

import csv
import json

from lyzortx.pipeline.track_i.steps.build_tier_b_weak_label_ingest import (
    build_canonical_resolution_index,
    normalize_ncbi_virus_rows,
    normalize_virus_host_db_rows,
    read_biosample_xml,
    main,
)


def _write_id_map(path, canonical_name_column, canonical_id_column, raw_names_column, rows) -> None:
    fieldnames = [canonical_name_column, canonical_id_column, raw_names_column]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_alias_map(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_name", "canonical_name", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def test_read_biosample_xml_extracts_host_metadata(tmp_path) -> None:
    biosample_path = tmp_path / "biosample.xml"
    biosample_path.write_text(
        """<?xml version="1.0"?>
<BioSampleSet>
  <BioSample accession="SAMN15394129">
    <Description>
      <Title>SARS-CoV-2/human/IND/GBRC225/2020</Title>
      <Organism taxonomy_id="2697049" taxonomy_name="Severe acute respiratory syndrome coronavirus 2">
        <OrganismName>Severe acute respiratory syndrome coronavirus 2</OrganismName>
      </Organism>
    </Description>
    <Attributes>
      <Attribute attribute_name="host" harmonized_name="host">Homo sapiens</Attribute>
      <Attribute attribute_name="isolation_source" harmonized_name="isolation_source">Oro-pharyngeal swab</Attribute>
    </Attributes>
  </BioSample>
</BioSampleSet>
""",
        encoding="utf-8",
    )

    rows = read_biosample_xml(biosample_path)

    assert rows[0]["biosample_accession"] == "SAMN15394129"
    assert rows[0]["biosample_host"] == "Homo sapiens"
    assert rows[0]["biosample_isolation_source"] == "Oro-pharyngeal swab"
    assert rows[0]["biosample_organism_name"] == "Severe acute respiratory syndrome coronavirus 2"


def test_normalize_virus_host_db_rows_expands_refseq_accessions_and_preserves_provenance(tmp_path) -> None:
    registry_row = {
        "source_id": "virus_host_db",
        "confidence_tier": "B",
        "confidence_basis": "metadata_inferred_without_uniform_wet_lab_assay",
        "notes": "Treat as weak labels",
    }
    bacteria_index = build_canonical_resolution_index(
        id_map_path=tmp_path / "unused_bacteria_id_map.csv",
        alias_path=tmp_path / "unused_bacteria_alias.csv",
        canonical_name_column="canonical_bacteria",
        canonical_id_column="canonical_bacteria_id",
        raw_names_column="raw_names",
    )
    phage_index = build_canonical_resolution_index(
        id_map_path=tmp_path / "unused_phage_id_map.csv",
        alias_path=tmp_path / "unused_phage_alias.csv",
        canonical_name_column="canonical_phage",
        canonical_id_column="canonical_phage_id",
        raw_names_column="raw_names",
    )
    rows = [
        {
            "virus tax id": "111",
            "virus name": "phage alpha",
            "virus lineage": "Viruses; Caudoviricetes",
            "refseq id": "NC_001, NC_002",
            "host tax id": "222",
            "host name": "E. coli",
            "host lineage": "Bacteria; Proteobacteria",
            "pmid": "12345",
            "evidence": "Literature",
            "sample type": "isolate",
            "source organism": "bacteriophage",
        }
    ]

    normalized = normalize_virus_host_db_rows(
        rows,
        registry_row=registry_row,
        bacteria_index=bacteria_index,
        phage_index=phage_index,
    )

    assert len(normalized) == 2
    assert normalized[0]["label_hard_any_lysis"] == "1"
    assert normalized[0]["source_system"] == "virus_host_db"
    assert normalized[0]["source_reference_id"] == "12345"
    assert normalized[0]["source_virus_accession"] == "NC_001"


def test_normalize_ncbi_virus_rows_merges_biosample_metadata_and_flags_conflicts(tmp_path) -> None:
    bacteria_id_map = tmp_path / "bacteria_id_map.csv"
    _write_id_map(
        bacteria_id_map,
        "canonical_bacteria",
        "canonical_bacteria_id",
        "raw_names",
        [
            {
                "canonical_bacteria": "Escherichia coli K-12",
                "canonical_bacteria_id": "BAC0001",
                "raw_names": "E. coli|Escherichia coli",
            },
        ],
    )
    bacteria_alias_map = tmp_path / "bacteria_alias_resolution.csv"
    _write_alias_map(
        bacteria_alias_map,
        [{"original_name": "E. coli", "canonical_name": "Escherichia coli K-12", "reason": "manual_alias"}],
    )
    phage_id_map = tmp_path / "phage_id_map.csv"
    _write_id_map(
        phage_id_map,
        "canonical_phage",
        "canonical_phage_id",
        "raw_names",
        [{"canonical_phage": "phage alpha", "canonical_phage_id": "PHG0001", "raw_names": "NC_001"}],
    )
    phage_alias_map = tmp_path / "phage_alias_resolution.csv"
    _write_alias_map(
        phage_alias_map,
        [{"original_name": "NC_001", "canonical_name": "phage alpha", "reason": "manual_alias"}],
    )
    bacteria_index = build_canonical_resolution_index(
        bacteria_id_map,
        bacteria_alias_map,
        canonical_name_column="canonical_bacteria",
        canonical_id_column="canonical_bacteria_id",
        raw_names_column="raw_names",
    )
    phage_index = build_canonical_resolution_index(
        phage_id_map,
        phage_alias_map,
        canonical_name_column="canonical_phage",
        canonical_id_column="canonical_phage_id",
        raw_names_column="raw_names",
    )
    biosample_lookup = {
        "SAMN1": {
            "biosample_accession": "SAMN1",
            "biosample_host": "Escherichia coli",
            "biosample_isolation_host": "Escherichia coli K-12",
            "biosample_isolation_source": "feces",
            "biosample_host_disease": "healthy",
            "biosample_title": "phage isolate",
        }
    }
    rows = [
        {
            "accession": "NC_001",
            "virus_name": "phage alpha",
            "host": "Escherichia coli",
            "biosample": "SAMN1",
            "taxid": "111",
            "virus_lineage": "Viruses; Caudoviricetes",
        },
        {
            "accession": "NC_001",
            "virus_name": "phage alpha",
            "host": "Salmonella enterica",
            "biosample": "SAMN1",
            "taxid": "111",
            "virus_lineage": "Viruses; Caudoviricetes",
        },
    ]
    normalized = normalize_ncbi_virus_rows(
        rows,
        registry_row={
            "source_id": "ncbi_virus_biosample",
            "confidence_tier": "B",
            "confidence_basis": "submitter_metadata_with_variable_validation",
            "notes": "Use only for weak-label expansion",
        },
        bacteria_index=bacteria_index,
        phage_index=phage_index,
        biosample_lookup=biosample_lookup,
    )

    assert normalized[0]["bacteria"] == "Escherichia coli K-12"
    assert normalized[0]["bacteria_id"] == "BAC0001"
    assert normalized[0]["phage"] == "phage alpha"
    assert normalized[0]["phage_id"] == "PHG0001"
    assert normalized[0]["source_qc_flag"] == "ok"
    assert normalized[0]["source_biosample_host_disease"] == "healthy"
    assert normalized[1]["source_qc_flag"] == "host_conflict"
    assert normalized[1]["source_disagreement_flag"] == "1"


def test_main_emits_combined_weak_label_outputs(tmp_path) -> None:
    source_registry = tmp_path / "source_registry.csv"
    source_registry.write_text(
        "\n".join(
            [
                "source_id,confidence_tier,confidence_basis,notes",
                "virus_host_db,B,metadata_inferred_without_uniform_wet_lab_assay,weak labels",
                "ncbi_virus_biosample,B,submitter_metadata_with_variable_validation,weak labels",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bacteria_id_map = tmp_path / "bacteria_id_map.csv"
    _write_id_map(
        bacteria_id_map,
        "canonical_bacteria",
        "canonical_bacteria_id",
        "raw_names",
        [{"canonical_bacteria": "Escherichia coli K-12", "canonical_bacteria_id": "BAC0001", "raw_names": "E. coli"}],
    )
    _write_alias_map(
        tmp_path / "bacteria_alias_resolution.csv",
        [{"original_name": "E. coli", "canonical_name": "Escherichia coli K-12", "reason": "manual_alias"}],
    )
    phage_id_map = tmp_path / "phage_id_map.csv"
    _write_id_map(
        phage_id_map,
        "canonical_phage",
        "canonical_phage_id",
        "raw_names",
        [{"canonical_phage": "phage alpha", "canonical_phage_id": "PHG0001", "raw_names": "NC_001"}],
    )
    _write_alias_map(
        tmp_path / "phage_alias_resolution.csv",
        [{"original_name": "NC_001", "canonical_name": "phage alpha", "reason": "manual_alias"}],
    )
    virus_host_db = tmp_path / "virushostdb.tsv"
    virus_host_db.write_text(
        "\t".join(
            [
                "virus tax id",
                "virus name",
                "virus lineage",
                "refseq id",
                "host tax id",
                "host name",
                "host lineage",
                "pmid",
                "evidence",
                "sample type",
                "source organism",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "111",
                "phage alpha",
                "Viruses; Caudoviricetes",
                "NC_001, NC_002",
                "222",
                "E. coli",
                "Bacteria; Proteobacteria",
                "12345",
                "Literature",
                "isolate",
                "bacteriophage",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ncbi_virus_report = tmp_path / "ncbi_virus_report.jsonl"
    ncbi_virus_report.write_text(
        json.dumps(
            {
                "accession": "NC_001",
                "virus_name": "phage alpha",
                "host": "Escherichia coli",
                "biosample": "SAMN1",
                "taxid": "111",
                "virus_lineage": "Viruses; Caudoviricetes",
            }
        )
        + "\n"
        + json.dumps(
            {
                "accession": "NC_001",
                "virus_name": "phage alpha",
                "host": "Salmonella enterica",
                "biosample": "SAMN1",
                "taxid": "111",
                "virus_lineage": "Viruses; Caudoviricetes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ncbi_biosample = tmp_path / "ncbi_biosample.xml"
    ncbi_biosample.write_text(
        """<?xml version="1.0"?>
<BioSampleSet>
  <BioSample accession="SAMN1">
    <Description>
      <Title>phage isolate</Title>
      <Organism taxonomy_id="111" taxonomy_name="phage alpha">
        <OrganismName>phage alpha</OrganismName>
      </Organism>
    </Description>
    <Attributes>
      <Attribute attribute_name="host" harmonized_name="host">Escherichia coli</Attribute>
      <Attribute attribute_name="isolation_host" harmonized_name="isolation_host">Escherichia coli K-12</Attribute>
      <Attribute attribute_name="isolation_source" harmonized_name="isolation_source">feces</Attribute>
      <Attribute attribute_name="host_disease" harmonized_name="host_disease">healthy</Attribute>
    </Attributes>
  </BioSample>
</BioSampleSet>
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--source-registry-path",
            str(source_registry),
            "--virus-host-db-path",
            str(virus_host_db),
            "--ncbi-virus-report-path",
            str(ncbi_virus_report),
            "--ncbi-biosample-path",
            str(ncbi_biosample),
            "--track-a-bacteria-id-map-path",
            str(bacteria_id_map),
            "--track-a-bacteria-alias-path",
            str(tmp_path / "bacteria_alias_resolution.csv"),
            "--track-a-phage-id-map-path",
            str(phage_id_map),
            "--track-a-phage-alias-path",
            str(tmp_path / "phage_alias_resolution.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "ti06_weak_label_ingested_pairs.csv").open("r", encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert len(combined) == 4
    assert {row["source_system"] for row in combined} == {"virus_host_db", "ncbi_virus_biosample"}
    assert any(row["source_qc_flag"] == "host_conflict" for row in combined)
    assert any(row["source_biosample_host_disease"] == "healthy" for row in combined)
    assert any(row["bacteria_id"] == "BAC0001" for row in combined)
    assert any(row["phage_id"] == "PHG0001" for row in combined)

    with (output_dir / "ti06_weak_label_summary.csv").open("r", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert {row["slice_type"] for row in summary} == {"source_system", "qc_flag"}

    manifest = json.loads((output_dir / "ti06_weak_label_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_sources"] == ["virus_host_db", "ncbi_virus_biosample"]
