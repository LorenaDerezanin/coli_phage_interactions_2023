import csv
import json
from pathlib import Path

import pytest

from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import (
    build_protein_sets,
    extract_genbank_translations,
    read_panel_phages,
)


def _write_panel_metadata(path: Path, phages: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phage"], delimiter=";")
        writer.writeheader()
        for phage in phages:
            writer.writerow({"phage": phage})


def test_extract_genbank_translations_supports_multiline_qualifiers(tmp_path: Path) -> None:
    genbank_path = tmp_path / "P1.gbk"
    genbank_path.write_text(
        "\n".join(
            [
                "LOCUS       P1               42 bp    DNA     linear   PHG 01-JAN-2000",
                "FEATURES             Location/Qualifiers",
                "     CDS             1..12",
                '                     /translation="MKT"',
                "     CDS             13..42",
                '                     /translation="MQQ',
                '                     AAA"',
                "ORIGIN",
                "        1 atgaaaacgaccatgcagcaggctgctgcctga",
                "//",
            ]
        ),
        encoding="utf-8",
    )

    proteins = extract_genbank_translations(genbank_path)

    assert [protein.sequence for protein in proteins] == ["MKT", "MQQAAA"]


def test_build_protein_sets_uses_fasta_or_gene_calling_and_writes_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "panel.csv"
    input_root = tmp_path / "inputs"
    output_dir = tmp_path / "out"
    (input_root / "FNA").mkdir(parents=True)
    (input_root / "FAA").mkdir(parents=True)

    _write_panel_metadata(metadata_path, ["P1", "P2"])
    (input_root / "FAA" / "P1.faa").write_text(">P1_protein\nMKK\n", encoding="utf-8")
    (input_root / "FNA" / "P2.fna").write_text(
        ">contig1\nATG" + ("AAA" * 150) + "TAA\n",
        encoding="utf-8",
    )
    (input_root / "FNA" / "EXTRA.fna").write_text(">extra\nATGAAATAG\n", encoding="utf-8")

    panel_phages = read_panel_phages(metadata_path, expected_panel_count=2)
    manifest = build_protein_sets(
        panel_phages=panel_phages,
        input_root=input_root,
        output_dir=output_dir,
        metadata_path=metadata_path,
    )

    p1_fasta = (output_dir / "protein_fastas" / "P1.faa").read_text(encoding="utf-8")
    p2_fasta = (output_dir / "protein_fastas" / "P2.faa").read_text(encoding="utf-8")
    summary_rows = list(csv.DictReader((output_dir / "phage_protein_summary.csv").open(encoding="utf-8")))
    manifest_json = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert ">P1_protein" in p1_fasta
    assert ">P2|prot_0001" in p2_fasta
    assert {row["phage"] for row in summary_rows} == {"P1", "P2"}
    assert next(row for row in summary_rows if row["phage"] == "P1")["protein_source"] == "protein_fasta"
    assert next(row for row in summary_rows if row["phage"] == "P2")["protein_source"] == "genome_pyrodigal"
    assert manifest["counts"]["ignored_non_panel_input_count"] == 1
    assert manifest_json["ignored_non_panel_inputs"] == ["EXTRA"]


def test_read_panel_phages_rejects_wrong_panel_size(tmp_path: Path) -> None:
    metadata_path = tmp_path / "panel.csv"
    _write_panel_metadata(metadata_path, ["P1"])

    with pytest.raises(ValueError, match="Unexpected phage panel size"):
        read_panel_phages(metadata_path, expected_panel_count=2)
