import json
from pathlib import Path

import pytest

from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15


def _blank_projection_row(bacteria: str) -> dict[str, object]:
    row = {column: "" for column in tl15.PROJECTION_COLUMNS}
    row["bacteria"] = bacteria
    row["assembly_accession"] = f"{bacteria}_ACC"
    return row


def _blank_expected_row(bacteria: str) -> dict[str, object]:
    row = {"bacteria": bacteria}
    for family in tl15.SUPPORTED_SURFACE_FAMILIES:
        row[family.output_present_column] = 0
        row[family.output_label_column] = ""
    return row


def test_best_seed_hits_by_family_prefers_highest_bit_score() -> None:
    hits = tl15.best_seed_hits_by_family(
        [
            {
                "query": "BTUB|btuB|seed",
                "target": "prot_a",
                "pident": "98.0",
                "qcov": "1.0",
                "tcov": "1.0",
                "bits": "100",
                "evalue": "1e-50",
            },
            {
                "query": "BTUB|btuB|seed",
                "target": "prot_b",
                "pident": "99.0",
                "qcov": "1.0",
                "tcov": "1.0",
                "bits": "120",
                "evalue": "1e-60",
            },
            {
                "query": "OMPC|ompC|seed",
                "target": "prot_c",
                "pident": "97.0",
                "qcov": "1.0",
                "tcov": "1.0",
                "bits": "80",
                "evalue": "1e-30",
            },
        ]
    )

    assert hits["BTUB"]["target"] == "prot_b"
    assert hits["OMPC"]["target"] == "prot_c"


def test_project_host_surface_row_distinguishes_absent_from_not_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly_path = tmp_path / "host.fna"
    assembly_path.write_text(">contig\nATGCGTATGCGTATGCGTATGCGT\n", encoding="utf-8")
    proteins_dir = tmp_path / "proteins"
    projection_dir = tmp_path / "projection"
    seed_fasta = tmp_path / "seeds.faa"
    seed_fasta.write_text(">seed\nM" + ("A" * 80) + "\n", encoding="utf-8")
    btub_reference = tmp_path / "btub_refs.faa"
    tl15._write_fasta(btub_reference, [("BTUB|99_1|panel_host", "M" + ("A" * 100))])

    def fake_predict_host_proteins(assembly_path: Path, *, proteins_path: Path) -> dict[str, object]:
        proteins_path.parent.mkdir(parents=True, exist_ok=True)
        proteins_path.write_text(">prot_btub\nM" + ("A" * 100) + "\n", encoding="utf-8")
        return {"protein_count": 1}

    monkeypatch.setattr(tl15, "predict_host_proteins", fake_predict_host_proteins)
    monkeypatch.setattr(tl15, "run_mmseqs_easy_search", lambda *args, **kwargs: kwargs["output_tsv"])

    read_calls = {"count": 0}

    def fake_read_mmseqs_rows(path: Path) -> list[dict[str, str]]:
        read_calls["count"] += 1
        if read_calls["count"] == 1:
            return [
                {
                    "query": "BTUB|btuB|seed",
                    "target": "prot_btub",
                    "pident": "99.0",
                    "qcov": "1.0",
                    "tcov": "1.0",
                    "bits": "150",
                    "evalue": "1e-80",
                }
            ]
        return []

    monkeypatch.setattr(tl15, "read_mmseqs_rows", fake_read_mmseqs_rows)

    row = tl15.project_host_surface_row(
        bacteria="B1",
        assembly_accession="GCF_TEST",
        assembly_path=assembly_path,
        seed_fasta_path=seed_fasta,
        family_reference_paths={"BTUB": btub_reference},
        proteins_dir=proteins_dir,
        projection_dir=projection_dir,
        threads=1,
    )

    assert row["host_receptor_btub_present"] == ""
    assert row["host_receptor_btub_variant"] == ""
    assert row["host_receptor_btub_call_status"] == "family_detected_variant_unresolved"
    assert row["host_receptor_fadL_present"] == 0
    assert row["host_receptor_fadL_call_status"] == "called_absent"


def test_summarize_projection_agreement_counts_callable_and_not_callable() -> None:
    expected_a = _blank_expected_row("B1")
    expected_b = _blank_expected_row("B2")
    expected_a["host_receptor_btub_present"] = 1
    expected_a["host_receptor_btub_variant"] = "99_1"
    expected_b["host_receptor_btub_present"] = 0

    projected_a = _blank_projection_row("B1")
    projected_b = _blank_projection_row("B2")
    projected_a["host_receptor_btub_present"] = 1
    projected_a["host_receptor_btub_variant"] = "99_1"
    projected_a["host_receptor_btub_call_status"] = "called_present"
    projected_b["host_receptor_btub_call_status"] = "family_detected_variant_unresolved"

    agreement_rows, mismatch_rows, support_rows = tl15.summarize_projection_agreement(
        projected_rows=[projected_a, projected_b],
        expected_rows=[expected_a, expected_b],
        family_reference_paths={"BTUB": Path("btub_refs.faa")},
    )

    btub_agreement = next(row for row in agreement_rows if row["feature_family"] == "receptor_btub")
    assert btub_agreement["callable_count"] == 1
    assert btub_agreement["not_callable_count"] == 1
    assert btub_agreement["exact_match_count"] == 1
    assert btub_agreement["agreement_rate_on_callable"] == 1.0

    btub_support = next(row for row in support_rows if row["feature_family"] == "receptor_btub")
    assert btub_support["support_status"] == "approximated"

    assert any(row["feature_family"] == "receptor_btub" and row["bacteria"] == "B2" for row in mismatch_rows)


def test_build_manifest_uses_relative_runtime_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    runtime_dir = output_dir / "runtime_assets"
    raw_dir = output_dir / "raw"
    runtime_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    seed_fasta = runtime_dir / "seeds.faa"
    seed_fasta.write_text(">seed\nMAAA\n", encoding="utf-8")
    assembly_summary = raw_dir / "assembly_summary.txt"
    assembly_summary.write_text("summary\n", encoding="utf-8")
    assembly_catalog = output_dir / "catalog.csv"
    assembly_catalog.write_text("catalog\n", encoding="utf-8")
    projection = output_dir / "projection.csv"
    projection.write_text("projection\n", encoding="utf-8")
    agreement = output_dir / "agreement.csv"
    agreement.write_text("agreement\n", encoding="utf-8")
    mismatch = output_dir / "mismatch.csv"
    mismatch.write_text("mismatch\n", encoding="utf-8")
    support = output_dir / "support.csv"
    support.write_text("support\n", encoding="utf-8")
    btub_refs = runtime_dir / "btub_refs.faa"
    btub_refs.write_text(">ref\nMAAA\n", encoding="utf-8")

    args = tl15.parse_args(["--output-dir", str(output_dir)])
    manifest = tl15.build_manifest(
        args=args,
        output_dir=output_dir,
        assembly_summary_path=assembly_summary,
        assembly_catalog_path=assembly_catalog,
        projection_path=projection,
        agreement_path=agreement,
        mismatch_path=mismatch,
        support_path=support,
        runtime_dir=runtime_dir,
        family_reference_paths={"BTUB": btub_refs},
        seed_fasta_path=seed_fasta,
        seed_metadata=[{"feature_family": "receptor_btub", "seed_fasta": "btub.faa", "seed_identifier": "BTUB|seed"}],
        catalog_rows=[{"bacteria": "B1", "assembly_match_status": "matched"}],
    )

    assert manifest["inputs"]["assembly_summary_local_path"] == "raw/assembly_summary.txt"
    assert manifest["runtime_assets"]["directory"] == "runtime_assets"
    assert manifest["runtime_assets"]["seed_fasta"] == "runtime_assets/seeds.faa"
    assert manifest["runtime_assets"]["family_reference_fastas"]["BTUB"] == "runtime_assets/btub_refs.faa"

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["outputs"]["projection_sha256"]
