import csv
import json
from pathlib import Path

from lyzortx.pipeline.autoresearch import derive_phage_stats_features


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phage_stats"


def test_build_phage_stats_schema_marks_numeric_columns() -> None:
    schema = derive_phage_stats_features.build_phage_stats_schema()

    assert schema["columns"] == [
        {"name": "phage", "dtype": "string"},
        {"name": "phage_sequence_record_count", "dtype": "int64"},
        {"name": "phage_genome_length_nt", "dtype": "int64"},
        {"name": "phage_gc_content", "dtype": "float64"},
        {"name": "phage_n50_contig_length_nt", "dtype": "int64"},
    ]
    assert schema["numeric_columns"] == [
        "phage_sequence_record_count",
        "phage_genome_length_nt",
        "phage_gc_content",
        "phage_n50_contig_length_nt",
    ]


def test_build_phage_stats_feature_row_computes_gc_and_n50_from_fixture() -> None:
    row = derive_phage_stats_features.build_phage_stats_feature_row(
        FIXTURE_DIR / "example_phage.fasta",
        phage_id="P1",
    )

    assert row == {
        "phage": "P1",
        "phage_sequence_record_count": 2,
        "phage_genome_length_nt": 16,
        "phage_gc_content": 0.5,
        "phage_n50_contig_length_nt": 10,
    }


def test_derive_phage_stats_features_writes_schema_feature_csv_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    result = derive_phage_stats_features.derive_phage_stats_features(
        FIXTURE_DIR / "example_phage.fasta",
        phage_id="P1",
        output_dir=output_dir,
    )

    assert result["feature_row"]["phage_gc_content"] == 0.5
    schema = json.loads((output_dir / "schema_manifest.json").read_text(encoding="utf-8"))
    assert schema["feature_block"] == "phage_stats"
    with (output_dir / "phage_stats_features.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "phage": "P1",
            "phage_sequence_record_count": "2",
            "phage_genome_length_nt": "16",
            "phage_gc_content": "0.5",
            "phage_n50_contig_length_nt": "10",
        }
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["guardrails"]["rebuildable_from_raw_fastas"] is True
    assert manifest["guardrails"]["low_cost_baseline_feature_family"] is True
    assert manifest["summary_stats"]["phage_n50_contig_length_nt"] == 10
