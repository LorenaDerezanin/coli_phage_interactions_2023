import csv
import json
from pathlib import Path

from lyzortx.pipeline.deployment_paired_features import derive_host_stats_features


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "host_stats"


def test_build_host_stats_schema_marks_numeric_columns() -> None:
    schema = derive_host_stats_features.build_host_stats_schema()

    assert schema["columns"] == [
        {"name": "bacteria", "dtype": "string"},
        {"name": "host_sequence_record_count", "dtype": "int64"},
        {"name": "host_genome_length_nt", "dtype": "int64"},
        {"name": "host_gc_content", "dtype": "float64"},
        {"name": "host_n50_contig_length_nt", "dtype": "int64"},
    ]
    assert schema["numeric_columns"] == [
        "host_sequence_record_count",
        "host_genome_length_nt",
        "host_gc_content",
        "host_n50_contig_length_nt",
    ]


def test_build_host_stats_feature_row_computes_gc_and_n50_from_fixture() -> None:
    row = derive_host_stats_features.build_host_stats_feature_row(FIXTURE_DIR / "example_host.fasta", bacteria_id="B1")

    assert row == {
        "bacteria": "B1",
        "host_sequence_record_count": 2,
        "host_genome_length_nt": 16,
        "host_gc_content": 0.5,
        "host_n50_contig_length_nt": 10,
    }


def test_derive_host_stats_features_writes_schema_feature_csv_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    result = derive_host_stats_features.derive_host_stats_features(
        FIXTURE_DIR / "example_host.fasta",
        bacteria_id="B1",
        output_dir=output_dir,
    )

    assert result["feature_row"]["host_gc_content"] == 0.5
    schema = json.loads((output_dir / "schema_manifest.json").read_text(encoding="utf-8"))
    assert schema["feature_block"] == "host_stats"
    with (output_dir / "host_stats_features.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "bacteria": "B1",
            "host_sequence_record_count": "2",
            "host_genome_length_nt": "16",
            "host_gc_content": "0.5",
            "host_n50_contig_length_nt": "10",
        }
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["guardrails"]["rebuildable_from_raw_fastas"] is True
    assert manifest["summary_stats"]["host_n50_contig_length_nt"] == 10
