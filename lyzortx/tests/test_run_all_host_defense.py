"""Tests for lyzortx.pipeline.deployment_paired_features.run_all_host_defense."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from lyzortx.pipeline.deployment_paired_features.run_all_host_defense import (
    AGGREGATED_CSV_PATH,
    EXPECTED_HOST_COUNT,
    aggregate_host_defense_csvs,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "host_defense"
PANEL_DEFENSE_SUBTYPES_PATH = Path("data/genomics/bacteria/defense_finder/370+host_defense_systems_subtypes.csv")
FIXTURE_HOST_IDS = ("001-023", "003-026", "013-008")


@pytest.fixture()
def three_host_output_dir(tmp_path: Path) -> Path:
    """Copy the 3 real per-host fixture directories into a tmp layout.

    Mirrors the real output structure: {host_id}/host_defense_gene_counts.csv
    """
    for host_id in FIXTURE_HOST_IDS:
        src = FIXTURES_DIR / host_id / "host_defense_gene_counts.csv"
        dst_dir = tmp_path / host_id
        dst_dir.mkdir()
        shutil.copy2(src, dst_dir / "host_defense_gene_counts.csv")
    return tmp_path


class TestAggregateHostDefenseCsvs:
    def test_aggregates_three_hosts_sorted(self, three_host_output_dir: Path, tmp_path: Path) -> None:
        out_csv = tmp_path / "out" / "aggregated.csv"
        count = aggregate_host_defense_csvs(three_host_output_dir, out_csv, PANEL_DEFENSE_SUBTYPES_PATH)
        assert count == 3
        assert out_csv.exists()

        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        assert [r["bacteria"] for r in rows] == ["001-023", "003-026", "013-008"]

    def test_column_count_matches_schema(self, three_host_output_dir: Path, tmp_path: Path) -> None:
        out_csv = tmp_path / "out" / "aggregated.csv"
        aggregate_host_defense_csvs(three_host_output_dir, out_csv, PANEL_DEFENSE_SUBTYPES_PATH)

        with out_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None
            list(reader)

        # 80 columns: bacteria + 79 retained subtypes
        assert len(fieldnames) == 80
        assert fieldnames[0] == "bacteria"

    def test_integer_counts_preserved(self, three_host_output_dir: Path, tmp_path: Path) -> None:
        out_csv = tmp_path / "out" / "aggregated.csv"
        aggregate_host_defense_csvs(three_host_output_dir, out_csv, PANEL_DEFENSE_SUBTYPES_PATH)

        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = {r["bacteria"]: r for r in csv.DictReader(f)}

        # 001-023 has RM_Type_IV=2 (not binary 1)
        assert rows["001-023"]["RM_Type_IV"] == "2"
        # 013-008 has RM_Type_IIG=2
        assert rows["013-008"]["RM_Type_IIG"] == "2"
        # 003-026 has Druantia_III=1
        assert rows["003-026"]["Druantia_III"] == "1"
        # Zero is preserved
        assert rows["001-023"]["AbiD"] == "0"

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_csv = tmp_path / "aggregated.csv"
        with pytest.raises(FileNotFoundError, match="No per-host output directories"):
            aggregate_host_defense_csvs(empty_dir, out_csv, PANEL_DEFENSE_SUBTYPES_PATH)

    def test_missing_columns_filled_with_zero(self, tmp_path: Path) -> None:
        """If a per-host CSV is missing some schema columns, they default to 0."""
        host_dir = tmp_path / "per_host" / "sparse-host"
        host_dir.mkdir(parents=True)
        csv_path = host_dir / "host_defense_gene_counts.csv"
        csv_path.write_text("bacteria,RM_Type_I\nsparse-host,5\n", encoding="utf-8")

        out_csv = tmp_path / "aggregated.csv"
        count = aggregate_host_defense_csvs(tmp_path / "per_host", out_csv, PANEL_DEFENSE_SUBTYPES_PATH)
        assert count == 1

        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["bacteria"] == "sparse-host"
        assert rows[0]["RM_Type_I"] == "5"
        assert rows[0]["AbiD"] == "0"

    def test_can_reaggregate_requested_subset_without_rerunning_workers(
        self,
        three_host_output_dir: Path,
        tmp_path: Path,
    ) -> None:
        out_csv = tmp_path / "out" / "aggregated.csv"
        count = aggregate_host_defense_csvs(
            three_host_output_dir,
            out_csv,
            PANEL_DEFENSE_SUBTYPES_PATH,
            bacteria_ids=["003-026", "013-008"],
        )

        assert count == 2
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["bacteria"] for row in rows] == ["003-026", "013-008"]

    def test_requested_subset_fails_when_per_host_output_is_missing(
        self,
        three_host_output_dir: Path,
        tmp_path: Path,
    ) -> None:
        out_csv = tmp_path / "out" / "aggregated.csv"
        with pytest.raises(FileNotFoundError, match="requested bacteria: missing-host"):
            aggregate_host_defense_csvs(
                three_host_output_dir,
                out_csv,
                PANEL_DEFENSE_SUBTYPES_PATH,
                bacteria_ids=["001-023", "missing-host"],
            )


class TestAggregatedCsvPath:
    def test_path_is_under_lyzortx_data(self) -> None:
        assert "generated_outputs" not in str(AGGREGATED_CSV_PATH)
        assert str(AGGREGATED_CSV_PATH).startswith("lyzortx/data/")

    def test_expected_host_count_is_403(self) -> None:
        assert EXPECTED_HOST_COUNT == 403
