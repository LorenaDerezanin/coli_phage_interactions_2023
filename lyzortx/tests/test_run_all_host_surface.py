"""Tests for run_all_host_surface pure functions."""

from __future__ import annotations

import csv

import pytest

from lyzortx.pipeline.deployment_paired_features.run_all_host_surface import (
    _translate_o_antigen_alleles,
    aggregate_host_surface_csvs,
    best_o_antigen_call,
    build_surface_feature_row,
)
from lyzortx.pipeline.deployment_paired_features.derive_host_surface_features import (
    _capsule_score_column_name,
    build_host_surface_schema,
)


class TestBestOAntigenCall:
    def test_no_hits_returns_empty(self):
        o_type, score = best_o_antigen_call({})
        assert o_type == ""
        assert score == 0.0

    def test_single_hit_extracts_o_type(self):
        o_type, score = best_o_antigen_call({"O157__wzt__O157-1-wzt": 45.3})
        assert o_type == "O157"
        assert score == 45.3

    def test_picks_highest_score(self):
        hits = {
            "O9__wzt__O9-20-wzt": 98.2,
            "O9__wzt__O9-23-wzt": 92.4,
            "O89__wzt__O89-4-wzt": 33.4,
        }
        o_type, score = best_o_antigen_call(hits)
        assert o_type == "O9"
        assert score == 98.2

    def test_different_o_types_picks_best(self):
        hits = {
            "O6__wzx__O6-1-wzx": 10.0,
            "O104__wzy__O104-3-wzy": 50.0,
        }
        o_type, score = best_o_antigen_call(hits)
        assert o_type == "O104"
        assert score == 50.0


class TestBuildSurfaceFeatureRow:
    def test_basic_row_structure(self):
        row = build_surface_feature_row(
            bacteria_id="test-host",
            o_hits={"O157__wzt__O157-1": 42.5},
            receptor_scores={"BTUB": 100.0, "OMPC": 50.0},
            capsule_scores={"KpsC": 30.0},
            lps_lookup={"O157": {"proxy_type": "R3"}},
            capsule_profile_names=["KpsC", "KpsD"],
        )
        assert row["bacteria"] == "test-host"
        assert row["host_o_antigen_type"] == "O157"
        assert row["host_o_antigen_score"] == 42.5
        assert row["host_lps_core_type"] == "R3"
        assert row["host_receptor_btub_score"] == 100.0
        assert row["host_receptor_ompC_score"] == 50.0
        # Capsule scores: KpsC present, KpsD absent
        assert row["host_capsule_profile_kpsc_score"] == 30.0
        assert row["host_capsule_profile_kpsd_score"] == 0.0

    def test_empty_hits_gives_empty_o_type(self):
        row = build_surface_feature_row(
            bacteria_id="empty",
            o_hits={},
            receptor_scores={},
            capsule_scores={},
            lps_lookup={},
            capsule_profile_names=[],
        )
        assert row["host_o_antigen_type"] == ""
        assert row["host_o_antigen_score"] == 0.0
        assert row["host_lps_core_type"] == ""

    def test_unknown_o_type_gives_empty_lps(self):
        row = build_surface_feature_row(
            bacteria_id="x",
            o_hits={"O999__wzt__O999-1": 10.0},
            receptor_scores={},
            capsule_scores={},
            lps_lookup={"O157": {"proxy_type": "R3"}},
            capsule_profile_names=[],
        )
        assert row["host_o_antigen_type"] == "O999"
        assert row["host_lps_core_type"] == ""

    def test_all_12_receptors_present(self):
        row = build_surface_feature_row(
            bacteria_id="x",
            o_hits={},
            receptor_scores={},
            capsule_scores={},
            lps_lookup={},
            capsule_profile_names=[],
        )
        receptor_cols = [k for k in row if k.startswith("host_receptor_")]
        assert len(receptor_cols) == 12

    def test_capsule_column_names_match_real_profiles(self):
        """Verify column naming for realistic profile names (KfiA, cluster_94)."""
        real_profiles = ["KfiA", "KpsC", "cluster_94", "KfoF"]
        row = build_surface_feature_row(
            bacteria_id="x",
            o_hits={},
            receptor_scores={},
            capsule_scores={"cluster_94": 12.5, "KfiA": 8.0},
            lps_lookup={},
            capsule_profile_names=real_profiles,
        )
        assert row["host_capsule_profile_kfia_score"] == 8.0
        assert row["host_capsule_profile_kpsc_score"] == 0.0
        assert row["host_capsule_profile_cluster_94_score"] == 12.5
        assert row["host_capsule_profile_kfof_score"] == 0.0

    def test_capsule_score_not_in_profile_list_ignored(self):
        """A capsule hit for a profile not in the schema should not appear in the row."""
        row = build_surface_feature_row(
            bacteria_id="x",
            o_hits={},
            receptor_scores={},
            capsule_scores={"UnknownProfile": 99.0},
            lps_lookup={},
            capsule_profile_names=["KpsC"],
        )
        assert row["host_capsule_profile_kpsc_score"] == 0.0
        assert "host_capsule_profile_unknownprofile_score" not in row


class TestCapsuleScoreColumnName:
    def test_simple_name(self):
        assert _capsule_score_column_name("KpsC") == "host_capsule_profile_kpsc_score"

    def test_cluster_with_number(self):
        assert _capsule_score_column_name("cluster_94") == "host_capsule_profile_cluster_94_score"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Unable to derive"):
            _capsule_score_column_name("---")


class TestTranslateOAntigenAlleles:
    def test_translates_dna_to_protein(self, tmp_path):
        dna_fasta = tmp_path / "alleles.fna"
        # ATG AAA TAA = Met-Lys-Stop
        dna_fasta.write_text(">O6__wzx__O6-1\nATGAAATAA\n>O9__wzt__O9-1\nATGGCGTAA\n")
        out = tmp_path / "alleles.faa"

        count = _translate_o_antigen_alleles(dna_fasta, out)
        assert count == 2

        lines = out.read_text().strip().split("\n")
        assert lines[0] == ">O6__wzx__O6-1"
        assert lines[1] == "MK"  # ATG AAA, stop stripped
        assert lines[2] == ">O9__wzt__O9-1"
        assert lines[3] == "MA"  # ATG GCG, stop stripped

    def test_empty_input(self, tmp_path):
        dna_fasta = tmp_path / "empty.fna"
        dna_fasta.write_text("")
        out = tmp_path / "empty.faa"
        count = _translate_o_antigen_alleles(dna_fasta, out)
        assert count == 0


class TestAggregateHostSurfaceCsvs:
    def test_empty_rows_writes_header_only(self, tmp_path):
        schema = build_host_surface_schema(("KpsC",))
        out_csv = tmp_path / "empty.csv"
        count = aggregate_host_surface_csvs([], out_csv, schema)
        assert count == 0
        with out_csv.open() as f:
            reader = csv.DictReader(f)
            assert len(list(reader)) == 0
            # Header still written
        assert out_csv.read_text().startswith("bacteria,")

    def test_writes_sorted_csv_with_schema_columns(self, tmp_path):
        schema = build_host_surface_schema(("KpsC", "KpsD"))
        rows = [
            build_surface_feature_row(
                bacteria_id="host-B",
                o_hits={"O6__wzx__O6-1": 20.0},
                receptor_scores={},
                capsule_scores={},
                lps_lookup={},
                capsule_profile_names=["KpsC", "KpsD"],
            ),
            build_surface_feature_row(
                bacteria_id="host-A",
                o_hits={},
                receptor_scores={"BTUB": 100.0},
                capsule_scores={"KpsC": 5.0},
                lps_lookup={},
                capsule_profile_names=["KpsC", "KpsD"],
            ),
        ]
        out_csv = tmp_path / "aggregated.csv"
        count = aggregate_host_surface_csvs(rows, out_csv, schema)
        assert count == 2

        with out_csv.open() as f:
            reader = csv.DictReader(f)
            result_rows = list(reader)

        # Sorted by bacteria
        assert result_rows[0]["bacteria"] == "host-A"
        assert result_rows[1]["bacteria"] == "host-B"
        # Schema columns present
        assert "host_o_antigen_type" in result_rows[0]
        assert "host_receptor_btub_score" in result_rows[0]
        assert "host_capsule_profile_kpsc_score" in result_rows[0]
        # Values correct
        assert float(result_rows[0]["host_receptor_btub_score"]) == 100.0
        assert float(result_rows[0]["host_capsule_profile_kpsc_score"]) == 5.0
        assert result_rows[1]["host_o_antigen_type"] == "O6"
