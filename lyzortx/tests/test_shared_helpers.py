"""Tests for shared helper modules used across check and step scripts."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from lyzortx.pipeline.steel_thread_v0.checks._check_helpers import (
    compare_dicts,
    count_csv_rows,
    load_json,
    read_csv_rows as check_read_csv_rows,
)
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import (
    load_json as step_load_json,
    parse_float,
    read_csv_rows as step_read_csv_rows,
    safe_round,
)


# ---------------------------------------------------------------------------
# compare_dicts
# ---------------------------------------------------------------------------


class TestCompareDicts:
    def test_identical_dicts(self) -> None:
        assert compare_dicts({"a": 1, "b": "x"}, {"a": 1, "b": "x"}) == []

    def test_mismatch_value(self) -> None:
        errors = compare_dicts({"a": 1}, {"a": 2})
        assert len(errors) == 1
        assert "Mismatch at a" in errors[0]

    def test_missing_key_in_actual(self) -> None:
        errors = compare_dicts({"a": 1, "b": 2}, {"a": 1})
        assert len(errors) == 1
        assert "Missing key in actual: b" in errors[0]

    def test_unexpected_key_in_actual(self) -> None:
        errors = compare_dicts({"a": 1}, {"a": 1, "b": 2})
        assert len(errors) == 1
        assert "Unexpected key in actual: b" in errors[0]

    def test_nested_comparison(self) -> None:
        expected = {"outer": {"inner": 10}}
        actual = {"outer": {"inner": 20}}
        errors = compare_dicts(expected, actual)
        assert len(errors) == 1
        assert "outer.inner" in errors[0]

    def test_numeric_tolerance_pass(self) -> None:
        errors = compare_dicts({"val": 1.0}, {"val": 1.000001}, numeric_tolerance=1e-5)
        assert errors == []

    def test_numeric_tolerance_fail(self) -> None:
        errors = compare_dicts({"val": 1.0}, {"val": 1.1}, numeric_tolerance=1e-5)
        assert len(errors) == 1
        assert "tolerance" in errors[0]

    def test_no_tolerance_exact_match_required(self) -> None:
        errors = compare_dicts({"val": 1.0}, {"val": 1.000001})
        assert len(errors) == 1

    def test_bool_not_treated_as_numeric(self) -> None:
        # bool is excluded from numeric tolerance path; True == 1 in Python
        # so this passes as equal via the non-numeric branch
        errors = compare_dicts({"flag": True}, {"flag": 1}, numeric_tolerance=1e-5)
        assert errors == []
        # But True != "true" (different types)
        errors = compare_dicts({"flag": True}, {"flag": "true"}, numeric_tolerance=1e-5)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# load_json (both modules)
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_check_load_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "value"}))
        assert load_json(p) == {"key": "value"}

    def test_step_load_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"num": 42}))
        assert step_load_json(p) == {"num": 42}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
    name,age,score
    Alice, 30 , 9.5
    Bob,25,8.0
""")


class TestReadCsvRows:
    def test_check_read_csv_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(SAMPLE_CSV)
        rows = check_read_csv_rows(p)
        assert len(rows) == 2
        assert rows[0]["age"] == "30"  # stripped

    def test_step_read_csv_rows_no_required(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(SAMPLE_CSV)
        rows = step_read_csv_rows(p)
        assert len(rows) == 2

    def test_step_read_csv_rows_with_required_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(SAMPLE_CSV)
        rows = step_read_csv_rows(p, required_columns=["name", "age"])
        assert len(rows) == 2

    def test_step_read_csv_rows_missing_required(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(SAMPLE_CSV)
        with pytest.raises(ValueError, match="Missing required columns"):
            step_read_csv_rows(p, required_columns=["name", "nonexistent"])

    def test_no_header_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError, match="No header"):
            check_read_csv_rows(p)


class TestCountCsvRows:
    def test_count(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text(SAMPLE_CSV)
        assert count_csv_rows(p) == 2


# ---------------------------------------------------------------------------
# parse_float / safe_round
# ---------------------------------------------------------------------------


class TestParseFloat:
    def test_valid(self) -> None:
        assert parse_float("3.14") == pytest.approx(3.14)

    def test_empty(self) -> None:
        assert parse_float("") is None

    def test_invalid(self) -> None:
        assert parse_float("abc") is None


class TestSafeRound:
    def test_round_value(self) -> None:
        assert safe_round(3.14159265) == pytest.approx(3.141593)

    def test_none_passthrough(self) -> None:
        assert safe_round(None) is None

    def test_custom_ndigits(self) -> None:
        assert safe_round(3.14159, ndigits=2) == pytest.approx(3.14)
