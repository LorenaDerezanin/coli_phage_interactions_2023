"""Shared helpers for regression check scripts."""

from __future__ import annotations

import csv
import json
from math import isclose
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file (excluding header)."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return sum(1 for _ in reader)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dicts with stripped string values."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def compare_dicts(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    prefix: str = "",
    numeric_tolerance: Optional[float] = None,
) -> List[str]:
    """Recursively compare two dicts and return a list of mismatch descriptions.

    Parameters
    ----------
    expected, actual:
        The reference and observed dictionaries.
    prefix:
        Dot-separated path prefix for error messages (used in recursion).
    numeric_tolerance:
        When set, numeric values are compared with ``math.isclose``
        using this as the absolute tolerance instead of exact equality.
    """
    errors: List[str] = []
    all_keys = sorted(set(expected.keys()) | set(actual.keys()))
    for key in all_keys:
        path = f"{prefix}.{key}" if prefix else key
        if key not in expected:
            errors.append(f"Unexpected key in actual: {path}")
            continue
        if key not in actual:
            errors.append(f"Missing key in actual: {path}")
            continue
        exp_val = expected[key]
        act_val = actual[key]
        if isinstance(exp_val, dict) and isinstance(act_val, dict):
            errors.extend(compare_dicts(exp_val, act_val, prefix=path, numeric_tolerance=numeric_tolerance))
            continue
        if numeric_tolerance is not None and _is_real_number(exp_val) and _is_real_number(act_val):
            if not isclose(float(exp_val), float(act_val), rel_tol=0.0, abs_tol=numeric_tolerance):
                errors.append(
                    f"Mismatch at {path}: expected={exp_val!r}, actual={act_val!r}, tolerance={numeric_tolerance}"
                )
            continue
        if exp_val != act_val:
            errors.append(f"Mismatch at {path}: expected={exp_val!r}, actual={act_val!r}")
    return errors


def _is_real_number(value: object) -> bool:
    """Return True for numeric types excluding bool."""
    return isinstance(value, Real) and not isinstance(value, bool)
