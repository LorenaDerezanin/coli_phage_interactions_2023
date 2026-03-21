"""Shared I/O helpers for steel-thread step scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path, required_columns: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dicts with stripped string values.

    Parameters
    ----------
    path:
        Path to the CSV file.
    required_columns:
        When provided, raise ``ValueError`` if any listed column is
        missing from the CSV header.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")
        if required_columns is not None:
            missing = [column for column in required_columns if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
        return [{k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()} for row in reader]


def parse_float(value: str) -> Optional[float]:
    """Parse a string to float, returning None for empty or unparseable values."""
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    """Round a float to *ndigits* decimal places, passing through None."""
    if value is None:
        return None
    return round(float(value), ndigits)
