#!/usr/bin/env python3
"""Input loading utilities for steel-thread steps."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator

REQUIRED_RAW_INTERACTION_COLUMNS = (
    "bacteria",
    "bacteria_index",
    "phage",
    "image",
    "replicate",
    "plate",
    "log_dilution",
    "X",
    "Y",
    "score",
)

ALLOWED_RAW_SCORE_VALUES = {"0", "1", "n"}


def iter_raw_interactions(raw_interactions_path: Path) -> Iterator[Dict[str, str]]:
    """Yield validated raw interaction rows from the semicolon-delimited source file."""
    with raw_interactions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {raw_interactions_path}.")

        missing = [c for c in REQUIRED_RAW_INTERACTION_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns in {raw_interactions_path}: {', '.join(sorted(missing))}"
            )

        for line_no, row in enumerate(reader, start=2):
            normalized = {k: (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            score_value = normalized["score"]
            if score_value not in ALLOWED_RAW_SCORE_VALUES:
                raise ValueError(
                    f"Unexpected score value '{score_value}' at {raw_interactions_path}:{line_no}"
                )
            yield normalized
