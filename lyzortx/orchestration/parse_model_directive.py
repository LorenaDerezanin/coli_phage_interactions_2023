#!/usr/bin/env python3
"""Extract model directive from a GitHub issue body.

Usage (CLI):
    gh issue view 42 --json body --jq '.body' | python -m lyzortx.orchestration.parse_model_directive

Prints the model ID (e.g. ``gpt-5.4-mini``) to stdout, or nothing if absent.
"""

from __future__ import annotations

import re
import sys

MODEL_DIRECTIVE_RE = re.compile(r"<!--\s*model:\s*(\S+)\s*-->")


def extract_model(body: str) -> str | None:
    """Return the model ID from an ``<!-- model: ... -->`` HTML comment, or *None*."""
    match = MODEL_DIRECTIVE_RE.search(body)
    return match.group(1) if match else None


if __name__ == "__main__":
    body = sys.stdin.read()
    model = extract_model(body)
    if model:
        print(model)
