#!/usr/bin/env python3
"""User-facing entry point for AUTORESEARCH candidate import and holdout replication."""

from __future__ import annotations

from lyzortx.pipeline.autoresearch.candidate_replay import main


if __name__ == "__main__":
    raise SystemExit(main())
