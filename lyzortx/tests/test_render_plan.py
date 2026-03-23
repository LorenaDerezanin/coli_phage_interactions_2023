"""Tests for render_plan.py.

Uses a small fixture plan with an expected .md snapshot to verify rendering,
plus a smoke test against the real plan.yml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lyzortx.orchestration.render_plan import render_plan

PLAN_PATH = Path("lyzortx/orchestration/plan.yml")
FIXTURE_PATH = Path("lyzortx/tests/fixtures/fixture_plan.yml")
EXPECTED_PATH = Path("lyzortx/tests/fixtures/fixture_plan_expected.md")


def test_fixture_plan_renders_to_expected_snapshot() -> None:
    """Render the fixture plan and compare to the expected .md file."""
    plan = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    rendered = render_plan(plan)
    expected = EXPECTED_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Fixture rendering does not match expected output. "
        "If the renderer changed intentionally, regenerate fixture_plan_expected.md."
    )


def test_render_does_not_crash_on_real_plan() -> None:
    """Smoke test: rendering the real plan.yml produces non-empty output."""
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    rendered = render_plan(plan)
    assert len(rendered) > 100
