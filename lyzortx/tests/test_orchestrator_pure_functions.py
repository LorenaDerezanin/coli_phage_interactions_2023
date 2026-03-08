"""Unit tests for pure helper logic in the orchestrator module."""

from __future__ import annotations

from lyzortx.orchestration.orchestrator import IssueRef
from lyzortx.orchestration.orchestrator import choose_preferred_issue
from lyzortx.orchestration.orchestrator import extract_task_id_from_issue


def test_extract_task_id_from_body_marker() -> None:
    issue = {
        "body": "<!-- ORCH_TASK_ID: IM07_SOURCE_REGISTRY -->\nTask details",
        "title": "[ORCH][SHOULD_NOT_BE_USED] placeholder",
    }

    assert extract_task_id_from_issue(issue) == "IM07_SOURCE_REGISTRY"


def test_extract_task_id_from_title_when_body_missing_marker() -> None:
    issue = {
        "body": "No orchestrator marker in this body",
        "title": "[ORCH][IM02_ST03B_SPLIT_SUITE] Implement split suite",
    }

    assert extract_task_id_from_issue(issue) == "IM02_ST03B_SPLIT_SUITE"


def test_extract_task_id_returns_none_when_no_marker_or_title_prefix() -> None:
    issue = {
        "body": "No marker here",
        "title": "Regular issue title",
    }

    assert extract_task_id_from_issue(issue) is None


def test_choose_preferred_issue_prefers_open_over_closed() -> None:
    open_issue = IssueRef(
        task_id="IM03_ST04B_ABLATIONS",
        number=11,
        state="open",
        title="open issue",
        html_url="https://example.com/open",
        updated_at="2026-03-08T22:00:00Z",
    )
    closed_issue = IssueRef(
        task_id="IM03_ST04B_ABLATIONS",
        number=10,
        state="closed",
        title="closed issue",
        html_url="https://example.com/closed",
        updated_at="2026-03-08T22:30:00Z",
    )

    assert choose_preferred_issue(closed_issue, open_issue) == open_issue


def test_choose_preferred_issue_prefers_more_recent_when_states_match() -> None:
    older_issue = IssueRef(
        task_id="IM04_ST05B_ST06C_DUAL_SLICE_BOOTSTRAP",
        number=20,
        state="closed",
        title="older",
        html_url="https://example.com/older",
        updated_at="2026-03-08T21:00:00Z",
    )
    newer_issue = IssueRef(
        task_id="IM04_ST05B_ST06C_DUAL_SLICE_BOOTSTRAP",
        number=21,
        state="closed",
        title="newer",
        html_url="https://example.com/newer",
        updated_at="2026-03-08T22:00:00Z",
    )

    assert choose_preferred_issue(newer_issue, older_issue) == newer_issue
