"""Tests for pr_feedback."""

import json
from types import SimpleNamespace

from lyzortx.orchestration.pr_feedback import collect_feedback
from lyzortx.orchestration.pr_feedback import feedback_count
from lyzortx.orchestration.pr_feedback import fetch_paginated_items
from lyzortx.orchestration.pr_feedback import filter_issue_comments
from lyzortx.orchestration.pr_feedback import filter_review_comments
from lyzortx.orchestration.pr_feedback import filter_reviews
from lyzortx.orchestration.pr_feedback import format_feedback_prompt


def _issue_comment(*, login: str = "reviewer", body: str = "Needs work") -> dict:
    return {"user": {"login": login}, "body": body}


def _review_comment(
    *,
    login: str = "reviewer",
    body: str = "Fix this",
    in_reply_to_id: int | None = None,
    path: str = "file.py",
    line: int = 10,
) -> dict:
    return {
        "user": {"login": login},
        "body": body,
        "in_reply_to_id": in_reply_to_id,
        "path": path,
        "line": line,
    }


def _review(*, login: str = "reviewer", body: str = "Looks wrong", state: str = "COMMENTED") -> dict:
    return {"user": {"login": login}, "body": body, "state": state}


def test_filter_issue_comments_ignores_empty_and_czarphage() -> None:
    comments = [
        _issue_comment(login="claude[bot]", body="Please fix this"),
        _issue_comment(login="claude[bot]", body="   "),
        _issue_comment(login="czarphage[bot]", body="Ready for human review."),
    ]
    filtered = filter_issue_comments(comments)
    assert filtered == [_issue_comment(login="claude[bot]", body="Please fix this")]


def test_filter_review_comments_ignores_replies_empty_and_czarphage() -> None:
    comments = [
        _review_comment(login="claude[bot]", body="Needs fix"),
        _review_comment(login="claude[bot]", body="reply", in_reply_to_id=1),
        _review_comment(login="claude[bot]", body=" "),
        _review_comment(login="czarphage[bot]", body="internal"),
    ]
    filtered = filter_review_comments(comments)
    assert filtered == [_review_comment(login="claude[bot]", body="Needs fix")]


def test_filter_reviews_ignores_pending_empty_and_czarphage() -> None:
    reviews = [
        _review(login="claude[bot]", body="Needs changes", state="COMMENTED"),
        _review(login="claude[bot]", body="", state="COMMENTED"),
        _review(login="claude[bot]", body="pending", state="PENDING"),
        _review(login="czarphage[bot]", body="internal", state="COMMENTED"),
    ]
    filtered = filter_reviews(reviews)
    assert filtered == [_review(login="claude[bot]", body="Needs changes", state="COMMENTED")]


def test_feedback_count_sums_all_comment_surfaces() -> None:
    feedback = {
        "issue_comments": [_issue_comment()],
        "review_comments": [_review_comment(), _review_comment(line=11)],
        "reviews": [_review()],
    }
    assert feedback_count(feedback) == 4


def test_format_feedback_prompt_includes_every_surface() -> None:
    feedback = {
        "issue_comments": [_issue_comment(login="claude[bot]", body="Top-level note")],
        "review_comments": [_review_comment(login="claude[bot]", body="Inline note", path="a.py", line=7)],
        "reviews": [_review(login="claude[bot]", body="Review summary", state="COMMENTED")],
    }
    prompt = format_feedback_prompt(331, feedback)
    assert "PR #331" in prompt
    assert "## Top-level PR comment by claude[bot]" in prompt
    assert "## Review comment by claude[bot] on a.py:7" in prompt
    assert "## Top-level review by claude[bot] (COMMENTED)" in prompt
    assert "Do not say there is nothing to do" in prompt


def test_fetch_paginated_items_parses_multi_page(monkeypatch) -> None:
    raw = json.dumps([_issue_comment(body="page 1")]) + "\n" + json.dumps([_issue_comment(body="page 2")])

    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool) -> SimpleNamespace:
        assert command == ["gh", "api", "--paginate", "repos/o/r/issues/7/comments"]
        assert capture_output is True
        assert text is True
        assert check is True
        return SimpleNamespace(stdout=raw)

    monkeypatch.setattr("lyzortx.orchestration.pr_feedback.subprocess.run", fake_run)
    items = fetch_paginated_items("repos/o/r/issues/7/comments")
    assert [item["body"] for item in items] == ["page 1", "page 2"]


def test_collect_feedback_combines_every_surface(monkeypatch) -> None:
    monkeypatch.setattr(
        "lyzortx.orchestration.pr_feedback.fetch_issue_comments",
        lambda owner, repo, pr_number: [_issue_comment(login="claude[bot]", body="Issue comment")],
    )
    monkeypatch.setattr(
        "lyzortx.orchestration.pr_feedback.fetch_review_comments",
        lambda owner, repo, pr_number: [_review_comment(login="claude[bot]", body="Inline comment")],
    )
    monkeypatch.setattr(
        "lyzortx.orchestration.pr_feedback.fetch_reviews",
        lambda owner, repo, pr_number: [_review(login="claude[bot]", body="Review body")],
    )

    feedback = collect_feedback("owner", "repo", 331)
    assert len(feedback["issue_comments"]) == 1
    assert len(feedback["review_comments"]) == 1
    assert len(feedback["reviews"]) == 1
