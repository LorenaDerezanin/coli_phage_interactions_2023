"""Tests for verify_review_replies."""

import json

from lyzortx.orchestration.verify_review_replies import find_unanswered_comments, parse_paginated_json


def _comment(
    id: int, user: str = "chatgpt-codex-connector[bot]", reply_to: int | None = None, review_id: int = 100
) -> dict:
    return {
        "id": id,
        "user": {"login": user},
        "in_reply_to_id": reply_to,
        "pull_request_review_id": review_id,
        "path": "file.py",
        "line": 10,
    }


def test_no_comments() -> None:
    assert find_unanswered_comments([]) == []


def test_all_answered() -> None:
    comments = [
        _comment(1),
        _comment(2, user="zoltanmaric", reply_to=1),
    ]
    assert find_unanswered_comments(comments) == []


def test_unanswered() -> None:
    comments = [
        _comment(1),
        _comment(2),
        _comment(3, user="zoltanmaric", reply_to=1),
    ]
    unanswered = find_unanswered_comments(comments)
    assert len(unanswered) == 1
    assert unanswered[0]["id"] == 2


def test_scoped_to_review_id() -> None:
    comments = [
        _comment(1, review_id=100),
        _comment(2, review_id=200),
    ]
    unanswered = find_unanswered_comments(comments, review_id=200)
    assert len(unanswered) == 1
    assert unanswered[0]["id"] == 2


def test_ignores_non_codex_comments() -> None:
    comments = [
        _comment(1, user="some-human"),
    ]
    assert find_unanswered_comments(comments) == []


def test_reply_from_any_user_counts() -> None:
    comments = [
        _comment(1),
        _comment(2, user="codex-action-bot", reply_to=1),
    ]
    assert find_unanswered_comments(comments) == []


def test_parse_single_page() -> None:
    raw = json.dumps([_comment(1), _comment(2)])
    assert len(parse_paginated_json(raw)) == 2


def test_parse_multi_page() -> None:
    page1 = json.dumps([_comment(1)])
    page2 = json.dumps([_comment(2), _comment(3)])
    raw = page1 + "\n" + page2
    assert len(parse_paginated_json(raw)) == 3


def test_parse_empty() -> None:
    assert parse_paginated_json("") == []
