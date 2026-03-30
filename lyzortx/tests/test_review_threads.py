"""Tests for review_threads."""

import json
from types import SimpleNamespace

from lyzortx.orchestration.review_threads import (
    extract_page_info,
    extract_threads,
    fetch_threads,
    filter_unresolved,
    format_prompt,
    format_thread,
)


def _graphql_response(threads: list[dict]) -> dict:
    return {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": threads}}}}}


def _thread(
    *,
    id: str = "PRRT_1",
    resolved: bool = False,
    outdated: bool = False,
    path: str = "file.py",
    line: int = 10,
    comments: list[dict] | None = None,
) -> dict:
    if comments is None:
        comments = [{"body": "Fix this", "author": {"login": "reviewer"}}]
    return {
        "id": id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "line": line,
        "comments": {"nodes": comments},
    }


def test_extract_threads() -> None:
    t = _thread()
    response = _graphql_response([t])
    assert extract_threads(response) == [t]


def test_extract_threads_empty_response() -> None:
    assert extract_threads({}) == []


def test_extract_page_info() -> None:
    response = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "pageInfo": {
                            "hasNextPage": True,
                            "endCursor": "CURSOR_1",
                        }
                    }
                }
            }
        }
    }
    assert extract_page_info(response) == {"hasNextPage": True, "endCursor": "CURSOR_1"}


def test_extract_page_info_empty_response() -> None:
    assert extract_page_info({}) == {}


def test_filter_unresolved_keeps_active() -> None:
    threads = [_thread(resolved=False, outdated=False)]
    assert len(filter_unresolved(threads)) == 1


def test_filter_unresolved_drops_resolved() -> None:
    threads = [_thread(resolved=True)]
    assert filter_unresolved(threads) == []


def test_filter_unresolved_drops_outdated() -> None:
    threads = [_thread(outdated=True)]
    assert filter_unresolved(threads) == []


def test_filter_unresolved_mixed() -> None:
    threads = [
        _thread(id="active", resolved=False, outdated=False),
        _thread(id="resolved", resolved=True, outdated=False),
        _thread(id="outdated", resolved=False, outdated=True),
        _thread(id="both", resolved=True, outdated=True),
    ]
    result = filter_unresolved(threads)
    assert len(result) == 1
    assert result[0]["id"] == "active"


def test_format_thread() -> None:
    t = _thread(
        path="src/main.py",
        line=42,
        comments=[
            {"body": "Bug here", "author": {"login": "alice"}},
            {"body": "Agreed", "author": {"login": "bob"}},
        ],
    )
    formatted = format_thread(t)
    assert "## src/main.py:42" in formatted
    assert "**alice:** Bug here" in formatted
    assert "**bob:** Agreed" in formatted


def test_format_thread_null_line() -> None:
    t = _thread(line=None)
    formatted = format_thread(t)
    assert ":0" in formatted


def test_format_prompt_structure() -> None:
    threads = [_thread(path="a.py"), _thread(path="b.py")]
    prompt = format_prompt(99, threads)
    assert "PR #99" in prompt
    assert "## a.py:" in prompt
    assert "## b.py:" in prompt
    assert "Address every unresolved thread" in prompt
    assert "pytest -q lyzortx/tests/" in prompt


def test_format_prompt_empty() -> None:
    prompt = format_prompt(1, [])
    assert "PR #1" in prompt
    assert "Address every unresolved thread" in prompt


def test_fetch_threads_paginates(monkeypatch) -> None:
    first_page = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [_thread(id="page1")],
                        "pageInfo": {
                            "hasNextPage": True,
                            "endCursor": "CURSOR_1",
                        },
                    }
                }
            }
        }
    }
    second_page = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [_thread(id="page2")],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    }
                }
            }
        }
    }
    responses = iter([first_page, second_page])
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool) -> SimpleNamespace:
        assert capture_output is True
        assert text is True
        assert check is True
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps(next(responses)))

    monkeypatch.setattr("lyzortx.orchestration.review_threads.subprocess.run", fake_run)

    result = fetch_threads("owner", "repo", 42)

    assert [thread["id"] for thread in extract_threads(result)] == ["page1", "page2"]
    assert not any(part == "cursor=CURSOR_1" for part in commands[0])
    assert any(part == "cursor=CURSOR_1" for part in commands[1])
