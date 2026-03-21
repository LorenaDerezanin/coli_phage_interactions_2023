"""Tests for review_threads."""

from lyzortx.orchestration.review_threads import (
    extract_threads,
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
