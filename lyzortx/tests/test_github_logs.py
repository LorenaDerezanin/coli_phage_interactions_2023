"""Tests for lyzortx.orchestration.github_logs."""

from __future__ import annotations

from lyzortx.orchestration.github_logs import (
    extract_codex_step,
    filter_codex_only,
    parse_url,
    strip_ansi,
    strip_diffs,
    strip_exec_output,
)

# ---------------------------------------------------------------------------
# Helpers to build realistic log lines
# ---------------------------------------------------------------------------

TS = "2026-04-03T13:01:06.123Z"


def _raw(job: str, step: str, content: str) -> str:
    """Build a raw GitHub Actions log line."""
    return f"{job}\t{step}\t{TS} {content}"


def _ts(content: str) -> str:
    """Build a timestamp-prefixed content line (already step-extracted)."""
    return f"{TS} {content}"


def _ansi(code: str, text: str) -> str:
    """Wrap text in ANSI escape codes."""
    return f"\x1b[{code}m{text}\x1b[0m"


# ---------------------------------------------------------------------------
# parse_url
# ---------------------------------------------------------------------------


def test_parse_url_with_job() -> None:
    url = "https://github.com/Org/repo/actions/runs/123/job/456"
    assert parse_url(url) == ("123", "456")


def test_parse_url_without_job() -> None:
    url = "https://github.com/Org/repo/actions/runs/789"
    assert parse_url(url) == ("789", None)


def test_parse_url_invalid() -> None:
    try:
        parse_url("https://github.com/Org/repo/pulls/1")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


def test_strip_ansi_removes_codes() -> None:
    assert strip_ansi("\x1b[35m\x1b[3mcodex\x1b[0m\x1b[0m") == "codex"


def test_strip_ansi_preserves_plain() -> None:
    assert strip_ansi("plain text") == "plain text"


# ---------------------------------------------------------------------------
# extract_codex_step
# ---------------------------------------------------------------------------


def test_extract_codex_step_named() -> None:
    log = "\n".join(
        [
            _raw("Job", "Set up job", "setup line"),
            _raw("Job", "Implement task with Codex", "codex line 1"),
            _raw("Job", "Implement task with Codex", "codex line 2"),
            _raw("Job", "Complete job", "done"),
        ]
    )
    result = extract_codex_step(log)
    assert len(result) == 2
    assert "codex line 1" in result[0]
    assert "codex line 2" in result[1]


def test_extract_codex_step_unknown_step_uses_session_markers() -> None:
    log = "\n".join(
        [
            _raw("Job", "UNKNOWN STEP", "setup stuff"),
            _raw("Job", "UNKNOWN STEP", 'Running: codex "exec" --model gpt-5.4'),
            _raw("Job", "UNKNOWN STEP", f"{_ansi('1', 'provider:')} proxy"),
            _raw("Job", "UNKNOWN STEP", f"{_ansi('35;3', 'codex')}"),
            _raw("Job", "UNKNOWN STEP", "I am thinking about the task."),
            _raw("Job", "UNKNOWN STEP", "##[error]The operation was canceled."),
        ]
    )
    result = extract_codex_step(log)
    # Should start at the "Running: codex" line, end before ##[error].
    assert any('codex "exec"' in line or "codex exec" in line.lower() for line in result)
    assert all("##[error]" not in line for line in result)


def test_extract_codex_step_no_match_returns_empty() -> None:
    log = "\n".join(
        [
            _raw("Job", "Build", "building..."),
            _raw("Job", "Test", "testing..."),
        ]
    )
    assert extract_codex_step(log) == []


# ---------------------------------------------------------------------------
# strip_diffs — apply patch blocks
# ---------------------------------------------------------------------------


def _make_patch_block() -> list[str]:
    """A realistic 'apply patch' block with a 2-file diff."""
    return [
        _ts(f"{_ansi('1', 'apply patch')}"),
        _ts(f"{_ansi('1', 'patch:')} completed"),
        _ts(f"{_ansi('2', '/__w/coli_phage_interactions_2023/coli_phage_interactions_2023/lyzortx/foo.py')}"),
        _ts(f"{_ansi('2', '/__w/coli_phage_interactions_2023/coli_phage_interactions_2023/lyzortx/bar.py')}"),
        _ts("diff --git a/lyzortx/foo.py b/lyzortx/foo.py"),
        _ts("new file mode 100644"),
        _ts("index 0000000..abc1234"),
        _ts("--- /dev/null"),
        _ts("+++ b/lyzortx/foo.py"),
        _ts("@@ -0,0 +1,3 @@"),
        _ts("+import os"),
        _ts("+import sys"),
        _ts("+print('hello')"),
        _ts("diff --git a/lyzortx/bar.py b/lyzortx/bar.py"),
        _ts("new file mode 100644"),
        _ts("--- /dev/null"),
        _ts("+++ b/lyzortx/bar.py"),
        _ts("@@ -0,0 +1,1 @@"),
        _ts("+# bar"),
    ]


def test_strip_diffs_replaces_patch_with_summary() -> None:
    codex_before = _ts(f"{_ansi('35;3', 'codex')}")
    codex_text = _ts("I am applying the patch now.")
    codex_after_marker = _ts(f"{_ansi('35;3', 'codex')}")
    codex_after_text = _ts("Patch applied, running tests.")

    lines = [codex_before, codex_text] + _make_patch_block() + [codex_after_marker, codex_after_text]
    result = strip_diffs(lines)

    # Original codex lines preserved.
    assert codex_before in result
    assert codex_text in result
    assert codex_after_marker in result
    assert codex_after_text in result

    # Diff lines removed.
    assert not any("diff --git" in ln for ln in result)
    assert not any("+import os" in ln for ln in result)

    # Summary inserted.
    summaries = [ln for ln in result if "[patched" in ln]
    assert len(summaries) == 1
    assert "lyzortx/foo.py" in summaries[0]
    assert "lyzortx/bar.py" in summaries[0]
    assert "2 file(s)" in summaries[0]


def test_strip_diffs_back_to_back_patches() -> None:
    """Two apply-patch blocks in a row should each get a summary."""
    patch1 = [
        _ts(f"{_ansi('1', 'apply patch')}"),
        _ts(f"{_ansi('1', 'patch:')} completed"),
        _ts(f"{_ansi('2', '/__w/coli_phage_interactions_2023/coli_phage_interactions_2023/lyzortx/a.py')}"),
        _ts("diff --git a/lyzortx/a.py b/lyzortx/a.py"),
        _ts("+# a"),
    ]
    patch2 = [
        _ts(f"{_ansi('1', 'apply patch')}"),
        _ts(f"{_ansi('1', 'patch:')} completed"),
        _ts(f"{_ansi('2', '/__w/coli_phage_interactions_2023/coli_phage_interactions_2023/lyzortx/b.py')}"),
        _ts("diff --git a/lyzortx/b.py b/lyzortx/b.py"),
        _ts("+# b"),
    ]
    after = _ts(f"{_ansi('35;3', 'codex')}")

    result = strip_diffs(patch1 + patch2 + [after])
    summaries = [ln for ln in result if "[patched" in ln]
    assert len(summaries) == 2
    assert "lyzortx/a.py" in summaries[0]
    assert "lyzortx/b.py" in summaries[1]


# ---------------------------------------------------------------------------
# strip_diffs — orphan diffs (git diff output without apply patch)
# ---------------------------------------------------------------------------


def test_strip_diffs_orphan_diff_removed() -> None:
    lines = [
        _ts(f"{_ansi('35;3', 'exec')}"),
        _ts("git diff -- foo.py"),
        _ts("succeeded in 0ms:"),
        _ts("diff --git a/foo.py b/foo.py"),
        _ts("index abc..def 100644"),
        _ts("--- a/foo.py"),
        _ts("+++ b/foo.py"),
        _ts("@@ -1,3 +1,4 @@"),
        _ts(" import os"),
        _ts("+import sys"),
        _ts(" print('hello')"),
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("Diff looks good."),
    ]
    result = strip_diffs(lines)
    assert not any("diff --git" in ln for ln in result)
    assert not any("+import sys" in ln for ln in result)
    # Context line (space-prefixed) also stripped.
    assert not any("import os" in ln and ln.strip().startswith(f"{TS}  import") for ln in result)
    # The codex marker and reasoning are kept.
    assert any("codex" in strip_ansi(ln) for ln in result)
    assert any("Diff looks good" in ln for ln in result)


# ---------------------------------------------------------------------------
# strip_exec_output
# ---------------------------------------------------------------------------


def test_strip_exec_output_keeps_command_and_status() -> None:
    lines = [
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("Starting work."),
        _ts(f"{_ansi('35;3', 'exec')}"),
        _ts("/bin/bash -lc 'cat big_file.py'"),
        _ts("succeeded in 0ms:"),
        _ts("line 1 of big file"),
        _ts("line 2 of big file"),
        _ts("line 3 of big file"),
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("Read the file."),
    ]
    result = strip_exec_output(lines)
    # Command and status kept.
    assert any("cat big_file.py" in ln for ln in result)
    assert any("succeeded" in ln for ln in result)
    # Output lines dropped.
    assert not any("line 1 of big file" in ln for ln in result)
    assert not any("line 2 of big file" in ln for ln in result)
    # Codex blocks kept.
    assert any("Starting work" in ln for ln in result)
    assert any("Read the file" in ln for ln in result)


def test_strip_exec_output_preserves_patch_summaries() -> None:
    lines = [
        _ts(f"{_ansi('35;3', 'exec')}"),
        _ts("some command"),
        _ts("succeeded"),
        "  [patched 1 file(s): lyzortx/foo.py]",
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("Done."),
    ]
    result = strip_exec_output(lines)
    assert any("[patched" in ln for ln in result)


# ---------------------------------------------------------------------------
# filter_codex_only
# ---------------------------------------------------------------------------


def test_filter_codex_only_keeps_reasoning() -> None:
    lines = [
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("I will read the file."),
        _ts("Then I will edit it."),
        _ts(f"{_ansi('35;3', 'exec')}"),
        _ts("cat file.py"),
        _ts("succeeded"),
        _ts("file contents here"),
        _ts(f"{_ansi('35;3', 'codex')}"),
        _ts("Now I will patch."),
        _ts(f"{_ansi('1', 'apply patch')}"),
        "  [patched 1 file(s): foo.py]",
    ]
    result = filter_codex_only(lines)
    texts = " ".join(result)
    assert "I will read the file" in texts
    assert "Then I will edit it" in texts
    assert "Now I will patch" in texts
    assert "[patched" in texts
    # Exec output not present.
    assert "cat file.py" not in texts
    assert "file contents here" not in texts


def test_filter_codex_only_drops_exec_entirely() -> None:
    lines = [
        _ts(f"{_ansi('35;3', 'exec')}"),
        _ts("ls -la"),
        _ts("succeeded"),
        _ts("total 42"),
    ]
    result = filter_codex_only(lines)
    assert result == []
