"""Tests for pure functions in ci_token_usage."""

from __future__ import annotations

from lyzortx.orchestration.ci_token_usage import (
    WasteReport,
    detect_waste,
    extract_token_count,
    format_table,
    strip_ansi,
)


class TestStripAnsi:
    def test_removes_color_codes(self) -> None:
        assert strip_ansi("\x1b[32mgreen\x1b[0m") == "green"

    def test_removes_bold_codes(self) -> None:
        assert strip_ansi("\x1b[1;31mbold red\x1b[0m") == "bold red"

    def test_passthrough_plain_text(self) -> None:
        assert strip_ansi("no escape here") == "no escape here"

    def test_multiple_codes(self) -> None:
        assert strip_ansi("\x1b[36mfoo\x1b[0m \x1b[33mbar\x1b[0m") == "foo bar"


class TestExtractTokenCount:
    def test_basic_extraction(self) -> None:
        log = "some output\ntokens used\n12345\nmore stuff"
        assert extract_token_count(log) == 12345

    def test_with_commas(self) -> None:
        log = "prefix\nTokens Used\n1,234,567\ntrailer"
        assert extract_token_count(log) == 1234567

    def test_with_ansi_codes(self) -> None:
        log = "\x1b[1mtokens used\x1b[0m\n\x1b[32m42000\x1b[0m\n"
        assert extract_token_count(log) == 42000

    def test_no_token_line(self) -> None:
        log = "just some random\nlog output\nwithout token info"
        assert extract_token_count(log) is None

    def test_token_header_but_no_number(self) -> None:
        log = "tokens used\n"
        assert extract_token_count(log) is None

    def test_case_insensitive(self) -> None:
        log = "TOKENS USED\n9999\n"
        assert extract_token_count(log) == 9999

    def test_singular_token(self) -> None:
        log = "token used\n500\n"
        assert extract_token_count(log) == 500

    def test_intervening_blank_line_still_finds_number(self) -> None:
        # The function looks at ALL subsequent lines, not just the immediate next.
        log = "tokens used\n\n7777\n"
        assert extract_token_count(log) == 7777


class TestDetectWaste:
    def test_env_discovery(self) -> None:
        log = "trying micromamba activate\ncommand not found: conda\nEnvironmentNameNotFound: foo"
        report = detect_waste(log)
        assert report.env_discovery == 3

    def test_failed_commands(self) -> None:
        log = "step exited 1\nstep exited 127\nstep exited 128\nstep exited 0\n"
        report = detect_waste(log)
        assert report.failed_commands == 3

    def test_git_config(self) -> None:
        log = "git config user.name\ngit config user.email\n"
        report = detect_waste(log)
        assert report.git_config_failures == 2

    def test_file_reads(self) -> None:
        log = "sed -n '1,10p' file.py\nsed -n '20,30p' other.py\n"
        report = detect_waste(log)
        assert report.file_reads == 2

    def test_shell_commands(self) -> None:
        log = "/bin/bash -lc 'echo hi'\n/bin/bash -lc 'ls'\n"
        report = detect_waste(log)
        assert report.shell_commands == 2

    def test_clean_log(self) -> None:
        log = "everything is fine\nno issues here\n"
        report = detect_waste(log)
        assert report == WasteReport()

    def test_ansi_codes_stripped(self) -> None:
        log = "\x1b[31mmicromamba\x1b[0m activate\n"
        report = detect_waste(log)
        assert report.env_discovery == 1


class TestFormatTable:
    def test_basic_table(self) -> None:
        headers = ["A", "B"]
        rows = [["x", "yy"], ["zzz", "w"]]
        result = format_table(rows, headers)
        lines = result.splitlines()
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "A" in lines[0]
        assert "---" in lines[1]

    def test_empty_rows(self) -> None:
        result = format_table([], ["Col1", "Col2"])
        lines = result.splitlines()
        assert len(lines) == 2  # header + separator only
