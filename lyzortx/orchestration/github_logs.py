#!/usr/bin/env python3
"""Extract and clean GitHub Actions logs from Codex implement runs.

Usage:
    # Extract Codex step, strip diffs, save to .scratch/
    python -m lyzortx.orchestration.github_logs extract-codex \\
        https://github.com/LyzorTx/coli_phage_interactions_2023/actions/runs/23946982907/job/69845252959

    # Same, but from already-downloaded log
    python -m lyzortx.orchestration.github_logs extract-codex --local \\
        .scratch/gh-actions-logs/23946982907/69845252959.log
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# GitHub Actions log prefix: "Job\tStep\tTimestampZ content"
LOG_PREFIX_RE = re.compile(r"^[^\t]*\t[^\t]*\t\S+Z\s?")

# Timestamp prefix on lines already extracted from the step column: "2026-...Z content"
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s?")

# Codex session markers (after ANSI stripping).
CODEX_MARKER = "codex"
EXEC_MARKER = "exec"
APPLY_PATCH_MARKER = "apply patch"

# Set of markers that start a new block in the Codex session.
ALL_MARKERS = {CODEX_MARKER, EXEC_MARKER, APPLY_PATCH_MARKER}

# Unified diff line patterns (after prefix/ANSI stripping).
DIFF_START_RE = re.compile(r"^diff --git ")
DIFF_LINE_RE = re.compile(
    r"^(?:diff --git |index [0-9a-f]|--- [ab/]|--- /dev/null|\+\+\+ [ab/]|\+\+\+ /dev/null"
    r"|@@ .* @@|new file mode|deleted file mode|old mode|new mode"
    r"|[+-](?![-+]{2}\s))"
)
# Patch metadata lines (file list after "patch: completed").
PATCH_META_RE = re.compile(r"^patch: |^/__w/")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def strip_log_prefix(line: str) -> str:
    return LOG_PREFIX_RE.sub("", line)


def parse_url(url: str) -> tuple[str, str | None]:
    """Extract (run_id, job_id) from a GitHub Actions URL.

    Supports:
      .../actions/runs/<run>/job/<job>
      .../actions/runs/<run>
    """
    m = re.search(r"/actions/runs/(\d+)(?:/job/(\d+))?", url)
    if not m:
        raise ValueError(f"Cannot parse run/job IDs from URL: {url}")
    return m.group(1), m.group(2)


def fetch_log(run_id: str, job_id: str | None) -> str:
    """Fetch log via gh CLI. Returns raw log text."""
    cmd = ["gh", "run", "view", run_id, "--log"]
    if job_id:
        cmd += ["--job", job_id]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"gh run view failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def extract_codex_step(log_text: str) -> list[str]:
    """Extract lines belonging to the 'Implement task with Codex' step.

    Falls back to all lines if step name is UNKNOWN STEP (common in long runs).
    """
    codex_lines: list[str] = []
    all_unknown = True

    for line in log_text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        step = parts[1]
        if step != "UNKNOWN STEP":
            all_unknown = False
        if step == "Implement task with Codex":
            codex_lines.append(parts[2])

    if codex_lines:
        return codex_lines

    # All UNKNOWN STEP — extract the Codex session by looking for session markers.
    if all_unknown:
        return _extract_codex_session_from_unknown(log_text)

    return codex_lines


def _extract_codex_session_from_unknown(log_text: str) -> list[str]:
    """When all steps are UNKNOWN STEP, find the Codex session boundaries."""
    lines = log_text.splitlines()
    content_lines = []
    for line in lines:
        parts = line.split("\t", 2)
        if len(parts) >= 3:
            content_lines.append(parts[2])
        else:
            content_lines.append(line)

    # Find session start: "Running: ... codex \"exec\""
    start = None
    for i, line in enumerate(content_lines):
        clean = strip_ansi(line)
        if 'codex "exec"' in clean or "codex exec" in clean.lower():
            start = i
            break

    if start is None:
        # Last resort: return everything after the first "provider:" line.
        for i, line in enumerate(content_lines):
            if "provider:" in strip_ansi(line):
                start = i
                break

    if start is None:
        return content_lines  # give up, return all

    # Find session end: "##[error]" at end, or "Process completed" line.
    end = len(content_lines)
    for i in range(len(content_lines) - 1, start, -1):
        clean = strip_ansi(content_lines[i])
        if "Process completed with exit code" in clean or "##[error]" in clean:
            end = i
            break

    return content_lines[start:end]


def strip_diffs(lines: list[str]) -> list[str]:
    """Remove unified diff blocks from extracted log lines.

    Handles two kinds of diff output:
    1. Patch blocks: start at 'apply patch', include file list + diff hunks.
    2. Orphan diffs: bare 'diff --git' blocks without a preceding 'apply patch'
       (e.g. git diff output streamed into the log).
    """
    result: list[str] = []
    in_patch = False  # Inside an 'apply patch' block.
    in_orphan_diff = False  # Inside a bare diff block.
    files_in_patch: list[str] = []

    for line in lines:
        clean = strip_ansi(line).strip()
        # Remove BOM if present.
        clean = clean.lstrip("\ufeff")
        # Strip timestamp for content matching (lines are already step-extracted).
        content = TIMESTAMP_RE.sub("", clean)

        if content == APPLY_PATCH_MARKER:
            in_orphan_diff = False
            # Flush summary from any previous patch block.
            if files_in_patch:
                result.append(f"  [patched {len(files_in_patch)} file(s): {', '.join(files_in_patch)}]")
            in_patch = True
            files_in_patch = []
            result.append(line)
            continue

        if in_patch:
            # Keep the "patch: completed/failed" status line.
            if PATCH_META_RE.match(content):
                if content.startswith("/__w/"):
                    # Extract just the repo-relative path.
                    path = content.split("/coli_phage_interactions_2023/", 1)
                    short = path[1] if len(path) > 1 else content
                    files_in_patch.append(short)
                else:
                    result.append(line)
                continue

            if DIFF_START_RE.match(content):
                continue

            if DIFF_LINE_RE.match(content):
                continue

            # Blank lines within diff blocks.
            if content == "" and files_in_patch:
                continue

            # We've exited the diff. Emit a summary of what was patched.
            if files_in_patch:
                result.append(f"  [patched {len(files_in_patch)} file(s): {', '.join(files_in_patch)}]")
                files_in_patch = []
            in_patch = False
            result.append(line)
            continue

        # Orphan diff detection: bare 'diff --git' without 'apply patch'.
        if DIFF_START_RE.match(content):
            in_orphan_diff = True
            continue

        if in_orphan_diff:
            # Exit orphan diff only on a known Codex session marker.
            if content in ALL_MARKERS:
                in_orphan_diff = False
                result.append(line)
                continue
            # Everything else inside a diff block is skipped: diff headers,
            # +/- lines, context lines (leading space), @@ hunks, blanks.
            continue

        result.append(line)

    # Flush trailing patch summary.
    if files_in_patch:
        result.append(f"  [patched {len(files_in_patch)} file(s): {', '.join(files_in_patch)}]")

    return result


def strip_exec_output(lines: list[str]) -> list[str]:
    """Keep exec command + status lines but drop their output.

    Exec blocks: ``exec`` marker -> command (bold) -> status -> output.
    We keep the marker, command, and status, then skip until the next marker.
    """
    result: list[str] = []
    in_exec = False
    exec_header_remaining = 0  # How many lines after 'exec' to keep (command + status).

    for line in lines:
        clean = strip_ansi(line).strip().lstrip("\ufeff")
        content = TIMESTAMP_RE.sub("", clean)

        if content in ALL_MARKERS:
            in_exec = content == EXEC_MARKER
            if in_exec:
                exec_header_remaining = 2  # Next 2 lines: command, status.
            result.append(line)
            continue

        # Synthetic patch summaries — always keep.
        if content.startswith("[patched "):
            result.append(line)
            continue

        if in_exec:
            if exec_header_remaining > 0:
                exec_header_remaining -= 1
                result.append(line)
            # else: skip output lines
            continue

        result.append(line)

    return result


def filter_codex_only(lines: list[str]) -> list[str]:
    """Keep only Codex reasoning blocks, dropping exec output and other noise.

    A codex block starts at a line whose ANSI-stripped, timestamp-stripped
    content equals "codex" and includes all following lines until the next
    marker (codex, exec, apply patch) or a synthetic [patched ...] summary.
    """
    result: list[str] = []
    in_codex_block = False

    for line in lines:
        clean = strip_ansi(line).strip().lstrip("\ufeff")
        content = TIMESTAMP_RE.sub("", clean)

        if content in ALL_MARKERS:
            in_codex_block = content == CODEX_MARKER
            continue

        # Synthetic patch summaries from strip_diffs — keep them.
        if content.startswith("[patched "):
            result.append(line)
            continue

        if in_codex_block:
            result.append(line)

    return result


def save_output(lines: list[str], run_id: str, job_id: str | None, *, codex_only: bool = False) -> Path:
    """Save cleaned log to .scratch/ and return the path."""
    scratch_dir = Path(".scratch/gh-actions-logs") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tag = "codex-only" if codex_only else "cleaned"
    suffix = f"{job_id}.{tag}.log" if job_id else f"{tag}.log"
    out_path = scratch_dir / suffix
    out_path.write_text("\n".join(strip_ansi(ln) for ln in lines) + "\n", encoding="utf-8")
    return out_path


def cmd_extract_codex(args: argparse.Namespace) -> int:
    """Extract the Codex implement step and strip diffs."""
    if args.local:
        log_path = Path(args.source)
        if not log_path.exists():
            print(f"File not found: {log_path}", file=sys.stderr)
            return 1
        log_text = log_path.read_text(encoding="utf-8")
        # Infer run_id/job_id from path: .scratch/gh-actions-logs/<run>/<job>.log
        parts = log_path.parts
        run_id = "local"
        job_id = log_path.stem
        for i, part in enumerate(parts):
            if part == "gh-actions-logs" and i + 1 < len(parts):
                run_id = parts[i + 1]
                break
    else:
        run_id, job_id = parse_url(args.source)
        log_text = fetch_log(run_id, job_id)

    codex_lines = extract_codex_step(log_text)
    if not codex_lines:
        print("No Codex step lines found in log.", file=sys.stderr)
        return 1

    cleaned = strip_diffs(codex_lines)
    if args.codex_only:
        cleaned = filter_codex_only(cleaned)
    elif not args.keep_exec_output:
        cleaned = strip_exec_output(cleaned)
    out_path = save_output(cleaned, run_id, job_id, codex_only=args.codex_only)
    print(f"Extracted {len(codex_lines)} lines -> {len(cleaned)} after filtering")
    print(f"Saved to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="github_logs",
        description="Extract and clean GitHub Actions logs.",
    )
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser(
        "extract-codex",
        help="Extract the Codex implement step from a run, stripping diff output.",
    )
    p_extract.add_argument(
        "source",
        help="GitHub Actions URL (run/job) or local log path (with --local).",
    )
    p_extract.add_argument(
        "--local",
        action="store_true",
        help="Treat source as a local file path instead of a URL.",
    )
    p_extract.add_argument(
        "--codex-only",
        action="store_true",
        help="Keep only Codex reasoning/commentary, dropping exec output and other noise.",
    )
    p_extract.add_argument(
        "--keep-exec-output",
        action="store_true",
        help="Keep full exec command output (default: strip output, keep command + status).",
    )
    p_extract.set_defaults(func=cmd_extract_codex)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
