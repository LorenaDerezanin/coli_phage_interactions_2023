#!/usr/bin/env python3
"""Analyze token usage across Codex CI workflow runs.

Usage:
    python -m lyzortx.orchestration.ci_token_usage            # recent overview
    python -m lyzortx.orchestration.ci_token_usage --runs 20   # last 20 runs
    python -m lyzortx.orchestration.ci_token_usage --ticket 42  # per-ticket breakdown
    python -m lyzortx.orchestration.ci_token_usage --waste      # token waste analysis
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass

REPO = "LorenaDerezanin/coli_phage_interactions_2023"
IMPLEMENT_WORKFLOW = "codex-implement.yml"
LIFECYCLE_WORKFLOW = "codex-pr-lifecycle.yml"

# Regex to strip ANSI escape codes from log output.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Pattern to find the token-usage line and extract the count from the NEXT line.
# Codex logs: a line containing "tokens used", then the next line has the number.
TOKEN_HEADER_RE = re.compile(r"tokens?\s+used", re.IGNORECASE)
TOKEN_VALUE_RE = re.compile(r"(\d[\d,]*)")

# Waste-detection patterns.
WASTE_PATTERNS: dict[str, re.Pattern[str]] = {
    "env_discovery": re.compile(
        r"(micromamba|EnvironmentNameNotFound|command not found.*(?:mamba|conda))",
        re.IGNORECASE,
    ),
    "failed_commands": re.compile(r"exited (?:1|127|128)\b"),
    "git_config_failures": re.compile(r"user\.name|user\.email", re.IGNORECASE),
    "file_reads": re.compile(r"sed -n"),
    "shell_commands": re.compile(r"/bin/bash\s+-lc"),
}


# ---------------------------------------------------------------------------
# Pure functions (unit-testable)
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return ANSI_RE.sub("", text)


def extract_token_count(log_text: str) -> int | None:
    """Return the token count from Codex log output, or None if not found.

    The Codex action logs a line containing "tokens used" and then the actual
    count appears on the following line.
    """
    lines = strip_ansi(log_text).splitlines()
    for i, line in enumerate(lines):
        if TOKEN_HEADER_RE.search(line):
            # Look at the remaining lines after the header for the first number.
            for subsequent in lines[i + 1 :]:
                match = TOKEN_VALUE_RE.search(subsequent)
                if match:
                    return int(match.group(1).replace(",", ""))
            # If the number is on the same line as the header, try that too.
            match = TOKEN_VALUE_RE.search(line)
            if match:
                return int(match.group(1).replace(",", ""))
    return None


@dataclass
class WasteReport:
    """Aggregated waste-pattern counts from a single run's log."""

    env_discovery: int = 0
    failed_commands: int = 0
    git_config_failures: int = 0
    file_reads: int = 0
    shell_commands: int = 0


def detect_waste(log_text: str) -> WasteReport:
    """Scan *log_text* for waste-indicator patterns and return counts."""
    clean = strip_ansi(log_text)
    return WasteReport(
        env_discovery=len(WASTE_PATTERNS["env_discovery"].findall(clean)),
        failed_commands=len(WASTE_PATTERNS["failed_commands"].findall(clean)),
        git_config_failures=len(WASTE_PATTERNS["git_config_failures"].findall(clean)),
        file_reads=len(WASTE_PATTERNS["file_reads"].findall(clean)),
        shell_commands=len(WASTE_PATTERNS["shell_commands"].findall(clean)),
    )


@dataclass
class RunInfo:
    """Metadata for a single workflow run."""

    run_id: str
    workflow: str
    date: str
    status: str
    conclusion: str
    head_branch: str
    tokens: int | None = None
    waste: WasteReport | None = None


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Produce a simple aligned text table from *headers* and *rows*."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Side-effect boundary: gh CLI helpers
# ---------------------------------------------------------------------------


def _gh(args: list[str], *, repo: str = REPO) -> str:
    """Run a ``gh`` CLI command and return its stdout."""
    cmd = ["gh"] + args + ["--repo", repo]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"gh command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result.stdout


def _gh_json(args: list[str], *, repo: str = REPO) -> list[dict] | dict:
    """Run a ``gh`` CLI command that returns JSON."""
    raw = _gh(args, repo=repo)
    if not raw.strip():
        return []
    return json.loads(raw)


def list_runs(workflow: str, limit: int = 50) -> list[dict]:
    """List recent workflow runs for *workflow*."""
    raw = _gh([
        "run", "list",
        "--workflow", workflow,
        "--limit", str(limit),
        "--json", "databaseId,workflowName,createdAt,status,conclusion,headBranch",
    ])
    if not raw.strip():
        return []
    return json.loads(raw)


def get_run_log(run_id: str) -> str:
    """Fetch the full log for a workflow run."""
    return _gh(["run", "view", str(run_id), "--log"])


def find_pr_for_branch(branch: str) -> dict | None:
    """Find the PR associated with a branch, if any."""
    raw = _gh([
        "pr", "list",
        "--head", branch,
        "--state", "all",
        "--json", "number,title,body,headRefName",
        "--limit", "1",
    ])
    if not raw.strip():
        return None
    items = json.loads(raw)
    return items[0] if items else None


def find_issue_from_pr_body(body: str) -> int | None:
    """Extract issue number from 'Closes #N' in a PR body."""
    match = re.search(r"Closes\s+#(\d+)", body or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------


def cmd_overview(limit: int) -> None:
    """Print a summary of recent Codex CI runs."""
    all_runs: list[RunInfo] = []
    for wf in [IMPLEMENT_WORKFLOW, LIFECYCLE_WORKFLOW]:
        for entry in list_runs(wf, limit=limit):
            all_runs.append(RunInfo(
                run_id=str(entry["databaseId"]),
                workflow=entry["workflowName"],
                date=entry["createdAt"][:10],
                status=entry["status"],
                conclusion=entry.get("conclusion") or "",
                head_branch=entry.get("headBranch") or "",
            ))

    # Sort by date descending, then take the requested limit.
    all_runs.sort(key=lambda r: r.date, reverse=True)
    all_runs = all_runs[:limit]

    if not all_runs:
        print("No Codex workflow runs found.")
        return

    # Fetch token counts.
    for run in all_runs:
        log = get_run_log(run.run_id)
        run.tokens = extract_token_count(log)

    # Build table.
    rows = []
    for r in all_runs:
        pr = find_pr_for_branch(r.head_branch)
        assoc = ""
        if pr:
            issue = find_issue_from_pr_body(pr.get("body", ""))
            assoc = f"PR#{pr['number']}"
            if issue:
                assoc += f" / Issue#{issue}"
        tokens_str = f"{r.tokens:,}" if r.tokens is not None else "N/A"
        rows.append([r.run_id, r.workflow[:30], r.date, r.conclusion or r.status, tokens_str, assoc])

    headers = ["Run ID", "Workflow", "Date", "Status", "Tokens", "PR/Issue"]
    print(format_table(rows, headers))

    # Summary stats.
    token_values = [r.tokens for r in all_runs if r.tokens is not None]
    success_tokens = [r.tokens for r in all_runs if r.tokens is not None and r.conclusion == "success"]
    failure_tokens = [r.tokens for r in all_runs if r.tokens is not None and r.conclusion == "failure"]

    print()
    print(f"Total tokens: {sum(token_values):,}" if token_values else "Total tokens: N/A")
    if success_tokens:
        print(f"Avg tokens (success): {sum(success_tokens) // len(success_tokens):,}")
    if failure_tokens:
        print(f"Avg tokens (failure): {sum(failure_tokens) // len(failure_tokens):,}")


def cmd_ticket(issue_number: int) -> None:
    """Print token spend breakdown for a given orchestrator ticket."""
    # Find the implement run that created the PR.
    implement_runs = list_runs(IMPLEMENT_WORKFLOW, limit=100)
    lifecycle_runs = list_runs(LIFECYCLE_WORKFLOW, limit=100)

    # Find PR that closes this issue.
    raw = _gh([
        "pr", "list",
        "--search", f"Closes #{issue_number}",
        "--state", "all",
        "--json", "number,title,headRefName,body",
        "--limit", "5",
    ])
    prs = json.loads(raw) if raw.strip() else []
    target_pr = None
    for pr in prs:
        if find_issue_from_pr_body(pr.get("body", "")) == issue_number:
            target_pr = pr
            break

    if not target_pr:
        print(f"No PR found that closes issue #{issue_number}.")
        return

    pr_branch = target_pr["headRefName"]
    pr_number = target_pr["number"]
    print(f"Issue #{issue_number} -> PR #{pr_number} (branch: {pr_branch})")
    print()

    # Match implement runs to this issue's PR branch.
    impl_matched = [r for r in implement_runs if r.get("headBranch") == pr_branch]
    life_matched = [r for r in lifecycle_runs if r.get("headBranch") == pr_branch]

    impl_tokens = 0
    life_tokens = 0

    if impl_matched:
        print("=== Implementation runs ===")
        for entry in impl_matched:
            rid = str(entry["databaseId"])
            log = get_run_log(rid)
            tokens = extract_token_count(log)
            tokens_str = f"{tokens:,}" if tokens else "N/A"
            print(f"  Run {rid}: {entry.get('conclusion', entry['status'])} — {tokens_str} tokens")
            if tokens:
                impl_tokens += tokens
        print()

    if life_matched:
        print("=== Review lifecycle runs ===")
        for entry in life_matched:
            rid = str(entry["databaseId"])
            log = get_run_log(rid)
            tokens = extract_token_count(log)
            tokens_str = f"{tokens:,}" if tokens else "N/A"
            print(f"  Run {rid}: {entry.get('conclusion', entry['status'])} — {tokens_str} tokens")
            if tokens:
                life_tokens += tokens
        print()

    total = impl_tokens + life_tokens
    print(f"Implementation tokens: {impl_tokens:,}")
    print(f"Review tokens:        {life_tokens:,}")
    print(f"Total tokens:         {total:,}")
    if total > 0:
        print(f"Review overhead:      {life_tokens / total * 100:.1f}%")


def cmd_waste(limit: int) -> None:
    """Print token waste analysis across recent runs."""
    all_runs: list[RunInfo] = []
    for wf in [IMPLEMENT_WORKFLOW, LIFECYCLE_WORKFLOW]:
        for entry in list_runs(wf, limit=limit):
            all_runs.append(RunInfo(
                run_id=str(entry["databaseId"]),
                workflow=entry["workflowName"],
                date=entry["createdAt"][:10],
                status=entry["status"],
                conclusion=entry.get("conclusion") or "",
                head_branch=entry.get("headBranch") or "",
            ))

    all_runs.sort(key=lambda r: r.date, reverse=True)
    all_runs = all_runs[:limit]

    if not all_runs:
        print("No Codex workflow runs found.")
        return

    for run in all_runs:
        log = get_run_log(run.run_id)
        run.tokens = extract_token_count(log)
        run.waste = detect_waste(log)

    rows = []
    for r in all_runs:
        w = r.waste or WasteReport()
        tokens_str = f"{r.tokens:,}" if r.tokens is not None else "N/A"
        rows.append([
            r.run_id,
            r.date,
            r.conclusion or r.status,
            tokens_str,
            str(w.env_discovery),
            str(w.failed_commands),
            str(w.git_config_failures),
            str(w.file_reads),
            str(w.shell_commands),
        ])

    headers = [
        "Run ID", "Date", "Status", "Tokens",
        "Env Fumble", "Failed Cmd", "Git Cfg", "File Reads", "Shell Cmds",
    ]
    print(format_table(rows, headers))

    # Aggregate.
    total_env = sum((r.waste.env_discovery if r.waste else 0) for r in all_runs)
    total_fail = sum((r.waste.failed_commands if r.waste else 0) for r in all_runs)
    total_git = sum((r.waste.git_config_failures if r.waste else 0) for r in all_runs)
    total_reads = sum((r.waste.file_reads if r.waste else 0) for r in all_runs)
    total_shell = sum((r.waste.shell_commands if r.waste else 0) for r in all_runs)
    print()
    print(f"Totals across {len(all_runs)} runs:")
    print(f"  Environment discovery issues: {total_env}")
    print(f"  Failed commands:              {total_fail}")
    print(f"  Git config issues:            {total_git}")
    print(f"  File reads (sed -n):          {total_reads}")
    print(f"  Shell commands (/bin/bash):    {total_shell}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token usage across Codex CI workflow runs.",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=10,
        help="Number of recent runs to analyze (default: 10).",
    )
    parser.add_argument(
        "--ticket", "-t",
        type=int,
        default=None,
        help="Analyze all token spend for a given orchestrator issue number.",
    )
    parser.add_argument(
        "--waste", "-w",
        action="store_true",
        help="Show token waste analysis (env fumbling, failed commands, etc.).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.ticket is not None:
        cmd_ticket(args.ticket)
    elif args.waste:
        cmd_waste(args.runs)
    else:
        cmd_overview(args.runs)


if __name__ == "__main__":
    main()
