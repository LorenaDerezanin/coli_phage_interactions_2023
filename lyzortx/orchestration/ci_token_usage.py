#!/usr/bin/env python3
"""Analyze token usage and cost across Codex CI and Claude PR review runs.

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
REVIEW_WORKFLOW = "claude-pr-review.yml"

ALL_WORKFLOWS = [IMPLEMENT_WORKFLOW, LIFECYCLE_WORKFLOW, REVIEW_WORKFLOW]

# Regex to strip ANSI escape codes from log output.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# GitHub Actions log lines have a prefix: "Job\tStep\tTimestampZ "
# Strip this to get just the content portion.
GH_LOG_PREFIX_RE = re.compile(r"^[^\t]*\t[^\t]*\t\S+Z\s*")

# Pattern to find the token-usage line and extract the count from the NEXT line.
# Codex logs: a line containing "tokens used", then the next line has the number.
TOKEN_HEADER_RE = re.compile(r"tokens?\s+used", re.IGNORECASE)
TOKEN_VALUE_RE = re.compile(r"^(\d[\d,]*)$")

# Claude code action logs a JSON result block with "total_cost_usd".
COST_USD_RE = re.compile(r'"total_cost_usd"\s*:\s*([\d.]+)')
NUM_TURNS_RE = re.compile(r'"num_turns"\s*:\s*(\d+)')
DURATION_MS_RE = re.compile(r'"duration_ms"\s*:\s*(\d+)')

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


def strip_log_prefix(line: str) -> str:
    """Remove the GitHub Actions log prefix (Job\\tStep\\tTimestamp) from a line."""
    return GH_LOG_PREFIX_RE.sub("", line)


def extract_token_count(log_text: str) -> int | None:
    """Return the token count from Codex log output, or None if not found.

    The Codex action logs a line containing "tokens used" and then the actual
    count appears on the following line.  Each log line is prefixed with
    ``Job\\tStep\\tTimestampZ`` which must be stripped before matching numbers.
    """
    raw_lines = strip_ansi(log_text).splitlines()
    # Strip GitHub Actions log prefix from every line so timestamps don't
    # produce false-positive number matches (e.g. "2026" from a date).
    lines = [strip_log_prefix(line) for line in raw_lines]
    for i, line in enumerate(lines):
        if TOKEN_HEADER_RE.search(line):
            # Look at subsequent lines for a standalone number.
            for subsequent in lines[i + 1 :]:
                content = subsequent.strip()
                match = TOKEN_VALUE_RE.match(content)
                if match:
                    return int(match.group(1).replace(",", ""))
                # Stop if we hit non-empty non-numeric content.
                if content:
                    break
    return None


@dataclass
class ClaudeActionResult:
    """Parsed result block from a Claude code action run."""

    cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0


def extract_claude_action_result(log_text: str) -> ClaudeActionResult | None:
    """Extract cost/turns/duration from a Claude code action JSON result block.

    The claude-code-action logs a JSON object with ``total_cost_usd``,
    ``num_turns``, and ``duration_ms`` near the end of the run.
    """
    clean = strip_ansi(log_text)
    # Strip log prefixes so we're working with raw content.
    lines = [strip_log_prefix(line) for line in clean.splitlines()]
    text = "\n".join(lines)

    cost_match = COST_USD_RE.search(text)
    if not cost_match:
        return None

    turns_match = NUM_TURNS_RE.search(text)
    duration_match = DURATION_MS_RE.search(text)

    return ClaudeActionResult(
        cost_usd=float(cost_match.group(1)),
        num_turns=int(turns_match.group(1)) if turns_match else 0,
        duration_ms=int(duration_match.group(1)) if duration_match else 0,
    )


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
    display_title: str = ""
    event: str = ""
    tokens: int | None = None
    claude_result: ClaudeActionResult | None = None
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
    raw = _gh(
        [
            "run",
            "list",
            "--workflow",
            workflow,
            "--limit",
            str(limit),
            "--json",
            "databaseId,workflowName,createdAt,status,conclusion,headBranch,displayTitle,event",
        ]
    )
    if not raw.strip():
        return []
    return json.loads(raw)


def get_run_log(run_id: str) -> str:
    """Fetch the full log for a workflow run."""
    return _gh(["run", "view", str(run_id), "--log"])


def find_pr_for_branch(branch: str) -> dict | None:
    """Find the PR associated with a branch, if any."""
    raw = _gh(
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            "number,title,body,headRefName",
            "--limit",
            "1",
        ]
    )
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


def _build_run_info(entry: dict) -> RunInfo:
    """Create a RunInfo from a gh run list JSON entry."""
    return RunInfo(
        run_id=str(entry["databaseId"]),
        workflow=entry["workflowName"],
        date=entry["createdAt"][:10],
        status=entry["status"],
        conclusion=entry.get("conclusion") or "",
        head_branch=entry.get("headBranch") or "",
        display_title=entry.get("displayTitle") or "",
        event=entry.get("event") or "",
    )


def _run_association(run: RunInfo) -> str:
    """Derive a short PR/Issue label for a run.

    Implement runs trigger on ``main`` so we can't look up the PR by branch.
    Instead we extract a task ID from the display title (e.g. ``[ORCH][TB05] ...``).
    Lifecycle runs have a real feature branch we can look up.
    """
    # For lifecycle runs, look up the PR by branch.
    if run.head_branch and run.head_branch != "main":
        pr = find_pr_for_branch(run.head_branch)
        if pr:
            issue = find_issue_from_pr_body(pr.get("body", ""))
            label = f"PR#{pr['number']}"
            if issue:
                label += f" / Issue#{issue}"
            return label

    # For implement runs, extract task ID from display title.
    m = re.search(r"\[ORCH\]\[(\w+)\]", run.display_title)
    if m:
        return m.group(1)

    return ""


def _is_review_workflow(run: RunInfo) -> bool:
    """Return True if *run* is from the Claude PR review workflow."""
    return run.workflow == "Claude PR Review"


def _enrich_run(run: RunInfo, log: str) -> None:
    """Extract token count and/or Claude action result from a run log."""
    run.tokens = extract_token_count(log)
    run.claude_result = extract_claude_action_result(log)


def _usage_str(run: RunInfo) -> str:
    """Return a human-readable usage string for a run.

    Codex runs report token counts; Claude review runs report USD cost.
    """
    if run.tokens is not None:
        return f"{run.tokens:,} tok"
    if run.claude_result and run.claude_result.cost_usd > 0:
        return f"${run.claude_result.cost_usd:.2f}"
    if run.conclusion in ("skipped", "cancelled"):
        return run.conclusion
    return "no LLM"


def cmd_overview(limit: int) -> None:
    """Print a summary of recent CI runs (Codex + Claude review)."""
    all_runs: list[RunInfo] = []
    for wf in ALL_WORKFLOWS:
        for entry in list_runs(wf, limit=limit):
            all_runs.append(_build_run_info(entry))

    # Sort by date descending, then take the requested limit.
    all_runs.sort(key=lambda r: r.date, reverse=True)
    all_runs = all_runs[:limit]

    if not all_runs:
        print("No workflow runs found.")
        return

    # Fetch token counts / costs.
    for run in all_runs:
        log = get_run_log(run.run_id)
        _enrich_run(run, log)

    # Build table.
    rows = []
    for r in all_runs:
        assoc = _run_association(r)
        rows.append([r.run_id, r.workflow[:30], r.date, r.conclusion or r.status, _usage_str(r), assoc])

    headers = ["Run ID", "Workflow", "Date", "Status", "Usage", "PR/Issue"]
    print(format_table(rows, headers))

    # Summary stats — split by workflow type.
    codex_runs = [r for r in all_runs if not _is_review_workflow(r)]
    review_runs = [r for r in all_runs if _is_review_workflow(r)]

    token_values = [r.tokens for r in codex_runs if r.tokens is not None]
    success_tokens = [r.tokens for r in codex_runs if r.tokens is not None and r.conclusion == "success"]
    failure_tokens = [r.tokens for r in codex_runs if r.tokens is not None and r.conclusion == "failure"]

    print()
    if token_values:
        print(f"Codex total tokens: {sum(token_values):,}")
    if success_tokens:
        print(f"Codex avg tokens (success): {sum(success_tokens) // len(success_tokens):,}")
    if failure_tokens:
        print(f"Codex avg tokens (failure): {sum(failure_tokens) // len(failure_tokens):,}")

    cost_values = [r.claude_result.cost_usd for r in review_runs if r.claude_result and r.claude_result.cost_usd > 0]
    if cost_values:
        print(f"Claude review total cost: ${sum(cost_values):.2f}")
        print(f"Claude review avg cost:   ${sum(cost_values) / len(cost_values):.2f}")
        print(f"Claude review runs:       {len(cost_values)}")


def cmd_ticket(issue_number: int) -> None:
    """Print token spend breakdown for a given orchestrator ticket."""
    implement_runs = list_runs(IMPLEMENT_WORKFLOW, limit=100)
    lifecycle_runs = list_runs(LIFECYCLE_WORKFLOW, limit=100)
    review_runs = list_runs(REVIEW_WORKFLOW, limit=100)

    # Get issue title to match implement runs by display title.
    issue_raw = _gh(["issue", "view", str(issue_number), "--json", "title"])
    issue_title = json.loads(issue_raw).get("title", "") if issue_raw.strip() else ""

    # Find PR that closes this issue.
    raw = _gh(
        [
            "pr",
            "list",
            "--search",
            f"Closes #{issue_number}",
            "--state",
            "all",
            "--json",
            "number,title,headRefName,body",
            "--limit",
            "5",
        ]
    )
    prs = json.loads(raw) if raw.strip() else []
    target_pr = None
    for pr in prs:
        if find_issue_from_pr_body(pr.get("body", "")) == issue_number:
            target_pr = pr
            break

    pr_branch = target_pr["headRefName"] if target_pr else None
    pr_number = target_pr["number"] if target_pr else None
    if target_pr:
        print(f"Issue #{issue_number} -> PR #{pr_number} (branch: {pr_branch})")
    else:
        print(f"Issue #{issue_number}: {issue_title}")
        print("  (no PR found)")
    print()

    # Match implement runs by display title (they all run on main).
    impl_matched = [r for r in implement_runs if issue_title and issue_title in r.get("displayTitle", "")]

    # Match lifecycle runs by PR branch.
    life_matched = [r for r in lifecycle_runs if pr_branch and r.get("headBranch") == pr_branch]

    # Match Claude review runs by PR branch.
    claude_matched = [r for r in review_runs if pr_branch and r.get("headBranch") == pr_branch]

    def _print_codex_runs(label: str, runs: list[dict]) -> int:
        total = 0
        if not runs:
            return 0
        print(f"=== {label} ===")
        for entry in runs:
            rid = str(entry["databaseId"])
            conclusion = entry.get("conclusion") or entry.get("status", "")
            if conclusion in ("skipped",):
                print(f"  Run {rid}: {conclusion} — skipped")
                continue
            log = get_run_log(rid)
            tokens = extract_token_count(log)
            if tokens:
                print(f"  Run {rid}: {conclusion} — {tokens:,} tokens")
            else:
                print(f"  Run {rid}: {conclusion} — no LLM invocation")
            if tokens:
                total += tokens
        print()
        return total

    def _print_claude_runs(label: str, runs: list[dict]) -> float:
        total_cost = 0.0
        if not runs:
            return 0.0
        print(f"=== {label} ===")
        for entry in runs:
            rid = str(entry["databaseId"])
            conclusion = entry.get("conclusion") or entry.get("status", "")
            if conclusion in ("skipped",):
                print(f"  Run {rid}: {conclusion} — skipped")
                continue
            log = get_run_log(rid)
            result = extract_claude_action_result(log)
            if result and result.cost_usd > 0:
                print(f"  Run {rid}: {conclusion} — ${result.cost_usd:.2f} ({result.num_turns} turns)")
                total_cost += result.cost_usd
            else:
                print(f"  Run {rid}: {conclusion} — N/A")
        print()
        return total_cost

    impl_tokens = _print_codex_runs("Implementation runs", impl_matched)
    life_tokens = _print_codex_runs("Review lifecycle runs (Codex)", life_matched)
    claude_cost = _print_claude_runs("Claude review runs", claude_matched)

    codex_total = impl_tokens + life_tokens
    print(f"Implementation tokens:  {impl_tokens:,}")
    print(f"Lifecycle tokens:       {life_tokens:,}")
    print(f"Codex total tokens:     {codex_total:,}")
    if claude_cost > 0:
        print(f"Claude review cost:     ${claude_cost:.2f}")
    if codex_total > 0:
        print(f"Lifecycle overhead:     {life_tokens / codex_total * 100:.1f}%")


def cmd_waste(limit: int) -> None:
    """Print token waste analysis across recent runs."""
    all_runs: list[RunInfo] = []
    for wf in ALL_WORKFLOWS:
        for entry in list_runs(wf, limit=limit):
            all_runs.append(_build_run_info(entry))

    all_runs.sort(key=lambda r: r.date, reverse=True)
    all_runs = all_runs[:limit]

    if not all_runs:
        print("No workflow runs found.")
        return

    for run in all_runs:
        log = get_run_log(run.run_id)
        _enrich_run(run, log)
        run.waste = detect_waste(log)

    rows = []
    for r in all_runs:
        w = r.waste or WasteReport()
        rows.append(
            [
                r.run_id,
                r.workflow[:20],
                r.date,
                r.conclusion or r.status,
                _usage_str(r),
                str(w.env_discovery),
                str(w.failed_commands),
                str(w.git_config_failures),
                str(w.file_reads),
                str(w.shell_commands),
            ]
        )

    headers = [
        "Run ID",
        "Workflow",
        "Date",
        "Status",
        "Usage",
        "Env Fumble",
        "Failed Cmd",
        "Git Cfg",
        "File Reads",
        "Shell Cmds",
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
        "--runs",
        "-n",
        type=int,
        default=10,
        help="Number of recent runs to analyze (default: 10).",
    )
    parser.add_argument(
        "--ticket",
        "-t",
        type=int,
        default=None,
        help="Analyze all token spend for a given orchestrator issue number.",
    )
    parser.add_argument(
        "--waste",
        "-w",
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
