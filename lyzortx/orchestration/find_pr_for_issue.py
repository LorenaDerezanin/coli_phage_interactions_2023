#!/usr/bin/env python3
"""Find an open PR that closes a given issue number.

Usage (CLI):
    python -m lyzortx.orchestration.find_pr_for_issue 120

Prints JSON ``{"number": 125, "headRefName": "branch-name"}`` to stdout,
or exits with code 1 if no matching PR is found.

Designed for CI workflows where ``gh pr list --search`` suffers from
GitHub search index lag.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

CLOSES_RE = re.compile(r"[Cc]loses\s+#(\d+)\b")


def pr_closes_issue(pr_body: str, issue_number: int) -> bool:
    """Return True if *pr_body* contains a ``Closes #N`` reference matching *issue_number*."""
    return any(int(m.group(1)) == issue_number for m in CLOSES_RE.finditer(pr_body))


def find_pr_for_issue(issue_number: int) -> dict[str, object] | None:
    """Find an open PR whose body contains ``Closes #<issue_number>``.

    Returns ``{"number": int, "headRefName": str}`` or *None*.
    """
    result = subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--json", "number,headRefName,body"],
        capture_output=True,
        text=True,
        check=True,
    )
    prs = json.loads(result.stdout)
    for pr in prs:
        if pr_closes_issue(pr.get("body", ""), issue_number):
            return {"number": pr["number"], "headRefName": pr["headRefName"]}
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <issue_number>", file=sys.stderr)
        sys.exit(2)
    issue_num = int(sys.argv[1])
    match = find_pr_for_issue(issue_num)
    if match is None:
        sys.exit(1)
    print(json.dumps(match))
