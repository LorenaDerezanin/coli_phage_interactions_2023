#!/usr/bin/env python3
"""Fetch actionable PR feedback from comment surfaces only."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from lyzortx.orchestration.verify_review_replies import parse_paginated_json

DEFAULT_IGNORED_LOGINS = frozenset({"czarphage[bot]"})
IGNORED_REVIEW_STATES = frozenset({"PENDING"})


def fetch_paginated_items(endpoint: str) -> list[dict[str, Any]]:
    """Fetch a paginated GitHub REST collection via gh api."""
    result = subprocess.run(
        ["gh", "api", "--paginate", endpoint],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_paginated_json(result.stdout)


def fetch_issue_comments(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch top-level PR conversation comments."""
    return fetch_paginated_items(f"repos/{owner}/{repo}/issues/{pr_number}/comments")


def fetch_review_comments(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch inline PR review comments."""
    return fetch_paginated_items(f"repos/{owner}/{repo}/pulls/{pr_number}/comments")


def fetch_reviews(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch top-level PR review summaries."""
    return fetch_paginated_items(f"repos/{owner}/{repo}/pulls/{pr_number}/reviews")


def _login(item: dict[str, Any]) -> str:
    return item.get("user", {}).get("login", "unknown")


def _body(item: dict[str, Any]) -> str:
    return (item.get("body") or "").strip()


def filter_issue_comments(
    comments: list[dict[str, Any]],
    ignored_logins: frozenset[str] = DEFAULT_IGNORED_LOGINS,
) -> list[dict[str, Any]]:
    """Keep non-empty top-level PR comments from non-ignored authors."""
    return [comment for comment in comments if _login(comment) not in ignored_logins and _body(comment)]


def filter_review_comments(
    comments: list[dict[str, Any]],
    ignored_logins: frozenset[str] = DEFAULT_IGNORED_LOGINS,
) -> list[dict[str, Any]]:
    """Keep non-empty top-level review comments from non-ignored authors."""
    return [
        comment
        for comment in comments
        if _login(comment) not in ignored_logins and _body(comment) and comment.get("in_reply_to_id") is None
    ]


def filter_reviews(
    reviews: list[dict[str, Any]],
    ignored_logins: frozenset[str] = DEFAULT_IGNORED_LOGINS,
) -> list[dict[str, Any]]:
    """Keep non-empty review summaries from non-ignored authors."""
    return [
        review
        for review in reviews
        if _login(review) not in ignored_logins and _body(review) and review.get("state") not in IGNORED_REVIEW_STATES
    ]


def collect_feedback(owner: str, repo: str, pr_number: int) -> dict[str, list[dict[str, Any]]]:
    """Collect PR feedback across top-level comments, review comments, and reviews."""
    return {
        "issue_comments": filter_issue_comments(fetch_issue_comments(owner, repo, pr_number)),
        "review_comments": filter_review_comments(fetch_review_comments(owner, repo, pr_number)),
        "reviews": filter_reviews(fetch_reviews(owner, repo, pr_number)),
    }


def feedback_count(feedback: dict[str, list[dict[str, Any]]]) -> int:
    """Return the total number of feedback artifacts on the PR."""
    return len(feedback["issue_comments"]) + len(feedback["review_comments"]) + len(feedback["reviews"])


def format_issue_comment(comment: dict[str, Any]) -> str:
    """Format a top-level PR comment for the Codex prompt."""
    return "\n".join(
        [
            f"## Top-level PR comment by {_login(comment)}",
            "",
            _body(comment),
        ]
    )


def format_review_comment(comment: dict[str, Any]) -> str:
    """Format an inline PR review comment for the Codex prompt."""
    path = comment.get("path", "unknown")
    line = comment.get("line") or comment.get("original_line") or 0
    return "\n".join(
        [
            f"## Review comment by {_login(comment)} on {path}:{line}",
            "",
            _body(comment),
        ]
    )


def format_review(review: dict[str, Any]) -> str:
    """Format a top-level review body for the Codex prompt."""
    return "\n".join(
        [
            f"## Top-level review by {_login(review)} ({review.get('state', 'UNKNOWN')})",
            "",
            _body(review),
        ]
    )


def format_feedback_prompt(pr_number: int, feedback: dict[str, list[dict[str, Any]]]) -> str:
    """Build the full Codex prompt from PR comment surfaces."""
    parts = [
        f"# Review feedback to address for PR #{pr_number}",
        "",
        "These are all PR comments and review bodies visible to the lifecycle workflow.",
        "Do not say there is nothing to do without reading them.",
        "",
    ]

    for comment in feedback["issue_comments"]:
        parts.append(format_issue_comment(comment))
        parts.append("")

    for comment in feedback["review_comments"]:
        parts.append(format_review_comment(comment))
        parts.append("")

    for review in feedback["reviews"]:
        parts.append(format_review(review))
        parts.append("")

    parts.extend(
        [
            "# Instructions",
            "",
            "Read every section above before deciding what action is needed.",
            "Use review-thread replies for inline review comments when you respond on GitHub.",
            "Use a top-level PR comment for top-level PR comments or top-level review summaries.",
            "For each actionable item: either fix it and explain what you changed,",
            "or push back explaining why the feedback is wrong or unnecessary.",
            "End every reply with a signature line: 'Posted by Codex <model>'",
            "where <model> is the model you are running as (e.g., gpt-5.4).",
            "Run tests: pytest -q lyzortx/tests/",
            "Commit and push fixes to the current branch.",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner", help="Repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument("output_path", help="Path to write the feedback prompt")
    parser.add_argument("--count-only", action="store_true", help="Only print the feedback count and exit")
    parser.add_argument("--json", action="store_true", help="Print the collected feedback payload as JSON")
    args = parser.parse_args()

    feedback = collect_feedback(args.owner, args.repo, args.pr_number)
    total = feedback_count(feedback)

    if args.json:
        print(json.dumps(feedback, indent=2))
        sys.exit(0)

    if args.count_only:
        print(total)
        sys.exit(0)

    if total == 0:
        print("0")
        sys.exit(0)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write(format_feedback_prompt(args.pr_number, feedback))
    print(total)


if __name__ == "__main__":
    main()
