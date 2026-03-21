#!/usr/bin/env python3
"""Fetch and format unresolved PR review threads via GitHub GraphQL API."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

# TODO: reviewThreads(first: 100) and comments(first: 10) will silently
# truncate PRs with >100 threads or >10 comments per thread. Add cursor-based
# pagination if this becomes an issue in practice.
REVIEW_THREADS_QUERY = """
query($owner: String!, $repo: String!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 10) {
            nodes {
              body
              author { login }
            }
          }
        }
      }
    }
  }
}
"""


def extract_threads(graphql_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the review thread nodes from a GraphQL response."""
    return (
        graphql_response.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("reviewThreads", {})
        .get("nodes", [])
    )


def filter_unresolved(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only unresolved, non-outdated threads."""
    return [t for t in threads if not t.get("isResolved") and not t.get("isOutdated")]


def format_thread(thread: dict[str, Any]) -> str:
    """Format a single thread as markdown for the Codex prompt."""
    path = thread.get("path", "unknown")
    line = thread.get("line") or 0
    comments = thread.get("comments", {}).get("nodes", [])
    comment_lines = []
    for c in comments:
        author = c.get("author", {}).get("login", "unknown")
        body = c.get("body", "")
        comment_lines.append(f"**{author}:** {body}")
    return f"## {path}:{line}\n\n" + "\n\n".join(comment_lines)


def format_prompt(pr_number: int, threads: list[dict[str, Any]]) -> str:
    """Build the full Codex feedback prompt from unresolved threads."""
    parts = [
        f"# Review feedback to address for PR #{pr_number}",
        "",
        "These are ALL unresolved review threads. Address every one.",
        "",
    ]
    for t in threads:
        parts.append(format_thread(t))
        parts.append("")
    parts.extend(
        [
            "# Instructions",
            "",
            "Address every unresolved thread above. Follow AGENTS.md policies.",
            "You MUST reply to every review comment using gh api to post a reply.",
            "For each comment: either fix the issue and explain what you changed,",
            "or push back explaining why the feedback is wrong or unnecessary.",
            "Run tests: pytest -q lyzortx/tests/",
            "Commit and push fixes to the current branch.",
        ]
    )
    return "\n".join(parts)


def fetch_threads(owner: str, repo: str, pr_number: int) -> dict[str, Any]:
    """Call gh api graphql to fetch review threads. Requires gh CLI."""
    result = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={REVIEW_THREADS_QUERY}",
            "-f",
            f"owner={owner}",
            "-f",
            f"repo={repo}",
            "-F",
            f"pr={pr_number}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    """CLI entry point. Usage: python -m lyzortx.orchestration.review_threads OWNER REPO PR_NUMBER OUTPUT_PATH."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner", help="Repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument("output_path", help="Path to write the feedback prompt")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print the unresolved thread count and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print all threads as JSON (includes id, isResolved, path, comments)",
    )
    args = parser.parse_args()

    response = fetch_threads(args.owner, args.repo, args.pr_number)
    threads = extract_threads(response)

    if args.json:
        print(json.dumps(threads, indent=2))
        sys.exit(0)

    unresolved = filter_unresolved(threads)

    if args.count_only:
        print(len(unresolved))
        sys.exit(0)

    if not unresolved:
        print("0")
        sys.exit(0)

    prompt = format_prompt(args.pr_number, unresolved)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(len(unresolved))


if __name__ == "__main__":
    main()
