#!/usr/bin/env python3
"""Verify that all Codex review comments on a PR have been replied to."""

from __future__ import annotations

import json
import sys
from typing import Any

CODEX_BOT = "chatgpt-codex-connector[bot]"


def find_unanswered_comments(
    review_comments: list[dict[str, Any]],
    review_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return Codex review comments that have no replies.

    Args:
        review_comments: All PR review comments (from GitHub API).
        review_id: If set, only consider comments from this specific review.

    Returns:
        List of top-level Codex comments with zero replies.
    """
    # Find top-level Codex comments (not replies themselves)
    top_level = [
        c for c in review_comments
        if c.get("user", {}).get("login") == CODEX_BOT
        and c.get("in_reply_to_id") is None
        and (review_id is None or c.get("pull_request_review_id") == review_id)
    ]

    # Build set of comment IDs that have at least one reply
    replied_to = {
        c["in_reply_to_id"]
        for c in review_comments
        if c.get("in_reply_to_id") is not None
    }

    return [c for c in top_level if c["id"] not in replied_to]


def main() -> None:
    """Read review comments JSON from stdin, check for unanswered comments."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", type=int, default=None, help="Scope to a specific review ID")
    args = parser.parse_args()

    comments = json.load(sys.stdin)
    unanswered = find_unanswered_comments(comments, review_id=args.review_id)

    if unanswered:
        for c in unanswered:
            path = c.get("path", "?")
            line = c.get("line") or c.get("original_line", "?")
            print(f"::warning::Unanswered comment on {path}:{line} (id={c['id']})")
        print(f"::error::{len(unanswered)} review comment(s) were not answered.")
        sys.exit(1)

    print(f"All review comments have replies.")


if __name__ == "__main__":
    main()
