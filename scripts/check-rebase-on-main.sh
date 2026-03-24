#!/usr/bin/env bash
# Pre-push hook: blocks pushes when the branch is not rebased on origin/main.
# Installed via pre-commit (stages: [pre-push]).
#
# Checks two things:
#   1. origin/main's tip is an ancestor of HEAD (branch is up to date).
#   2. No merge commits exist between origin/main and HEAD (linear history).

set -euo pipefail

branch=$(git rev-parse --abbrev-ref HEAD)

# Pushing main itself is always allowed.
if [ "$branch" = "main" ]; then
    exit 0
fi

# Fetch latest main so the check uses current remote state.
git fetch origin main --quiet

merge_base=$(git merge-base origin/main HEAD)
main_tip=$(git rev-parse origin/main)

if [ "$merge_base" != "$main_tip" ]; then
    echo "ERROR: Branch '$branch' is not rebased on origin/main."
    echo "Run: git fetch origin main && git rebase origin/main"
    exit 1
fi

# Reject merge commits — the policy requires rebase, not merge.
if git log --merges --oneline origin/main..HEAD | grep -q .; then
    echo "ERROR: Branch '$branch' has merge commits. Use rebase, not merge."
    echo "Run: git rebase origin/main"
    exit 1
fi
