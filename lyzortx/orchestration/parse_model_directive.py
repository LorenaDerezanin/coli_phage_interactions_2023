#!/usr/bin/env python3
"""Extract model directive from a GitHub issue body and resolve to a concrete model ID.

Usage (CLI — extract only):
    gh issue view 42 --json body --jq '.body' | python -m lyzortx.orchestration.parse_model_directive

Usage (CLI — extract and resolve for a provider):
    gh issue view 42 --json body --jq '.body' | python -m lyzortx.orchestration.parse_model_directive --provider claude

Prints the concrete model ID (e.g. ``claude-opus-4-6``) to stdout, or nothing if absent.
"""

from __future__ import annotations

import re
import sys

MODEL_DIRECTIVE_RE = re.compile(r"<!--\s*model:\s*(\S+)\s*-->")

# Semantic model tiers used in plan.yml.  Each tier maps to a concrete model
# ID per provider.  "smart" = highest-capability tier for complex/fragile tasks;
# "simple" = cost-efficient tier for bounded mechanical work.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "smart": "claude-opus-4-6",
        "simple": "claude-sonnet-4-6",
    },
    "codex": {
        "smart": "gpt-5.4",
        "simple": "gpt-5.4-mini",
    },
}

VALID_TIERS = frozenset(("smart", "simple"))


def resolve_model(tier: str, provider: str) -> str:
    """Map a semantic tier (``smart`` / ``simple``) to a concrete model ID.

    Raises ``ValueError`` for unknown tiers or providers.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Unknown model tier '{tier}'. Valid tiers: {', '.join(sorted(VALID_TIERS))}.")
    try:
        return PROVIDER_MODELS[provider][tier]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{provider}'. Known providers: {', '.join(sorted(PROVIDER_MODELS))}."
        ) from None


def extract_model(body: str) -> str | None:
    """Return the model tier from an ``<!-- model: ... -->`` HTML comment, or *None*."""
    match = MODEL_DIRECTIVE_RE.search(body)
    return match.group(1) if match else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default=None,
        help="Resolve tier to concrete model ID for this provider (claude, codex).",
    )
    args = parser.parse_args()

    body = sys.stdin.read()
    model = extract_model(body)
    if model:
        if args.provider:
            model = resolve_model(model, args.provider)
        print(model)
