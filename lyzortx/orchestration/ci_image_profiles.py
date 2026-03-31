#!/usr/bin/env python3
"""Shared CI image profile metadata for orchestrator and workflow routing."""

from __future__ import annotations

CI_IMAGE_BASE = "ghcr.io/lyzortx/coli-phage-interactions-2023-codex-ci"
VALID_CI_IMAGE_PROFILES = {
    "base",
    "host-typing",
    "full-bio",
}
CI_IMAGE_LABEL_PREFIX = "ci-image:"


def normalize_ci_image_profile(profile: str) -> str:
    """Return a validated CI image profile."""
    if profile not in VALID_CI_IMAGE_PROFILES:
        valid = ", ".join(sorted(VALID_CI_IMAGE_PROFILES))
        raise ValueError(f"Unknown ci_image_profile {profile!r}; expected one of: {valid}")
    return profile


def ci_image_profile_label(profile: str) -> str:
    """Return the GitHub label used to mirror the CI image profile."""
    normalized = normalize_ci_image_profile(profile)
    return f"{CI_IMAGE_LABEL_PREFIX}{normalized}"


def ci_image_profile_from_labels(labels: list[str]) -> str:
    """Resolve exactly one CI image profile from a set of GitHub labels."""
    matches = [label for label in labels if label.startswith(CI_IMAGE_LABEL_PREFIX)]
    if not matches:
        raise ValueError("Missing required ci-image:* label")
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one ci-image:* label, found: {matches}")
    return normalize_ci_image_profile(matches[0].removeprefix(CI_IMAGE_LABEL_PREFIX))


def ci_image_for_profile(profile: str, image_tag: str = "main") -> str:
    """Return the fully qualified GHCR image reference for a profile/tag pair."""
    normalized = normalize_ci_image_profile(profile)
    return f"{CI_IMAGE_BASE}:{normalized}-{image_tag}"
