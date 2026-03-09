"""Unit tests for pure helper logic in ST0.3b split-suite builder."""

from __future__ import annotations

from lyzortx.pipeline.steel_thread_v0.steps.st03b_build_split_suite import family_key
from lyzortx.pipeline.steel_thread_v0.steps.st03b_build_split_suite import select_holdout_items


def test_family_key_uses_placeholder_for_missing_values() -> None:
    assert family_key("") == "__MISSING_PHAGE_FAMILY__"
    assert family_key("Myoviridae") == "Myoviridae"


def test_select_holdout_items_is_deterministic_and_nonempty() -> None:
    items = ["A", "B", "C", "D", "E"]
    first = select_holdout_items(items, holdout_fraction=0.2, split_salt="salt-v1")
    second = select_holdout_items(items, holdout_fraction=0.2, split_salt="salt-v1")

    assert first == second
    assert len(first) == 1


def test_select_holdout_items_changes_with_salt() -> None:
    items = ["A", "B", "C", "D", "E"]
    first = select_holdout_items(items, holdout_fraction=0.4, split_salt="salt-v1")
    second = select_holdout_items(items, holdout_fraction=0.4, split_salt="salt-v2")

    assert first != second
