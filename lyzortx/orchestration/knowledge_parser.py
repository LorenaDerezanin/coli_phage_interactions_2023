#!/usr/bin/env python3
"""Parse knowledge.yml and provide validation and diff logic for the knowledge model.

The knowledge model is a structured representation of consolidated project knowledge,
distilled from lab notebook entries. It follows the same YAML-source-of-truth pattern
as plan.yml / plan_parser.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

VALID_STATUSES = frozenset({"active", "superseded", "dead-end"})
VALID_CONFIDENCES = frozenset({"validated", "preliminary"})


@dataclass(frozen=True)
class KnowledgeUnit:
    """A single consolidated fact or finding."""

    id: str
    statement: str
    sources: list[str]
    status: str  # "active" | "superseded" | "dead-end"
    confidence: str | None = None
    context: str | None = None
    relates_to: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeTheme:
    """A thematic grouping of knowledge units."""

    key: str
    title: str
    description: str | None = None
    units: list[KnowledgeUnit] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeModel:
    """The complete consolidated knowledge model."""

    last_consolidated: str
    source_dir: str
    themes: list[KnowledgeTheme] = field(default_factory=list)

    def all_units(self) -> list[KnowledgeUnit]:
        return [unit for theme in self.themes for unit in theme.units]

    def all_unit_ids(self) -> set[str]:
        return {unit.id for unit in self.all_units()}

    def find_unit(self, unit_id: str) -> KnowledgeUnit | None:
        for unit in self.all_units():
            if unit.id == unit_id:
                return unit
        return None

    def units_by_status(self, status: str) -> list[KnowledgeUnit]:
        return [unit for unit in self.all_units() if unit.status == status]


@dataclass(frozen=True)
class KnowledgeDiff:
    """Differences between two knowledge models."""

    added: list[KnowledgeUnit]
    removed: list[KnowledgeUnit]
    updated: list[tuple[KnowledgeUnit, KnowledgeUnit]]  # (old, new)


def load_knowledge(path: Path) -> KnowledgeModel:
    """Parse knowledge.yml into a KnowledgeModel."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return _parse_model(data)


def _parse_model(data: dict[str, Any]) -> KnowledgeModel:
    themes = []
    for theme_raw in data.get("themes", []):
        units = []
        for unit_raw in theme_raw.get("units", []):
            relates_to = unit_raw.get("relates_to") or []
            units.append(
                KnowledgeUnit(
                    id=unit_raw["id"],
                    statement=unit_raw["statement"].strip(),
                    sources=list(unit_raw.get("sources", [])),
                    status=unit_raw.get("status", "active"),
                    confidence=unit_raw.get("confidence"),
                    context=unit_raw.get("context"),
                    relates_to=list(relates_to),
                )
            )
        themes.append(
            KnowledgeTheme(
                key=theme_raw["key"],
                title=theme_raw["title"],
                description=theme_raw.get("description"),
                units=units,
            )
        )
    return KnowledgeModel(
        last_consolidated=str(data.get("last_consolidated", "")),
        source_dir=str(data.get("source_dir", "")),
        themes=themes,
    )


def validate_knowledge(model: KnowledgeModel) -> list[str]:
    """Validate a knowledge model. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []
    all_ids: set[str] = set()

    for theme in model.themes:
        if not theme.key:
            errors.append("Theme has empty key")
        if not theme.title:
            errors.append(f"Theme {theme.key!r} has empty title")

        for unit in theme.units:
            # Duplicate IDs
            if unit.id in all_ids:
                errors.append(f"Duplicate unit ID: {unit.id!r}")
            all_ids.add(unit.id)

            # Empty statement
            if not unit.statement.strip():
                errors.append(f"Unit {unit.id!r} has empty statement")

            # Invalid status
            if unit.status not in VALID_STATUSES:
                errors.append(f"Unit {unit.id!r} has invalid status {unit.status!r}")

            # Invalid confidence
            if unit.confidence is not None and unit.confidence not in VALID_CONFIDENCES:
                errors.append(f"Unit {unit.id!r} has invalid confidence {unit.confidence!r}")

            # Empty sources
            if not unit.sources:
                errors.append(f"Unit {unit.id!r} has no sources")

    # Validate relates_to references (second pass — need all IDs collected first)
    for unit in model.all_units():
        for ref in unit.relates_to:
            if ref not in all_ids:
                errors.append(f"Unit {unit.id!r} relates_to unknown ID {ref!r}")

    return errors


def diff_knowledge(old: KnowledgeModel, new: KnowledgeModel) -> KnowledgeDiff:
    """Compute the difference between two knowledge models."""
    old_by_id = {unit.id: unit for unit in old.all_units()}
    new_by_id = {unit.id: unit for unit in new.all_units()}

    added = [new_by_id[uid] for uid in new_by_id if uid not in old_by_id]
    removed = [old_by_id[uid] for uid in old_by_id if uid not in new_by_id]
    updated = [
        (old_by_id[uid], new_by_id[uid]) for uid in old_by_id if uid in new_by_id and old_by_id[uid] != new_by_id[uid]
    ]

    return KnowledgeDiff(added=added, removed=removed, updated=updated)
