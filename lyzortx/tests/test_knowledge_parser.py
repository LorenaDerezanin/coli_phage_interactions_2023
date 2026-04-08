"""Tests for knowledge_parser and render_knowledge."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lyzortx.orchestration.knowledge_parser import (
    KnowledgeModel,
    KnowledgeTheme,
    KnowledgeUnit,
    diff_knowledge,
    load_knowledge,
    validate_knowledge,
)
from lyzortx.orchestration.render_knowledge import render_knowledge

MINIMAL_KNOWLEDGE = textwrap.dedent("""\
    last_consolidated: "2026-04-08"
    source_dir: lyzortx/research_notes/lab_notebooks

    themes:
      - key: data-labels
        title: Data & Labels
        description: Labeling policy and data quality findings.
        units:
          - id: label-policy-v1
            statement: >
              Binary lysis labels derived from matrix_score > 0; borderline score=0
              pairs are downweighted rather than excluded.
            sources: [ST0.1, TA11]
            status: active
            confidence: validated

          - id: confidence-tiers
            statement: >
              Three confidence tiers based on replicate agreement.
            sources: [ST0.1b]
            status: active

      - key: dead-ends
        title: Dead Ends
        units:
          - id: external-data-lift
            statement: >
              VHRdb, BASEL, and KlebPhaCol external data showed neutral cumulative
              lift over the internal-only baseline.
            sources: [TK01, TK02, TK03]
            status: dead-end
            context: May revisit with different integration strategies.
            relates_to: [label-policy-v1]
""")


@pytest.fixture
def minimal_knowledge_path(tmp_path: Path) -> Path:
    p = tmp_path / "knowledge.yml"
    p.write_text(MINIMAL_KNOWLEDGE, encoding="utf-8")
    return p


def test_load_knowledge_roundtrip(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)

    assert model.last_consolidated == "2026-04-08"
    assert model.source_dir == "lyzortx/research_notes/lab_notebooks"
    assert len(model.themes) == 2
    assert model.themes[0].key == "data-labels"
    assert model.themes[0].title == "Data & Labels"
    assert len(model.themes[0].units) == 2
    assert model.themes[0].units[0].id == "label-policy-v1"
    assert model.themes[0].units[0].status == "active"
    assert model.themes[0].units[0].confidence == "validated"
    assert model.themes[0].units[0].sources == ["ST0.1", "TA11"]

    # Verify relates_to parsed
    dead_end_unit = model.themes[1].units[0]
    assert dead_end_unit.relates_to == ["label-policy-v1"]
    assert dead_end_unit.context == "May revisit with different integration strategies."


def test_all_units(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)
    assert len(model.all_units()) == 3
    assert model.all_unit_ids() == {"label-policy-v1", "confidence-tiers", "external-data-lift"}


def test_find_unit(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)
    unit = model.find_unit("label-policy-v1")
    assert unit is not None
    assert unit.status == "active"
    assert model.find_unit("nonexistent") is None


def test_units_by_status(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)
    active = model.units_by_status("active")
    assert len(active) == 2
    dead_ends = model.units_by_status("dead-end")
    assert len(dead_ends) == 1


def test_validate_valid_model(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)
    errors = validate_knowledge(model)
    assert errors == []


def test_validate_catches_duplicate_ids() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(id="dupe", statement="First", sources=["X"], status="active"),
                    KnowledgeUnit(id="dupe", statement="Second", sources=["Y"], status="active"),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("Duplicate unit ID" in e for e in errors)


def test_validate_catches_empty_statement() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(id="empty", statement="   ", sources=["X"], status="active"),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("empty statement" in e for e in errors)


def test_validate_catches_invalid_status() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(id="bad", statement="Something", sources=["X"], status="bogus"),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("invalid status" in e for e in errors)


def test_validate_catches_invalid_confidence() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(
                        id="bad-conf",
                        statement="Something",
                        sources=["X"],
                        status="active",
                        confidence="maybe",
                    ),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("invalid confidence" in e for e in errors)


def test_validate_catches_empty_sources() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(id="nosrc", statement="Something", sources=[], status="active"),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("no sources" in e for e in errors)


def test_validate_catches_broken_relates_to() -> None:
    model = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme 1",
                units=[
                    KnowledgeUnit(
                        id="orphan",
                        statement="Something",
                        sources=["X"],
                        status="active",
                        relates_to=["nonexistent-id"],
                    ),
                ],
            )
        ],
    )
    errors = validate_knowledge(model)
    assert any("relates_to unknown ID" in e for e in errors)


def test_diff_knowledge_detects_additions() -> None:
    old = KnowledgeModel(
        last_consolidated="2026-04-01",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[KnowledgeUnit(id="a", statement="A", sources=["X"], status="active")],
            )
        ],
    )
    new = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[
                    KnowledgeUnit(id="a", statement="A", sources=["X"], status="active"),
                    KnowledgeUnit(id="b", statement="B", sources=["Y"], status="active"),
                ],
            )
        ],
    )
    diff = diff_knowledge(old, new)
    assert len(diff.added) == 1
    assert diff.added[0].id == "b"
    assert len(diff.removed) == 0
    assert len(diff.updated) == 0


def test_diff_knowledge_detects_removals() -> None:
    old = KnowledgeModel(
        last_consolidated="2026-04-01",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[
                    KnowledgeUnit(id="a", statement="A", sources=["X"], status="active"),
                    KnowledgeUnit(id="b", statement="B", sources=["Y"], status="active"),
                ],
            )
        ],
    )
    new = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[KnowledgeUnit(id="a", statement="A", sources=["X"], status="active")],
            )
        ],
    )
    diff = diff_knowledge(old, new)
    assert len(diff.removed) == 1
    assert diff.removed[0].id == "b"


def test_diff_knowledge_detects_updates() -> None:
    old = KnowledgeModel(
        last_consolidated="2026-04-01",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[KnowledgeUnit(id="a", statement="Old text", sources=["X"], status="active")],
            )
        ],
    )
    new = KnowledgeModel(
        last_consolidated="2026-04-08",
        source_dir="test",
        themes=[
            KnowledgeTheme(
                key="t1",
                title="Theme",
                units=[KnowledgeUnit(id="a", statement="New text", sources=["X", "Y"], status="active")],
            )
        ],
    )
    diff = diff_knowledge(old, new)
    assert len(diff.updated) == 1
    old_unit, new_unit = diff.updated[0]
    assert old_unit.statement == "Old text"
    assert new_unit.statement == "New text"


def test_render_produces_valid_markdown(minimal_knowledge_path: Path) -> None:
    model = load_knowledge(minimal_knowledge_path)
    rendered = render_knowledge(model)

    # Basic structural checks
    assert rendered.startswith("# Project Knowledge Model")
    assert "Last consolidated: 2026-04-08" in rendered
    assert "## Data & Labels" in rendered
    assert "## Dead Ends" in rendered
    assert "3 knowledge units" in rendered
    assert "2 active" in rendered
    assert "1 dead ends" in rendered

    # Source references present
    assert "ST0.1" in rendered
    assert "TK01" in rendered

    # Cross-references rendered
    assert "label-policy-v1" in rendered

    # Context rendered
    assert "revisit with different integration" in rendered
