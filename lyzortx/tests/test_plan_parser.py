"""Tests for plan_parser and render_plan."""

from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

from lyzortx.orchestration.plan_parser import (
    PlanGraph,
    PlanTask,
    is_task_ready,
    is_track_complete,
    load_plan,
    mark_task_done,
    select_ready_tasks,
)
from lyzortx.orchestration.render_plan import load_plan_yaml, render_plan

MINIMAL_PLAN = textwrap.dedent("""\
    tracks:
      A:
        name: Foundation
        stage: 0
        depends_on: []
        tasks:
          - id: TA01
            title: First task
            status: done
          - id: TA02
            title: Second task
            status: pending
      B:
        name: Build-Out
        stage: 1
        depends_on: [A]
        tasks:
          - id: TB01
            title: Depends on A
            status: pending
        gates:
          - id: GB01
            title: Gate for B
            status: pending
""")


def _write_plan(tmp_path: Path, content: str = MINIMAL_PLAN) -> Path:
    p = tmp_path / "plan.yml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_plan_counts(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    assert len(graph.tasks) == 4
    assert graph.track_deps == {"A": [], "B": ["A"]}


def test_track_complete(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    # A has one done + one pending
    assert not is_track_complete(graph, "A")


def test_task_readiness_within_track(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    ta02 = next(t for t in graph.tasks if t.task_id == "TA02")
    # TA01 is done, so TA02 is ready
    assert is_task_ready(ta02, graph)


def test_task_blocked_by_cross_track_dep(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    tb01 = next(t for t in graph.tasks if t.task_id == "TB01")
    # Track A is not complete (TA02 pending), so TB01 is blocked
    assert not is_task_ready(tb01, graph)


def test_gate_blocked_by_prior_task(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    gb01 = next(t for t in graph.tasks if t.task_id == "GB01")
    # TB01 is pending, so gate GB01 is blocked
    assert not is_task_ready(gb01, graph)


def test_select_ready_tasks(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    ready = select_ready_tasks(graph)
    assert [t.task_id for t in ready] == ["TA02"]


def test_mark_task_done(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    mark_task_done(plan_path, "TA02")
    graph = load_plan(plan_path)
    ta02 = next(t for t in graph.tasks if t.task_id == "TA02")
    assert ta02.status == "done"
    # Now TB01 should be ready (Track A is complete)
    tb01 = next(t for t in graph.tasks if t.task_id == "TB01")
    assert is_task_ready(tb01, graph)


def test_render_plan_contains_tracks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_plan_yaml(plan_path)
    md = render_plan(plan)
    assert "## Track A: Foundation" in md
    assert "## Track B: Build-Out" in md
    assert "- [x] First task" in md
    assert "- [ ] Second task" in md
    assert "```mermaid" in md
    assert "ta --> tb" in md


def test_render_plan_includes_gates(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_plan_yaml(plan_path)
    md = render_plan(plan)
    assert "### Go / No-Go Gates" in md
    assert "- [ ] Gate for B" in md
