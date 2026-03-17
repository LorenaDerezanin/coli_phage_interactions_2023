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
from lyzortx.orchestration.render_plan import load_plan_yaml, render_plan, write_rendered_plan

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
          - id: TB02
            title: Also depends on A
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
    assert not is_track_complete(graph, "A")


def test_task_readiness_within_track(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    ta02 = next(t for t in graph.tasks if t.task_id == "TA02")
    assert is_task_ready(ta02, graph)


def test_task_blocked_by_cross_track_dep(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    tb01 = next(t for t in graph.tasks if t.task_id == "TB01")
    assert not is_task_ready(tb01, graph)


def test_within_track_sequencing(tmp_path: Path) -> None:
    graph = load_plan(_write_plan(tmp_path))
    tb02 = next(t for t in graph.tasks if t.task_id == "TB02")
    # TB01 is pending, so TB02 is blocked even if cross-track deps were met
    assert not is_task_ready(tb02, graph)


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


def test_render_plan_wraps_long_task_metadata(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        tracks:
          A:
            name: Foundation
            stage: 0
            depends_on: []
            tasks:
              - id: TA01
                title: Define a task title that is long enough to force wrapping once implementation metadata is appended
                status: done
                implemented_in: lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py
                baseline: lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json
    """)
    plan = load_plan_yaml(_write_plan(tmp_path, content))
    md = render_plan(plan)
    assert (
        "- [x] Define a task title that is long enough to force wrapping once implementation metadata is appended."
    ) in md
    assert "      `lyzortx/pipeline/steel_thread_v0/steps/st01_label_policy.py`." in md
    assert "      `lyzortx/pipeline/steel_thread_v0/baselines/st01_expected_metrics.json`." in md
    assert all(len(line) <= 120 for line in md.splitlines())


def test_render_plan_wraps_guiding_principle(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        tracks:
          A:
            name: Foundation
            stage: 0
            depends_on: []
            description: A guiding principle description that is intentionally long enough to exceed the markdown prose width limit and must wrap cleanly onto a continuation line.
            tasks:
              - id: TA01
                title: First task
                status: pending
    """)
    plan = load_plan_yaml(_write_plan(tmp_path, content))
    md = render_plan(plan)
    assert "- **Guiding Principle:** A guiding principle description that is intentionally long enough" in md
    assert "  width limit and must wrap cleanly onto a continuation line." in md
    assert all(len(line) <= 120 for line in md.splitlines())


def test_write_rendered_plan_uses_pymarkdown_fix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_path = tmp_path / "PLAN.md"
    captured: dict[str, str] = {}

    class _FakeApi:
        def configuration_file_path(self, config_path: str) -> "_FakeApi":
            captured["config_path"] = config_path
            return self

        def fix_path(self, file_path: str) -> None:
            captured["fixed_path"] = file_path
            path = Path(file_path)
            path.write_text(path.read_text(encoding="utf-8") + "\n<!-- fixed -->\n", encoding="utf-8")

    monkeypatch.setattr("lyzortx.orchestration.render_plan.PyMarkdownApi", lambda: _FakeApi())
    write_rendered_plan(output_path, "# Title\n")

    assert captured["config_path"].endswith(".pymarkdown.yaml")
    assert output_path.read_text(encoding="utf-8") == "# Title\n\n<!-- fixed -->\n"
