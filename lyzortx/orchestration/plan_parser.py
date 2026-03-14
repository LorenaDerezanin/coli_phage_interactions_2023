#!/usr/bin/env python3
"""Parse plan.yml and provide task readiness logic for the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PlanTask:
    task_id: str
    track: str
    title: str
    status: str  # "done" | "pending"
    ordinal: int  # 1-based position within track
    implemented_in: str | None = None
    baseline: str | None = None


@dataclass(frozen=True)
class PlanGraph:
    tasks: list[PlanTask]
    track_deps: dict[str, list[str]]  # track key -> list of prerequisite track keys

    def tasks_for_track(self, track: str) -> list[PlanTask]:
        return [t for t in self.tasks if t.track == track]

    def track_keys(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.tasks:
            seen.setdefault(t.track, None)
        return list(seen)


def load_plan(plan_path: Path) -> PlanGraph:
    text = plan_path.read_text(encoding="utf-8")
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    data = yaml.safe_load(text)
    tracks_raw = data.get("tracks", {})

    all_tasks: list[PlanTask] = []
    track_deps: dict[str, list[str]] = {}

    for key, track in tracks_raw.items():
        track_deps[key] = list(track.get("depends_on", []))
        for i, raw in enumerate(track.get("tasks", [])):
            all_tasks.append(
                PlanTask(
                    task_id=raw["id"],
                    track=key,
                    title=raw["title"],
                    status=raw.get("status", "pending"),
                    ordinal=i + 1,
                    implemented_in=raw.get("implemented_in"),
                    baseline=raw.get("baseline"),
                )
            )

    return PlanGraph(tasks=all_tasks, track_deps=track_deps)


def is_track_complete(graph: PlanGraph, track: str) -> bool:
    return all(t.status == "done" for t in graph.tasks_for_track(track))


def is_task_ready(task: PlanTask, graph: PlanGraph) -> bool:
    if task.status == "done":
        return False

    # Within-track: all prior items must be done.
    for t in graph.tasks_for_track(task.track):
        if t.ordinal < task.ordinal and t.status != "done":
            return False

    # Cross-track: all items in all prerequisite tracks must be done.
    for dep_track in graph.track_deps.get(task.track, []):
        if not is_track_complete(graph, dep_track):
            return False

    return True


def select_ready_tasks(graph: PlanGraph) -> list[PlanTask]:
    return [t for t in graph.tasks if is_task_ready(t, graph)]


def mark_task_done(plan_path: Path, task_id: str) -> None:
    """Set a task's status to 'done' in plan.yml."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    text = plan_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    for track in data.get("tracks", {}).values():
        for task in track.get("tasks", []):
            if task["id"] == task_id:
                task["status"] = "done"
                _write_plan_yaml(plan_path, data)
                return

    raise ValueError(f"Task {task_id!r} not found in {plan_path}")


def _write_plan_yaml(plan_path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    plan_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
