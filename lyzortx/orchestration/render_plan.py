#!/usr/bin/env python3
"""Render plan.yml to PLAN.md.

Usage:
    python -m lyzortx.orchestration.render_plan
    python -m lyzortx.orchestration.render_plan --plan-path path/to/plan.yml --output path/to/PLAN.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
import textwrap
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from lyzortx.orchestration.render_utils import (
    MAX_PROSE_WIDTH,
    REPO_ROOT,
    write_rendered_markdown,
)

DEFAULT_PLAN_PATH = REPO_ROOT / "lyzortx/orchestration/plan.yml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "lyzortx/research_notes/PLAN.md"

# Map track keys to Mermaid node IDs (lowercase prefix "t" + lowercase track key).
TRACK_KEY_TO_MERMAID = {
    "ST": "tst",
    "A": "ta",
    "B": "tb",
    "C": "tc",
    "D": "td",
    "E": "te",
    "F": "tf",
    "G": "tg",
    "H": "th",
    "I": "ti",
    "J": "tj",
    "K": "tk",
}


def load_plan_yaml(plan_path: Path) -> dict[str, Any]:
    text = plan_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    raise ImportError("PyYAML is required: pip install pyyaml")


def _render_mermaid(tracks: dict[str, Any]) -> str:
    stages: dict[int, list[str]] = defaultdict(list)
    for key, track in tracks.items():
        stage = track.get("stage", 0)
        stages[stage].append(key)

    stage_labels = {
        0: "Stage 0 (Serial Foundation)",
        1: "Stage 1 (Parallelizable Build-Out)",
        2: "Stage 2 (Parallelizable Integration)",
        3: "Stage 3 (Release and Audit)",
    }

    lines = ["```mermaid", "graph LR"]

    for stage_num in sorted(stages):
        label = stage_labels.get(stage_num, f"Stage {stage_num}")
        lines.append(f'  subgraph s{stage_num}["{label}"]')
        for key in stages[stage_num]:
            node_id = TRACK_KEY_TO_MERMAID.get(key, f"t{key.lower()}")
            name = tracks[key]["name"]
            lines.append(f'    {node_id}["Track {key}: {name}"]')
        lines.append("  end")
        lines.append("")

    for key, track in tracks.items():
        target = TRACK_KEY_TO_MERMAID.get(key, f"t{key.lower()}")
        for dep in track.get("depends_on", []):
            source = TRACK_KEY_TO_MERMAID.get(dep, f"t{dep.lower()}")
            lines.append(f"  {source} --> {target}")

    lines.append("```")
    return "\n".join(lines)


def _render_task_line(task: dict[str, Any]) -> str:
    check = "x" if task.get("status") == "done" else " "
    task_id = task.get("id", "")
    title = task["title"]
    prefix = f"**{task_id}** " if task_id else ""
    has_metadata = (
        task.get("implemented_in")
        or task.get("baseline")
        or task.get("model")
        or task.get("depends_on_tasks")
        or task.get("ci_image_profile")
    )
    parts = [f"{prefix}{title}." if has_metadata else f"{prefix}{title}"]
    impl = task.get("implemented_in")
    if impl and task.get("status") == "done":
        parts.append(f"Implemented in `{impl}`.")
    baseline = task.get("baseline")
    if baseline and task.get("status") == "done":
        parts.append(f"Regression baseline: `{baseline}`.")
    model = task.get("model")
    if model:
        parts.append(f"Model: `{model}`.")
    ci_image_profile = task.get("ci_image_profile")
    if ci_image_profile:
        parts.append(f"CI image profile: `{ci_image_profile}`.")
    task_dependencies = task.get("depends_on_tasks")
    if task_dependencies:
        rendered_dependencies = ", ".join(f"`{dependency_id}`" for dependency_id in task_dependencies)
        parts.append(f"Depends on tasks: {rendered_dependencies}.")

    lines = [
        textwrap.fill(
            " ".join(parts),
            width=MAX_PROSE_WIDTH,
            initial_indent=f"- [{check}] ",
            subsequent_indent="      ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]

    criteria = task.get("acceptance_criteria")
    if criteria:
        for criterion in criteria:
            lines.append(
                textwrap.fill(
                    criterion,
                    width=MAX_PROSE_WIDTH,
                    initial_indent="  - ",
                    subsequent_indent="    ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    return "\n".join(lines)


def render_plan(plan: dict[str, Any]) -> str:
    tracks = plan["tracks"]
    sections: list[str] = []

    sections.append("# Lyzor Tx In-Silico Pipeline Plan")
    sections.append("")
    sections.append("## Parallel Execution View")
    sections.append("")
    sections.append(
        "- Tracks in the same stage box can run in parallel unless blocked by their own incoming dependencies."
    )
    sections.append("")
    sections.append(_render_mermaid(tracks))
    sections.append("")

    for key, track in tracks.items():
        name = track["name"]
        sections.append(f"## Track {key}: {name}")
        sections.append("")

        description = track.get("description")
        if description:
            sections.append(
                textwrap.fill(
                    f"**Guiding Principle:** {description}",
                    width=MAX_PROSE_WIDTH,
                    initial_indent="- ",
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

        for task in track.get("tasks", []):
            sections.append(_render_task_line(task))
        sections.append("")

    return "\n".join(sections)


def write_rendered_plan(output_path: Path, rendered: str) -> None:
    write_rendered_markdown(output_path, rendered)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Path to plan.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write rendered PLAN.md (use --stdout to print instead)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = load_plan_yaml(args.plan_path)
    rendered = render_plan(plan)

    if args.stdout:
        sys.stdout.write(rendered)
    else:
        write_rendered_plan(args.output, rendered)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
