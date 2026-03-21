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
import tempfile
import textwrap
from typing import Any

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = REPO_ROOT / "lyzortx/orchestration/plan.yml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "lyzortx/research_notes/PLAN.md"
DEFAULT_PYMARKDOWN_CONFIG = REPO_ROOT / ".pymarkdown.yaml"
MAX_PROSE_WIDTH = 120

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
    title = task["title"]
    parts = [
        f"{title}." if task.get("status") == "done" and (task.get("implemented_in") or task.get("baseline")) else title
    ]
    impl = task.get("implemented_in")
    if impl and task.get("status") == "done":
        parts.append(f"Implemented in `{impl}`.")
    baseline = task.get("baseline")
    if baseline and task.get("status") == "done":
        parts.append(f"Regression baseline: `{baseline}`.")
    return textwrap.fill(
        " ".join(parts),
        width=MAX_PROSE_WIDTH,
        initial_indent=f"- [{check}] ",
        subsequent_indent="      ",
        break_long_words=False,
        break_on_hyphens=False,
    )


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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f"{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(rendered)

    try:
        api = PyMarkdownApi().configuration_file_path(str(DEFAULT_PYMARKDOWN_CONFIG))
        api.fix_path(str(temp_path))
        temp_path.replace(output_path)
    except PyMarkdownApiException as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"PyMarkdown fix failed for {output_path}: {exc}") from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


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
