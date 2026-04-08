#!/usr/bin/env python3
"""Render knowledge.yml to KNOWLEDGE.md.

Usage:
    python -m lyzortx.orchestration.render_knowledge
    python -m lyzortx.orchestration.render_knowledge --knowledge-path path/to/knowledge.yml --output path/to/KNOWLEDGE.md
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from lyzortx.orchestration.knowledge_parser import (
    KnowledgeModel,
    KnowledgeTheme,
    KnowledgeUnit,
    load_knowledge,
    validate_knowledge,
)
from lyzortx.orchestration.render_utils import (
    MAX_PROSE_WIDTH,
    REPO_ROOT,
    write_rendered_markdown,
)

DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "lyzortx/orchestration/knowledge.yml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "lyzortx/KNOWLEDGE.md"


def _render_unit(unit: KnowledgeUnit) -> str:
    """Render a single knowledge unit as a markdown list item."""
    parts = [unit.statement]

    annotations: list[str] = []
    if unit.confidence:
        annotations.append(unit.confidence)
    annotations.append(f"source: {', '.join(unit.sources)}")
    if unit.relates_to:
        annotations.append(f"see also: {', '.join(unit.relates_to)}")

    parts.append(f"[{'; '.join(annotations)}]")

    text = " ".join(parts)
    if unit.status == "superseded":
        text = f"~~{text}~~"

    lines = textwrap.fill(
        text,
        width=MAX_PROSE_WIDTH,
        initial_indent="- ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )

    if unit.context:
        context_line = textwrap.fill(
            unit.context,
            width=MAX_PROSE_WIDTH,
            initial_indent="  - *",
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines += f"\n{context_line}*"

    return lines


def _render_theme(theme: KnowledgeTheme) -> str:
    """Render a theme section."""
    sections: list[str] = []
    sections.append(f"## {theme.title}")
    sections.append("")

    if theme.description:
        sections.append(
            textwrap.fill(
                theme.description,
                width=MAX_PROSE_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        sections.append("")

    active_units = [u for u in theme.units if u.status in ("active", "dead-end")]
    superseded_units = [u for u in theme.units if u.status == "superseded"]

    for unit in active_units:
        sections.append(_render_unit(unit))

    if superseded_units:
        sections.append("")
        sections.append("### Superseded")
        sections.append("")
        for unit in superseded_units:
            sections.append(_render_unit(unit))

    sections.append("")
    return "\n".join(sections)


def render_knowledge(model: KnowledgeModel) -> str:
    """Render a KnowledgeModel to markdown."""
    sections: list[str] = []

    sections.append("# Project Knowledge Model")
    sections.append("")
    sections.append(f"<!-- Last consolidated: {model.last_consolidated} -->")
    sections.append(f"<!-- Source: {model.source_dir} -->")
    sections.append("")

    # Summary counts
    all_units = model.all_units()
    active_count = len(model.units_by_status("active"))
    dead_end_count = len(model.units_by_status("dead-end"))
    superseded_count = len(model.units_by_status("superseded"))
    sections.append(
        f"**{len(all_units)} knowledge units** across {len(model.themes)} themes"
        f" ({active_count} active, {dead_end_count} dead ends, {superseded_count} superseded)"
    )
    sections.append("")

    for theme in model.themes:
        sections.append(_render_theme(theme))

    return "\n".join(sections)


def write_rendered_knowledge(output_path: Path, rendered: str) -> None:
    """Write rendered markdown, applying pymarkdown fixes."""
    write_rendered_markdown(output_path, rendered)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-path",
        type=Path,
        default=DEFAULT_KNOWLEDGE_PATH,
        help="Path to knowledge.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write rendered KNOWLEDGE.md (use --stdout to print instead)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = load_knowledge(args.knowledge_path)

    errors = validate_knowledge(model)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    rendered = render_knowledge(model)

    if args.stdout:
        sys.stdout.write(rendered)
    else:
        write_rendered_knowledge(args.output, rendered)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
