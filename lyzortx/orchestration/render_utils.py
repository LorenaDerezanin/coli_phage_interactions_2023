"""Shared utilities for YAML-to-markdown renderers (plan, knowledge, etc.)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYMARKDOWN_CONFIG = REPO_ROOT / ".pymarkdown.yaml"
MAX_PROSE_WIDTH = 120


def write_rendered_markdown(output_path: Path, rendered: str) -> None:
    """Write rendered markdown, applying pymarkdown fixes via atomic temp-file swap."""
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
