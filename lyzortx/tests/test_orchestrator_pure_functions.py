"""Unit tests for pure helper logic in the orchestrator module."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyzortx.orchestration.ci_image_profiles import ci_image_for_profile
from lyzortx.orchestration.ci_image_profiles import ci_image_profile_from_labels
from lyzortx.orchestration.ci_image_profiles import ci_image_profile_label
from lyzortx.orchestration.orchestrator import IssueRef
from lyzortx.orchestration.orchestrator import Task
from lyzortx.orchestration.orchestrator import choose_preferred_issue
from lyzortx.orchestration.orchestrator import extract_task_id_from_issue
from lyzortx.orchestration.orchestrator import sync_status_from_issues
from lyzortx.orchestration.find_pr_for_issue import pr_closes_issue
from lyzortx.orchestration.parse_model_directive import extract_model


def test_extract_task_id_from_body_marker() -> None:
    issue = {
        "body": "<!-- ORCH_TASK_ID: IM07_SOURCE_REGISTRY -->\nTask details",
        "title": "[ORCH][SHOULD_NOT_BE_USED] placeholder",
    }

    assert extract_task_id_from_issue(issue) == "IM07_SOURCE_REGISTRY"


def test_extract_task_id_from_title_when_body_missing_marker() -> None:
    issue = {
        "body": "No orchestrator marker in this body",
        "title": "[ORCH][IM02_ST03B_SPLIT_SUITE] Implement split suite",
    }

    assert extract_task_id_from_issue(issue) == "IM02_ST03B_SPLIT_SUITE"


def test_extract_task_id_returns_none_when_no_marker_or_title_prefix() -> None:
    issue = {
        "body": "No marker here",
        "title": "Regular issue title",
    }

    assert extract_task_id_from_issue(issue) is None


def test_choose_preferred_issue_prefers_open_over_closed() -> None:
    open_issue = IssueRef(
        task_id="IM03_ST04B_ABLATIONS",
        number=11,
        state="open",
        state_reason="",
        title="open issue",
        html_url="https://example.com/open",
        updated_at="2026-03-08T22:00:00Z",
    )
    closed_issue = IssueRef(
        task_id="IM03_ST04B_ABLATIONS",
        number=10,
        state="closed",
        state_reason="completed",
        title="closed issue",
        html_url="https://example.com/closed",
        updated_at="2026-03-08T22:30:00Z",
    )

    assert choose_preferred_issue(closed_issue, open_issue) == open_issue


def test_choose_preferred_issue_prefers_more_recent_when_states_match() -> None:
    older_issue = IssueRef(
        task_id="IM04_ST05B_ST06C_DUAL_SLICE_BOOTSTRAP",
        number=20,
        state="closed",
        state_reason="completed",
        title="older",
        html_url="https://example.com/older",
        updated_at="2026-03-08T21:00:00Z",
    )
    newer_issue = IssueRef(
        task_id="IM04_ST05B_ST06C_DUAL_SLICE_BOOTSTRAP",
        number=21,
        state="closed",
        state_reason="completed",
        title="newer",
        html_url="https://example.com/newer",
        updated_at="2026-03-08T22:00:00Z",
    )

    assert choose_preferred_issue(newer_issue, older_issue) == newer_issue


def _make_task(track: str, task_id: str = "TX01") -> Task:
    return Task(
        task_id=task_id,
        title="Test task",
        description="desc",
        dependencies=[],
        executor="agent",
        command=None,
        expected_paths=[],
        acceptance_criteria=[],
        plan_checkbox_text=None,
        track=track,
        model="gpt-5.4-mini",
        ci_image_profile="host-typing",
    )


@pytest.mark.parametrize(
    "track,expected_path",
    [
        ("B", "lyzortx/research_notes/lab_notebooks/track_B.md"),
        ("ST", "lyzortx/research_notes/lab_notebooks/track_ST.md"),
        ("I", "lyzortx/research_notes/lab_notebooks/track_I.md"),
        ("A", "lyzortx/research_notes/lab_notebooks/track_A.md"),
    ],
)
def test_agent_instruction_uses_track_for_notebook_path(track: str, expected_path: str) -> None:
    task = _make_task(track=track)
    # Reproduce the f-string from create_agent_task_issue (orchestrator.py line 311)
    instruction = (
        f"2. Write findings and interpretation to `lyzortx/research_notes/lab_notebooks/track_{task.track}.md`"
        " following the existing entry format (ordered by task code, earliest first)."
    )
    assert f"`{expected_path}`" in instruction


def test_task_requires_track_and_model() -> None:
    """Task requires track and model — omitting either raises TypeError."""
    with pytest.raises(TypeError):
        Task(
            task_id="X01",
            title="t",
            description="d",
            dependencies=[],
            executor="agent",
            command=None,
            expected_paths=[],
            acceptance_criteria=[],
            plan_checkbox_text=None,
        )


def test_extract_model_from_issue_body() -> None:
    body = "<!-- ORCH_TASK_ID: TG04 -->\n<!-- model: gpt-5.4-mini -->\n## Task"
    assert extract_model(body) == "gpt-5.4-mini"


def test_extract_model_full_model() -> None:
    body = "<!-- model: gpt-5.4 -->\nSome content"
    assert extract_model(body) == "gpt-5.4"


def test_extract_model_missing() -> None:
    body = "<!-- ORCH_TASK_ID: TG04 -->\n## Task with no model"
    assert extract_model(body) is None


def test_extract_model_whitespace_tolerance() -> None:
    body = "<!--  model:  gpt-5.4-mini  -->"
    assert extract_model(body) == "gpt-5.4-mini"


def test_ci_image_profile_from_labels_prefers_prefixed_label() -> None:
    labels = ["orchestrator-task", "ci-image:host-typing", "model-gpt-5.4"]
    assert ci_image_profile_from_labels(labels) == "host-typing"


def test_ci_image_profile_from_labels_requires_explicit_label() -> None:
    with pytest.raises(ValueError, match="Missing required ci-image:\\* label"):
        ci_image_profile_from_labels(["orchestrator-task"])


def test_ci_image_profile_label_and_image_ref() -> None:
    assert ci_image_profile_label("full-bio") == "ci-image:full-bio"
    assert ci_image_for_profile("host-typing") == (
        "ghcr.io/lyzortx/coli-phage-interactions-2023-codex-ci:host-typing-main"
    )


def _write_pending_task_plan(tmp_path: Path, **task_fields: str | list[str]) -> Path:
    """Write a minimal plan.yml with one pending task. Override fields via kwargs."""
    import yaml

    task = {"id": "TA01", "title": "Test task", "status": "pending"}
    task.update(task_fields)
    content = yaml.dump(
        {"tracks": {"A": {"name": "Foundation", "stage": 0, "depends_on": [], "tasks": [task]}}},
        default_flow_style=False,
    )
    plan_path = tmp_path / "plan.yml"
    plan_path.write_text(content, encoding="utf-8")
    return plan_path


def test_load_pending_tasks_raises_on_missing_model(tmp_path: Path) -> None:
    """Pending tasks without a model field must cause a ValueError."""
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = _write_pending_task_plan(
        tmp_path,
        acceptance_criteria=["Some criterion"],
        ci_image_profile="base",
    )
    with pytest.raises(ValueError, match="missing required 'model' field"):
        load_pending_tasks(plan_path)


def test_load_pending_tasks_raises_on_missing_acceptance_criteria(tmp_path: Path) -> None:
    """Pending tasks without acceptance_criteria must cause a ValueError."""
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = _write_pending_task_plan(tmp_path, model="gpt-5.4-mini", ci_image_profile="base")
    with pytest.raises(ValueError, match="missing required 'acceptance_criteria'"):
        load_pending_tasks(plan_path)


def test_load_pending_tasks_accepts_complete_task(tmp_path: Path) -> None:
    """Pending tasks with model and acceptance_criteria should load without error."""
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = _write_pending_task_plan(
        tmp_path,
        model="gpt-5.4-mini",
        acceptance_criteria=["Output exists"],
        ci_image_profile="base",
    )
    tasks = load_pending_tasks(plan_path)
    assert len(tasks) == 1
    assert tasks[0].model == "gpt-5.4-mini"
    assert tasks[0].acceptance_criteria == ["Output exists"]
    assert tasks[0].ci_image_profile == "base"


def test_load_pending_tasks_raises_on_missing_ci_image_profile(tmp_path: Path) -> None:
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = _write_pending_task_plan(tmp_path, model="gpt-5.4-mini", acceptance_criteria=["Output exists"])
    with pytest.raises(ValueError, match="missing required 'ci_image_profile' field"):
        load_pending_tasks(plan_path)


def test_load_pending_tasks_parses_ci_image_profile(tmp_path: Path) -> None:
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = _write_pending_task_plan(
        tmp_path,
        model="gpt-5.4-mini",
        acceptance_criteria=["Output exists"],
        ci_image_profile="host-typing",
    )
    tasks = load_pending_tasks(plan_path)
    assert tasks[0].ci_image_profile == "host-typing"


def test_load_pending_tasks_works_on_real_plan() -> None:
    """Smoke test: load_pending_tasks succeeds on the real plan.yml."""
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    tasks = load_pending_tasks(Path("lyzortx/orchestration/plan.yml"))
    for t in tasks:
        assert t.model, f"Task {t.task_id} missing model"
        assert t.acceptance_criteria, f"Task {t.task_id} missing acceptance_criteria"


def test_load_pending_tasks_preserves_explicit_task_dependencies(tmp_path: Path) -> None:
    from lyzortx.orchestration.orchestrator import load_pending_tasks

    plan_path = tmp_path / "plan.yml"
    plan_path.write_text(
        """
tracks:
  L:
    name: Mechanistic Features and Generalized Inference
    stage: 1
    depends_on: []
    tasks:
      - id: TL15
        title: Build raw-host surface projector
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Surface projector exists
        depends_on_tasks: []
      - id: TL16
        title: Build host typing projector
        status: pending
        model: gpt-5.4
        ci_image_profile: host-typing
        acceptance_criteria:
          - Host typing projector exists
        depends_on_tasks: []
      - id: TL17
        title: Build phage compatibility projector
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Phage compatibility projector exists
        depends_on_tasks: []
      - id: TL18
        title: Rebuild deployable bundle
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Bundle rebuild completes
        depends_on_tasks: [TL15, TL16, TL17]
""".strip(),
        encoding="utf-8",
    )

    tasks = {task.task_id: task for task in load_pending_tasks(plan_path)}
    assert tasks["TL15"].dependencies == []
    assert tasks["TL16"].dependencies == []
    assert tasks["TL17"].dependencies == []
    assert tasks["TL18"].dependencies == ["TL15", "TL16", "TL17"]


def test_run_once_dispatches_tl15_tl17_in_parallel_before_tl18(tmp_path: Path) -> None:
    from lyzortx.orchestration.orchestrator import initialize_state
    from lyzortx.orchestration.orchestrator import load_pending_tasks
    from lyzortx.orchestration.orchestrator import run_once

    plan_path = tmp_path / "plan.yml"
    plan_path.write_text(
        """
tracks:
  L:
    name: Mechanistic Features and Generalized Inference
    stage: 1
    depends_on: []
    tasks:
      - id: TL15
        title: Build raw-host surface projector
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Surface projector exists
        depends_on_tasks: []
      - id: TL16
        title: Build host typing projector
        status: pending
        model: gpt-5.4
        ci_image_profile: host-typing
        acceptance_criteria:
          - Host typing projector exists
        depends_on_tasks: []
      - id: TL17
        title: Build phage compatibility projector
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Phage compatibility projector exists
        depends_on_tasks: []
      - id: TL18
        title: Rebuild deployable bundle
        status: pending
        model: gpt-5.4
        ci_image_profile: full-bio
        acceptance_criteria:
          - Bundle rebuild completes
        depends_on_tasks: [TL15, TL16, TL17]
""".strip(),
        encoding="utf-8",
    )

    tasks = load_pending_tasks(plan_path)
    state = initialize_state(tasks, tmp_path / "runtime_state.json")

    result = run_once(
        tasks=tasks,
        state=state,
        issues_by_task={},
        github_client=None,
        max_active_tasks=4,
        plan_path=plan_path,
    )

    assert result["action"] == "tasks_waiting_for_agent"
    assert result["task_ids"] == ["TL15", "TL16", "TL17"]
    assert state["task_status"]["TL15"] == "in_progress"
    assert state["task_status"]["TL16"] == "in_progress"
    assert state["task_status"]["TL17"] == "in_progress"
    assert state["task_status"]["TL18"] == "pending"


# --- sync_status_from_issues: state_reason handling ---


def _make_issue_ref(
    task_id: str = "TX01",
    state: str = "open",
    state_reason: str = "",
) -> IssueRef:
    return IssueRef(
        task_id=task_id,
        number=1,
        state=state,
        state_reason=state_reason,
        title="test",
        html_url="https://example.com",
        updated_at="2026-03-22T00:00:00Z",
    )


def test_sync_marks_completed_when_closed_as_completed() -> None:
    """Issue closed with state_reason='completed' (PR merged) marks task done."""
    task = _make_task(track="G", task_id="TG04")
    task_status: dict[str, str] = {"TG04": "in_progress"}
    issues = {"TG04": _make_issue_ref("TG04", state="closed", state_reason="completed")}
    sync_status_from_issues([task], task_status, issues)
    assert task_status["TG04"] == "completed"


def test_sync_does_not_complete_when_closed_as_not_planned() -> None:
    """Issue closed with state_reason='not_planned' (manual close) must NOT mark task done."""
    task = _make_task(track="G", task_id="TG04")
    task_status: dict[str, str] = {"TG04": "in_progress"}
    issues = {"TG04": _make_issue_ref("TG04", state="closed", state_reason="not_planned")}
    sync_status_from_issues([task], task_status, issues)
    assert task_status["TG04"] == "pending"


def test_sync_preserves_blocked_when_closed_as_not_planned() -> None:
    """A blocked task closed as not_planned should stay blocked."""
    task = _make_task(track="G", task_id="TG04")
    task_status: dict[str, str] = {"TG04": "blocked"}
    issues = {"TG04": _make_issue_ref("TG04", state="closed", state_reason="not_planned")}
    sync_status_from_issues([task], task_status, issues)
    assert task_status["TG04"] == "blocked"


def test_sync_marks_in_progress_when_open() -> None:
    """Open issue means task is in progress."""
    task = _make_task(track="G", task_id="TG04")
    task_status: dict[str, str] = {"TG04": "pending"}
    issues = {"TG04": _make_issue_ref("TG04", state="open")}
    sync_status_from_issues([task], task_status, issues)
    assert task_status["TG04"] == "in_progress"


def test_sync_pending_when_no_issue() -> None:
    """Task with no matching issue stays pending."""
    task = _make_task(track="G", task_id="TG04")
    task_status: dict[str, str] = {"TG04": "pending"}
    sync_status_from_issues([task], task_status, {})
    assert task_status["TG04"] == "pending"


# --- pr_closes_issue ---


def test_pr_closes_issue_exact_match() -> None:
    assert pr_closes_issue("Closes #120", 120)


def test_pr_closes_issue_lowercase() -> None:
    assert pr_closes_issue("closes #120", 120)


def test_pr_closes_issue_no_prefix_collision() -> None:
    """#12 must not match #120."""
    assert not pr_closes_issue("Closes #120", 12)


def test_pr_closes_issue_no_suffix_collision() -> None:
    """#120 must not match #12."""
    assert not pr_closes_issue("Closes #12", 120)


def test_pr_closes_issue_in_longer_body() -> None:
    body = "## Summary\nFixes the thing.\n\nCloses #42\n"
    assert pr_closes_issue(body, 42)


def test_pr_closes_issue_missing() -> None:
    assert not pr_closes_issue("No closure reference here", 42)


def test_pr_closes_issue_multiple_references() -> None:
    body = "Closes #10\nCloses #20\n"
    assert pr_closes_issue(body, 20)
    assert not pr_closes_issue(body, 200)
