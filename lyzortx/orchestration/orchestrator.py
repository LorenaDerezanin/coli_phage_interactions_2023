#!/usr/bin/env python3
"""Issue-driven orchestrator for PLAN task execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = REPO_ROOT / "lyzortx/orchestration/plan.yml"
DEFAULT_PLAN_MD_PATH = REPO_ROOT / "lyzortx/research_notes/PLAN.md"
DEFAULT_STATE_PATH = REPO_ROOT / "lyzortx/generated_outputs/orchestration/runtime_state.json"
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}
READ_ONLY_COMMANDS = {"status"}
ISSUE_LABEL = "orchestrator-task"
TASK_ID_MARKER = "ORCH_TASK_ID"
ISSUE_TITLE_PATTERN = re.compile(r"^\[ORCH\]\[([A-Za-z0-9_.-]+)\]")
TASK_ID_BODY_PATTERN = re.compile(r"ORCH_TASK_ID:\s*([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class Task:
    """Structured task metadata loaded from the static task registry."""

    task_id: str
    title: str
    description: str
    dependencies: list[str]
    executor: str
    command: list[str] | None
    expected_paths: list[str]
    acceptance_criteria: list[str]
    plan_checkbox_text: str | None
    track: str = ""
    model: str = ""


@dataclass(frozen=True)
class IssueRef:
    """Normalized issue metadata used for task-state reconciliation."""

    task_id: str
    number: int
    state: str
    title: str
    html_url: str
    updated_at: str


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object at {path}, got {type(payload).__name__}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_acceptance_criteria(plan_path: Path, task_id: str) -> list[str]:
    """Load acceptance criteria for a task from plan.yml."""
    import yaml

    data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    for track in data.get("tracks", {}).values():
        for task_list in [track.get("tasks", []), track.get("gates", [])]:
            for task in task_list:
                if task["id"] == task_id:
                    return task.get("acceptance_criteria", [])
    return []


def load_pending_tasks(plan_path: Path) -> list[Task]:
    """Load all pending tasks from plan.yml as Task objects."""
    from lyzortx.orchestration.plan_parser import load_plan

    graph = load_plan(plan_path)
    tasks = [
        Task(
            task_id=pt.task_id,
            title=pt.title,
            description=f"Track {pt.track} task. {pt.title}",
            dependencies=[],
            executor="agent",
            command=None,
            expected_paths=[],
            acceptance_criteria=_load_acceptance_criteria(plan_path, pt.task_id),
            plan_checkbox_text=pt.title,
            track=pt.track,
            model=pt.model or "",
        )
        for pt in graph.tasks
        if pt.status != "done"
    ]
    missing_model = [t.task_id for t in tasks if not t.model]
    if missing_model:
        raise ValueError(f"Pending tasks missing required 'model' field in plan.yml: {missing_model}")
    return tasks


def mark_task_done_in_plan(plan_path: Path, plan_md_path: Path, task_id: str) -> None:
    """Mark a task done in plan.yml and regenerate PLAN.md."""
    from lyzortx.orchestration.plan_parser import mark_task_done
    from lyzortx.orchestration.render_plan import load_plan_yaml, render_plan, write_rendered_plan

    mark_task_done(plan_path, task_id)
    plan = load_plan_yaml(plan_path)
    write_rendered_plan(plan_md_path, render_plan(plan))


def initialize_state(tasks: list[Task], state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        state = load_json(state_path)
    else:
        state = {
            "schema_version": 1,
            "paused": False,
            "task_status": {},
            "history": [],
            "issue_index": {},
            "updated_at": utc_now_iso(),
        }

    task_status = state.setdefault("task_status", {})
    if not isinstance(task_status, dict):
        raise ValueError("State field 'task_status' must be an object")

    for task in tasks:
        if task_status.get(task.task_id) not in VALID_STATUSES:
            task_status[task.task_id] = "pending"

    state.setdefault("paused", False)
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("State field 'history' must be an array")
    issue_index = state.setdefault("issue_index", {})
    if not isinstance(issue_index, dict):
        raise ValueError("State field 'issue_index' must be an object")

    return state


def append_history(state: dict[str, Any], event_type: str, **fields: Any) -> None:
    entry = {
        "timestamp": utc_now_iso(),
        "event_type": event_type,
    }
    entry.update(fields)
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("State field 'history' must be an array")
    history.append(entry)


def extract_task_id_from_issue(issue: dict[str, Any]) -> str | None:
    body = str(issue.get("body", ""))
    body_match = TASK_ID_BODY_PATTERN.search(body)
    if body_match:
        return body_match.group(1)

    title = str(issue.get("title", ""))
    title_match = ISSUE_TITLE_PATTERN.match(title)
    if title_match:
        return title_match.group(1)

    return None


def choose_preferred_issue(candidate: IssueRef, current: IssueRef) -> IssueRef:
    if candidate.state == "open" and current.state != "open":
        return candidate
    if current.state == "open" and candidate.state != "open":
        return current
    return candidate if candidate.updated_at > current.updated_at else current


class GitHubClient:
    """Thin GitHub REST client for issue-backed orchestration."""

    def __init__(self, token: str, repo: str, api_base: str = "https://api.github.com") -> None:
        self.token = token
        self.repo = repo
        self.api_base = api_base.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        allow_not_found: bool = False,
    ) -> Any:
        url = f"{self.api_base}{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed ({exc.code}): {details}") from exc

    def ensure_task_label_exists(self) -> None:
        encoded_name = urllib.parse.quote(ISSUE_LABEL, safe="")
        existing = self._request(
            "GET",
            f"/repos/{self.repo}/labels/{encoded_name}",
            allow_not_found=True,
        )
        if existing is not None:
            return

        self._request(
            "POST",
            f"/repos/{self.repo}/labels",
            {
                "name": ISSUE_LABEL,
                "color": "1D76DB",
                "description": "Orchestrator-managed task issue",
            },
        )

    def list_task_issues(self) -> dict[str, IssueRef]:
        issues_by_task: dict[str, IssueRef] = {}
        page = 1
        encoded_label = urllib.parse.quote(ISSUE_LABEL)

        while True:
            path = f"/repos/{self.repo}/issues?state=all&labels={encoded_label}&per_page=100&page={page}"
            page_items = self._request("GET", path)
            if not isinstance(page_items, list):
                raise RuntimeError("Unexpected GitHub API payload while listing issues")
            if not page_items:
                break

            for item in page_items:
                if not isinstance(item, dict):
                    continue
                if "pull_request" in item:
                    continue
                task_id = extract_task_id_from_issue(item)
                if not task_id:
                    continue
                issue_ref = IssueRef(
                    task_id=task_id,
                    number=int(item.get("number", 0)),
                    state=str(item.get("state", "")),
                    title=str(item.get("title", "")),
                    html_url=str(item.get("html_url", "")),
                    updated_at=str(item.get("updated_at", "")),
                )
                existing = issues_by_task.get(task_id)
                if existing is None:
                    issues_by_task[task_id] = issue_ref
                else:
                    issues_by_task[task_id] = choose_preferred_issue(issue_ref, existing)
            page += 1

        return issues_by_task

    def create_agent_task_issue(self, task: Task) -> IssueRef:
        title = f"[ORCH][{task.task_id}] {task.title}"

        acceptance = "\n".join(f"- {c}" for c in task.acceptance_criteria) or "- Define in implementation PR"
        body_parts = [
            f"<!-- {TASK_ID_MARKER}: {task.task_id} -->",
            f"<!-- model: {task.model} -->" if task.model else "",
            "## Orchestrator Task",
            f"- Task ID: `{task.task_id}`",
            f"- Executor: `{task.executor}`",
            "- Plan Source: `lyzortx/orchestration/plan.yml`",
            "",
            "## Description",
            task.description or "No description provided.",
            "",
            "## Acceptance Criteria",
            acceptance,
            "",
            "## Agent Instructions",
            "",
            "1. Implement the task described above, following `AGENTS.md` policies.",
            f"2. Write findings and interpretation to `lyzortx/research_notes/lab_notebooks/track_{task.track}.md`"
            " following the existing entry format (ordered by task code, earliest first).",
            "3. Create the PR using `gh pr create` (NOT any built-in PR tool).",
            "4. PR body MUST include `Closes #<this-issue-number>` for auto-close on merge.",
            "5. Add `--label orchestrator-task` to the `gh pr create` command.",
            "",
            "## Completion",
            "This issue closes automatically when the linked PR merges.",
        ]
        body = "\n".join(body_parts)

        payload = {
            "title": title,
            "body": body,
            "labels": [ISSUE_LABEL],
        }
        created = self._request("POST", f"/repos/{self.repo}/issues", payload)
        if not isinstance(created, dict):
            raise RuntimeError("Unexpected GitHub API payload while creating issue")

        return IssueRef(
            task_id=task.task_id,
            number=int(created.get("number", 0)),
            state=str(created.get("state", "open")),
            title=str(created.get("title", title)),
            html_url=str(created.get("html_url", "")),
            updated_at=str(created.get("updated_at", utc_now_iso())),
        )

    def add_comment_to_issue(self, issue_number: int, body: str) -> dict[str, Any]:
        """Post a comment on an existing issue."""
        result = self._request(
            "POST",
            f"/repos/{self.repo}/issues/{issue_number}/comments",
            {"body": body},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected payload while commenting on issue #{issue_number}")
        return result


def sync_status_from_issues(
    tasks: list[Task],
    task_status: dict[str, str],
    issues_by_task: dict[str, IssueRef],
) -> dict[str, dict[str, Any]]:
    """Reconcile task status from authoritative GitHub issue state."""

    issue_index: dict[str, dict[str, Any]] = {}
    for task in tasks:
        issue_ref = issues_by_task.get(task.task_id)
        previous = task_status.get(task.task_id, "pending")
        if issue_ref is None:
            task_status[task.task_id] = previous if previous == "blocked" else "pending"
            continue

        task_status[task.task_id] = "completed" if issue_ref.state == "closed" else "in_progress"
        issue_index[task.task_id] = {
            "number": issue_ref.number,
            "state": issue_ref.state,
            "url": issue_ref.html_url,
            "title": issue_ref.title,
            "updated_at": issue_ref.updated_at,
        }

    return issue_index


def summarize(
    tasks: list[Task], task_status: dict[str, str], paused: bool, plan_path: Path | None = None
) -> dict[str, Any]:
    counts = {status: 0 for status in VALID_STATUSES}
    for task in tasks:
        status = task_status.get(task.task_id, "pending")
        counts[status] = counts.get(status, 0) + 1

    in_progress = [task.task_id for task in tasks if task_status.get(task.task_id) == "in_progress"]

    # Use plan graph for accurate readiness if available.
    if plan_path is not None:
        from lyzortx.orchestration.plan_parser import load_plan, select_ready_tasks

        graph = load_plan(plan_path)
        ready = [t.task_id for t in select_ready_tasks(graph)]
    else:
        ready = [
            task.task_id
            for task in tasks
            if task_status.get(task.task_id) == "pending"
            and all(task_status.get(dep) == "completed" for dep in task.dependencies)
        ]

    return {
        "paused": paused,
        "counts": counts,
        "in_progress": in_progress,
        "ready": ready,
        "total_tasks": len(tasks),
    }


def _dispatch_one_agent_task(
    task: Task,
    state: dict[str, Any],
    issues_by_task: dict[str, IssueRef],
    github_client: GitHubClient,
) -> dict[str, Any]:
    """Dispatch a single agent task as a GitHub issue. Returns a result dict."""
    task_status = state["task_status"]

    existing_issue = issues_by_task.get(task.task_id)
    if existing_issue is not None and existing_issue.state == "open":
        task_status[task.task_id] = "in_progress"
        append_history(
            state,
            "task_dispatched_existing_issue",
            task_id=task.task_id,
            issue_number=existing_issue.number,
            issue_url=existing_issue.html_url,
        )
        return {"task_id": task.task_id, "issue_number": existing_issue.number, "reused": True}

    created_issue = github_client.create_agent_task_issue(task)
    task_status[task.task_id] = "in_progress"
    state.setdefault("issue_index", {})[task.task_id] = {
        "number": created_issue.number,
        "state": created_issue.state,
        "url": created_issue.html_url,
        "title": created_issue.title,
        "updated_at": created_issue.updated_at,
    }

    append_history(
        state,
        "task_dispatched_new_issue",
        task_id=task.task_id,
        issue_number=created_issue.number,
        issue_url=created_issue.html_url,
    )
    return {"task_id": task.task_id, "issue_number": created_issue.number, "reused": False}


def run_once(
    tasks: list[Task],
    state: dict[str, Any],
    issues_by_task: dict[str, IssueRef],
    github_client: GitHubClient | None,
    max_active_tasks: int,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    task_status = state["task_status"]
    if state.get("paused"):
        append_history(state, "run_once_skipped", reason="paused")
        return {"action": "skipped", "reason": "paused"}

    active_count = sum(1 for s in task_status.values() if s == "in_progress")
    slots = max_active_tasks - active_count
    if slots <= 0:
        append_history(state, "run_once_skipped", reason="max_active_tasks_reached")
        return {"action": "skipped", "reason": "max_active_tasks_reached"}

    # Get all ready tasks from the plan graph.
    from lyzortx.orchestration.plan_parser import load_plan, select_ready_tasks

    effective_plan_path = plan_path or DEFAULT_PLAN_PATH
    graph = load_plan(effective_plan_path)
    ready = select_ready_tasks(graph)

    # Filter out tasks already in progress.
    in_progress_ids = {tid for tid, s in task_status.items() if s == "in_progress"}
    candidates = [pt for pt in ready if pt.task_id not in in_progress_ids][:slots]

    if not candidates:
        append_history(state, "run_once_skipped", reason="no_ready_tasks")
        return {"action": "skipped", "reason": "no_ready_tasks"}

    if github_client is None:
        dispatched = []
        for pt in candidates:
            task_status[pt.task_id] = "in_progress"
            append_history(
                state,
                "task_waiting_for_agent",
                task_id=pt.task_id,
                note="No GitHub token/repo configured; issue dispatch skipped.",
            )
            dispatched.append(pt.task_id)
        return {
            "action": "tasks_waiting_for_agent",
            "task_ids": dispatched,
            "warning": "No GitHub token/repo configured",
        }

    try:
        github_client.ensure_task_label_exists()
    except Exception as exc:
        append_history(state, "run_once_skipped", reason="label_setup_failed", note=str(exc))
        return {"action": "skipped", "reason": "label_setup_failed", "note": str(exc)}

    dispatched: list[dict[str, Any]] = []
    for pt in candidates:
        criteria = _load_acceptance_criteria(effective_plan_path, pt.task_id)
        task = Task(
            task_id=pt.task_id,
            title=pt.title,
            description=f"Track {pt.track} task. {pt.title}",
            dependencies=[],
            executor="agent",
            command=None,
            expected_paths=[],
            acceptance_criteria=criteria,
            plan_checkbox_text=pt.title,
            track=pt.track,
        )
        result = _dispatch_one_agent_task(task, state, issues_by_task, github_client)
        dispatched.append(result)

    return {"action": "tasks_dispatched", "dispatched": dispatched}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        choices=[
            "status",
            "run_once",
            "pause",
            "resume",
        ],
        default="status",
        help="Command to execute.",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Path to plan.yml.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to the runtime state JSON file.",
    )
    parser.add_argument("--task-id", type=str, help="Task id for set_task_status command.")
    parser.add_argument(
        "--status",
        type=str,
        choices=sorted(VALID_STATUSES),
        help="New task status for set_task_status command.",
    )
    parser.add_argument("--note", type=str, default="", help="Optional note attached to history events.")
    parser.add_argument(
        "--max-active-tasks",
        type=int,
        default=1,
        help="Maximum number of tasks that may be in progress at once.",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in owner/repo form for issue-backed orchestration.",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token used for issue API operations.",
    )
    parser.add_argument(
        "--github-api-url",
        type=str,
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        help="GitHub API base URL.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_active_tasks < 1:
        raise ValueError("--max-active-tasks must be >= 1")

    plan_path: Path = args.plan_path
    plan_md_path = plan_path.parent.parent / "research_notes" / "PLAN.md"
    tasks = load_pending_tasks(plan_path)
    state_path_exists = args.state_path.exists()
    state = initialize_state(tasks, args.state_path)

    github_client: GitHubClient | None = None
    issues_by_task: dict[str, IssueRef] = {}
    if args.github_repo and args.github_token:
        github_client = GitHubClient(
            token=args.github_token,
            repo=args.github_repo,
            api_base=args.github_api_url,
        )
        issues_by_task = github_client.list_task_issues()
        issue_index = sync_status_from_issues(tasks, state["task_status"], issues_by_task)
        state["issue_index"] = issue_index

        # Mark tasks done in plan.yml when their issues are closed.
        for task_id, status in list(state["task_status"].items()):
            if status == "completed":
                try:
                    mark_task_done_in_plan(plan_path, plan_md_path, task_id)
                except (ValueError, KeyError):
                    pass  # Task may already be done or not found in plan.

    result: dict[str, Any]
    if args.command == "status":
        result = {
            "action": "status",
            "summary": summarize(tasks, state["task_status"], bool(state.get("paused")), plan_path),
        }
    elif args.command == "pause":
        state["paused"] = True
        append_history(state, "orchestrator_paused", note=args.note)
        result = {"action": "paused"}
    elif args.command == "resume":
        state["paused"] = False
        append_history(state, "orchestrator_resumed", note=args.note)
        result = {"action": "resumed"}
    elif args.command == "run_once":
        result = run_once(
            tasks=tasks,
            state=state,
            issues_by_task=issues_by_task,
            github_client=github_client,
            max_active_tasks=args.max_active_tasks,
            plan_path=plan_path,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    if args.command not in READ_ONLY_COMMANDS or not state_path_exists:
        state["updated_at"] = utc_now_iso()
        save_json(args.state_path, state)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"orchestrator_error: {exc}", file=sys.stderr)
        raise
