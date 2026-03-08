#!/usr/bin/env python3
"""Issue-driven orchestrator for PLAN task execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_PATH = REPO_ROOT / "lyzortx/orchestration/tasks.json"
DEFAULT_STATE_PATH = REPO_ROOT / "lyzortx/generated_outputs/orchestration/runtime_state.json"
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}
READ_ONLY_COMMANDS = {"status", "suggest_plan_sync"}
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


def parse_tasks(tasks_path: Path) -> list[Task]:
    data = load_json(tasks_path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Task registry must define a top-level 'tasks' array")

    tasks: list[Task] = []
    task_ids: set[str] = set()
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise ValueError("Each task entry must be a JSON object")

        task_id = str(item.get("id", "")).strip()
        if not task_id:
            raise ValueError("Task is missing required 'id'")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task id detected: {task_id}")
        task_ids.add(task_id)

        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(f"Task {task_id} has non-list 'dependencies'")

        command = item.get("command")
        if command is not None:
            if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
                raise ValueError(f"Task {task_id} has invalid 'command'; expected list[str]")

        expected_paths = item.get("expected_paths", [])
        if not isinstance(expected_paths, list) or not all(isinstance(path, str) for path in expected_paths):
            raise ValueError(f"Task {task_id} has invalid 'expected_paths'; expected list[str]")

        acceptance_criteria = item.get("acceptance_criteria", [])
        if not isinstance(acceptance_criteria, list) or not all(
            isinstance(criterion, str) for criterion in acceptance_criteria
        ):
            raise ValueError(f"Task {task_id} has invalid 'acceptance_criteria'; expected list[str]")

        plan_checkbox_text = item.get("plan_checkbox_text")
        if plan_checkbox_text is not None and not isinstance(plan_checkbox_text, str):
            raise ValueError(f"Task {task_id} has invalid 'plan_checkbox_text'; expected string")

        tasks.append(
            Task(
                task_id=task_id,
                title=str(item.get("title", "")).strip() or task_id,
                description=str(item.get("description", "")).strip(),
                dependencies=[str(dep) for dep in dependencies],
                executor=str(item.get("executor", "agent")).strip() or "agent",
                command=command,
                expected_paths=expected_paths,
                acceptance_criteria=[criterion.strip() for criterion in acceptance_criteria if criterion.strip()],
                plan_checkbox_text=plan_checkbox_text.strip() if isinstance(plan_checkbox_text, str) else None,
            )
        )

    known_task_ids = {task.task_id for task in tasks}
    unknown_dependencies = {dep for task in tasks for dep in task.dependencies if dep not in known_task_ids}
    if unknown_dependencies:
        missing = ", ".join(sorted(unknown_dependencies))
        raise ValueError(f"Task registry contains unknown dependencies: {missing}")

    return tasks


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

        dependencies = "\n".join(f"- `{dependency}`" for dependency in task.dependencies) or "- None"
        acceptance = "\n".join(f"- {criterion}" for criterion in task.acceptance_criteria) or "- Define in implementation PR"
        body_parts = [
            f"<!-- {TASK_ID_MARKER}: {task.task_id} -->",
            "## Orchestrator Task",
            f"- Task ID: `{task.task_id}`",
            f"- Executor: `{task.executor}`",
            "- Plan Source: `lyzortx/research_notes/PLAN.md`",
            "",
            "## Description",
            task.description or "No description provided.",
            "",
            "## Dependencies",
            dependencies,
            "",
            "## Acceptance Criteria",
            acceptance,
            "",
        ]
        if task.expected_paths:
            expected = "\n".join(f"- `{path}`" for path in task.expected_paths)
            body_parts.extend(["## Expected Paths", expected, ""])

        body_parts.extend(
            [
                "## Completion",
                "Close this issue once implementation and verification are complete.",
            ]
        )
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


def select_next_ready_task(tasks: list[Task], task_status: dict[str, str]) -> Task | None:
    for task in tasks:
        if task_status.get(task.task_id) != "pending":
            continue
        if any(task_status.get(dependency) != "completed" for dependency in task.dependencies):
            continue
        return task
    return None


def validate_expected_paths(task: Task) -> list[str]:
    missing_paths: list[str] = []
    for expected_path in task.expected_paths:
        resolved = REPO_ROOT / expected_path
        if not resolved.exists():
            missing_paths.append(expected_path)
    return missing_paths


def run_shell_task(task: Task) -> tuple[bool, str]:
    if not task.command:
        return False, "Task is configured for shell execution but has no command"

    completed = subprocess.run(
        task.command,
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-10:]) if completed.stdout else ""
    stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-10:]) if completed.stderr else ""

    message_parts = [f"command_exit_code={completed.returncode}"]
    if stdout_tail:
        message_parts.append(f"stdout_tail=\n{stdout_tail}")
    if stderr_tail:
        message_parts.append(f"stderr_tail=\n{stderr_tail}")

    success = completed.returncode == 0
    return success, "\n\n".join(message_parts)


def summarize(tasks: list[Task], task_status: dict[str, str], paused: bool) -> dict[str, Any]:
    counts = {status: 0 for status in VALID_STATUSES}
    for task in tasks:
        status = task_status.get(task.task_id, "pending")
        counts[status] = counts.get(status, 0) + 1

    in_progress = [task.task_id for task in tasks if task_status.get(task.task_id) == "in_progress"]
    ready = [
        task.task_id
        for task in tasks
        if task_status.get(task.task_id) == "pending"
        and all(task_status.get(dependency) == "completed" for dependency in task.dependencies)
    ]

    return {
        "paused": paused,
        "counts": counts,
        "in_progress": in_progress,
        "ready": ready,
        "total_tasks": len(tasks),
    }


def run_once(
    tasks: list[Task],
    state: dict[str, Any],
    issues_by_task: dict[str, IssueRef],
    github_client: GitHubClient | None,
    max_active_tasks: int,
) -> dict[str, Any]:
    task_status = state["task_status"]
    if state.get("paused"):
        append_history(state, "run_once_skipped", reason="paused")
        return {"action": "skipped", "reason": "paused"}

    active_tasks = [task_id for task_id, status in task_status.items() if status == "in_progress"]
    if len(active_tasks) >= max_active_tasks:
        append_history(
            state,
            "run_once_skipped",
            reason="max_active_tasks_reached",
            active_tasks=active_tasks,
            max_active_tasks=max_active_tasks,
        )
        return {
            "action": "skipped",
            "reason": "max_active_tasks_reached",
            "active_tasks": active_tasks,
            "max_active_tasks": max_active_tasks,
        }

    next_task = select_next_ready_task(tasks, task_status)
    if next_task is None:
        append_history(state, "run_once_skipped", reason="no_ready_tasks")
        return {"action": "skipped", "reason": "no_ready_tasks"}

    append_history(state, "task_started", task_id=next_task.task_id, executor=next_task.executor)

    if next_task.executor == "shell":
        task_status[next_task.task_id] = "in_progress"
        success, execution_note = run_shell_task(next_task)
        if not success:
            task_status[next_task.task_id] = "blocked"
            append_history(
                state,
                "task_blocked",
                task_id=next_task.task_id,
                reason="shell_command_failed",
                note=execution_note,
            )
            return {
                "action": "task_blocked",
                "task_id": next_task.task_id,
                "reason": "shell_command_failed",
                "note": execution_note,
            }

        missing_paths = validate_expected_paths(next_task)
        if missing_paths:
            task_status[next_task.task_id] = "blocked"
            append_history(
                state,
                "task_blocked",
                task_id=next_task.task_id,
                reason="missing_expected_paths",
                missing_paths=missing_paths,
                note=execution_note,
            )
            return {
                "action": "task_blocked",
                "task_id": next_task.task_id,
                "reason": "missing_expected_paths",
                "missing_paths": missing_paths,
                "note": execution_note,
            }

        task_status[next_task.task_id] = "completed"
        append_history(state, "task_completed", task_id=next_task.task_id, note=execution_note)
        return {
            "action": "task_completed",
            "task_id": next_task.task_id,
            "note": execution_note,
        }

    if github_client is None:
        task_status[next_task.task_id] = "in_progress"
        append_history(
            state,
            "task_waiting_for_agent",
            task_id=next_task.task_id,
            note="No GitHub token/repo configured; issue dispatch skipped.",
        )
        return {
            "action": "task_waiting_for_agent",
            "task_id": next_task.task_id,
            "executor": next_task.executor,
            "warning": "No GitHub token/repo configured",
        }

    existing_issue = issues_by_task.get(next_task.task_id)
    if existing_issue is not None and existing_issue.state == "open":
        task_status[next_task.task_id] = "in_progress"
        append_history(
            state,
            "task_dispatched_existing_issue",
            task_id=next_task.task_id,
            issue_number=existing_issue.number,
            issue_url=existing_issue.html_url,
        )
        return {
            "action": "task_dispatched_existing_issue",
            "task_id": next_task.task_id,
            "issue_number": existing_issue.number,
            "issue_url": existing_issue.html_url,
        }

    try:
        github_client.ensure_task_label_exists()
    except Exception as exc:
        task_status[next_task.task_id] = "blocked"
        append_history(
            state,
            "task_blocked",
            task_id=next_task.task_id,
            reason="label_setup_failed",
            note=str(exc),
        )
        return {
            "action": "task_blocked",
            "task_id": next_task.task_id,
            "reason": "label_setup_failed",
            "note": str(exc),
        }

    created_issue = github_client.create_agent_task_issue(next_task)
    task_status[next_task.task_id] = "in_progress"
    state.setdefault("issue_index", {})[next_task.task_id] = {
        "number": created_issue.number,
        "state": created_issue.state,
        "url": created_issue.html_url,
        "title": created_issue.title,
        "updated_at": created_issue.updated_at,
    }
    append_history(
        state,
        "task_dispatched_new_issue",
        task_id=next_task.task_id,
        issue_number=created_issue.number,
        issue_url=created_issue.html_url,
    )
    return {
        "action": "task_dispatched_new_issue",
        "task_id": next_task.task_id,
        "issue_number": created_issue.number,
        "issue_url": created_issue.html_url,
    }


def suggest_plan_sync(tasks: list[Task], task_status: dict[str, str]) -> dict[str, Any]:
    suggestions: list[dict[str, str]] = []
    for task in tasks:
        if task.plan_checkbox_text and task_status.get(task.task_id) == "completed":
            suggestions.append(
                {
                    "task_id": task.task_id,
                    "checkbox_text": task.plan_checkbox_text,
                }
            )

    return {
        "action": "suggest_plan_sync",
        "completed_task_count": len(suggestions),
        "suggestions": suggestions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        choices=[
            "status",
            "run_once",
            "pause",
            "resume",
            "set_task_status",
            "suggest_plan_sync",
        ],
        default="status",
        help="Command to execute.",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to the static task registry JSON.",
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

    tasks = parse_tasks(args.tasks_path)
    task_ids = {task.task_id for task in tasks}
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
        state["issue_index"] = sync_status_from_issues(tasks, state["task_status"], issues_by_task)

    result: dict[str, Any]
    if args.command == "status":
        result = {
            "action": "status",
            "summary": summarize(tasks, state["task_status"], bool(state.get("paused"))),
        }
    elif args.command == "suggest_plan_sync":
        result = suggest_plan_sync(tasks, state["task_status"])
    elif args.command == "pause":
        state["paused"] = True
        append_history(state, "orchestrator_paused", note=args.note)
        result = {"action": "paused"}
    elif args.command == "resume":
        state["paused"] = False
        append_history(state, "orchestrator_resumed", note=args.note)
        result = {"action": "resumed"}
    elif args.command == "set_task_status":
        if not args.task_id or not args.status:
            raise ValueError("set_task_status requires both --task-id and --status")
        if args.task_id not in task_ids:
            raise ValueError(f"Unknown task id: {args.task_id}")
        state["task_status"][args.task_id] = args.status
        append_history(
            state,
            "task_status_overridden",
            task_id=args.task_id,
            status=args.status,
            note=args.note,
        )
        result = {
            "action": "task_status_updated",
            "task_id": args.task_id,
            "status": args.status,
        }
    elif args.command == "run_once":
        result = run_once(
            tasks=tasks,
            state=state,
            issues_by_task=issues_by_task,
            github_client=github_client,
            max_active_tasks=args.max_active_tasks,
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
