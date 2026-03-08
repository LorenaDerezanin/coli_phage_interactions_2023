# PLAN Orchestrator (Issue-Driven)

This directory contains an event-driven orchestrator for executing [PLAN.md](../research_notes/PLAN.md) tasks with
GitHub Issues as the authoritative sequencing state.

## What Is Authoritative

- Source of truth for task progression: GitHub issue state (`open`/`closed`) on issues labeled `orchestrator-task`.
- Local convenience state: `lyzortx/generated_outputs/orchestration/runtime_state.json`.
- Task catalog and DAG: `lyzortx/orchestration/tasks.json`.

In GitHub Actions, `runtime_state.json` is ephemeral per run and uploaded only as an artifact. It is not used as durable
sequencing state.

## Components

- CLI runner: `lyzortx/orchestration/orchestrator.py`.
- Task registry: `lyzortx/orchestration/tasks.json`.
- CI trigger workflow: `.github/workflows/orchestrator.yml`.

## CLI Usage

Run from repository root.

```bash
python -m lyzortx.orchestration.orchestrator --command status
```

Dispatch one ready task (`executor="agent"` creates/uses a GitHub issue when `GITHUB_TOKEN` and `GITHUB_REPOSITORY` are
set):

```bash
python -m lyzortx.orchestration.orchestrator --command run_once
```

Pause/resume orchestration:

```bash
python -m lyzortx.orchestration.orchestrator --command pause --note "maintenance"
python -m lyzortx.orchestration.orchestrator --command resume
```

Manual override (local convenience only; issue state remains authoritative):

```bash
python -m lyzortx.orchestration.orchestrator \
  --command set_task_status \
  --task-id IM01_LOCK_DENOMINATOR_POLICY \
  --status blocked \
  --note "Waiting on data"
```

Emit PLAN checkbox suggestions (does not edit files):

```bash
python -m lyzortx.orchestration.orchestrator --command suggest_plan_sync
```

## Task Registry Schema (v2)

Each task entry in `tasks.json` supports:

- `id`: stable task id.
- `title`: short task title.
- `description`: execution context.
- `dependencies`: upstream task ids that must be completed.
- `executor`: `agent` or `shell`.
- `command`: optional `list[str]` for shell tasks.
- `expected_paths`: optional output path checks (primarily for `shell` tasks).
- `acceptance_criteria`: checklist included in dispatched issues.
- `plan_checkbox_text`: human-facing checkbox text for PLAN sync suggestion.

## GitHub Actions Trigger Model

Workflow: `.github/workflows/orchestrator.yml`

- `workflow_dispatch`: manual commands.
- `repository_dispatch`: API/CLI command trigger.
- `issues.closed`: when a labeled orchestrator task issue is closed, run a new tick to dispatch the next ready task.

No cron trigger is used in this mode. Default `max_active_tasks` in workflow dispatch is `1`. The `orchestrator-task`
label is created automatically on first dispatch if missing.

## Dispatch Payload Example

```json
{
  "event_type": "orchestrator_cmd",
  "client_payload": {
    "command": "set_task_status",
    "task_id": "IM01_LOCK_DENOMINATOR_POLICY",
    "status": "blocked",
    "note": "Waiting for benchmark definitions",
    "max_active_tasks": 1
  }
}
```
