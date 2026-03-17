# Replace PR Review Cycle with LangGraph State Machine

## Context

The PR review cycle (`codex-pr-lifecycle.yml`) is a 190-line GitHub Actions workflow that implements a
review→fix→verify→rereview loop using webhook event chaining and GitHub labels as state. This is fragile: context is
lost between workflow runs, the review loop requires re-triggering via events, and there's no way to enforce that the
LLM actually completed required tool calls. Replacing this with a LangGraph state machine gives us deterministic state
transitions, a native loop with round tracking, and tool-call enforcement — all testable in Python.

## Composio agent-orchestrator vs. LangGraph: Evaluation

[Composio agent-orchestrator](https://github.com/ComposioHQ/agent-orchestrator) was evaluated as an alternative. It's a
TypeScript-based system for managing fleets of AI coding agents with a plugin architecture (runtime, agent, workspace,
tracker, SCM, notifier, terminal, lifecycle — 8 slots). It includes a polling-based lifecycle manager, a reaction system
(CI failure → auto-fix, changes-requested → send-to-agent), review fingerprinting, and escalation logic.

**What it solves that overlaps with our needs:**

- Review comment detection + auto-dispatch to agent (their `changes-requested` reaction)
- CI failure → retry loop with escalation (their `ci-failed` reaction)
- Round/retry counting with configurable escalation thresholds
- Multi-agent session management with worktree isolation

**Why it's not the right fit here:**

- **Language mismatch**: TypeScript/Node.js ecosystem; our orchestration is Python. Adopting it means running a Node.js
  sidecar or rewriting our orchestrator in TS — neither is justified for one pipeline project.
- **Polling model**: It runs a persistent server polling GitHub every 30s. Our system uses GitHub Actions event triggers
  (zero infrastructure). Composio would require a running process somewhere.
- **Agent coupling**: It assumes agents are long-running processes in tmux/Docker sessions. Our agents are ephemeral —
  invoked per-task in CI, do their work, exit. The session lifecycle model doesn't map.
- **Black box review logic**: The review cycle is embedded in the lifecycle manager's `checkSession()` loop. We want
  explicit, testable state machine nodes — especially the non-LLM `verify_replies` step.
- **Overkill**: We need 1 review graph for 1 repo. Composio is designed for fleet management across many repos. The
  plugin architecture adds complexity we'd never use.
- **No tool-call enforcement**: Composio sends messages to agents and checks outcomes (did a PR appear?). LangGraph lets
  us define tool schemas and verify the LLM actually called each required tool.

**Verdict**: Composio solves a different problem (fleet management for teams running many agents across many repos).
LangGraph is the right tool for our problem (a testable state machine governing a multi-step LLM workflow within a
single CI run).

## Decisions

- LLM provider: OpenAI (gpt-5.4), matching current system
- Fix strategy: LLM with LangChain tools (read_file, write_file, run_command, gh_reply)
- Rollout: Direct replacement of `codex-pr-lifecycle.yml`

## Files to Create

```
lyzortx/orchestration/review_graph/
    __init__.py          # graph factory: build_review_graph()
    __main__.py          # CLI entry point
    state.py             # ReviewCycleState TypedDict
    nodes.py             # node functions (collect, fix, verify, rereview, escalate)
    tools.py             # LangChain tool definitions for the fix agent
    github_helpers.py    # review-specific GitHub API functions (extend GitHubClient)
    AGENTS.md            # architecture docs for this subpackage
    CLAUDE.md            # imports @AGENTS.md
lyzortx/tests/
    test_review_graph.py              # routing + node + integration tests
    test_review_graph_tools.py        # tool unit tests
    test_review_graph_github_helpers.py  # GitHub helper tests
```

## Files to Modify

- `requirements.txt` — add `langchain-core`, `langchain-openai`, `langgraph` (alphabetically)
- `.github/workflows/codex-pr-lifecycle.yml` — replace with slim workflow that invokes the Python graph

## Reuse

- `verify_review_replies.find_unanswered_comments()` — import directly in `verify_replies` node
- `orchestrator.GitHubClient` — import and extend via helper functions in `github_helpers.py`
- Test pattern from `test_verify_review_replies.py` — `_comment()` factory, pure function tests

---

## Step 1: Add Dependencies

Add to `requirements.txt` (alphabetical):

```
langchain-core
langchain-openai
langgraph
```

These slot between `joblib` and `matplotlib`. CI (`unit-tests.yml`) already does `pip install -r requirements.txt`.

## Step 2: `state.py` — Define Graph State

```python
class ReviewCycleState(TypedDict):
    repo: str                        # owner/repo
    pr_number: int
    review_id: int | None            # triggering review ID
    github_token: str
    round: int                       # current fix round (starts 0)
    max_rounds: int                  # cap (default 3)
    feedback_comments: list[dict]    # collected from GitHub
    review_body_feedback: str        # substantive review body text
    has_feedback: bool
    unanswered_comments: list[dict]  # from verify step
    approved: bool                   # rereview outcome
    outcome: str                     # terminal: "clean" | "approved" | "escalated"
```

## Step 3: `github_helpers.py` — Review-Specific GitHub API

Helper functions that accept a `GitHubClient` instance (from `orchestrator.py`):

- `fetch_review_comments(client, repo, pr, review_id) -> list[dict]` — paginated inline comments
- `fetch_review_body(client, repo, pr, review_id) -> str | None` — body text, filtering boilerplate
- `fetch_issue_comments(client, repo, pr) -> list[dict]` — general PR comments
- `post_pr_comment(client, repo, pr, body)` — post a comment
- `add_pr_label(client, repo, pr, label)` — add a label
- `reply_to_review_comment(client, repo, pr, comment_id, body)` — reply to inline comment

Pagination: follow the pattern in `GitHubClient.list_task_issues()` (lines 249-285 of orchestrator.py) — loop pages with
`per_page=100&page=N` until empty.

## Step 4: `tools.py` — LangChain Tools for Fix Agent

Define tools for the `address_feedback` LLM agent:

```python
@tool
def read_file(path: str) -> str: ...

@tool
def write_file(path: str, content: str) -> str: ...

@tool
def patch_file(path: str, search: str, replace: str) -> str: ...

@tool
def run_command(command: str, timeout_seconds: int = 120) -> str: ...

@tool
def reply_to_comment(comment_id: int, body: str) -> str: ...
```

- Tools are instantiated with `working_directory` and `GitHubClient` via closure/factory
- `run_command` uses `subprocess.run` with timeout, captures stdout+stderr
- Security: runs inside GH Actions, already sandboxed (matches current `danger-full-access`)
- Max iterations guard: 50 tool calls per invocation to prevent runaway loops

## Step 5: `nodes.py` — Node Functions

Each node takes `ReviewCycleState`, returns a partial state update dict.

### `collect_feedback(state) -> dict`

- Creates `GitHubClient` from state
- Calls `fetch_review_comments`, `fetch_review_body`, `fetch_issue_comments`
- Determines `has_feedback` (same logic as workflow lines 46-77)
- Returns `{feedback_comments, review_body_feedback, has_feedback}`

### `address_feedback(state) -> dict`

- Constructs prompt from `feedback_comments` + `review_body_feedback` (same markdown format as current
  `/tmp/codex_feedback.txt` built at workflow lines 102-133)
- Creates `ChatOpenAI(model="gpt-5.4")` bound to tools from `tools.py`
- Runs in a tool-calling loop (LangGraph's `create_react_agent` or manual `while` loop with `model.bind_tools()`)
- LLM reads flagged files, fixes code, runs tests, commits, pushes, replies to each comment
- Returns `{round: state["round"] + 1}`

### `verify_replies(state) -> dict`

- NON-LLM node
- Fetches current review comments from GitHub
- Calls `find_unanswered_comments()` from `verify_review_replies.py` (direct import)
- Returns `{unanswered_comments: [...]}`

### `rereview(state) -> dict`

- LLM node: reviews the current PR diff against acceptance criteria
- If no new issues found: returns `{approved: True}`
- If new issues: posts comments on PR, returns `{approved: False, feedback_comments: [new]}`

### `mark_clean(state) -> dict`

- Terminal: adds `ready-for-human-review` label, posts success comment
- Returns `{outcome: "clean"}`

### `escalate(state) -> dict`

- Terminal: adds `needs-human-review` label, posts max-rounds message
- Returns `{outcome: "escalated"}`

## Step 6: `__init__.py` — Graph Wiring

<!-- markdownlint-disable MD010 MD033 -->

```
                    +------------------+
                    | collect_feedback |
                    +--------+---------+
                             |
                 +-----------+-----------+
                 | no feedback           | has feedback
                 v                       v
            mark_clean          +-- round >= max? --+
                                | no                | yes
                                v                   v
                        address_feedback          escalate
                                |
                                v
                         verify_replies
                                |
                +---------------+---------------+
                | unanswered comments           | all replied
                v                               v
        round >= max? --+                    rereview
        | no            | yes                    |
        v               v                        v
  address_feedback   escalate        +-----------+-----------+
                                     | approved              | has_issues
                                     v                       v
                                mark_clean          round >= max? --+
                                                      | no          | yes
                                                      v             v
                                               address_feedback  escalate
```

<!-- markdownlint-enable MD010 MD033 -->

Routing functions (pure, testable):

```python
def route_after_collect(state) -> str:
    if not state["has_feedback"]:
        return "mark_clean"
    if state["round"] >= state["max_rounds"]:
        return "escalate"
    return "address_feedback"

def route_after_verify_replies(state) -> str:
    if state["unanswered_comments"]:
        if state["round"] >= state["max_rounds"]:
            return "escalate"
        return "address_feedback"
    return "rereview"

def route_after_rereview(state) -> str:
    if state["approved"]:
        return "mark_clean"
    if state["round"] >= state["max_rounds"]:
        return "escalate"
    return "address_feedback"
```

Compile with `MemorySaver` checkpointer (sufficient for single GH Actions run).

## Step 7: `__main__.py` — CLI Entry Point

```bash
python -m lyzortx.orchestration.review_graph \
    --pr-number 42 \
    --review-id 123456 \
    --repo owner/repo \
    --max-rounds 3
```

- Reads `GITHUB_TOKEN` and `OPENAI_API_KEY` from environment
- Constructs initial state, invokes graph, prints JSON outcome
- Exit code 0 on success, 1 on error

## Step 8: Replace `codex-pr-lifecycle.yml`

Replace the 190-line workflow with ~40 lines:

```yaml
name: PR Review Cycle (LangGraph)
on:
  pull_request_review:
    types: [submitted]
  workflow_dispatch:
    inputs:
      pr_number:
        description: "PR number"
        required: true
        type: string
permissions:
  contents: write
  pull-requests: write
  issues: write
jobs:
  review-cycle:
    if: >-
      github.event_name == 'workflow_dispatch' || (github.event.review.state == 'commented'
          && github.event.review.user.login == 'chatgpt-codex-connector[bot]')
    runs-on: ubuntu-24.04
    timeout-minutes: 30
    env:
      GITHUB_TOKEN: ${{ secrets.ORCHESTRATOR_PAT }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v6
        with:
          ref: <pr-branch> # resolved from pr number
          fetch-depth: 0
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12.12"
      - run: pip install -r requirements.txt
      - run: |
          python -m lyzortx.orchestration.review_graph \
            --pr-number $PR_NUM \
            --review-id $REVIEW_ID \
            --repo ${{ github.repository }}
```

The workflow just sets up the environment and delegates everything to the Python graph. No more label-based round
counting, no more shell-scripted API calls, no more Codex action.

## Step 9: Tests

### `test_review_graph.py`

- **Routing tests** (pure functions, no mocking):
  - `test_route_after_collect_no_feedback` -> "mark_clean"
  - `test_route_after_collect_has_feedback` -> "address_feedback"
  - `test_route_after_collect_round_exceeded` -> "escalate"
  - `test_route_after_verify_replies_with_unanswered_comments` -> "address_feedback"
  - `test_route_after_verify_replies_round_exceeded` -> "escalate"
  - `test_route_after_verify_replies_all_replied` -> "rereview"
  - `test_route_after_rereview_approved` -> "mark_clean"
  - `test_route_after_rereview_has_issues` -> "address_feedback"
  - `test_route_after_rereview_round_exceeded` -> "escalate"
- **Node tests** (mock GitHubClient):
  - `test_collect_feedback_no_comments`
  - `test_collect_feedback_with_inline_and_body`
  - `test_verify_replies_reuses_find_unanswered_comments`
- **Graph integration** (mock LLM, verify full state flow):
  - `test_graph_no_feedback_path` -> collect -> mark_clean
  - `test_graph_unanswered_comments_loop_back_to_address_feedback`
  - `test_graph_escalation_after_max_rounds`

### `test_review_graph_tools.py`

- `test_read_file`, `test_write_file`, `test_patch_file` (use `tmp_path`)
- `test_run_command_captures_output`, `test_run_command_timeout`

### `test_review_graph_github_helpers.py`

- Mock `GitHubClient._request`, verify pagination and filtering logic

## Verification

1. **Unit tests**: `micromamba activate phage_env && pytest lyzortx/tests/test_review_graph*.py -v`
2. **Local dry run**: `python -m lyzortx.orchestration.review_graph --pr-number 1 --repo test/test --max-rounds 1` with
   mock env vars to verify CLI plumbing
3. **CI check**: Push to a branch, open a test PR, trigger `workflow_dispatch` with the PR number
4. **Pre-commit**: `pre-commit run --all-files` to verify formatting
