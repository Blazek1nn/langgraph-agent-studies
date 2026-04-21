# langgraph-agent-studies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, production-minded LangGraph ReAct agent with three tools, SQLite persistence, typed state, CLI chat, and a full test suite.

**Architecture:** StateGraph with two nodes (agent + tools), conditional routing via `tools_condition`, and `SqliteSaver` checkpointer for persistence across restarts. Package lives in `src/agent_studies/` using a src-layout so imports are isolated and testable.

**Tech Stack:** Python 3.11+, LangGraph 0.x, langchain-anthropic, SqliteSaver (langgraph-checkpoint-sqlite), pytest, uv

---

### Task 1: Scaffold project with uv and git

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/agent_studies/__init__.py`
- Create: `tests/__init__.py`
- Create: `examples/` (empty dir placeholder)
- Create: `notes/` placeholder via .gitkeep

- [ ] **Step 1: Initialize git repo**

```bash
cd C:/Users/USER/Desktop/langgraph-agent-studies
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "langgraph-agent-studies"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.3",
    "langchain-core>=0.3",
    "langchain-anthropic>=0.3",
    "python-dotenv>=1.0",
    "langgraph-checkpoint-sqlite>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agent_studies"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: end-to-end tests requiring ANTHROPIC_API_KEY",
]
```

- [ ] **Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 4: Create .gitignore**

```
.env
*.sqlite
notes/
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
dist/
.ruff_cache/
```

- [ ] **Step 5: Create package init files and directory stubs**

`src/agent_studies/__init__.py` — empty file.
`tests/__init__.py` — empty file.

Also create:
- `examples/` directory
- `notes/.gitkeep`

- [ ] **Step 6: Run uv sync**

```bash
uv sync --extra dev
```
Expected: resolves and installs all dependencies with no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/ tests/ examples/ notes/
git commit -m "chore: scaffold project with uv"
```

---

### Task 2: Define typed agent state

**Files:**
- Create: `src/agent_studies/state.py`

- [ ] **Step 1: Write the failing import test**

In `tests/test_tools.py`, add at the top (we'll build on this file):
```python
from agent_studies.state import AgentState
```
Run: `uv run pytest tests/test_tools.py -v`
Expected: ImportError (module doesn't exist yet).

- [ ] **Step 2: Create state.py**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

- [ ] **Step 3: Verify import resolves**

```bash
uv run python -c "from agent_studies.state import AgentState; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/agent_studies/state.py
git commit -m "feat: add typed agent state"
```

---

### Task 3: Implement three tools with safe eval

**Files:**
- Create: `src/agent_studies/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests first**

Full content of `tests/test_tools.py`:

```python
import os
import pytest
from pathlib import Path


# ── search_docs ──────────────────────────────────────────────────────────────

def test_search_docs_finds_match(tmp_path, monkeypatch):
    """search_docs returns snippet when query matches a .md file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text("LangGraph uses a StateGraph to define the flow of messages.")
    monkeypatch.setattr("agent_studies.tools.DOCS_DIR", docs_dir)

    from agent_studies.tools import search_docs
    result = search_docs.invoke({"query": "stategraph"})
    assert "StateGraph" in result
    assert "sample.md" in result


def test_search_docs_no_match(tmp_path, monkeypatch):
    """search_docs returns explicit no-match string when query misses."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "empty.md").write_text("nothing relevant here")
    monkeypatch.setattr("agent_studies.tools.DOCS_DIR", docs_dir)

    from agent_studies.tools import search_docs
    result = search_docs.invoke({"query": "xyzzy_not_found_12345"})
    assert "no matches found" in result.lower()


# ── calculate ─────────────────────────────────────────────────────────────────

def test_calculate_basic():
    from agent_studies.tools import calculate
    result = calculate.invoke({"expression": "2 + 3 * 4"})
    assert result == "14"


def test_calculate_rejects_import():
    from agent_studies.tools import calculate
    result = calculate.invoke({"expression": "__import__('os')"})
    assert result.startswith("Error:")


def test_calculate_rejects_exec():
    from agent_studies.tools import calculate
    result = calculate.invoke({"expression": "exec('import os')"})
    assert result.startswith("Error:")


# ── save_note ─────────────────────────────────────────────────────────────────

def test_save_note_creates_file(tmp_path, monkeypatch):
    """save_note writes slugified file into notes dir."""
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("agent_studies.tools.NOTES_DIR", notes_dir)

    from agent_studies.tools import save_note
    result = save_note.invoke({"title": "My Test Note", "content": "hello world"})
    expected_path = notes_dir / "my-test-note.md"
    assert expected_path.exists()
    assert expected_path.read_text() == "hello world"
    assert "my-test-note.md" in result


def test_save_note_slugifies(tmp_path, monkeypatch):
    """save_note handles special chars in title."""
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("agent_studies.tools.NOTES_DIR", notes_dir)

    from agent_studies.tools import save_note
    save_note.invoke({"title": "Hello!! World  2024", "content": "test"})
    assert (notes_dir / "hello--world--2024.md").exists() or any(
        f.name.startswith("hello") for f in notes_dir.iterdir()
    )
```

Run: `uv run pytest tests/test_tools.py -v`
Expected: ImportError / multiple failures.

- [ ] **Step 2: Create tools.py**

```python
import ast
import re
from pathlib import Path
from langchain_core.tools import tool

# Resolved at import time; tests monkeypatch these module-level variables.
DOCS_DIR: Path = Path(__file__).resolve().parents[3] / "docs"
NOTES_DIR: Path = Path(__file__).resolve().parents[3] / "notes"

_CONTEXT_WINDOW = 300
_MAX_RESULTS = 3

# Allowed AST node types for safe math eval
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


@tool
def search_docs(query: str) -> str:
    """Search local markdown docs for a query string. Returns up to 3 snippets."""
    query_lower = query.lower()
    results: list[str] = []

    if not DOCS_DIR.exists():
        return "Error: docs directory not found"

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        idx = text.lower().find(query_lower)
        if idx == -1:
            continue
        start = max(0, idx - _CONTEXT_WINDOW // 2)
        end = min(len(text), idx + _CONTEXT_WINDOW // 2)
        snippet = text[start:end].replace("\n", " ")
        results.append(f"[{md_file.name}] ...{snippet}...")
        if len(results) >= _MAX_RESULTS:
            break

    if not results:
        return f"no matches found for '{query}'"
    return "\n\n".join(results)


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Supports + - * / // % **."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid syntax — {exc}"

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return f"Error: disallowed operation '{type(node).__name__}' in expression"

    try:
        result = eval(compile(tree, "<string>", "eval"))  # noqa: S307 — AST-validated
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


@tool
def save_note(title: str, content: str) -> str:
    """Save a note as a markdown file in the notes/ directory."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    if not slug:
        return "Error: title produces empty slug"
    path = NOTES_DIR / f"{slug}.md"
    try:
        path.write_text(content, encoding="utf-8")
        return str(path)
    except OSError as exc:
        return f"Error: {exc}"
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_tools.py -v
```
Expected: all 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/agent_studies/tools.py tests/test_tools.py
git commit -m "feat: add three tools with safe eval"
```

---

### Task 4: Add SQLite checkpointer factory

**Files:**
- Create: `src/agent_studies/memory.py`

- [ ] **Step 1: Create memory.py**

`SqliteSaver.from_conn_string` is a context manager; expose a helper that yields so callers can use `with get_checkpointer() as cp:`.

```python
from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def get_checkpointer(db_path: str = "agent_memory.sqlite"):
    """Yield a SqliteSaver checkpointer. Use as a context manager."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from agent_studies.memory import get_checkpointer; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/agent_studies/memory.py
git commit -m "feat: add sqlite checkpointer"
```

---

### Task 5: Assemble ReAct agent graph

**Files:**
- Create: `src/agent_studies/agent.py`

- [ ] **Step 1: Create agent.py**

```python
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent_studies.state import AgentState
from agent_studies.tools import calculate, save_note, search_docs

_TOOLS = [search_docs, calculate, save_note]
_LLM = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).bind_tools(_TOOLS)


def _call_model(state: AgentState) -> dict:
    response = _LLM.invoke(state["messages"])
    return {"messages": [response]}


def build_agent(checkpointer):
    """Compile and return the ReAct agent graph with the given checkpointer."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", _call_model)
    builder.add_node("tools", ToolNode(_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 2: Smoke-test import (no API call)**

```bash
uv run python -c "from agent_studies.agent import build_agent; print('OK')"
```
Expected: `OK` (LLM is instantiated but not called).

- [ ] **Step 3: Commit**

```bash
git add src/agent_studies/agent.py
git commit -m "feat: assemble react agent graph"
```

---

### Task 6: Add interactive CLI chat example

**Files:**
- Create: `examples/run_chat.py`

- [ ] **Step 1: Create run_chat.py**

```python
"""Interactive CLI chat with the LangGraph ReAct agent."""
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent_studies.agent import build_agent
from agent_studies.memory import get_checkpointer

_THREAD_CONFIG = {"configurable": {"thread_id": "cli-session"}}
_DB_PATH = "agent_memory.sqlite"
_BANNER = "langgraph-agent-studies | ReAct agent with SQLite memory | type 'exit' to quit"


def main() -> None:
    print(_BANNER)
    with get_checkpointer(_DB_PATH) as checkpointer:
        agent = build_agent(checkpointer)
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if user_input.lower() in {"exit", "quit"}:
                print("Bye.")
                break
            if not user_input:
                continue
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=_THREAD_CONFIG,
            )
            last = result["messages"][-1]
            print(f"\nAgent: {last.content}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it launches without error (no API call)**

```bash
echo "exit" | uv run python examples/run_chat.py
```
Expected: prints banner then exits cleanly. If `ANTHROPIC_API_KEY` is missing, it may raise on the LLM import — that's acceptable; the test for this is the integration test.

- [ ] **Step 3: Commit**

```bash
git add examples/run_chat.py
git commit -m "feat: add interactive chat example"
```

---

### Task 7: Add integration test

**Files:**
- Create: `tests/test_agent.py`

- [ ] **Step 1: Create test_agent.py**

```python
import os
import pytest
from langchain_core.messages import HumanMessage
from agent_studies.agent import build_agent
from agent_studies.memory import get_checkpointer


@pytest.mark.integration
def test_agent_calculates_17_times_23():
    """End-to-end: agent uses the calculate tool and returns 391."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    with get_checkpointer(":memory:") as checkpointer:
        agent = build_agent(checkpointer)
        result = agent.invoke(
            {"messages": [HumanMessage(content="What is 17 times 23?")]},
            config={"configurable": {"thread_id": "test-session"}},
        )

    last_message = result["messages"][-1]
    assert "391" in last_message.content
```

- [ ] **Step 2: Run unit tests only (no API)**

```bash
uv run pytest -m "not integration" -v
```
Expected: all tests pass (test_agent.py has no non-integration tests, so only test_tools.py runs).

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent.py
git commit -m "test: add tool unit tests and integration test"
```

---

### Task 8: Write docs (study notes)

**Files:**
- Create: `docs/langgraph_basics.md`
- Create: `docs/multiagent_patterns.md`

- [ ] **Step 1: Create langgraph_basics.md**

```markdown
# LangGraph Basics

LangGraph models agent logic as a directed graph. The two core abstractions are
**nodes** (Python callables that receive state and return a state patch) and
**edges** (explicit routing between nodes). State is a TypedDict with reducer
annotations; reducers define how each field merges incoming updates — `add_messages`
appends rather than overwrites, which is what you want for a conversation.

Conditional edges let you inspect the state after a node runs and choose the next
node dynamically. The canonical use: check whether the last message has tool calls,
route to the tool executor if yes, exit if no. `tools_condition` from
`langgraph.prebuilt` is a pre-built function for exactly this.

Checkpointers serialize the full state to a store (SQLite, Postgres, Redis) after
every step. `SqliteSaver` is the local option — fine for development and small
deployments. The checkpointer is passed at `compile()` time; the caller then passes
`thread_id` in the invoke config to identify which conversation to load or continue.

The graph compiles to an executable via `builder.compile(checkpointer=...)`.
Streaming is built in: swap `invoke` for `stream` to get intermediate node outputs
as a generator.
```

- [ ] **Step 2: Create multiagent_patterns.md**

```markdown
# Multi-Agent Patterns

Three patterns cover most production multi-agent needs.

**Supervisor.** A single orchestrator node receives the task, decides which
specialist agent to call next, and synthesizes the result. Good when tasks are
heterogeneous and you want a single point of control. The supervisor becomes a
bottleneck and single point of failure — worth it when routing logic is complex
and specialists don't need to communicate directly.

**Swarm.** Agents hand off to each other peer-to-peer based on local decisions.
No central router. Works well for pipelines where each stage knows which stage
should follow. Harder to debug because control flow is distributed. LangGraph's
`Command` primitive (return a `Command(goto="next_agent")` from a node) is the
natural implementation.

**Hierarchical.** Supervisors that themselves delegate to sub-supervisors. Useful
when the task space is large enough that one router can't hold all the routing logic
in its context. Depth increases latency and token cost at each level — only add
a tier when the routing problem genuinely needs it.

Rule of thumb: start with a supervisor, flatten to a swarm when the supervisor
becomes a routing table, add hierarchy only when the swarm graph becomes unreadable.
```

- [ ] **Step 3: Verify search_docs finds content**

```bash
uv run python -c "
from agent_studies.tools import search_docs
print(search_docs.invoke({'query': 'checkpointer'}))
"
```
Expected: returns a snippet from one of the docs files.

- [ ] **Step 4: Commit**

```bash
git add docs/langgraph_basics.md docs/multiagent_patterns.md
git commit -m "docs: add study notes on langgraph basics and multiagent patterns"
```

---

### Task 9: Write README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md** with the full template from the spec (ASCII diagram included).

- [ ] **Step 2: Verify no broken markdown** by reading it back.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run all unit tests**

```bash
uv run pytest -m "not integration" -v
```
Expected: all green, zero failures.

- [ ] **Step 2: Run integration test if API key is set**

```bash
uv run pytest -m integration -v
```
Expected: passes (or skipped if key absent).

- [ ] **Step 3: Verify CLI launches**

```bash
echo "exit" | uv run python examples/run_chat.py
```
Expected: banner printed, clean exit.

- [ ] **Step 4: Check git log**

```bash
git log --oneline
```
Expected: 9+ semantic commits visible.
