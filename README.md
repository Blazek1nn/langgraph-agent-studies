# langgraph-agent-studies

A minimal but production-minded LangGraph agent built to study the core
primitives of the framework: **graph**, **state**, **tools**, and
**persistence**. Uses Anthropic Claude as the LLM backbone.

> **Context.** I maintain [`multi-agent-core`](https://github.com/BlazeK1nn/multi-agent-core),
> a from-scratch Python multi-agent orchestration framework built **without**
> LangChain, to understand the primitives in depth. This repo is the
> complementary exercise: porting that mental model into LangGraph to
> demonstrate fluency with the framework most production teams adopt.

## What it does

A ReAct-style agent with:

- Persistent conversation memory via `SqliteSaver` checkpointer
- Three tools: document search (local markdown corpus), safe calculator,
  and note persistence
- Structured state with typed reducers (`TypedDict` + `add_messages`)
- Unit tests for each tool and one end-to-end integration test

## Architecture

```
          ┌─────────────────────────────────────┐
          │           SqliteSaver               │
          │        (agent_memory.sqlite)        │
          └──────────────┬──────────────────────┘
                         │ checkpoint
                         ▼
  user ──► [ agent node ] ──has tool calls?──► [ tools node ]
                 ▲                                    │
                 └────────────────────────────────────┘
                              loop back
                         │
                    no tool calls
                         │
                         ▼
                        END
```

## Quickstart

```bash
git clone https://github.com/BlazeK1nn/langgraph-agent-studies
cd langgraph-agent-studies
cp .env.example .env  # add your GROQ_API_KEY
uv sync
uv run python examples/run_chat.py
```

## Tests

```bash
uv run pytest -m "not integration"   # fast unit tests
uv run pytest -m integration         # end-to-end (requires API key)
```

## Design choices

- **SQLite over in-memory checkpointer** — keeps conversations across
  restarts, matches how I handle persistence in `multi-agent-core`.
- **Anthropic Claude instead of OpenAI** — I know Claude's behavior in depth
  from building production systems with it.
- **Type-hinted state** — catches schema drift early when the graph grows.
- **`ast`-based safe eval** — never `eval()` user input, even inside a tool.

## What I'd add next

- Vector store backend (pgvector or Qdrant) for `search_docs`
- LangGraph Studio integration for visual debugging
- LangMem for long-term memory beyond the conversation checkpoint
- Observability: OpenTelemetry traces on each node

## Why I built this

Studying LangGraph by reading the docs is one thing; writing a working
agent with persistent state and tests is another. This repo is the second
thing.
