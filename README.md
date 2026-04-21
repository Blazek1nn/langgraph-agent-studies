# langgraph-agent-studies

> 🇧🇷 [Português](#português) | 🇺🇸 [English](#english)

---

## Português

Um agente LangGraph mínimo mas com mentalidade de produção, construído para estudar os primitivos centrais do framework: **grafo**, **estado**, **ferramentas** e **persistência**.

> **Contexto.** Eu mantenho o [`multi-agent-core`](https://github.com/BlazeK1nn/multi-agent-core), um framework de orquestração multi-agente em Python construído **do zero, sem LangChain**, para entender os primitivos em profundidade. Este repositório é o exercício complementar: portando esse modelo mental para o LangGraph, demonstrando fluência no framework que a maioria dos times de produção adota.

### O que faz

Um agente no estilo ReAct com:

- Memória de conversa persistente via checkpointer `SqliteSaver`
- Três ferramentas: busca em documentos (corpus markdown local), calculadora segura e persistência de notas
- Estado estruturado com reducers tipados (`TypedDict` + `add_messages`)
- Testes unitários para cada ferramenta e um teste de integração end-to-end

### Arquitetura

```
          ┌─────────────────────────────────────┐
          │           SqliteSaver               │
          │        (agent_memory.sqlite)        │
          └──────────────┬──────────────────────┘
                         │ checkpoint
                         ▼
  user ──► [ nó agente ] ──tem tool calls?──► [ nó ferramentas ]
                 ▲                                    │
                 └────────────────────────────────────┘
                              loop back
                         │
                    sem tool calls
                         │
                         ▼
                        END
```

### Início rápido

```bash
git clone https://github.com/Blazek1nn/langgraph-agent-studies
cd langgraph-agent-studies
cp .env.example .env  # adicione sua GROQ_API_KEY
uv sync --extra dev
uv run python examples/run_chat.py
```

### Testes

```bash
uv run --extra dev pytest -m "not integration"   # testes unitários rápidos
uv run --extra dev pytest -m integration         # end-to-end (requer GROQ_API_KEY)
```

### Decisões de design

- **SQLite em vez de checkpointer in-memory** — mantém conversas entre reinicializações, alinhado com como trato persistência no `multi-agent-core`.
- **Groq (Llama 3.3-70b) em vez de OpenAI** — API gratuita, latência baixa, suporte robusto a tool calling.
- **Estado com type hints** — detecta drift de schema cedo, quando o grafo cresce.
- **Safe eval baseado em `ast`** — nunca `eval()` em input do usuário, mesmo dentro de uma ferramenta.

### O que eu adicionaria depois

- Backend de vector store (pgvector ou Qdrant) para `search_docs`
- Integração com LangGraph Studio para debug visual
- LangMem para memória de longo prazo além do checkpoint de conversa
- Observabilidade: traces OpenTelemetry em cada nó

### Por que construí isso

Estudar LangGraph lendo a documentação é uma coisa; escrever um agente funcional com estado persistente e testes é outra. Este repositório é a segunda coisa.

---

## English

A minimal but production-minded LangGraph agent built to study the core primitives of the framework: **graph**, **state**, **tools**, and **persistence**.

> **Context.** I maintain [`multi-agent-core`](https://github.com/BlazeK1nn/multi-agent-core), a from-scratch Python multi-agent orchestration framework built **without LangChain**, to understand the primitives in depth. This repo is the complementary exercise: porting that mental model into LangGraph to demonstrate fluency with the framework most production teams adopt.

### What it does

A ReAct-style agent with:

- Persistent conversation memory via `SqliteSaver` checkpointer
- Three tools: document search (local markdown corpus), safe calculator, and note persistence
- Structured state with typed reducers (`TypedDict` + `add_messages`)
- Unit tests for each tool and one end-to-end integration test

### Architecture

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

### Quickstart

```bash
git clone https://github.com/Blazek1nn/langgraph-agent-studies
cd langgraph-agent-studies
cp .env.example .env  # add your GROQ_API_KEY
uv sync --extra dev
uv run python examples/run_chat.py
```

### Tests

```bash
uv run --extra dev pytest -m "not integration"   # fast unit tests
uv run --extra dev pytest -m integration         # end-to-end (requires GROQ_API_KEY)
```

### Design choices

- **SQLite over in-memory checkpointer** — keeps conversations across restarts, matches how I handle persistence in `multi-agent-core`.
- **Groq (Llama 3.3-70b) instead of OpenAI** — free API tier, low latency, robust tool calling support.
- **Type-hinted state** — catches schema drift early when the graph grows.
- **`ast`-based safe eval** — never `eval()` user input, even inside a tool.

### What I'd add next

- Vector store backend (pgvector or Qdrant) for `search_docs`
- LangGraph Studio integration for visual debugging
- LangMem for long-term memory beyond the conversation checkpoint
- Observability: OpenTelemetry traces on each node

### Why I built this

Studying LangGraph by reading the docs is one thing; writing a working agent with persistent state and tests is another. This repo is the second thing.
