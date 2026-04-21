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
