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
