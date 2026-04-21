from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent_studies.state import AgentState
from agent_studies.tools import calculate, save_note, search_docs

_TOOLS = [search_docs, calculate, save_note]


def build_agent(checkpointer):
    """Compile and return the ReAct agent graph with the given checkpointer."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(_TOOLS)

    def _call_model(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", _call_model)
    builder.add_node("tools", ToolNode(_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
