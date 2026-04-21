import os
import pytest
from langchain_core.messages import HumanMessage
from agent_studies.agent import build_agent
from agent_studies.memory import get_checkpointer


@pytest.mark.integration
def test_agent_calculates_17_times_23():
    """End-to-end: agent uses the calculate tool and returns 391."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    with get_checkpointer(":memory:") as checkpointer:
        agent = build_agent(checkpointer)
        result = agent.invoke(
            {"messages": [HumanMessage(content="What is 17 times 23?")]},
            config={"configurable": {"thread_id": "test-session"}},
        )

    last_message = result["messages"][-1]
    assert "391" in last_message.content
