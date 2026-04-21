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
