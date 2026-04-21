from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def get_checkpointer(db_path: str = "agent_memory.sqlite"):
    """Yield a SqliteSaver checkpointer. Use as a context manager."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer
