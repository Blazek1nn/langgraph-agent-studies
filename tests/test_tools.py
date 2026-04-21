import pytest
from pathlib import Path


# ── search_docs ───────────────────────────────────────────────────────────────

def test_search_docs_finds_match(tmp_path, monkeypatch):
    """search_docs returns snippet when query matches a .md file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "LangGraph uses a StateGraph to define the flow of messages."
    )
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
    assert any(f.name.startswith("hello") for f in notes_dir.iterdir())
