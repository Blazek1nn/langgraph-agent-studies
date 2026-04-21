import ast
import re
from pathlib import Path
from langchain_core.tools import tool

# Resolved at import time; tests monkeypatch these module-level variables.
DOCS_DIR: Path = Path(__file__).resolve().parents[2] / "docs"
NOTES_DIR: Path = Path(__file__).resolve().parents[2] / "notes"

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
