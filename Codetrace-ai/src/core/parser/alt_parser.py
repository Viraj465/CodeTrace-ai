"""
alt_parser.py — Alternative parsers for config/data languages that do not have
a stable Tree-sitter Python binding compatible with tree-sitter>=0.25.

Supported languages and their strategies:
  - YAML      → PyYAML  (parses into a Python dict; walks keys as symbols)
  - TOML      → tomllib (built-in, Python 3.11+; walks keys as symbols)
  - SQL       → sqlglot  (full multi-dialect AST; extracts table names and CTEs)
  - Dockerfile → Regex/line-by-line (INSTRUCTION-based; no call graph needed)
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: extension → language name (mirrors EXTENSIONS_MAP in file_extension.py)
# ---------------------------------------------------------------------------

ALT_EXTENSIONS_MAP: dict[str, str] = {
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".toml": "toml",
    ".sql":  "sql",
    # Dockerfile has no extension; handled by basename matching in language_for_file
}

# Dockerfile basenames that should trigger dockerfile parsing
DOCKERFILE_BASENAMES: frozenset[str] = frozenset({
    "dockerfile",
    "dockerfile.dev",
    "dockerfile.prod",
    "dockerfile.test",
})


def language_for_file(file_path: str) -> Optional[str]:
    """Return the alt-parser language name for a file, or None if not supported."""
    p = Path(file_path)
    # Check extension first
    lang = ALT_EXTENSIONS_MAP.get(p.suffix.lower())
    if lang:
        return lang
    # Check for Dockerfile by basename (case-insensitive)
    if p.name.lower() in DOCKERFILE_BASENAMES:
        return "dockerfile"
    return None


# ---------------------------------------------------------------------------
# Shared output shape helpers
# ---------------------------------------------------------------------------

def _make_symbol(name: str, symbol_type: str, line: int) -> dict:
    """Produce a symbol dict compatible with CodeParser.extract_symbols_and_calls output."""
    return {
        "name": name,
        "qualified_name": name,
        "type": symbol_type,
        "start_line": line,
        # byte_range is not available for alt parsers; use (0, 0) as sentinel.
        "byte_range": (0, 0),
    }


# ---------------------------------------------------------------------------
# YAML parser  (PyYAML)
# ---------------------------------------------------------------------------

def _parse_yaml(content: str) -> tuple[list[dict], list[dict]]:
    """
    Walk the top-level keys of a YAML document as "property" symbols.
    No call graph is produced (config files have no call semantics).
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "PyYAML is not installed. Install it with: pip install pyyaml\n"
            "YAML files will not be indexed."
        )
        return [], []

    symbols: list[dict] = []
    try:
        docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError as exc:
        logger.debug("YAML parse error: %s", exc)
        return [], []

    for doc in docs:
        if isinstance(doc, dict):
            for key in doc:
                symbols.append(_make_symbol(str(key), "property", 1))

    return symbols, []


# ---------------------------------------------------------------------------
# TOML parser  (tomllib — stdlib in Python 3.11+)
# ---------------------------------------------------------------------------

def _parse_toml(content: str) -> tuple[list[dict], list[dict]]:
    """
    Walk the top-level section names of a TOML document as "property" symbols.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore  # backport for 3.10
        except ImportError:
            logger.warning(
                "tomllib is unavailable and tomli is not installed.\n"
                "Install the backport with: pip install tomli\n"
                "TOML files will not be indexed."
            )
            return [], []

    symbols: list[dict] = []
    try:
        data = tomllib.loads(content)
    except Exception as exc:
        logger.debug("TOML parse error: %s", exc)
        return [], []

    for key in data:
        symbols.append(_make_symbol(key, "property", 1))

    return symbols, []


# ---------------------------------------------------------------------------
# SQL parser  (sqlglot)
# ---------------------------------------------------------------------------

def _parse_sql(content: str) -> tuple[list[dict], list[dict]]:
    """
    Use sqlglot to extract:
      - Table names referenced in FROM / JOIN clauses  → "class" symbols
      - CTE names (WITH … AS)                         → "function" symbols
      - Stored procedure / function names (CREATE …)  → "function" symbols
    Calls are modelled as CTE → table references so the graph gets useful edges.
    """
    try:
        import sqlglot  # type: ignore
        import sqlglot.expressions as exp
    except ImportError:
        logger.warning(
            "sqlglot is not installed. Install it with: pip install sqlglot\n"
            "SQL files will not be indexed."
        )
        return [], []

    symbols: list[dict] = []
    calls: list[dict] = []

    try:
        statements = sqlglot.parse(content, error_level=sqlglot.ErrorLevel.WARN)
    except Exception as exc:
        logger.debug("SQL parse error: %s", exc)
        return [], []

    for stmt in statements:
        if stmt is None:
            continue

        # --- CTEs: WITH cte_name AS (...) ---
        for cte in stmt.find_all(exp.CTE):
            cte_name = cte.alias
            if cte_name:
                line = getattr(cte, "line", 1) or 1
                symbols.append(_make_symbol(cte_name, "function", line))

        # --- CREATE TABLE / VIEW ---
        for create in stmt.find_all(exp.Create):
            tbl = create.find(exp.Table)
            if tbl and tbl.name:
                line = getattr(create, "line", 1) or 1
                symbols.append(_make_symbol(tbl.name, "class", line))

        # --- CREATE PROCEDURE / FUNCTION ---
        for func_def in stmt.find_all(exp.Anonymous):
            name = func_def.name
            if name:
                line = getattr(func_def, "line", 1) or 1
                symbols.append(_make_symbol(name, "function", line))

        # --- Table references (FROM / JOIN) → treated as "calls" to those tables ---
        for table in stmt.find_all(exp.Table):
            if table.name and not any(s["name"] == table.name for s in symbols):
                line = getattr(table, "line", 1) or 1
                symbols.append(_make_symbol(table.name, "class", line))

    return symbols, calls


# ---------------------------------------------------------------------------
# Dockerfile parser  (regex / line-by-line)
# ---------------------------------------------------------------------------

# Dockerfile instructions that are semantically meaningful for indexing
_DOCKERFILE_SYMBOL_INSTRUCTIONS = frozenset({
    "FROM", "ARG", "ENV", "LABEL", "EXPOSE", "VOLUME", "ENTRYPOINT", "CMD",
})
_DOCKERFILE_CALL_INSTRUCTIONS = frozenset({"RUN", "COPY", "ADD"})

_INSTRUCTION_RE = re.compile(
    r"^\s*(?P<instruction>[A-Z]+)\s+(?P<rest>.+)", re.IGNORECASE
)


def _parse_dockerfile(content: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a Dockerfile line-by-line:
      - FROM, ARG, ENV, LABEL … → symbols (type "property")
      - RUN, COPY, ADD          → calls (modelled as shell command invocations)
    """
    symbols: list[dict] = []
    calls: list[dict] = []
    current_stage: Optional[str] = None

    for lineno, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        # Skip comments and blank lines
        if not line or line.startswith("#"):
            continue
        # Skip continuation lines
        if line == "\\":
            continue

        m = _INSTRUCTION_RE.match(line)
        if not m:
            continue

        instruction = m.group("instruction").upper()
        rest = m.group("rest").strip()

        if instruction == "FROM":
            # FROM <image> [AS <stage>]
            parts = rest.split()
            image = parts[0]
            stage = None
            if len(parts) >= 3 and parts[1].upper() == "AS":
                stage = parts[2]
                current_stage = stage
            name = stage or image
            symbols.append(_make_symbol(name, "class", lineno))

        elif instruction in _DOCKERFILE_SYMBOL_INSTRUCTIONS:
            # Capture the first token of the value as the symbol name
            name = rest.split()[0] if rest.split() else rest
            symbols.append(_make_symbol(f"{instruction}:{name}", "property", lineno))

        elif instruction in _DOCKERFILE_CALL_INSTRUCTIONS:
            # Model RUN/COPY/ADD as a call from the current build stage
            caller = current_stage or "global"
            # Extract the first command token for RUN
            if instruction == "RUN":
                cmd = rest.split()[0] if rest.split() else rest
                calls.append({
                    "caller": caller,
                    "callee": cmd,
                    "line": lineno,
                })
            else:
                calls.append({
                    "caller": caller,
                    "callee": f"{instruction}:{rest[:60]}",
                    "line": lineno,
                })

    return symbols, calls


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

_PARSERS = {
    "yaml":       _parse_yaml,
    "toml":       _parse_toml,
    "sql":        _parse_sql,
    "dockerfile": _parse_dockerfile,
}


def extract_symbols_and_calls(
    content: str, language_name: str
) -> tuple[list[dict], list[dict]]:
    """
    Parse *content* using the alternative parser registered for *language_name*.

    Returns (symbols, calls) using the same shape as CodeParser.extract_symbols_and_calls.
    Raises ValueError for unsupported language names.
    """
    parser_fn = _PARSERS.get(language_name)
    if parser_fn is None:
        raise ValueError(
            f"No alternative parser registered for language '{language_name}'. "
            f"Supported: {sorted(_PARSERS)}"
        )
    return parser_fn(content)
