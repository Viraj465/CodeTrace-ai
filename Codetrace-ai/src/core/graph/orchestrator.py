"""
Ties the parsers and the graph together — parses files and turns what comes out
into graph nodes/edges and vector-store entries.
"""

from typing import Iterable, Iterator, Optional
# Relative imports so IDEs resolve them correctly within the package.
from ..parser.parser import CodeParser
from ..parser import alt_parser
from .builder import CodeGraph

def _read_source(file_path: str) -> str:
    """
    Read a source file as text. UTF-8 (with or without BOM) covers nearly all
    code; anything else — typically a legacy Latin-1 / cp1252 file — falls back
    to Latin-1, which maps every byte, so the file still gets indexed instead
    of crashing the parse and dropping out of the graph.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# Languages that call into each other directly, so a callee may resolve across
# them. Anything not listed here only resolves within its own language.
_LANGUAGE_FAMILIES = {
    "javascript": "js", "typescript": "js", "tsx": "js",
    "c": "c", "cpp": "c",
}


class GraphOrchestrator:
    def __init__(self):
        self.parser = CodeParser()
        self.graph = CodeGraph()
        self._family_cache: dict[str, Optional[str]] = {}

    def _family_for_file(self, file_path: str) -> Optional[str]:
        """Language family used to keep call resolution from crossing languages."""
        if file_path not in self._family_cache:
            language, _ = self._language_for_file(file_path)
            self._family_cache[file_path] = (
                _LANGUAGE_FAMILIES.get(language, language) if language else None
            )
        return self._family_cache[file_path]

    def build_batch(
        self,
        all_symbols: list[tuple[str, dict]],
        all_calls: list[tuple[str, dict]],
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
        """
        Turn freshly parsed (file, symbol) and (file, call) pairs into graph
        node and edge batches, resolving each callee to a symbol ID.

        Resolution is order-independent — the result never depends on which
        parse thread finished first — and deliberately conservative:
          1. the caller's own class (``self.helper()`` → ``Service.helper``);
          2. a symbol with that name defined in the same file, if exactly one;
          3. a symbol with that name anywhere in the same language family, if
             exactly one.
        Anything else (ambiguous or unknown, e.g. ``list.append``) stays a
        bare-name node rather than being wired to an arbitrary guess.

        Candidates cover this batch plus every symbol already in the graph, so a
        call in a changed file still resolves into an unchanged file.
        """
        by_qualified: dict[str, set[str]] = {}
        by_simple: dict[str, set[str]] = {}
        node_file: dict[str, str] = {}

        def _register(node_id: str, file_path: str, qualified: str) -> None:
            node_file[node_id] = file_path
            by_qualified.setdefault(qualified, set()).add(node_id)
            by_simple.setdefault(qualified.rsplit(".", 1)[-1], set()).add(node_id)

        for node_id, data in self.graph.direct_graph.nodes(data=True):
            file_path = data.get("file")
            if not file_path or file_path == "unknown":
                continue  # unresolved bare-name nodes are not symbols
            _register(node_id, file_path, node_id.rsplit(":", 1)[-1])

        nodes_batch: list[tuple[str, str, str]] = []
        for file_path, s in all_symbols:
            qualified_name = s.get("qualified_name") or s["name"]
            symbol_id = f"{file_path}:{qualified_name}"
            nodes_batch.append((symbol_id, s["type"], file_path))
            _register(symbol_id, file_path, qualified_name)

        def _unique(candidates) -> Optional[str]:
            candidates = list(candidates)
            return candidates[0] if len(candidates) == 1 else None

        def _resolve(file_path: str, caller: str, callee: str) -> Optional[str]:
            # 1. Method on the caller's own class.
            if "." in caller:
                own = f"{file_path}:{caller.rsplit('.', 1)[0]}.{callee}"
                if own in node_file:
                    return own

            named = by_qualified.get(callee) or by_simple.get(callee) or set()
            if not named:
                return None

            # 2. Defined in the same file.
            same_file = _unique(n for n in named if node_file[n] == file_path)
            if same_file:
                return same_file

            # 3. Unique within the same language family.
            family = self._family_for_file(file_path)
            return _unique(
                n for n in named if self._family_for_file(node_file[n]) == family
            )

        edges_batch: list[tuple[str, str]] = []
        for file_path, c in all_calls:
            caller_id = f"{file_path}:{c['caller']}"
            callee_id = _resolve(file_path, c["caller"], c["callee"])
            edges_batch.append((caller_id, callee_id or c["callee"]))

        return nodes_batch, edges_batch

    def _language_for_file(self, file_path: str) -> tuple[Optional[str], bool]:
        """
        Returns (language_name, is_alt_parser).
        Checks the alt-parser registry first (YAML, TOML, SQL, Dockerfile),
        then falls back to the Tree-sitter registry.
        """
        alt_lang = alt_parser.language_for_file(file_path)
        if alt_lang:
            return alt_lang, True
        ts_lang = self.parser.language_for_file(file_path)
        return ts_lang, False

    def iter_supported_files(self, files: Iterable[str]) -> Iterator[tuple[str, str]]:
        """
        Yield (file_path, language_name) for every file we can parse — covering
        both the Tree-sitter registry and the alt-parser one (YAML, TOML, SQL,
        Dockerfile).

        The indexer has to gate on this rather than CodeParser.iter_supported_files,
        which only knows the Tree-sitter languages and would quietly drop every
        alt-parser file from the index.
        """
        for file_path in files:
            language_name, _is_alt = self._language_for_file(file_path)
            if language_name:
                yield file_path, language_name

    def extract_from_file(self, file_path: str) -> dict:
        """
        Extract and return symbols, calls, and vector store data 
        without modifying the graph or vector store.
        """
        code_file = _read_source(file_path)

        language_name, is_alt = self._language_for_file(file_path)
        if not language_name:
            return {"symbols": [], "calls": [], "vs_data": []}

        if is_alt:
            symbols, calls = alt_parser.extract_symbols_and_calls(
                code_file, language_name=language_name
            )
        else:
            symbols, calls = self.parser.extract_symbols_and_calls(
                code_file, language_name=language_name
            )

        source_bytes = code_file.encode("utf-8")
        vs_data = []
        for s in symbols:
            start_byte, end_byte = s["byte_range"]
            if start_byte == 0 and end_byte == 0:
                # Alt-parser symbols have no byte range, so index the whole file.
                content = code_file
            else:
                content = source_bytes[start_byte:end_byte].decode("utf-8")
            vs_data.append({
                "id": f"{file_path}:{s.get('qualified_name') or s['name']}",
                "content": content,
                "metadata": {
                    "file_path": file_path,
                    "symbol_name": s["name"],
                    "qualified_name": s.get("qualified_name") or s["name"],
                    "type": s["type"],
                    "start_line": s["start_line"]
                }
            })

        return {"symbols": symbols, "calls": calls, "vs_data": vs_data}
    
