import threading
from importlib import import_module
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tree_sitter import Language, Parser, Query, QueryCursor

# Relative imports are used to resolve correctly for IDEs (like Pyrefly or Pylance) inside the package.
from .ast_utility import build_qualified_name, resolve_enclosing_function
from ...file_extension import EXTENSIONS_MAP


LANGUAGE_MODULES: dict[str, tuple[str, str, str]] = {
    "python": ("tree_sitter_python", "tree-sitter-python", "language"),
    "javascript": ("tree_sitter_javascript", "tree-sitter-javascript", "language"),
    "typescript": ("tree_sitter_typescript", "tree-sitter-typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "tree-sitter-typescript", "language_tsx"),
    "java": ("tree_sitter_java", "tree-sitter-java", "language"),
    "go": ("tree_sitter_go", "tree-sitter-go", "language"),
    "c": ("tree_sitter_c", "tree-sitter-c", "language"),
    "cpp": ("tree_sitter_cpp", "tree-sitter-cpp", "language"),
    "rust": ("tree_sitter_rust", "tree-sitter-rust", "language"),
    "php": ("tree_sitter_php", "tree-sitter-php", "language"),
    "html": ("tree_sitter_html", "tree-sitter-html", "language"),
    "json": ("tree_sitter_json", "tree-sitter-json", "language"),
    "css": ("tree_sitter_css", "tree-sitter-css", "language"),
    # New languages
    "c_sharp": ("tree_sitter_c_sharp", "tree-sitter-c-sharp", "language"),
    "swift": ("tree_sitter_swift", "tree-sitter-swift", "language"),
    "kotlin": ("tree_sitter_kotlin", "tree-sitter-kotlin", "language"),
    "bash": ("tree_sitter_bash", "tree-sitter-bash", "language"),
    # "dart": ("tree_sitter_dart_orchard", "tree-sitter-dart-orchard", "language"),
}


class CodeParser:
    def __init__(self, default_extensions: str = ".py"):
        self.default_extensions = default_extensions
        self.query_dir = Path(__file__).parent / "queries"
        self.default_language_name = EXTENSIONS_MAP.get(default_extensions, "python")

        # Language objects are immutable after creation, making them safe to share across threads.
        self._language_cache: dict[str, object] = {}
        self._language_lock = threading.Lock()

        # Parser objects hold mutable C-level state such as incremental parse buffers.
        # Since they are not thread-safe, we use threading.local() so each worker
        # thread in the ThreadPoolExecutor gets its own Parser instance.
        self._thread_local = threading.local()

        # Query objects are read-only once compiled, making them safe to share.
        # However, compilation itself is not thread-safe, so it is protected with a lock.
        self._query_cache: dict[str, Query] = {}
        self._query_lock = threading.Lock()

        self.supported_languages = set(LANGUAGE_MODULES.keys())
        self.supported_languages.update({p.stem for p in self.query_dir.glob("*.scm")})
        self.default_language = self._get_language(self.default_language_name)
        self.default_parser = self._get_parser(self.default_language_name)

    def _get_language(self, language_name: str):
        # Double-checked locking is used here: the fast path avoids the lock on a cache hit.
        if language_name in self._language_cache:
            return self._language_cache[language_name]

        with self._language_lock:
            # Re-check inside the lock in case another thread populated it.
            if language_name in self._language_cache:
                return self._language_cache[language_name]

            module_entry = LANGUAGE_MODULES.get(language_name)
            if module_entry is None:
                module_name = ""
                package_name = ""
                language_attr = "language"
            else:
                module_name, package_name, language_attr = module_entry
            if not module_name:
                raise ValueError(f"No official tree-sitter binding configured for '{language_name}'")

            try:
                language_module = import_module(module_name)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"Missing dependency '{package_name}' for language '{language_name}'. "
                    f"Install it to enable parsing for this language."
                ) from exc

            fallback_attrs = [
                language_attr,
                "language",
                f"language_{language_name}",
            ]
            if language_name in {"typescript", "tsx"}:
                fallback_attrs.extend(["language_typescript", "language_tsx"])

            language_fn = None
            for attr in fallback_attrs:
                if attr and hasattr(language_module, attr):
                    language_fn = getattr(language_module, attr)
                    break

            if language_fn is None:
                raise RuntimeError(
                    f"Language factory not found for '{language_name}'. "
                    f"Tried attributes: {', '.join(attr for attr in fallback_attrs if attr)}"
                )

            self._language_cache[language_name] = Language(language_fn())
            return self._language_cache[language_name]

    def _get_parser(self, language_name: str):
        """
        Return a Parser for the specified language that is private to the current thread.

        Tree-sitter's Parser holds mutable C-level state for its incremental parse buffers. 
        Sharing a single Parser across ThreadPoolExecutor workers causes state corruption.

        By storing parsers in threading.local(), each worker thread lazily creates 
        its own set of Parser instances.
        """
        # Each thread has its own parser dictionary stored inside self._thread_local.
        thread_parsers: dict = getattr(self._thread_local, "_parsers", None)
        if thread_parsers is None:
            thread_parsers = {}
            self._thread_local._parsers = thread_parsers

        if language_name not in thread_parsers:
            parser = Parser()
            language = self._get_language(language_name)
            if hasattr(parser, "set_language"):
                parser.set_language(language)
            else:
                parser.language = language
            thread_parsers[language_name] = parser
        return thread_parsers[language_name]

    def _query_captures(self, language: Language, query_scm: str, root_node):
        if not query_scm.strip():
            return []

        # Query objects are read-only after compilation and can be shared.
        # Since two threads might try to compile the same query simultaneously,
        # the cache-miss path is protected with a lock.
        cache_key = f"{id(language)}:{hash(query_scm)}"
        query = self._query_cache.get(cache_key)
        if query is None:
            with self._query_lock:
                query = self._query_cache.get(cache_key)
                if query is None:
                    query = Query(language, query_scm)
                    self._query_cache[cache_key] = query

        cursor = QueryCursor(query)
        captures: list[tuple[object, str]] = []
        for _pattern_index, match_captures in cursor.matches(root_node):
            for capture_name, nodes in match_captures.items():
                for node in nodes:
                    captures.append((node, capture_name))
        return captures

    def _load_query(self, language_name: str) -> str:
        """Load a .scm query file from the queries directory."""
        query_path = self.query_dir / f"{language_name}.scm"
        if not query_path.exists():
            return ""
        # Explicit encoding is added to avoid platform-dependent decoding issues.
        return query_path.read_text(encoding="utf-8")

    def language_for_file(self, file_path: str) -> Optional[str]:
        language = EXTENSIONS_MAP.get(Path(file_path).suffix.lower())
        if language in self.supported_languages:
            return language
        return None

    def iter_supported_files(self, files: Iterable[str]) -> Iterator[tuple[str, str]]:
        # Added missing space in type annotation
        for file_path in files:
            language_name = self.language_for_file(file_path)
            if language_name:
                yield file_path, language_name

    def _symbol_type_from_capture(self, capture_name: str, parent_type: str) -> str:
        if capture_name.startswith("class."):
            return "class"
        if capture_name.startswith("function."):
            return "function"
        if capture_name.startswith("symbol."):
            lowered = parent_type.lower()
            class_hints = ("class", "struct", "interface", "enum", "type")
            return "class" if any(hint in lowered for hint in class_hints) else "function"
        lowered = parent_type.lower()
        class_hints = ("class", "struct", "interface", "enum", "type")
        return "class" if any(hint in lowered for hint in class_hints) else "function"

    def _is_symbol_capture(self, capture_name: str) -> bool:
        # Explicitly exclude call.name — it ends with '.name' but is a call capture.
        return capture_name.endswith(".name") and not capture_name.startswith("call.")

    def _is_call_capture(self, capture_name: str) -> bool:
        return capture_name == "call.name" or capture_name.endswith(".call")

    def extract_symbols_and_calls(self, code: str, language_name: Optional[str] = None):
        """
        Acts as a bridge between the raw code text and the relation graph.
        Used to determine what is defined in the code and where it is called.

        Converts raw source code into structured relationships and returns symbols and calls.
        """
        language_name = language_name or self.default_language_name
        parser = self._get_parser(language_name)
        language = self._get_language(language_name)

        source_bytes = code.encode("utf-8")
        tree = parser.parse(source_bytes)

        query_scm = self._load_query(language_name)

        symbols = []
        calls = []

        captures = self._query_captures(language, query_scm, tree.root_node)

        for node, capture_name in captures:
            # Extract symbols
            if self._is_symbol_capture(capture_name):
                definition_node = node.parent or node
                symbol_type = self._symbol_type_from_capture(capture_name, definition_node.type)
                if symbol_type == "function":
                    qualified_name = build_qualified_name(
                        definition_node,
                        source_bytes,
                        language_name=language_name,
                    )
                else:
                    qualified_name = node.text.decode("utf8")

                symbols.append({
                    "name": node.text.decode("utf8"),
                    "qualified_name": qualified_name,
                    "type": symbol_type,
                    "start_line": node.start_point[0] + 1,
                    "byte_range": (definition_node.start_byte, definition_node.end_byte)
                })

            # Extract call relations
            elif self._is_call_capture(capture_name):
                callee = node.text.decode("utf8")

                # Find which function this call belongs to
                caller = resolve_enclosing_function(node, source_bytes, language_name=language_name)

                if caller:
                    calls.append({
                        "caller": caller,
                        "callee": callee,
                        "line": node.start_point[0] + 1
                    })

        return symbols, calls
