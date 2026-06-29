"""
Coordinates and maps the code systems.
"""

from typing import Optional, Any
# Use relative imports so IDEs resolve them correctly within the package.
from ..parser.parser import CodeParser
from ..parser import alt_parser
from .builder import CodeGraph

class GraphOrchestrator:
    def __init__(self):
        self.parser = CodeParser()
        self.graph = CodeGraph()

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

    def extract_from_file(self, file_path: str) -> dict:
        """
        Extract and return symbols, calls, and vector store data 
        without modifying the graph or vector store.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            code_file = f.read()

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
                # Alt-parser symbols: index the full file content as context
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
    
    def build_from_file(self, file_path: str, vector_store: Optional[Any] = None):
        """
        Extract code elements from the file and build the relation map.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            code_file = f.read()

        language_name, is_alt = self._language_for_file(file_path)
        if not language_name:
            return

        if is_alt:
            symbols, calls = alt_parser.extract_symbols_and_calls(
                code_file, language_name=language_name
            )
        else:
            symbols, calls = self.parser.extract_symbols_and_calls(
                code_file, language_name=language_name
            )
        
        # Step 1: Map definition nodes
        name_to_id: dict[str, str] = {}
        qualified_to_id: dict[str, str] = {}

        # Prepare batched lists for vector store insertion
        vs_ids = []
        vs_contents = []
        vs_metadatas = []

        # Use raw bytes to accurately slice code content for the vector database
        source_bytes = code_file.encode("utf-8")

        for s in symbols:
            qualified_name = s.get("qualified_name") or s["name"]
            symbol_id = f"{file_path}:{qualified_name}"

            self.graph.add_nodes(symbol_id, s['type'], file_path)
            name_to_id[s["name"]] = symbol_id
            qualified_to_id[qualified_name] = symbol_id

            # Prepare data for vector database
            if vector_store:
                start_byte, end_byte = s["byte_range"]
                if start_byte == 0 and end_byte == 0:
                    # Alt-parser: index full file content for semantic search
                    content = code_file
                else:
                    content = source_bytes[start_byte:end_byte].decode("utf-8")
                
                vs_ids.append(symbol_id)
                vs_contents.append(content)
                vs_metadatas.append({
                    "file_path": file_path,
                    "symbol_name": s["name"],
                    "qualified_name": qualified_name,
                    "type": s["type"],
                    "start_line": s["start_line"]
                })
        
        # Step 2: Map call relation edges
        for c in calls:
            # Prefix the caller ID with the file path to maintain uniqueness
            caller_id = f"{file_path}:{c['caller']}"
            
            # Prefer an exact qualified name match; fallback to a simple name match.
            callee_id = qualified_to_id.get(c["callee"]) or name_to_id.get(c["callee"])

            if not callee_id:
                # Attempt to resolve using a unique suffix match (e.g., Class.method).
                candidates = [
                    symbol_id
                    for qualified, symbol_id in qualified_to_id.items()
                    if qualified.endswith(f".{c['callee']}")
                ]
                if len(candidates) == 1:
                    callee_id = candidates[0]

            # If the callee remains unresolved, retain the raw name for edge information.
            self.graph.add_edges(caller_id, callee_id or c["callee"])
        
        # Step 3: Execute batched insert into the vector database
        if vector_store and vs_ids:
            # Use batch insertion for rapid file processing
            vector_store.add_symbols_batch(vs_ids, vs_contents, vs_metadatas)
