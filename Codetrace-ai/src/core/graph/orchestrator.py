"""
Co-ordinator to map systems
"""

from typing import Optional, Any
from src.core.parser.parser import CodeParser
from src.core.graph.builder import CodeGraph

class GraphOrchestrator:
    def __init__(self):
        self.parser = CodeParser()
        self.graph = CodeGraph()
    
    def build_from_file(self, file_path: str, vector_store: Optional[Any] = None):
        """
        Extracting code from file and building map
        """
        with open(file_path, "r", encoding="utf-8") as f:
            code_file = f.read()

        language_name = self.parser.language_for_file(file_path)
        if not language_name:
            return

        symbols, calls = self.parser.extract_symbols_and_calls(code_file, 
                                                               language_name=language_name)
        
        # 1. Map Definitions (Nodes)
        name_to_id: dict[str, str] = {}
        qualified_to_id: dict[str, str] = {}

        # Batching lists for the VectorStore
        vs_ids = []
        vs_contents = []
        vs_metadatas = []

        # We need raw bytes to slice the exact code content for the VectorDB
        source_bytes = code_file.encode("utf-8")

        for s in symbols:
            qualified_name = s.get("qualified_name") or s["name"]
            symbol_id = f"{file_path}:{qualified_name}"

            self.graph.add_nodes(symbol_id, s['type'], file_path)
            name_to_id[s["name"]] = symbol_id
            qualified_to_id[qualified_name] = symbol_id

            # --- Vector DB Prep ---
            if vector_store:
                # Extract the exact string content of the function/class
                start_byte, end_byte = s["byte_range"]
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
        
        # 2. Map Call Relations (Edges)
        for c in calls:
            # prefix with file_path to keep it unique in the graph
            caller_id = f"{file_path}:{c['caller']}"
            
            # Prefer exact qualified name match, fallback to unique name match.
            callee_id = qualified_to_id.get(c["callee"]) or name_to_id.get(c["callee"])

            if not callee_id:
                # Try resolving a unique suffix match like `Class.method`.
                candidates = [
                    symbol_id
                    for qualified, symbol_id in qualified_to_id.items()
                    if qualified.endswith(f".{c['callee']}")
                ]
                if len(candidates) == 1:
                    callee_id = candidates[0]

            # If unresolved, leave as raw callee to keep edge info.
            self.graph.add_edges(caller_id, callee_id or c["callee"])
        
        # 3. Execute Vector DB Batch Insert
        if vector_store and vs_ids:
            # We use the batch method to process files rapidly
            vector_store.add_symbols_batch(vs_ids, vs_contents, vs_metadatas)
