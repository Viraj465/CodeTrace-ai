import networkx as nx
import sqlite3
import logging
from pathlib import Path
from typing import Optional
from src.core.database.db_utils import get_db_connection

logger = logging.getLogger(__name__)

# Graph builder

class CodeGraph:

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        file TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS graph_edges (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        relation TEXT NOT NULL,
        PRIMARY KEY (source, target, relation)
    );
    CREATE INDEX IF NOT EXISTS idx_graph_nodes_file ON graph_nodes(file);
    CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
    CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
    """
    
    def __init__(self, db_dir: Optional[str | Path] = None):
        """
        Directed graph creation with optional SQLite persistence.
        If db_dir is provided, it automatically restores the previous state.
        """
        self.direct_graph = nx.DiGraph()
        self.db_path = None
        
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)
            self.db_path = Path(db_dir) / "graph_metadata.db"
            self._init_db()
            self.load_from_db()
    
    def _init_db(self) -> None:
     with get_db_connection(self.db_path) as conn:
         conn.executescript(self.SCHEMA)
         # PRAGMAs are now handled by get_db_connection
         conn.commit()
    
    # creation of nodes
    def add_nodes(self, symbol_id: str, symbol_type: str, file: str):
        """
        Creation of nodes to the graph
        """
        # Pass as keyword arguments for correct metadata indexing
        self.direct_graph.add_node(symbol_id,
                                   type = symbol_type,
                                   file = file)
    
    def add_edges(self, caller: str, callee: str):
        """
        Function A (caller) -> Function B (callee)
        """
        self.direct_graph.add_edge(caller,
                                   callee,
                                   relation = "calls")
    
    def add_ownership(self, cls: str, method: str):
        """
        Class own method
        """
        self.direct_graph.add_edge(cls,
                                   method,
                                   relation = "defines")
        
    def persist_to_db(self):
        """
        1. Graph Serialization:
        Dumps the entire RAM graph to SQLite. Uses UPSERTs to prevent duplicate errors.
        Should be called at the end of the CLI `index` command.
        """
        if not self.db_path: 
            return
        
        # Extract data from NetworkX
        nodes_data = [(n, d.get("type", "unknown"), d.get("file", "unknown")) 
                      for n, d in self.direct_graph.nodes(data=True)]
        
        edges_data = [(u, v, d.get("relation", "calls")) 
                      for u, v, d in self.direct_graph.edges(data=True)]
        
        with get_db_connection(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            
            conn.executemany("""
                INSERT INTO graph_nodes (node_id, type, file) VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET type=excluded.type, file=excluded.file
            """, nodes_data)
            
            conn.executemany("""
                INSERT INTO graph_edges (source, target, relation) VALUES (?, ?, ?)
                ON CONFLICT(source, target, relation) DO NOTHING
            """, edges_data)
            
            conn.commit()
            logger.info(f"Persisted {len(nodes_data)} nodes and {len(edges_data)} edges to DB.")
            
    def load_from_db(self):
        """
        2. Graph Restoration:
        Wipes RAM and rebuilds the DiGraph perfectly from SQLite.
        Essential for instantly booting the CLI `chat` command.
        """
        if not self.db_path: 
            return
        
        self.direct_graph.clear()
        with get_db_connection(self.db_path) as conn:
            nodes = conn.execute("SELECT node_id, type, file FROM graph_nodes").fetchall()
            edges = conn.execute("SELECT source, target, relation FROM graph_edges").fetchall()
            
        for node_id, n_type, n_file in nodes:
            self.direct_graph.add_node(node_id, type=n_type, file=n_file)
            
        for src, tgt, rel in edges:
            self.direct_graph.add_edge(src, tgt, relation=rel)

    def prune_file(self, filepath: str):
        """
        3. Edge & Node Pruning (Delta Cleanup):
        If a file is modified or deleted, this safely removes its old nodes AND 
        any edges where it was the caller or the callee.
        """
        if not self.db_path: 
            return
        
        with get_db_connection(self.db_path) as conn:
            # 1. Identify all nodes defined in this file
            nodes = conn.execute("SELECT node_id FROM graph_nodes WHERE file = ?", (filepath,)).fetchall()
            nodes_to_delete = [row[0] for row in nodes]
            
            if not nodes_to_delete: 
                return
            
            placeholders = ",".join("?" * len(nodes_to_delete))
            
            # 2. Manual Cascade Delete: Remove edges touching these nodes
            conn.execute(f"""
                DELETE FROM graph_edges 
                WHERE source IN ({placeholders}) OR target IN ({placeholders})
            """, nodes_to_delete * 2)
            
            # 3. Delete the nodes themselves
            conn.execute("DELETE FROM graph_nodes WHERE file = ?", (filepath,))
            conn.commit()
            
        # 4. Sync memory to perfectly match the cleaned DB
        self.load_from_db()
    
    # Query utility
    def get_dependencies(self, symbol: str):
        """
        What does the symbol calls?
        """
        return list(self.direct_graph.successors(symbol))

    def get_callers(self, symbol: str):
        """
        Who calls this symbol?
        """
        return list(self.direct_graph.predecessors(symbol))
    
    def shortest_path(self, start: str, end: str):
        """
        Trace execution path
        """
        return nx.shortest_path(self.direct_graph, start, end)

    def get_all_downstream_dependents(self, symbol: str) -> list[dict]:
        """
        Find ALL symbols that transitively depend on `symbol`.
        Uses BFS over reversed edges (who calls this?) to trace the full
        impact chain.  Returns a list of dicts sorted by depth:
        [{"symbol": ..., "type": ..., "file": ..., "depth": ...}]
        """
        if symbol not in self.direct_graph:
            return []

        dependents = []
        visited = {symbol}
        queue = [(caller, 1) for caller in self.direct_graph.predecessors(symbol)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            node_data = self.direct_graph.nodes.get(current, {})
            dependents.append({
                "symbol": current,
                "type": node_data.get("type", "unknown"),
                "file": node_data.get("file", "unknown"),
                "depth": depth,
            })

            for caller in self.direct_graph.predecessors(current):
                if caller not in visited:
                    queue.append((caller, depth + 1))

        return sorted(dependents, key=lambda d: d["depth"])

    def list_files_in_graph(self) -> list[str]:
        """Return a deduplicated list of all files tracked in the graph."""
        files = set()
        for _, data in self.direct_graph.nodes(data=True):
            f = data.get("file")
            if f:
                files.add(f)
        return sorted(files)

    def export_format(self):
        """
        return context in json for LLM use.
        """

        return nx.node_link_data(self.direct_graph)