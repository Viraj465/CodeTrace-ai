import networkx as nx
import sqlite3
import logging
from pathlib import Path
from typing import Optional
# Use relative imports so IDEs resolve them correctly within the package.
from ..database.db_utils import get_db_connection
from ...ignore import read_gitignore
import fnmatch

logger = logging.getLogger(__name__)


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
        Creates a directed graph with optional SQLite persistence.
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
    
    def add_nodes(self, symbol_id: str, symbol_type: str, file: str):
        """
        Add a node to the graph.
        """
        # Pass as keyword arguments for correct metadata indexing
        self.direct_graph.add_node(symbol_id,
                                   type = symbol_type,
                                   file = file)

    def add_edges(self, caller: str, callee: str):
        """
        Add an edge indicating that the caller invokes the callee.
        """
        self.direct_graph.add_edge(caller,
                                   callee,
                                   relation = "calls")

    def add_nodes_batch(self, nodes: list[tuple[str, str, str]]):
        """Add multiple nodes from a list of (symbol_id, symbol_type, file)."""
        for symbol_id, symbol_type, file in nodes:
            self.direct_graph.add_node(symbol_id, type=symbol_type, file=file)

    def add_edges_batch(self, edges: list[tuple[str, str]]):
        """Add multiple edges from a list of (caller_id, callee_id)."""
        for caller, callee in edges:
            self.direct_graph.add_edge(caller, callee, relation="calls")
    
    def add_ownership(self, cls: str, method: str):
        """
        Add an edge indicating that a class defines a method.
        """
        self.direct_graph.add_edge(cls,
                                   method,
                                   relation = "defines")
        
    def persist_to_db(self):
        """
        Serialize the entire RAM graph to SQLite.
        Uses UPSERTs to prevent duplicate errors.
        Should be called at the end of the index command.
        """
        if not self.db_path: 
            return
        
        # Extract nodes and edges from NetworkX
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
        Clear the graph in memory and rebuild it from the SQLite database.
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



    def prune_files(self, filepaths: list[str]):
        """Remove nodes/edges for multiple files, then reload graph once."""
        if not self.db_path or not filepaths:
            return
        
        # Chunk the IN clauses to stay under SQLite's variable limit.
        CHUNK_SIZE = 900

        all_nodes = []
        with get_db_connection(self.db_path) as conn:
            # Collect node IDs in chunks to avoid exceeding SQL variable limit
            for i in range(0, len(filepaths), CHUNK_SIZE):
                chunk = filepaths[i:i + CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT node_id FROM graph_nodes WHERE file IN ({placeholders})",
                    chunk
                ).fetchall()
                all_nodes.extend(r[0] for r in rows)
            
            if all_nodes:
                # Delete edges in chunks. 
                # The OR clause doubles the variable count, so halve the chunk size.
                edge_chunk = CHUNK_SIZE // 2
                for i in range(0, len(all_nodes), edge_chunk):
                    chunk = all_nodes[i:i + edge_chunk]
                    ph = ",".join("?" * len(chunk))
                    conn.execute(f"""
                        DELETE FROM graph_edges 
                        WHERE source IN ({ph}) OR target IN ({ph})
                    """, chunk * 2)

                # Delete nodes in chunks
                for i in range(0, len(filepaths), CHUNK_SIZE):
                    chunk = filepaths[i:i + CHUNK_SIZE]
                    ph = ",".join("?" * len(chunk))
                    conn.execute(f"DELETE FROM graph_nodes WHERE file IN ({ph})", chunk)

                conn.commit()
        
        self.load_from_db()
    
    def get_dependencies(self, symbol: str):
        """
        Return the list of symbols that this symbol calls.
        """
        return list(self.direct_graph.successors(symbol))

    def get_callers(self, symbol: str):
        """
        Return the list of symbols that call this symbol.
        """
        return list(self.direct_graph.predecessors(symbol))
    
    def shortest_path(self, start: str, end: str):
        """
        Return the shortest execution path between start and end.
        """
        return nx.shortest_path(self.direct_graph, start, end)

    def get_all_downstream_dependents(self, symbol: str) -> list[dict]:
        """
        Find all symbols that transitively depend on the given symbol.
        Returns a list of dictionaries sorted by depth.
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
        Return the graph context in JSON format.
        """

        return nx.node_link_data(self.direct_graph)
    
    def _load_gitignore(self, repo_root: Path):
        """
        Load the .gitignore file and return a set of raw patterns.
        """
        return read_gitignore(str(repo_root))
    
    def is_ignored(self, rel_path: str, ignore_spec) -> bool:
        """
        Return True if the relative path matches any pattern from the pathspec.
        """
        return ignore_spec.match_file(rel_path)
    
    def filter_paths(self, paths:list[Path], repo_root:Path)->list[Path]:
        """
        Return a list of paths that do not match the .gitignore rules.
        """
        ignore_spec = self._load_gitignore(repo_root)
        kept: list[Path] = []
        for p in paths:
            # Compute a POSIX‑style path relative to the repo root
            rel = p.relative_to(repo_root).as_posix()
            if not self.is_ignored(rel, ignore_spec):
                kept.append(p)
        return kept
