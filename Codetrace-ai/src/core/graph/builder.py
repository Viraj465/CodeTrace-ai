import networkx as nx
import sqlite3
import logging
from pathlib import Path
from typing import Optional
# Relative imports so IDEs resolve them correctly within the package.
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
         # get_db_connection already sets the PRAGMAs for us.
         conn.commit()
    
    def add_nodes(self, symbol_id: str, symbol_type: str, file: str):
        """
        Add a node to the graph.
        """
        # Keyword args so NetworkX stores them as node metadata.
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
        Dump the whole in-memory graph to SQLite. UPSERTs keep duplicate keys
        from blowing up. Call this at the end of the index command.
        """
        if not self.db_path:
            return

        # Pull the nodes and edges out of NetworkX.
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



    def prune_files(self, filepaths: list[str]) -> list[tuple[str, str, str]]:
        """
        Remove nodes/edges for multiple files, then reload graph once.

        Returns the *inbound* edges that were removed — edges whose source lives
        in some other (unpruned) file but whose target is a symbol of a pruned
        file. When a changed file is re-indexed, only that file is re-parsed, so
        these calls from unchanged files would otherwise be lost for good. The
        indexer hands them to ``restore_inbound_edges`` after re-adding the
        file's symbols. For deleted files, just ignore the return value.
        """
        if not self.db_path or not filepaths:
            return []

        # Chunk the IN clauses to stay under SQLite's variable limit.
        CHUNK_SIZE = 900

        all_nodes = []
        inbound: list[tuple[str, str, str]] = []
        with get_db_connection(self.db_path) as conn:
            # Gather the node IDs in chunks so we don't blow past SQLite's variable limit.
            for i in range(0, len(filepaths), CHUNK_SIZE):
                chunk = filepaths[i:i + CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT node_id FROM graph_nodes WHERE file IN ({placeholders})",
                    chunk
                ).fetchall()
                all_nodes.extend(r[0] for r in rows)
            
            if all_nodes:
                # Remember inbound edges from other files before deleting them.
                pruned = set(all_nodes)
                for i in range(0, len(all_nodes), CHUNK_SIZE):
                    chunk = all_nodes[i:i + CHUNK_SIZE]
                    ph = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT source, target, relation FROM graph_edges WHERE target IN ({ph})",
                        chunk,
                    ).fetchall()
                    inbound.extend(
                        (src, tgt, rel) for src, tgt, rel in rows if src not in pruned
                    )

                # Delete edges in chunks too. The OR clause binds each id twice,
                # so halve the chunk size to stay under the limit.
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
        return inbound

    def restore_inbound_edges(self, edges: list[tuple[str, str, str]]) -> int:
        """
        Re-add inbound edges saved by ``prune_files`` whose target symbol exists
        again after re-parsing. Edges to symbols that were renamed or removed
        are dropped, which is exactly what should happen to them.
        Returns the number of edges restored.
        """
        restored = 0
        for src, tgt, rel in edges:
            # Only real symbol nodes carry a "file" attribute; checking it keeps
            # us from resurrecting an edge into an implicit, attribute-less node.
            if src in self.direct_graph and "file" in self.direct_graph.nodes.get(tgt, {}):
                self.direct_graph.add_edge(src, tgt, relation=rel)
                restored += 1
        return restored

    def resolve_symbol_id(self, symbol_id: str) -> list[str]:
        """
        Map a user/model-supplied symbol ID onto the node IDs stored in the graph.

        Node IDs are stored as ``"{absolute file path}:{qualified name}"``, but
        callers (the LLM, following the tool schema) usually pass a
        project-relative path, forward slashes, or just a qualified name. This
        accepts all of those. Returns every matching node ID: exactly one means
        resolved, more than one means ambiguous, none means not found.
        """
        symbol_id = (symbol_id or "").strip()
        if not symbol_id:
            return []
        if symbol_id in self.direct_graph:
            return [symbol_id]

        # Split on the LAST colon so Windows drive letters survive.
        if ":" in symbol_id:
            path_part, qualified = symbol_id.rsplit(":", 1)
        else:
            path_part, qualified = "", symbol_id
        want_path = Path(path_part).as_posix().lower() if path_part else ""
        while want_path.startswith("./"):
            want_path = want_path[2:]

        matches: list[str] = []
        for node_id, data in self.direct_graph.nodes(data=True):
            node_file = data.get("file")
            if not node_file or node_file == "unknown":
                continue
            node_qualified = node_id.rsplit(":", 1)[-1]
            if node_qualified != qualified:
                continue
            if want_path:
                # Case-insensitive so Windows paths match however they're typed;
                # require a path-separator boundary so "b.py" never matches "ab.py".
                node_posix = Path(node_file).as_posix().lower()
                if not (node_posix == want_path or node_posix.endswith("/" + want_path)):
                    continue
            matches.append(node_id)
        return sorted(matches)

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
        Load all .gitignore rules for the repo and return a compiled
        ``pathspec.PathSpec`` (usable via ``spec.match_file(rel_posix_path)``).
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
            # Repo-root-relative path, POSIX-style separators.
            rel = p.relative_to(repo_root).as_posix()
            if not self.is_ignored(rel, ignore_spec):
                kept.append(p)
        return kept
