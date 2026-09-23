# CodeTrace AI — Repository Analysis

## 1. Executive summary

CodeTrace AI is a Python CLI and Model Context Protocol (MCP) server that turns a software repository into a locally persisted code-intelligence workspace. Its central idea is to give an AI coding agent a reliable, inspectable model of a codebase instead of allowing it to answer from vague semantic similarity or language-model memory alone.

The system builds two complementary views of a project:

1. **A deterministic structural view**: Tree-sitter AST queries and alternative parsers extract symbols, qualified names, call sites, callers, dependencies, and transitive downstream dependents. NetworkX provides in-memory graph traversal and SQLite persists the graph.
2. **A semantic retrieval view**: Code symbols are embedded with both BGE and E5 sentence-transformer models, stored in ChromaDB, combined with Reciprocal Rank Fusion, and optionally reranked with FlashRank.

An agent loop communicates with an LLM through plain `httpx` requests, invokes seven local tools, and turns their results into evidence-backed answers. The CLI also supports indexing, interactive chat, persistent sessions, architecture visualization, provider configuration, model context management, GitHub/GitLab cloning, and MCP registration for IDEs.

The project is designed to be local-first and air-gap capable. Parsing, graph traversal, embeddings, indexing, snapshots, and chat persistence happen on the user’s machine. However, when a cloud provider is selected, indexed source code is sent to that provider; only an Ollama configuration keeps the LLM interaction on-device.

## 2. Product concept and problem being solved

The project addresses a common failure mode of AI coding assistants: they can produce plausible answers while lacking verified knowledge of callers, dependencies, file locations, or change impact. CodeTrace’s answer is a governed investigation pipeline:

```text
inspect index → discover semantically → verify structurally → read evidence
→ analyze impact → propose an edit or report findings
```

The system prompt calls this the **Governed Pipeline Protocol**. Repository claims are classified as:

- **CONFIRMED** — established by a current tool call.
- **INFERRED** — reasoning based on confirmed evidence and explicitly labelled as inference.
- **UNRESOLVED** — evidence is missing or ambiguous, so the agent must abstain rather than invent an answer.

The intended behavior is “discover with embeddings, validate with the graph and indexed snapshots.” Semantic search helps locate an unknown concept; graph traversal verifies structural relationships; file snapshots verify implementation details and line-level citations.

## 3. High-level architecture

```text
                         ┌──────────────────────────────┐
                         │        Typer CLI              │
                         │ init / index / chat / config  │
                         │ visualize / history / export  │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │      AgentOrchestrator        │
                         │ plain HTTP LLM tool loop      │
                         └───────┬──────────┬────────────┘
                                 │          │
                    ┌────────────▼───┐  ┌───▼────────────────┐
                    │ Shared tools   │  │ Token/output layer │
                    │ 7 JSON tools   │  │ budgets + Markdown │
                    └──────┬─────────┘  └────────────────────┘
                           │
       ┌───────────────────┼────────────────────┐
       │                   │                    │
┌──────▼──────┐    ┌───────▼────────┐    ┌──────▼─────────┐
│ VectorStore │    │ CodeGraph      │    │ SyncManager    │
│ ChromaDB    │    │ NetworkX+SQLite│    │ SHA-256+SQLite │
│ BGE + E5    │    │ symbols/edges  │    │ snapshots      │
└──────┬──────┘    └───────▲────────┘    └──────▲─────────┘
       │                    │                    │
       └────────────────────┴──────────┬─────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │ GraphOrchestrator          │
                         │ Tree-sitter + alt parsers  │
                         └─────────────────────────────┘

                         MCP server exposes the same
                         shared tools to IDEs and agents.
```

## 4. Indexing and parsing pipeline

### 4.1 File discovery and exclusion

`src/ignore.py` combines nested `.gitignore` files with built-in exclusions. It prunes version-control directories, virtual environments, caches, dependency directories, build output, IDE metadata, coverage folders, and CodeTrace’s own `.codetrace` directory. Dot-directories are skipped automatically. PDFs are explicitly excluded as non-source files.

`src/file_extension.py` maps source extensions to parser language names. `src/cli/project_helpers.py` validates local projects and can shallow-clone GitHub/GitLab URLs, including branch URLs, into a temporary directory.

### 4.2 Tree-sitter parser

`src/core/parser/parser.py`:

- Loads language bindings dynamically.
- Caches immutable language objects and compiled queries.
- Gives each worker thread its own mutable Tree-sitter parser.
- Loads `.scm` queries from `src/core/parser/queries`.
- Extracts symbol definitions, symbol type, qualified name, start line, byte range, and call sites.
- Resolves enclosing classes and functions through `ast_utility.py`.

Qualified names distinguish methods such as `AuthService.login` from top-level functions such as `login`. Calls retain the enclosing caller and source line.

The active Tree-sitter language set is:

| Language | Notes |
|---|---|
| Python | symbols and calls |
| JavaScript / JSX | JavaScript grammar; JSX extension is mapped to JavaScript |
| TypeScript / TSX | separate TypeScript and TSX grammars |
| Java | symbols and calls |
| C / C++ | symbols and calls |
| C# | symbols and calls |
| Go | symbols and calls |
| Rust | symbols and calls |
| PHP | symbols and calls |
| Swift | symbols and calls |
| Kotlin | symbols and calls |
| Bash | symbols and calls where query captures permit |
| HTML / CSS / JSON | structural symbols based on grammar queries |

That is 17 language families when JSX is treated as a JavaScript variant, or 18 extension-level entries in the README-style list. A `dart.scm` query file is present, but Dart is commented out of `LANGUAGE_MODULES` and `file_extension.py`, so Dart is not active support.

### 4.3 Alternative parsers

`src/core/parser/alt_parser.py` handles formats not represented by the active Tree-sitter registry:

- **YAML** — PyYAML loads documents and top-level keys become `property` symbols.
- **TOML** — `tomllib`/fallback parsing walks keys as symbols.
- **SQL** — SQLGlot parses SQL ASTs and extracts tables and common table expressions.
- **Dockerfile** — line-oriented parsing identifies build-stage and instruction symbols. `RUN`, `COPY`, and `ADD` are modelled as call-like operations.

These formats provide useful search and structural coverage, but they do not provide the same full function-level call graph semantics as Tree-sitter code languages.

### 4.4 Graph construction

`GraphOrchestrator` combines parser output and creates IDs in the form:

```text
<file path>:<qualified symbol name>
```

Definitions become graph nodes with type and file metadata. Calls become directed `calls` edges from caller to callee. Resolution attempts exact qualified names, plain names, and unique qualified-name suffixes. Unresolved names are retained as raw edge targets rather than discarded, which preserves the fact that a call was observed but also means unresolved references must not be treated as confirmed definitions.

`CodeGraph` in `src/core/graph/builder.py` maintains a NetworkX directed graph and can persist/load nodes and edges in `.codetrace/graph_metadata.db`. It supports:

- caller lookup;
- direct dependency lookup;
- shortest paths;
- transitive downstream traversal with depth labels;
- file listing;
- JSON/node-link export;
- batch node/edge updates;
- deletion of all graph records belonging to changed files;
- `.gitignore`-aware path filtering.

## 5. Semantic retrieval (“Hybrid Brain”)

`src/backend/vector_store.py` implements the semantic side of the index.

### Ingestion

Each extracted symbol is stored as a small document containing its source slice and metadata such as file path, symbol name, qualified name, type, and start line. Alternative-parser symbols have no byte range, so the whole source file is indexed for those records.

Two persistent ChromaDB collections are maintained:

- `code_bge` using `BAAI/bge-small-en-v1.5`;
- `code_e5` using `intfloat/e5-small-v2`.

### Search

For a query, the two embedding models search concurrently. Their ranked results are fused with Reciprocal Rank Fusion. The fused candidates are sent through the local FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) in small batches. If reranking fails because of memory pressure or a model/runtime problem, the system falls back to RRF order instead of failing the search.

The store supports batched insertion, deletion by symbol, deletion by file, collection counts, and persistent storage under `.codetrace/chroma`.

Device detection from `src/core/system_info.py` chooses CUDA, Apple MPS, or CPU and sets embedding batch sizes of 128, 64, or 32 respectively. RAM, CPU count, operating system, Python version, GPU information, and terminal Unicode capability are also detected and displayed by the CLI.

## 6. Incremental indexing and persistence

`SyncManager` in `src/core/database/sync_manager.py` prevents unnecessary parsing and embedding:

- Computes SHA-256 hashes in chunks.
- Compares current hashes with `.codetrace/sync_metadata.db`.
- Hashes files in parallel for larger repositories.
- Identifies new/changed and deleted files.
- Stores file hashes and index metadata.
- Stores source snapshots for later DB-backed `read_file` operations.
- Limits snapshots to one million bytes and records line count, size, and truncation state.
- Maintains a project manifest hash from tracked file/hash pairs.

The typical changed-file flow is to remove old graph nodes and vector records for affected files, parse and index the new content, then commit the new hashes and snapshots. This prevents renamed or deleted symbols from remaining as stale search results.

`db_utils.py` provides a SQLite context manager with WAL mode, foreign keys, normal synchronous mode, memory cache/temp-store settings, mmap support, incremental vacuum, query optimization, and guaranteed connection closure.

`ChatStore` stores sessions in `.codetrace/chat_history.db`. It supports:

- short UUID session IDs;
- project association;
- message persistence for user and assistant roles;
- latest-session lookup;
- recent-session listing with message counts;
- limited history retrieval for the LLM;
- text search over all sessions;
- Markdown export;
- explicit connection closing.

## 7. Agent loop and model providers

`src/core/agents/retriever.py` contains `AgentOrchestrator`. It deliberately avoids LangChain and uses plain HTTP, making the provider boundary explicit and lightweight.

Supported provider configurations are:

- Anthropic native Messages API;
- OpenAI-compatible OpenAI endpoint;
- Groq;
- Gemini’s OpenAI-compatible endpoint;
- OpenRouter;
- Ollama;
- custom OpenAI- or Anthropic-compatible endpoints.

Provider configuration includes provider, API key, model name, base URL, and API style. Keys are optional for Ollama and custom local/unauthenticated endpoints. Setup can query model listings and masks recognizable API-key formats in output.

The loop supports tool calls, streaming/non-streaming responses, provider-specific request differences, Gemini thought-signature round-tripping, Gemini field cleanup, empty-choice/error handling, and endpoint-specific attribution headers. It dispatches all tool calls to the shared implementation in `tools.py`, so the CLI and MCP server use the same behavior.

### Context and memory management

`TokenBudgetManager`:

- counts tokens with LiteLLM provider-aware tokenizers;
- falls back to estimates when a provider reports no usage;
- resolves model context windows dynamically;
- queries Ollama’s native API for the loaded model’s context size;
- classifies models into tier 1, tier 2, or tier 3;
- changes history compression, search/result limits, tool-result caps, and output budgets by tier;
- preserves recent turns and compresses older history;
- trims oversized tool results;
- records per-turn input/output/cache usage and tool-call counts;
- warns or adapts before hard context limits are exceeded.

Ollama has additional hardware-aware context sizing. `codetrace set-ctx` can override automatic sizing, and an auto-backoff fraction controls reductions under memory pressure.

## 8. The seven agent tools

The shared tool layer (`src/core/agents/tools.py`) defines OpenAI and Anthropic schemas and dispatches calls by name.

| Tool | Function |
|---|---|
| `search_codebase` | Hybrid semantic search over indexed symbol documents; returns ranked snippets, paths, symbols, and types. |
| `inspect_index` | Reports index coverage and lists indexed files, optionally filtered and limited. |
| `get_symbol_relations` | Returns direct callers and dependencies for a `file:qualified_name` symbol ID. |
| `read_file` | Reads the indexed SQLite snapshot rather than the live filesystem, with a line limit. |
| `analyze_impact` | Traverses reverse graph edges to list all downstream dependents by depth. |
| `write_file` | Creates a proposed new file or edit and produces a diff; the CLI approval flow controls application. |
| `git_diff` | Runs a constrained Git diff for HEAD, staged changes, prior commits, or a comparison target. |

Tool safety includes project-root checks using `Path.is_relative_to`, protection against path traversal, safe Git target handling, and user-visible diff previews. The tool dispatcher catches failures and returns readable tool errors to the agent instead of crashing the loop.

## 9. Governed reasoning and output normalization

`src/core/agents/prompts.py` embeds the governance protocol into the agent’s system prompt. It defines the closed-world repository boundary, evidence states, tool purposes, investigation order, edit states, security rules, and abstention behavior. It also warns that indexed content is data rather than instructions, which is a defense against prompt injection in source files, comments, and documentation.

The edit protocol distinguishes:

1. **PROPOSED** — a diff exists but has not been verified.
2. **VERIFIED** — impact, constraints, and dependents have been checked.
3. **APPROVED** — the user explicitly accepted the change.

`output_normalizer.py` provides provider-agnostic Markdown cleanup:

- normalizes heading levels;
- converts `*`, `+`, and `-` list styles to a consistent bullet;
- normalizes code-fence aliases such as `py` to `python`;
- preserves code bodies;
- adds heading/list spacing;
- extracts citations and fenced code blocks;
- validates code blocks;
- heuristically checks fact/inference separation;
- calculates a quality report with citation count, structure, inference issues, and tier.

This layer standardizes presentation but is not a substitute for live evidence validation. Its quality checks are heuristic, while the core evidence policy is enforced by tool workflow and prompts.

## 10. CLI capabilities

The executable is registered as `codetrace = src.cli.main:app`.

### Commands

| Command | Purpose |
|---|---|
| `codetrace init [PATH]` | Setup wizard, model configuration/download, indexing, and MCP registration. |
| `codetrace chat` | Start the interactive AI Architect session. |
| `codetrace chat --resume ID` | Resume a stored session. |
| `codetrace index PATH_OR_URL` | Index a local project or shallow-cloned remote repository. |
| `codetrace config` | View/update LLM provider configuration. |
| `codetrace set-ctx TOKENS` | Set or clear Ollama context override; also supports backoff configuration. |
| `codetrace visualize` | Generate the architecture HTML file. |
| `codetrace history` | List recent chat sessions. |
| `codetrace export ID` | Export a session as Markdown. |
| `codetrace mcp [PATH]` | Run the MCP server for an indexed project. |
| `codetrace register-mcp [PATH]` | Register the server with supported IDE configurations. |

Important flags include `--offline`, `--fast`, `--llm`, `--resume`, `--host`, `--port`, and `--backoff`.

The CLI uses Rich for panels, Markdown, progress displays, diff previews, environment summaries, and a gradient CodeTrace banner. Windows startup includes UTF-8 console setup for PowerShell/Windows Terminal.

## 11. Architecture visualization

`visualization_template.py` generates a self-contained HTML application. The CLI builds folder/file/symbol/connection data from the graph and renders it into the template.

The visualization includes:

- a force-directed folder dependency graph;
- cross-folder call edges;
- folder nodes sized/coloured by content or relationship data;
- zoom, pan, and automatic zoom-to-fit;
- search that fades nonmatching folders and edges;
- a click-open sidebar;
- folder connection details;
- expandable files and symbols;
- symbol type colours;
- project statistics;
- empty-index guidance.

The generated HTML escapes `</` in embedded JSON so repository content cannot accidentally terminate the script tag. The template references hosted fonts and D3 resources as part of the visual presentation, so the generated file is structurally self-contained but may still rely on those external browser resources depending on the template section used.

## 12. MCP and IDE integration

`codetrace_mcp/server.py` wraps the same seven shared tool implementations in an MCP server. It can run over stdio for Cursor, VS Code, Claude Desktop/Code, Windsurf, and other MCP-capable clients.

At startup it:

1. Resolves the project path.
2. Requires a `.codetrace` directory.
3. Changes into the project root.
4. Loads ChromaDB and the persisted graph.
5. Exposes the seven tools through FastMCP (with a compatibility fallback import).

The setup flow resolves the MCP server from the installed package and registers the current Python interpreter, avoiding the earlier failure mode where the path was incorrectly derived from the indexed project. Configuration support covers user-level Cursor, Claude, and VS Code files plus a workspace `.vscode/mcp.json` format.

## 13. On-disk layout

After indexing, a target project typically contains:

```text
.codetrace/
├── chroma/                 # BGE and E5 vector collections
├── graph_metadata.db       # persisted graph nodes and calls
├── sync_metadata.db        # hashes, snapshots, manifest metadata
└── chat_history.db         # sessions and messages
```

The generated architecture visualization is stored in the project’s `.codetrace` area. Global LLM configuration is stored at `~/.codetrace/config.json`.

## 14. Dependencies and packaging

The project targets Python 3.10–3.12 and is built with setuptools. Runtime dependencies cover:

- Typer, Rich, Pydantic, dotenv, and HTTPX for the CLI/configuration/UI;
- Tree-sitter language bindings and query files for parsing;
- PyYAML, SQLGlot, and TOML support for alternative formats;
- Torch, Transformers, Sentence Transformers, ChromaDB, NetworkX, and FlashRank for indexing/search/graph work;
- LiteLLM, tiktoken, and HTTPX for model metadata and token accounting;
- FastAPI/Uvicorn/watchfiles and MCP-related packages for service/integration support;
- psutil with standard-library fallbacks for system information.

`MANIFEST.in` and setuptools package data include the `.scm` query files in built distributions. Package `__init__.py` files are present in the key package directories so the installed wheel includes the CLI, backend, agents, and database modules.

## 15. Tests currently present

The repository contains focused tests rather than a complete end-to-end indexing suite:

- `tests/test_mcp_server.py` verifies that all seven MCP tools are registered and that uninitialized stores return a graceful error.
- `tests/test_output_normalizer.py` covers citation extraction, code blocks, language aliases, Markdown normalization, quality checks, provider-style variations, and tier-specific expectations.
- `tests/test_token_mnager.py` checks that an unknown cloud model uses the safer 32K/4K fallback tier rather than the small local fallback.

There are no visible comprehensive tests for the full parser matrix, incremental synchronization, vector-store ingestion, graph resolution accuracy, CLI workflows, provider API calls, or visualization rendering. Those are important areas for future test expansion.

## 16. Project status and caveats

### Strengths

- Strong separation between semantic discovery and deterministic structural verification.
- Shared tool implementations prevent CLI/MCP behavior drift.
- Incremental SHA-256 indexing is appropriate for repeated local use.
- SQLite persistence makes the graph, snapshots, and sessions inspectable.
- Path traversal checks and approval-oriented edit previews reduce accidental writes.
- Dynamic model context handling is better suited to varied providers than a fixed model registry.
- The governance prompt explicitly favours abstention over fabricated repository facts.

### Caveats

- The graph is based on static parsing and name resolution, not runtime tracing. Dynamic dispatch, reflection, aliases, generated code, and unresolved external symbols can produce incomplete or approximate relationships.
- An empty relationship result does not prove that a symbol is dead code.
- Alternative parsers intentionally provide weaker structural semantics than Tree-sitter.
- Semantic search quality depends on downloaded models, available RAM/VRAM, and reranker availability.
- “Zero cloud dependency” is an operating mode, not a guarantee for every configuration: cloud providers receive indexed code.
- `constants/prompts.txt` contains an older prompt-style constant while `src/core/agents/prompts.py` contains the newer governed prompts; the latter is the active agent implementation path.
- `refactor.py` is only a placeholder script that prints `Refactoring...`; it is not part of the CodeTrace indexing or agent architecture.
- Generated/build/cache directories shown in the original folder listing are artifacts and should not be treated as source features.

## 17. End-to-end usage model

The intended lifecycle is:

```text
1. Install codetrace-ai.
2. Run `codetrace init .`.
3. Configure a cloud or local LLM and download/cache embedding models.
4. Scan the repository while applying ignore rules.
5. Hash files and skip unchanged content.
6. Parse changed files and create graph nodes/edges.
7. Embed symbols into BGE/E5 Chroma collections.
8. Persist snapshots, graph state, hashes, and metadata.
9. Start `codetrace chat` or connect an IDE through MCP.
10. Let the agent inspect, search, verify relations, read snapshots, and analyze impact.
11. Review proposed diffs and explicitly approve any writes.
12. Re-index later; SHA-256 delta sync updates only changed/deleted files.
```

In short, CodeTrace AI is an evidence-oriented local codebase operating layer for AI agents: Tree-sitter and graph analysis provide structural truth, embeddings provide discovery, SQLite provides durable project state, an HTTP tool-calling loop provides reasoning, and governance plus approval controls constrain what the agent may claim or change.
