# Codetrace-AI — Command Reference

## Installation

```bash
pip install codetrace-ai
```

---

## Quick Start (2 commands)

```bash
cd /path/to/your/project
codetrace init
codetrace chat
```

---

## All Commands
### `codetrace init [PATH]`

One-command setup — does everything automatically:
1. Configures your LLM provider (interactive wizard)
2. Downloads embedding models
3. Indexes your codebase
4. Registers MCP for Cursor & Claude Code

```bash
codetrace init                    # setup current directory
codetrace init /path/to/project   # setup a specific project
codetrace init --fast             # use smaller models (low RAM)
codetrace init --llm ollama       # pre-select provider
codetrace init --llm groq         # skip provider selection
codetrace init --offline          # strict air-gapped mode (requires cached models)
```

---

### `codetrace chat`

Interactive AI chat loop — ask questions about your codebase.

```bash
codetrace chat                      # new session
codetrace chat --resume abc123      # resume a past session
codetrace chat --offline            # strict air-gapped mode
```

**In-chat commands:**
- `/clear` — start a fresh session without exiting
- `exit` or `quit` — close the chat

---

### `codetrace index [PATH or URL]`

Re-index a codebase. Supports local directories and GitHub URLs.

```bash
codetrace index .                                          # re-index current dir
codetrace index /path/to/project                           # index a specific dir
codetrace index https://github.com/user/repo               # clone + index
codetrace index https://github.com/user/repo/tree/develop  # specific branch
```

---

### `codetrace config`

View or change your LLM provider configuration.

```bash
codetrace config
```

Supported providers: `groq`, `openai`, `anthropic`, `gemini`, `ollama`

---

### `codetrace history`

List all past chat sessions for the current project.

```bash
codetrace history
```

Output:
```
╭── Chat History ──────────────────────────────────────────╮
│ ID        First Question              Messages  Last     │
│ abc123    How does auth work?         12        2h ago   │
│ def456    Trace the call graph        8         1d ago   │
╰──────────────────────────────────────────────────────────╯
```

---

### `codetrace export <SESSION_ID>`

Export a chat session as a Markdown document.

```bash
codetrace export abc123                 # print to terminal
codetrace export abc123 -o notes.md     # save to file
```

---

### `codetrace visualize`

Generate an interactive HTML graph of your code architecture.

```bash
codetrace visualize
```

Opens a browser with an interactive node graph showing functions, classes, and their relationships.

---

## Offline / Local LLM Setup (Air-Gapped)

For privacy-conscious environments, Codetrace can run 100% offline using **Ollama**.
Zero data leaves your machine.

**Standard Offline Setup:**
```bash
# 1. Install Ollama (https://ollama.com)
ollama pull llama3.2

# 2. Setup Codetrace with local LLM
codetrace init --llm ollama

# 3. Chat — zero data leaves your machine
codetrace chat
```

**True Air-Gapped Installation (`--offline`)**
By default, `codetrace init` requires an internet connection on its first run to download the base embedding models (`bge-small` and `e5-small`) and it sends anonymous startup pings to TryChroma.

If your machine has **zero internet access**, use `--offline` to strictly kill all telemetry and external requests.

1. On an *internet-connected* computer, run `codetrace init` to download the models.
2. Transfer the HuggingFace cache folder to your offline machine via USB:
   - Mac/Linux: `~/.cache/huggingface/hub`
   - Windows: `C:\Users\<username>\.cache\huggingface\hub`
3. Initialize and chat safely:
```bash
codetrace init --offline
codetrace chat --offline
```

---

## IDE Integration (MCP)

`codetrace init` auto-registers MCP for **Cursor** and **Claude Code**.

After running `init`, your IDE will automatically see these tools:
- `search_codebase` — semantic code search
- `get_symbol_relations` — call graph analysis
- `read_file` — read source files
- `analyze_impact` — blast radius analysis
- `write_file` — propose code changes
- `git_diff` — view recent changes

No manual configuration needed.

---

## File Structure

After running `codetrace init`, your project will have:

```
your-project/
├── .codetrace/            ← created by codetrace
│   ├── chroma/            ← vector embeddings (ChromaDB)
│   ├── graph_metadata.db  ← code graph (SQLite)
│   └── chat_history.db    ← chat sessions (SQLite)
├── src/
├── ...
└── your code files
```

Global config is stored at `~/.codetrace/config.json`.
