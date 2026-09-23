"""
Codetrace-ai: Autonomous entry point to code engine.
"""
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Getting Rich's Unicode to render in PowerShell / Windows Terminal takes three
# separate nudges — the console code page, Python's own streams, and Rich itself
# all have to agree on UTF-8, and fixing one without the others still garbles output.
if sys.platform == "win32":
    # Flip the Win32 console code page to UTF-8.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    # Make Python's own stdout/stderr write UTF-8 bytes.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # reconfigure() didn't exist before 3.7.
    # Keep Rich off its LegacyWindowsTerm path. On older consoles Rich falls back
    # to _win32_console.py, which sidesteps everything above; RICH_LEGACY_WINDOWS=0
    # pins it to the normal ANSI renderer.
    os.environ.setdefault("RICH_LEGACY_WINDOWS", "0")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
import colorsys
from rich.progress import Progress, SpinnerColumn, TextColumn, ProgressColumn, Task
from rich.text import Text
from rich.prompt import Prompt

class GradientBarColumn(ProgressColumn):
    """A sleek progress bar that transitions from yellow to orange."""
    def __init__(self, bar_width: int = 40):
        self.bar_width = bar_width
        super().__init__()

    def render(self, task: "Task") -> Text:
        total = task.total if task.total is not None else 100
        progress = task.completed / total if total > 0 else 0

        # Yellow to Orange (hue from ~0.15 to ~0.05)
        hue = 0.15 - (progress * 0.10)
        r, g, b = colorsys.hsv_to_rgb(max(0.0, hue), 0.9, 1.0)
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        filled = int(self.bar_width * progress)
        empty = self.bar_width - filled

        return Text.assemble(
            ("[", "dim"),
            ("\u2500" * filled, f"bold {hex_color}"),
            ("\u2500" * empty, "dim white"),
            ("]", "dim"),
            (f" {int(progress * 100)}%", f"bold {hex_color}")
        )

from src.backend.chat_store import ChatStore
from src.backend.vector_store import VectorStore, VectorStoreConfig
from src.cli.config_helpers import (
    enable_offline_mode,
    ensure_config as _ensure_config,
    looks_like_api_key,
    mask_key,
    register_mcp as _register_mcp,
    run_setup_wizard as _run_setup_wizard_impl,
)
from src.cli.project_helpers import (
    clone_repo as _clone_repo,
    get_project_root as _get_project_root,
    repo_cache_dir as _repo_cache_dir,
    parse_github_url as _parse_github_url,
)
from src.cli.ui_helpers import (
    group_pending_writes_by_root_dir as _group_pending_writes_by_root_dir,
    print_banner as _print_banner,
    show_diff_panel as _show_diff_panel,
)
from src.core.agents.retriever import AgentOrchestrator
from src.core.agents.tools import (
    clear_pending_writes,
    get_pending_writes,
    replace_pending_writes,
    write_file_impl,
)
from src.core.database.sync_manager import SyncManager
from src.core.graph.builder import CodeGraph
from src.core.graph.orchestrator import GraphOrchestrator
from src.core.system_info import get_system_info
from src.config_io import write_private_json
from src.ignore import ALWAYS_IGNORE_DIRS, _is_always_ignored, _is_always_ignored_file

logger = logging.getLogger(__name__)

# Quiet down HuggingFace/Transformers weight-loading chatter.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ...and httpx request logs plus FlashRank's progress bars.
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("flashrank").setLevel(logging.ERROR)
logging.getLogger("src.backend.vector_store").setLevel(logging.WARNING)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

app = typer.Typer(
    name="codetrace",
    help="CodeTrace-ai: Autonomous System Architect",
    add_completion=False,
)

# legacy_windows=False for the same reason as above: keep Rich off the renderer that garbles Unicode.
console = Console(legacy_windows=False)


def print_banner() -> None:
    _print_banner(console)


def print_system_info() -> None:
    """Print a compact system environment panel after the banner."""
    sys_info = get_system_info()
    device_color = {"cuda": "green", "mps": "green", "cpu": "yellow"}.get(sys_info.device, "white")
    lines = [
        f"  [bold]OS[/bold]       [cyan]{sys_info.os_name}[/cyan]  [dim]{sys_info.os_version}  {sys_info.arch}[/dim]",
        f"  [bold]Compute[/bold]  [{device_color}]{sys_info.embed_device_label}[/{device_color}]"
        + (f"  [dim]{sys_info.gpu_name}[/dim]" if sys_info.gpu_name else ""),
        f"  [bold]RAM[/bold]      {sys_info.ram_available_gb:.1f} GB free / {sys_info.ram_total_gb:.1f} GB total  "
        f"[dim]· {sys_info.cpu_cores} CPU cores[/dim]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold dim]System Environment[/bold dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def _resolve_mcp_server_path(target_dir: Path) -> Path:
    """
    Locate the MCP server inside the *installed package*.

    It must not be derived from the project being indexed: target_dir only holds
    a codetrace_mcp/ directory when that project happens to be the Codetrace
    source tree itself, so every pip-installed user got a path that doesn't exist.

    codetrace_mcp ships without an __init__.py, which makes it a namespace
    package — those report ``origin is None`` and carry the directory in
    ``submodule_search_locations`` instead, so both shapes are handled here.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("codetrace_mcp")
    except (ImportError, ValueError):
        spec = None

    if spec:
        if spec.origin:
            candidate = Path(spec.origin).parent / "server.py"
            if candidate.is_file():
                return candidate
        for location in spec.submodule_search_locations or []:
            candidate = Path(location) / "server.py"
            if candidate.is_file():
                return candidate

    # Last resort: the layout that shipped before this was fixed.
    return target_dir / "codetrace_mcp" / "server.py"


def get_project_root(path: str) -> Path:
    return _get_project_root(path, console)


def ensure_config() -> None:
    _ensure_config(console)


def _run_setup_wizard(config_path: Path, is_reconfigure: bool = False) -> None:
    _run_setup_wizard_impl(config_path, console, is_reconfigure=is_reconfigure)

def collect_source_files(root: Path) -> list[Path]:
    """
    Collect every source file under ``root``, honoring both .gitignore and the
    hardcoded ALWAYS_IGNORE_DIRS.

    We prune ignored dirs in-place during the os.walk so we never step into
    node_modules/, venv/, __pycache__/ and friends \u2014 on a big repo that's the
    difference between touching a few hundred files and enumerating 20k+.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Drop ignored dirs in-place so os.walk won't recurse into them.
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for filename in filenames:
            if _is_always_ignored_file(filename):
                continue
            all_files.append(Path(dirpath) / filename)

    graph = CodeGraph()
    filtered_files = graph.filter_paths(all_files, repo_root=root)
    return filtered_files

@app.command()
def chat(
    resume: str = typer.Option("", "--resume", "-r", help="Resume a previous session by ID"),
    offline: bool = typer.Option(False, "--offline", help="Run in strict air-gapped mode (requires cached models)"),
):
    """
    Launch the interactive AI Architect chat loop.
    Use --resume <session_id> to continue a previous session.
    """
    if offline:
        enable_offline_mode()

    print_banner()
    print_system_info()
    target_dir = get_project_root(".")
    db_dir = target_dir / ".codetrace"

    if not db_dir.exists():
        console.print("[red]Error: Repository not indexed. Run 'codetrace index .' first.[/red]")
        raise typer.Exit(1)

    try:
        sync_manager_meta = SyncManager(db_dir=str(db_dir))
        manifest_hash = sync_manager_meta.get_metadata("manifest_hash")
        if not manifest_hash:
            console.print(
                "[yellow]Index metadata is missing. Run 'codetrace index .' to refresh DB coverage.[/yellow]"
            )
    except Exception:
        pass

    # No config yet? Kick off the setup wizard.
    ensure_config()

    if offline:
        config_path = Path.home() / ".codetrace" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = json.load(f)
            provider = cfg.get("provider", "")
            if provider and provider != "ollama":
                console.print(
                    f"\n[bold red]CRITICAL WARNING:[/bold red] You are in --offline mode, but your configured "
                    f"LLM ({provider}) requires an internet connection.\n"
                    f"Your code WILL be sent to external cloud servers.\n"
                )

    with console.status("[bold cyan]Waking up the Architect (loading Graph & Vectors)...", spinner="point"):
        # Open the vector store and graph DB.
        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)

        graph = CodeGraph()
        graph.db_path = db_dir / "graph_metadata.db"
        graph._init_db()
        graph.load_from_db()

        # Spin up the agent (uses httpx under the hood).
        try:
            agent = AgentOrchestrator(vector_store, graph)
        except ValueError as e:
            # The retriever raises ValueError when the API key is bad/missing.
            console.print(f"[red]Configuration Error: {e}[/red]")
            raise typer.Exit(1)

    console.print("[bold green]Architect is online! Type 'exit' or 'quit' to stop.[/bold green]")

    # For Ollama, show the context window in use (it adapts as the session runs —
    # watch for the ℹ notices when it auto-lowers under memory pressure).
    if getattr(agent, "_is_ollama", False):
        console.print(
            f"[dim]Ollama context window: {agent._ollama_num_ctx:,} tokens "
            f"({getattr(agent, '_ollama_ctx_source', 'auto')}). "
            f"Change with 'codetrace set-ctx'.[/dim]"
        )

    # Chat history lives in its own SQLite DB.
    chat_store = ChatStore(db_dir / "chat_history.db")

    if resume and chat_store.session_exists(resume):
        session_id = resume
        prev = chat_store.get_messages(session_id, limit=4)
        console.print(f"[dim]Resuming session [bold]{session_id}[/bold] ({len(prev)} messages loaded)[/dim]")
    else:
        session_id = chat_store.create_session(project=str(target_dir))
        if resume:
            console.print(f"[yellow]Session '{resume}' not found — starting a new session.[/yellow]")
        console.print(f"[dim]Session: {session_id}[/dim]")

    console.print("-" * 60)

    # Main chat loop — runs until the user exits.
    while True:
        query = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        if query.strip().lower() in ["exit", "quit"]:
            chat_store.close()
            console.print("[bold magenta]Shutting down Architect. Goodbye![/bold magenta]")
            break

        if query.strip().lower() == "/clear":
            session_id = chat_store.create_session(project=str(target_dir))
            console.print(f"[dim]New session started: {session_id}[/dim]")
            continue

        if not query.strip():
            continue

        # Catch someone pasting their API key into the prompt by mistake.
        if looks_like_api_key(query):
            console.print(
                "[bold yellow]⚠  That looks like an API key, not a question![/bold yellow]\n"
                "[dim]Your key was NOT sent to the model.\n"
                "To update your configuration, run:[/dim] [cyan]codetrace config[/cyan]"
            )
            continue

        # Run the agent.
        try:
            # Friendly, ASCII-safe labels for each tool the agent might call.
            tool_labels = {
                "search_codebase":      "Searching codebase",
                "inspect_index":        "Inspecting index",
                "get_symbol_relations": "Tracing symbol relations",
                "read_file":            "Reading file",
                "analyze_impact":       "Analyzing impact",
                "write_file":           "Proposing file change",
                "git_diff":             "Running git diff",
            }

            # On an Ollama out-of-memory the agent stops the turn, lowers num_ctx,
            # and emits a "memory_limit" event — we then ask whether to retry the
            # same question with the smaller window. Everything else runs once.
            full_response = ""
            memory_stopped = False

            while True:
                streaming_started = False
                full_response = ""
                live = None
                # Spinner shown while a tool runs; we stop and erase it once the
                # tool finishes or the model starts streaming its answer.
                _tool_status = None
                # Spinner tailing a thinking model's reasoning, plus the buffer it
                # renders from. Both reset per attempt.
                _reasoning_status = None
                _reasoning_buf = ""
                memory_event = None

                for event in agent.stream(query, chat_history=chat_store.get_history_for_llm(session_id)):
                    evt_type = event["type"]

                    if evt_type == "thought":
                        # Swap in a fresh spinner for whatever tool is now running.
                        if _tool_status:
                            _tool_status.stop()
                        # Reasoning ended the moment a tool call was decided.
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                            _reasoning_buf = ""
                        tool_name = event.get("tool", "")
                        label = tool_labels.get(tool_name, "Working")
                        # The event message carries the tool's argument detail.
                        detail = event.get("message", "")
                        _tool_status = console.status(
                            f"[dim]{detail}[/dim]",
                            spinner="dots",
                        )
                        _tool_status.start()

                    elif evt_type == "tool_end":
                        # Tool's done — clear the spinner so it leaves no trace.
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                            _reasoning_buf = ""

                    elif evt_type == "reasoning":
                        # Thinking models emit reasoning before any answer token.
                        # Show a live tail of it: without this the terminal sits
                        # blank for the whole reasoning phase, which on a local
                        # model can be a minute or more and reads as a freeze.
                        chunk = event.get("content", "")
                        if not isinstance(chunk, str) or not chunk:
                            continue
                        _reasoning_buf += chunk
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status is None:
                            _reasoning_status = console.status("", spinner="dots")
                            _reasoning_status.start()
                        # Last line, trimmed — a rolling window, not a transcript.
                        tail = " ".join(_reasoning_buf.split())[-90:]
                        _reasoning_status.update(f"[dim italic]thinking… {tail}[/dim italic]")

                    elif evt_type == "token":
                        token_text = event.get("content", "")
                        if not isinstance(token_text, str):
                            token_text = str(token_text)
                        if not token_text:
                            continue

                        # Reasoning is over once real content arrives.
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                            _reasoning_buf = ""

                        # Kill any leftover tool spinner before streaming text.
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None

                        if not streaming_started:
                            console.print("\n[bold dark_orange]Architect:[/bold dark_orange]")
                            streaming_started = True
                            live = Live(Markdown(""), console=console, refresh_per_second=10)
                            live.start()
                        full_response += token_text
                        live.update(Markdown(full_response))

                    elif evt_type == "notice":
                        # Informational retune (e.g. auto-lowered num_ctx). Print
                        # once, out of the way.
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                        console.print(f"[dim]ℹ {event.get('message', '')}[/dim]")

                    elif evt_type == "memory_limit":
                        # Hard OOM: the turn was stopped and num_ctx lowered. Note
                        # it and break out to ask whether to retry.
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                        if live:
                            live.stop()
                            live = None
                        memory_event = event
                        console.print(
                            f"\n[bold yellow]⚠ {event.get('message', '')}[/bold yellow]"
                        )
                        break

                    elif evt_type == "done":
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                        if live:
                            live.stop()
                        if not streaming_started and not full_response:
                            # Agent wrapped up without emitting any tokens.
                            pass

                    elif evt_type == "error":
                        # Something blew up in the async producer (API error, auth, …).
                        if _tool_status:
                            _tool_status.stop()
                            _tool_status = None
                        if _reasoning_status:
                            _reasoning_status.stop()
                            _reasoning_status = None
                        if live:
                            live.stop()
                            live = None
                        # Some exceptions carry an empty message and some carry a
                        # multi-line one (onnxruntime allocation dumps, provider
                        # stack traces). Empty printed as a bare "Architect
                        # Error:" with nothing after it; multi-line smeared
                        # across the spinner. Normalize both.
                        err_msg = " ".join(
                            str(event.get("message") or "").split()
                        ) or "Unknown error (no detail reported)"
                        console.print(f"\n[bold red]Architect Error:[/bold red] {err_msg}")
                        break

                    elif evt_type == "usage":
                        # The token counter — the one line we leave on screen.
                        turn_usage = event.get("turn")
                        if turn_usage:
                            console.print(turn_usage.format())
                    
                    elif evt_type == "normalized":
                        # Replace the raw accumulated text with the normalized
                        # version once streaming is complete. live may be None if
                        # no content tokens ever streamed (empty answer).
                        full_response = event.get("content", "")
                        if live:
                            live.update(Markdown(full_response))

                if memory_event is None:
                    break  # normal completion — no retry needed

                choice = Prompt.ask(
                    "[yellow]Continue this question with the smaller context window?[/yellow]",
                    choices=["y", "n"],
                    default="y",
                )
                if choice.lower() == "y":
                    console.print("[dim]Retrying with the reduced window…[/dim]")
                    continue

                # User declined — drop any half-formed proposed edits and skip
                # persisting this incomplete turn.
                memory_stopped = True
                clear_pending_writes()
                console.print(
                    "[dim]Stopped. The context window stays reduced for the rest "
                    "of the session (raise it with 'codetrace set-ctx').[/dim]"
                )
                break

            # Now deal with any file changes the agent proposed this turn. Nothing
            # gets written without the user approving it first (once per turn, after
            # all events are in).
            pending_writes = get_pending_writes()
            if pending_writes:
                batches = _group_pending_writes_by_root_dir(pending_writes)
                console.print(
                    f"\n[bold yellow]⚡ {len(pending_writes)} proposed change(s) "
                    f"across {len(batches)} batch(es):[/bold yellow]"
                )
                total_changed = 0
                total_skipped = 0
                total_failed = 0
                remaining_batches: list[tuple[str, list[dict]]] = []
                for idx, (batch_name, batch_items) in enumerate(batches, start=1):
                    console.print(
                        f"\n[bold cyan]Batch {idx}/{len(batches)}[/bold cyan] "
                        f"[dim]({batch_name}, {len(batch_items)} file(s))[/dim]"
                    )
                    batch_changed = 0
                    batch_skipped = 0
                    batch_failed = 0
                    for pw in batch_items:
                        approved = _show_diff_panel(console, pw)
                        if approved:
                            result = write_file_impl(
                                pw["file_path"],
                                pw["content"],
                                project_root=str(Path.cwd().resolve()),
                            )
                            if result.startswith("Successfully wrote"):
                                batch_changed += 1
                                console.print(f"  [bold green]✓ {result}[/bold green]")
                            else:
                                batch_failed += 1
                                console.print(f"  [bold red]✗ {result}[/bold red]")
                        else:
                            batch_skipped += 1
                            console.print(f"  [dim]✗ Skipped: {Path(pw['file_path']).name}[/dim]")

                    total_changed += batch_changed
                    total_skipped += batch_skipped
                    total_failed += batch_failed
                    console.print(
                        f"  [bold green]Batch {idx} complete:[/bold green] "
                        f"changed={batch_changed}, skipped={batch_skipped}, failed={batch_failed}"
                    )

                    if idx < len(batches):
                        next_batch = Prompt.ask(
                            "  [bold yellow]Proceed to next batch now?[/bold yellow]",
                            choices=["y", "n"],
                            default="y",
                        )
                        if next_batch.lower() != "y":
                            remaining_batches.extend(batches[idx:])
                            break

                if remaining_batches:
                    remaining = [pw for _, items in remaining_batches for pw in items]
                    replace_pending_writes(remaining)
                    next_batch_name = remaining_batches[0][0] if remaining_batches else "<none>"
                    console.print(
                        f"\n[yellow]Paused batch processing.[/yellow] "
                        f"[dim]Next batch queued: {next_batch_name} "
                        f"({len(remaining_batches[0][1]) if remaining_batches else 0} file(s)).[/dim]\n"
                        f"[dim]{len(remaining)} pending change(s) kept for the next run.[/dim]"
                    )
                else:
                    clear_pending_writes()
                console.print(
                    f"[bold cyan]Edit summary:[/bold cyan] "
                    f"changed={total_changed}, skipped={total_skipped}, failed={total_failed}"
                )

            console.print("-" * 60)

            # Persist this turn to history — unless it was aborted at an OOM stop,
            # in which case the turn never completed and shouldn't be saved.
            if not memory_stopped:
                chat_store.add_message(session_id, "user", query)
                if full_response:
                    chat_store.add_message(session_id, "assistant", full_response)

        except Exception as e:
            if _tool_status:
                try:
                    _tool_status.stop()
                except Exception:
                    pass
            if _reasoning_status:
                try:
                    _reasoning_status.stop()
                except Exception:
                    pass
            if live:
                try:
                    live.stop()
                except Exception:
                    pass
            error_text = str(e)
            if "Recursion limit" in error_text:
                console.print(
                    "[bold red]API Error:[/bold red] Workflow recursion limit reached before stop condition.\n"
                    "[yellow]Try asking for one edit batch at a time (the agent now processes edits in batches).[/yellow]"
                )
            else:
                console.print(f"[bold red]API Error: {e}[/bold red]")

@app.command()
def init(
    path: str = typer.Argument(".", help="Target directory to initialize and index"),
    fast: bool = typer.Option(False, "--fast", help="Use smaller models (less RAM, faster startup)"),
    llm: str = typer.Option("", "--llm", help="Pre-select LLM provider (groq/openai/anthropic/gemini/ollama)"),
    offline: bool = typer.Option(False, "--offline", help="Run in strict air-gapped mode (requires cached models)"),
):
    """
    One-command setup: config → download models → index codebase → register MCP.

    After this, just run 'codetrace chat' to start.
    """
    if offline:
        enable_offline_mode()

    print_banner()
    target_dir = get_project_root(path)
    db_dir = target_dir / ".codetrace"

    console.print("[bold]Starting Codetrace Setup[/bold]\n")

    # Step 1: Configuration.
    console.print("[bold cyan]Step 1/4[/bold cyan] – Configuration")
    global_dir = Path.home() / ".codetrace"
    global_dir.mkdir(parents=True, exist_ok=True)
    config_path = global_dir / "config.json"

    if config_path.exists():
        console.print("  [green]✓ Config already exists – skipping[/green]")
    else:
        if llm and llm.lower() == "custom":
            # A custom endpoint needs a base URL and wire format that no flag can
            # supply, so send it through the wizard rather than writing a config
            # that would fail on first use.
            console.print("  [dim]Custom provider needs an endpoint — starting setup.[/dim]\n")
            _run_setup_wizard(config_path, is_reconfigure=False)
        elif llm:
            # --llm was passed, so skip the wizard and write a default config.
            config_data = {"provider": llm.lower(), "api_key": "", "model_name": "", "base_url": ""}
            if llm.lower() != "ollama":
                api_key = Prompt.ask(f"  [cyan]Enter your {llm.upper()} API Key[/cyan]", password=True)
                config_data["api_key"] = api_key
            write_private_json(config_path, config_data)
            console.print(f"  [green]✓ Configured with {llm}[/green]")
        else:
            _run_setup_wizard(config_path, is_reconfigure=False)
    console.print()

    # Step 2: Download Models.
    console.print("[bold cyan]Step 2/4[/bold cyan] – Downloading Embedding Models")

    if fast:
        bge_model = "BAAI/bge-small-en-v1.5"
        e5_model = "intfloat/e5-small-v2"
    else:
        bge_model = "BAAI/bge-small-en-v1.5"
        e5_model = "intfloat/e5-small-v2"

    def _download_model(model_name):
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(model_name)
        return model_name

    with console.status(f"  [cyan]Downloading {bge_model} + {e5_model}...", spinner="dots"):
        for model_name in [bge_model, e5_model]:
            name = _download_model(model_name)
            console.print(f"  [green]✓ {name}[/green]")
    console.print()

    # Step 3: Index Codebase.
    console.print("[bold cyan]Step 3/4[/bold cyan] – Indexing Codebase")

    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        SyncManager(db_dir=str(db_dir))

    with console.status("  [cyan]Booting vector models and DB connections...", spinner="point"):
        sync_manager = SyncManager(db_dir=str(db_dir))
        orchestrator = GraphOrchestrator()

        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)

        orchestrator.graph.db_path = db_dir / "graph_metadata.db"
        orchestrator.graph._init_db()
        orchestrator.graph.load_from_db()

    # Walk the tree, pruning ALWAYS_IGNORE_DIRS as we go so we never enumerate
    # their contents. filter_paths() applies the .gitignore rules afterward.
    _raw_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for fn in filenames:
            if _is_always_ignored_file(fn):
                continue
            _raw_files.append(Path(dirpath) / fn)
    _raw_files = orchestrator.graph.filter_paths(_raw_files, repo_root=target_dir)
    all_files = [str(p) for p in _raw_files]
    supported_files = [f for f, _ in orchestrator.iter_supported_files(all_files)]
    supported_set = set(supported_files)

    changed_all_file_pairs = sync_manager.get_changed_files(all_files)
    deleted_files = sync_manager.get_deleted_files(all_files)

    if deleted_files:
        # Deleted files are gone from disk, so they're never in supported_set;
        # decide whether they were indexed from the path alone.
        _deleted_supported = [
            df for df, _ in orchestrator.iter_supported_files(deleted_files)
        ]
        if _deleted_supported:
            orchestrator.graph.prune_files(_deleted_supported)
            if vector_store:
                vector_store.delete_by_file(_deleted_supported)
        for df in deleted_files:
            sync_manager.remove_file_record(df)
            sync_manager.remove_file_snapshot(df)

    if not supported_files:
        console.print("  [yellow]No supported code files found.[/yellow]")

    changed_supported_file_pairs = [
        (file_path, file_hash)
        for file_path, file_hash in changed_all_file_pairs
        if file_path in supported_set
    ]

    if changed_all_file_pairs:
        for file_path, file_hash in changed_all_file_pairs:
            sync_manager.upsert_file_snapshot_from_disk(file_path, file_hash=file_hash)

    if changed_supported_file_pairs:
        start_time = time.time()

        # Wipe the old graph nodes/edges and vectors for each changed file first,
        # otherwise symbols that got renamed or deleted inside a file would stick
        # around (for brand-new files this is a no-op). Has to happen before we
        # parse and before we seed the lookup maps below.
        _changed_supported_files = [fp for fp, _ in changed_supported_file_pairs]
        if vector_store:
            vector_store.delete_by_file(_changed_supported_files)
        # Calls INTO these files from unchanged files get deleted with them;
        # keep them so they can be restored once the files are re-parsed.
        _inbound_edges = orchestrator.graph.prune_files(_changed_supported_files)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            GradientBarColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"  [cyan]Parsing & Indexing {len(changed_supported_file_pairs)} files...",
                total=len(changed_supported_file_pairs)
            )

            # Parse every changed file in parallel.
            MAX_WORKERS = max(4, os.cpu_count() or 4)
            all_symbols = []
            all_calls = []
            all_vs_ids = []
            all_vs_contents = []
            all_vs_metadatas = []
            _failed_files: set[str] = set()

            def _parse_one(file_path: str) -> dict:
                return orchestrator.extract_from_file(file_path)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(_parse_one, fp): fp for fp, _ in changed_supported_file_pairs}

                for fut in as_completed(futures):
                    fp = futures[fut]
                    try:
                        result = fut.result()
                        all_symbols.extend([(fp, s) for s in result["symbols"]])
                        all_calls.extend([(fp, c) for c in result["calls"]])
                        for v in result["vs_data"]:
                            all_vs_ids.append(v["id"])
                            all_vs_contents.append(v["content"])
                            all_vs_metadatas.append(v["metadata"])
                        progress.advance(task)
                    except Exception as e:
                        logger.error("Failed to parse %s: %s", fp, e)
                        _failed_files.add(fp)

            # Build the graph in one batch. Callees resolve against this batch
            # plus the unchanged files already in the graph (see build_batch).
            nodes_batch, edges_batch = orchestrator.build_batch(all_symbols, all_calls)
            orchestrator.graph.add_nodes_batch(nodes_batch)
            orchestrator.graph.add_edges_batch(edges_batch)
            orchestrator.graph.restore_inbound_edges(_inbound_edges)

            if vector_store and all_vs_ids:
                vector_store.add_symbols_batch(all_vs_ids, all_vs_contents, all_vs_metadatas)

        orchestrator.graph.persist_to_db()
        # Leave files that failed to parse unsynced so the next run retries
        # them, instead of silently treating them as indexed with no symbols.
        sync_manager.mark_files_synced_batch(
            [(fp, h) for fp, h in changed_all_file_pairs if fp not in _failed_files]
        )
        sync_manager.update_index_manifest(
            target_dir,
            sync_manager.get_all_tracked_file_hashes(),
            supported_file_count=len(supported_files),
        )
        elapsed = time.time() - start_time
        if _failed_files:
            console.print(
                f"[yellow]⚠ {len(_failed_files)} file(s) could not be parsed and will be "
                f"retried on the next index run (see the log for details).[/yellow]"
            )
        console.print(f"  [green]✓ Indexed {len(supported_files)} files in {elapsed:.2f}s[/green]")
    else:
        if changed_all_file_pairs:
            sync_manager.mark_files_synced_batch(changed_all_file_pairs)
            sync_manager.update_index_manifest(
                target_dir,
                sync_manager.get_all_tracked_file_hashes(),
                supported_file_count=len(supported_files),
            )
            console.print("  [green]✓ File snapshots updated (no supported code deltas)[/green]")
        else:
            sync_manager.update_index_manifest(
                target_dir,
                sync_manager.get_all_tracked_file_hashes(),
                supported_file_count=len(supported_files),
            )
            console.print("  [green]✓ Codebase is already up to date[/green]")
    console.print()

    # Step 4: Register MCP.
    console.print("[bold cyan]Step 4/4[/bold cyan] – Registering MCP for IDEs")
    
    # Resolve the server from the *installed package*, not the project being
    # indexed — target_dir only contains codetrace_mcp/ when the project happens
    # to be the CodeTrace source tree itself, so deriving the path from it wrote
    # a file:// that doesn't exist for every pip-installed user. sys.executable
    # keeps the IDE on the same interpreter codetrace is installed into.
    server_path = _resolve_mcp_server_path(target_dir)

    servers_config = {
        "codetrace": {
            "command": sys.executable,
            "args": [str(server_path), "--project", str(target_dir)],
        }
    }
    mcp_results = _register_mcp(servers_config, workspace_dir=target_dir)
    for msg in mcp_results:
        console.print(f"  {msg}")
    console.print()

    # Final Summary.
    console.print(Panel(
        f"[bold green]✅ Codetrace is ready![/bold green]\n\n"
        f"  Indexed:  [bold]{target_dir}[/bold]\n"
        f"  MCP:      registered for Claude Code, Cursor + VS Code (project config)\n\n"
        f"  [dim]Try:[/dim] [cyan]codetrace chat[/cyan]",
        title="Setup Complete",
        border_style="green"
    ))

@app.command(name="register-mcp", help="Register MCP server with IDEs (Cursor, Claude Code, VS Code).")
def register_mcp(
    path: str = typer.Argument(".", help="Path to the indexed project"),
):
    """Register the Codetrace MCP server with all supported IDEs."""
    target_dir = get_project_root(path)
    server_path = _resolve_mcp_server_path(target_dir)

    servers_config = {
        "codetrace": {
            "command": sys.executable,
            "args": [str(server_path), "--project", str(target_dir)],
        }
    }
    mcp_results = _register_mcp(servers_config, workspace_dir=target_dir)
    for msg in mcp_results:
        console.print(f"  {msg}")
    console.print()



@app.command()
def index(path: str = typer.Argument(".", help="Target directory or GitHub URL to index")):
    """
    Scan the codebase, extract AST, and index into the Graph & Vector databases.

    Supports both local directories and remote GitHub/GitLab URLs:
      codetrace index .
      codetrace index https://github.com/user/repo
      codetrace index https://github.com/user/repo/tree/develop
    """
    print_banner()
    print_system_info()

    # Is this a remote repo URL or a local path?
    cloned_dir = None
    repo_info = _parse_github_url(path)

    if repo_info:
        # Keep the checkout (and the index inside it) in a stable location so
        # `codetrace chat` can be run against it afterwards, and so indexing the
        # same URL again is an incremental update rather than a fresh clone.
        repo_dir = _repo_cache_dir(repo_info["clone_url"], repo_info["branch"])
        updating = (repo_dir / ".git").exists()
        action = "Updating" if updating else "Cloning"
        console.print(f"\n[bold blue]{action} remote repository...[/bold blue]")
        console.print(f"  URL:    [dim]{repo_info['clone_url']}[/dim]")
        if repo_info["branch"]:
            console.print(f"  Branch: [dim]{repo_info['branch']}[/dim]")
        console.print()

        try:
            status = "Fetching latest commit..." if updating else "Running git clone --depth 1..."
            with console.status(f"[bold cyan]{status}", spinner="dots"):
                cloned_dir = _clone_repo(repo_info["clone_url"], repo_info["branch"], dest=repo_dir)
            verb = "Updated" if updating else "Cloned to"
            console.print(f"  [green]\u2713 {verb}:[/green] [dim]{cloned_dir}[/dim]\n")
        except RuntimeError as e:
            console.print(f"[bold red]Clone failed:[/bold red] {e}")
            raise typer.Exit(1)

        target_dir = cloned_dir
    else:
        target_dir = get_project_root(path)

    db_dir = target_dir / ".codetrace"

    # Set up the .codetrace dir on the fly — mostly matters for fresh clones.
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        SyncManager(db_dir=str(db_dir))

    console.print(f"[bold blue]Starting Codetrace Indexer[/bold blue] \U0001f680")
    console.print(f"Target: [dim]{target_dir}[/dim]\n")

    # Bring up the databases and embedding models.
    with console.status("[bold cyan]Waking up Vector Models and DB connections...", spinner="point"):
        sync_manager = SyncManager(db_dir=str(db_dir))
        orchestrator = GraphOrchestrator()

        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)

        orchestrator.graph.db_path = db_dir / "graph_metadata.db"
        orchestrator.graph._init_db()
        orchestrator.graph.load_from_db()

    # Find the files to index.
    # Walk the tree, pruning ALWAYS_IGNORE_DIRS as we go so we never enumerate
    # their contents. filter_paths() applies the .gitignore rules afterward.
    _raw_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for fn in filenames:
            if _is_always_ignored_file(fn):
                continue
            _raw_files.append(Path(dirpath) / fn)
    _raw_files = orchestrator.graph.filter_paths(_raw_files, repo_root=target_dir)
    all_files = [str(p) for p in _raw_files]
    supported_files = [f for f, _ in orchestrator.iter_supported_files(all_files)]
    supported_set = set(supported_files)

    if not supported_files:
        console.print("[yellow]No supported code files found in this directory.[/yellow]")

    # Figure out what actually changed since the last index.
    changed_all_file_pairs = sync_manager.get_changed_files(all_files)
    changed_supported_file_pairs = [
        (file_path, file_hash)
        for file_path, file_hash in changed_all_file_pairs
        if file_path in supported_set
    ]
    deleted_files = sync_manager.get_deleted_files(all_files)

    if not changed_all_file_pairs and not deleted_files:
        sync_manager.update_index_manifest(
            target_dir,
            sync_manager.get_all_tracked_file_hashes(),
            supported_file_count=len(supported_files),
        )
        console.print("[bold green]\u2713 Codebase is fully up to date![/bold green]")
        if cloned_dir:
            _print_remote_chat_hint(cloned_dir)
        return

    # Drop anything that was deleted from disk.
    if deleted_files:
        with console.status(f"[bold red]Pruning {len(deleted_files)} deleted files..."):
            # Deleted files are gone from disk, so they're never in supported_set;
            # decide whether they were indexed from the path alone.
            _deleted_supported = [
                df for df, _ in orchestrator.iter_supported_files(deleted_files)
            ]
            if _deleted_supported:
                orchestrator.graph.prune_files(_deleted_supported)
                if vector_store:
                    vector_store.delete_by_file(_deleted_supported)
            for df in deleted_files:
                sync_manager.remove_file_record(df)
                sync_manager.remove_file_snapshot(df)

    if changed_all_file_pairs:
        with console.status(f"[bold cyan]Updating {len(changed_all_file_pairs)} file snapshot(s) in DB..."):
            for file_path, file_hash in changed_all_file_pairs:
                sync_manager.upsert_file_snapshot_from_disk(file_path, file_hash=file_hash)

    # Parse and embed everything that changed.
    if changed_supported_file_pairs:
        start_time = time.time()

        # Wipe the old graph nodes/edges and vectors for each changed file first,
        # otherwise symbols that got renamed or deleted inside a file would stick
        # around (for brand-new files this is a no-op). Has to happen before we
        # parse and before we seed the lookup maps below.
        _changed_supported_files = [fp for fp, _ in changed_supported_file_pairs]
        if vector_store:
            vector_store.delete_by_file(_changed_supported_files)
        # Calls INTO these files from unchanged files get deleted with them;
        # keep them so they can be restored once the files are re-parsed.
        _inbound_edges = orchestrator.graph.prune_files(_changed_supported_files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:

            task = progress.add_task(
                f"[cyan]Parsing & Indexing {len(changed_supported_file_pairs)} files...",
                total=len(changed_supported_file_pairs)
            )

            # Parse every changed file in parallel.
            MAX_WORKERS = max(4, os.cpu_count() or 4)
            all_symbols = []
            all_calls = []
            all_vs_ids = []
            all_vs_contents = []
            all_vs_metadatas = []
            _failed_files: set[str] = set()

            def _parse_one(file_path: str) -> dict:
                return orchestrator.extract_from_file(file_path)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(_parse_one, fp): fp for fp, _ in changed_supported_file_pairs}

                for fut in as_completed(futures):
                    fp = futures[fut]
                    try:
                        result = fut.result()
                        all_symbols.extend([(fp, s) for s in result["symbols"]])
                        all_calls.extend([(fp, c) for c in result["calls"]])
                        for v in result["vs_data"]:
                            all_vs_ids.append(v["id"])
                            all_vs_contents.append(v["content"])
                            all_vs_metadatas.append(v["metadata"])
                        progress.advance(task)
                    except Exception as e:
                        logger.error("Failed to parse %s: %s", fp, e)
                        _failed_files.add(fp)

            # Build the graph in one batch. Callees resolve against this batch
            # plus the unchanged files already in the graph (see build_batch).
            nodes_batch, edges_batch = orchestrator.build_batch(all_symbols, all_calls)
            orchestrator.graph.add_nodes_batch(nodes_batch)
            orchestrator.graph.add_edges_batch(edges_batch)
            orchestrator.graph.restore_inbound_edges(_inbound_edges)

            if vector_store and all_vs_ids:
                vector_store.add_symbols_batch(all_vs_ids, all_vs_contents, all_vs_metadatas)

        # Flush the graph and sync state to disk.
        with console.status("[bold magenta]Persisting Graph and Sync states..."):
            orchestrator.graph.persist_to_db()
            # Leave files that failed to parse unsynced so the next run retries
            # them, instead of silently treating them as indexed with no symbols.
            sync_manager.mark_files_synced_batch(
                [(fp, h) for fp, h in changed_all_file_pairs if fp not in _failed_files]
            )
            sync_manager.update_index_manifest(
                target_dir,
                sync_manager.get_all_tracked_file_hashes(),
                supported_file_count=len(supported_files),
            )

        elapsed = time.time() - start_time
        if _failed_files:
            console.print(
                f"[yellow]⚠ {len(_failed_files)} file(s) could not be parsed and will be "
                f"retried on the next index run (see the log for details).[/yellow]"
            )
        console.print(f"\n[bold green]\u2713 Indexing complete in {elapsed:.2f}s![/bold green]")
    else:
        if changed_all_file_pairs:
            sync_manager.mark_files_synced_batch(changed_all_file_pairs)
        sync_manager.update_index_manifest(
            target_dir,
            sync_manager.get_all_tracked_file_hashes(),
            supported_file_count=len(supported_files),
        )
        if changed_all_file_pairs:
            console.print("\n[bold green]\u2713 Snapshot update complete (no supported code deltas).[/bold green]")

    if cloned_dir:
        _print_remote_chat_hint(cloned_dir)


def _print_remote_chat_hint(repo_dir: Path) -> None:
    """Tell the user where an indexed remote repo lives and how to query it."""
    console.print(
        f"\n[dim]Repository and index kept at:[/dim] [bold]{repo_dir}[/bold]\n"
        f"[dim]To ask about it:[/dim] [cyan]cd \"{repo_dir}\" && codetrace chat[/cyan]"
    )

@app.command()
def config():
    """
    View or update your LLM provider configuration.
    """
    print_banner()

    global_dir = Path.home() / ".codetrace"
    config_path = global_dir / "config.json"

    if not config_path.exists():
        console.print("[yellow]No configuration found. Starting setup...[/yellow]\n")
        global_dir.mkdir(parents=True, exist_ok=True)
        _run_setup_wizard(config_path, is_reconfigure=False)
        return

    # Print what's configured, keeping the API key masked.
    with open(config_path) as f:
        cfg = json.load(f)

    summary = (
        f"[bold]Provider:[/bold]  {cfg.get('provider', 'N/A')}\n"
        f"[bold]API Key:[/bold]   {mask_key(cfg.get('api_key', ''))}\n"
        f"[bold]Model:[/bold]     {cfg.get('model_name') or '(default)'}"
    )
    # The endpoint and wire format are what define a custom provider, so they
    # belong in the summary — otherwise there's no way to tell two apart.
    if cfg.get("base_url"):
        summary += f"\n[bold]Endpoint:[/bold]  {cfg['base_url']}"
    if cfg.get("provider", "").lower() == "custom":
        summary += f"\n[bold]API Style:[/bold] {cfg.get('api_style', 'openai')}"
    if cfg.get("provider", "").lower() != "ollama":
        summary += (
            f"\n[bold]Context:[/bold]   "
            f"{cfg.get('context_window', 'auto-detect / fallback')}"
            f"\n[bold]Max output:[/bold] "
            f"{cfg.get('max_output_tokens', 'auto-detect')}"
        )

    console.print(Panel(
        summary,
        title="Current Configuration",
        border_style="cyan"
    ))

    # Don't clobber the existing config without a yes.
    overwrite = Prompt.ask(
        "\n[yellow]Overwrite this configuration?[/yellow]",
        choices=["y", "n"],
        default="n"
    )

    if overwrite.lower() == "y":
        _run_setup_wizard(config_path, is_reconfigure=True)
    else:
        console.print("[dim]Configuration unchanged.[/dim]")

@app.command(name="set-ctx")
def set_ctx(
    tokens: int = typer.Argument(
        -1,
        help="Ollama context window (num_ctx) in tokens, e.g. 16384. "
             "Use 0 to clear the override and return to hardware auto-detection. "
             "Omit to leave the window unchanged (e.g. when only setting --backoff).",
    ),
    backoff: float = typer.Option(
        None,
        "--backoff",
        help="Fraction of the window KEPT each time it auto-lowers under memory "
             "pressure (0.5 = halve, 0.75 = gentler). Range 0.25–0.9.",
    ),
):
    """
    Configure the Ollama context window (num_ctx) and its auto-lowering step.

    num_ctx overrides the automatic GPU-memory-based sizing. Larger windows
    retain more conversation history but reserve more GPU memory — too large can
    OOM or spill to CPU and slow generation. The back-off controls how gently the
    window shrinks when CodeTrace detects memory pressure.

    Examples:
      codetrace set-ctx 16384             # pin the window
      codetrace set-ctx 0                 # clear → auto-detect
      codetrace set-ctx --backoff 0.75    # gentler auto-lowering, window unchanged
    """
    print_banner()

    config_path = Path.home() / ".codetrace" / "config.json"
    if not config_path.exists():
        console.print("[red]No configuration found. Run 'codetrace config' first.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    if tokens == -1 and backoff is None:
        console.print(
            "[yellow]Nothing to change. Pass a token count and/or --backoff.[/yellow]"
        )
        raise typer.Exit(1)

    if cfg.get("provider", "").lower() != "ollama":
        console.print(
            "[yellow]Note: these settings only apply to the Ollama provider; "
            f"your current provider is '{cfg.get('provider', 'N/A')}'.[/yellow]"
        )

    # num_ctx (only when a value was supplied).
    if tokens == 0:
        cfg.pop("ollama_num_ctx", None)
        console.print(
            "[green]Cleared manual num_ctx — reverting to hardware auto-detection.[/green]"
        )
    elif tokens != -1:
        if tokens < 2048:
            console.print("[red]num_ctx must be at least 2048 tokens.[/red]")
            raise typer.Exit(1)
        cfg["ollama_num_ctx"] = tokens
        console.print(f"[green]Ollama num_ctx set to {tokens:,} tokens.[/green]")
        if tokens >= 65536:
            console.print(
                "[dim]Heads up: large windows need substantial GPU memory. "
                "If Ollama OOMs or slows down, lower this value.[/dim]"
            )

    # back-off factor.
    if backoff is not None:
        if not (0.25 <= backoff <= 0.9):
            console.print("[red]--backoff must be between 0.25 and 0.9.[/red]")
            raise typer.Exit(1)
        cfg["ollama_ctx_backoff"] = backoff
        console.print(
            f"[green]Auto-lower back-off set to {backoff:g} "
            f"(keeps {backoff:.0%} of the window per step).[/green]"
        )

    write_private_json(config_path, cfg)


@app.command(name="set-default-ctx")
def set_default_ctx(
    tokens: int = typer.Argument(
        -1,
        help="Default context window (in tokens) for cloud models that litellm "
             "doesn't recognise, e.g. 32000. Use 0 to clear and go back to the "
             "8K conservative fallback. Omit to just view the current value.",
    ),
):
    """
    Set the fallback context window for UNRECOGNISED cloud models.

    When a cloud provider's model isn't in litellm's registry (common for
    custom / self-hosted / brand-new models), CodeTrace normally falls back to
    a cramped 8K "unknown" tier. This lets you raise that fallback so an
    unmapped cloud model gets a realistic window instead.

    This ONLY affects cloud providers — never Ollama. Ollama sizes its window
    from GPU memory (num_ctx), and a large default there would OOM, so the 8K
    fallback is deliberately kept for Ollama.

    Examples:
      codetrace set-default-ctx 32000      # unmapped cloud models use 32K
      codetrace set-default-ctx 0          # clear -> back to 8K fallback
      codetrace set-default-ctx            # just show the current value
    """
    print_banner()

    config_path = Path.home() / ".codetrace" / "config.json"
    if not config_path.exists():
        console.print("[red]No configuration found. Run 'codetrace config' first.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    current = cfg.get("default_context_window")

    if tokens == -1:
        if current:
            console.print(
                f"[cyan]Current cloud default context window: {current:,} tokens.[/cyan]"
            )
        else:
            console.print(
                "[cyan]No cloud default set - unmapped cloud models fall back to 8K.[/cyan]"
            )
        return

    if tokens == 0:
        cfg.pop("default_context_window", None)
        console.print(
            "[green]Cleared cloud default - unmapped cloud models now fall back to 8K.[/green]"
        )
    else:
        if tokens < 2048:
            console.print("[red]Default context window must be at least 2048 tokens.[/red]")
            raise typer.Exit(1)
        cfg["default_context_window"] = tokens
        console.print(
            f"[green]Cloud default context window set to {tokens:,} tokens.[/green]\n"
            f"[dim]Applies only to cloud models litellm doesn't recognise; "
            f"Ollama is unaffected.[/dim]"
        )

    write_private_json(config_path, cfg)


@app.command(name="set-model-limits")
def set_model_limits(
    context_window: int | None = typer.Option(
    None,
    "--context-window",
    help="Actual maximum context window for the configured cloud model.",
    ),
    max_output_tokens: int | None = typer.Option(
        None,
        "--max-output-tokens",
        help="Maximum completion tokens for the configured cloud model.",
    ),
):
    """Set explicit limits for the currently configured cloud model."""
    config_path = Path.home() / ".codetrace" / "config.json"

    if not config_path.exists():
        console.print("[red]No configuration found. Run 'codetrace config' first.[/red]")
        raise typer.Exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("provider", "").lower() == "ollama":
        console.print(
            "[red]This command is for cloud models only. "
            "Use 'codetrace set-ctx' for Ollama.[/red]"
        )
        raise typer.Exit(1)

    if context_window is None and max_output_tokens is None:
        console.print(
            f"Model: {cfg.get('model_name', 'unknown')}\n"
            f"Context window: {cfg.get('context_window', 'auto')}\n"
            f"Max output tokens: {cfg.get('max_output_tokens', 'auto')}"
        )
        return

    if context_window is not None:
        if context_window < 2048:
            console.print("[red]--context-window must be at least 2048.[/red]")
            raise typer.Exit(1)
        cfg["context_window"] = context_window

    if max_output_tokens is not None:
        if max_output_tokens < 128:
            console.print("[red]--max-output-tokens must be at least 128.[/red]")
            raise typer.Exit(1)

        effective_window = context_window or cfg.get("context_window")
        if isinstance(effective_window, int) and max_output_tokens >= effective_window:
            console.print(
                "[red]--max-output-tokens must be smaller than the context window.[/red]"
            )
            raise typer.Exit(1)

        cfg["max_output_tokens"] = max_output_tokens

    write_private_json(config_path, cfg)

    console.print(
        f"[green]Saved limits for {cfg.get('model_name', 'configured model')}:[/green]\n"
        f"  Context window: {cfg.get('context_window', 'auto')}\n"
        f"  Max output:    {cfg.get('max_output_tokens', 'auto')}"
    )


@app.command()
def visualize(path: str = typer.Argument(".", help="Target directory")):
    """
    Export the code dependency graph as an interactive HTML visualization.

    Generates a self-contained HTML file showing a folder-level architecture
    map with cross-folder function call edges, hover tooltips, search, and
    a detail sidebar.
    """
    print_banner()
    target_dir = get_project_root(path)
    db_dir = target_dir / ".codetrace"

    if not db_dir.exists():
        console.print("[red]Error: Not a Codetrace repository. Run 'codetrace init' and 'codetrace index .' first.[/red]")
        raise typer.Exit(1)

    with console.status("[bold cyan]Loading graph...", spinner="point"):
        graph = CodeGraph()
        graph.db_path = db_dir / "graph_metadata.db"
        graph._init_db()
        graph.load_from_db()

    node_count = graph.direct_graph.number_of_nodes()
    edge_count = graph.direct_graph.number_of_edges()

    if node_count == 0:
        console.print("[yellow]Graph is empty. Run 'codetrace index .' first.[/yellow]")
        return

    with console.status("[bold cyan]Generating interactive visualization...", spinner="dots"):
        viz_data = _build_viz_data(graph, target_dir)

        from src.cli.visualization_template import render as render_viz
        html = render_viz(viz_data)

        output_path = db_dir / "graph_visualization.html"
        output_path.write_text(html, encoding="utf-8")

    cross_calls = viz_data["totalConnections"]
    folder_count = viz_data["totalFolders"]

    console.print(Panel(
        f"[green]Graph exported successfully![/green]\n\n"
        f"Nodes: [bold]{node_count}[/bold]  |  Edges: [bold]{edge_count}[/bold]\n"
        f"Folders: [bold]{folder_count}[/bold]  |  Cross-folder calls: [bold]{cross_calls}[/bold]\n\n"
        f"Open in browser: [bold cyan]{output_path}[/bold cyan]\n\n"
        f"[bold]Interactions:[/bold]\n"
        f"  [dim]Hover[/dim]  folder \u2192 highlight connections & show breakdown\n"
        f"  [dim]Hover[/dim]  edge \u2192 see which functions call each other\n"
        f"  [dim]Click[/dim]  folder \u2192 open detail sidebar (files, symbols, deps)\n"
        f"  [dim]Search[/dim] \u2192 find symbols and highlight their folder\n"
        f"  [dim]Drag[/dim]   \u2192 rearrange layout  |  [dim]Scroll[/dim] \u2192 zoom",
        title="Code Architecture Visualization",
        border_style="cyan"
    ))


def _build_viz_data(graph: "CodeGraph", target_dir: Path) -> dict:
    """
    Pull folder-level architecture data out of the CodeGraph.

    We bucket every symbol under its parent folder (relative to the project root),
    then roll up the cross-folder call edges — carrying the real function names
    along — so the visualization can show both the big-picture layout and the
    individual function-to-function calls behind each edge.
    """
    resolved_root = target_dir.resolve()

    # First pass: group nodes by file and folder.
    files_map: dict[str, dict] = {}   # relative_path -> {folder, name, symbols}
    folders_map: dict[str, dict] = {} # folder_id -> {files, types}

    for node_id, data in graph.direct_graph.nodes(data=True):
        n_type = data.get("type", "unknown")
        n_file = data.get("file", "")
        if not n_file:
            continue

        # Make the path relative and use forward slashes everywhere.
        try:
            rel_path = str(Path(n_file).resolve().relative_to(resolved_root))
        except ValueError:
            rel_path = Path(n_file).name
        rel_path = rel_path.replace("\\", "/")

        # Split off the folder.
        parts = rel_path.rsplit("/", 1)
        if len(parts) == 2:
            folder, file_name = parts
        else:
            folder, file_name = "(root)", parts[0]

        # The symbol name is the tail of the node_id (after the last ':').
        symbol_name = node_id.rsplit(":", 1)[-1] if ":" in node_id else node_id

        # Record the file.
        if rel_path not in files_map:
            files_map[rel_path] = {"folder": folder, "name": file_name, "symbols": []}
        files_map[rel_path]["symbols"].append({
            "id": node_id, "name": symbol_name, "type": n_type,
        })

        # ...and the folder.
        if folder not in folders_map:
            folders_map[folder] = {"files": set(), "types": {}}
        folders_map[folder]["files"].add(rel_path)
        folders_map[folder]["types"][n_type] = folders_map[folder]["types"].get(n_type, 0) + 1

    # Second pass: roll up the call edges that cross folder boundaries.
    folder_edges_map: dict[tuple[str, str], dict] = {}

    for src, tgt, data in graph.direct_graph.edges(data=True):
        if data.get("relation") != "calls":
            continue

        src_file = graph.direct_graph.nodes.get(src, {}).get("file", "")
        tgt_file = graph.direct_graph.nodes.get(tgt, {}).get("file", "")
        if not src_file or not tgt_file or src_file == tgt_file:
            continue

        try:
            src_rel = str(Path(src_file).resolve().relative_to(resolved_root)).replace("\\", "/")
            tgt_rel = str(Path(tgt_file).resolve().relative_to(resolved_root)).replace("\\", "/")
        except ValueError:
            continue

        src_parts = src_rel.rsplit("/", 1)
        tgt_parts = tgt_rel.rsplit("/", 1)
        src_folder = src_parts[0] if len(src_parts) == 2 else "(root)"
        tgt_folder = tgt_parts[0] if len(tgt_parts) == 2 else "(root)"

        if src_folder == tgt_folder:
            continue

        key = (src_folder, tgt_folder)
        if key not in folder_edges_map:
            folder_edges_map[key] = {"weight": 0, "calls": []}
        folder_edges_map[key]["weight"] += 1

        # Keep a few concrete calls behind each edge (capped so it stays light).
        if len(folder_edges_map[key]["calls"]) < 25:
            src_name = src.rsplit(":", 1)[-1] if ":" in src else src
            tgt_name = tgt.rsplit(":", 1)[-1] if ":" in tgt else tgt
            folder_edges_map[key]["calls"].append({
                "sourceName": src_name,
                "targetName": tgt_name,
                "sourceFile": src_parts[-1],
                "targetFile": tgt_parts[-1],
            })

    # Assemble the JSON payload the template consumes.
    folders_list = []
    for fid, info in sorted(folders_map.items()):
        symbol_count = sum(info["types"].values())
        dominant = max(info["types"], key=info["types"].get) if info["types"] else "unknown"
        folders_list.append({
            "id": fid, "name": fid,
            "fileCount": len(info["files"]),
            "symbolCount": symbol_count,
            "types": info["types"],
            "dominantType": dominant,
        })

    edges_list = [
        {"source": src, "target": tgt, "weight": info["weight"], "calls": info["calls"]}
        for (src, tgt), info in folder_edges_map.items()
    ]

    files_list = [
        {"path": p, "folder": info["folder"], "name": info["name"], "symbols": info["symbols"]}
        for p, info in sorted(files_map.items())
    ]

    return {
        "projectName": target_dir.name,
        "folders": folders_list,
        "folderEdges": edges_list,
        "files": files_list,
        "totalSymbols": sum(f["symbolCount"] for f in folders_list),
        "totalFiles": len(files_map),
        "totalFolders": len(folders_map),
        "totalConnections": sum(e["weight"] for e in edges_list),
    }

@app.command()
def history():
    """List recent chat sessions for this project."""
    target_dir = get_project_root(".")
    db_dir = target_dir / ".codetrace"
    db_path = db_dir / "chat_history.db"

    if not db_path.exists():
        console.print("[yellow]No chat history found. Run 'codetrace chat' first.[/yellow]")
        raise typer.Exit(0)

    store = ChatStore(db_path)
    sessions = store.list_sessions()
    store.close()

    if not sessions:
        console.print("[yellow]No chat sessions found.[/yellow]")
        raise typer.Exit(0)

    from rich.table import Table

    table = Table(title="Chat History", border_style="cyan")
    table.add_column("ID", style="bold cyan", width=10)
    table.add_column("First Question", style="white")
    table.add_column("Messages", justify="center", style="green")
    table.add_column("Last Active", style="dim")

    for s in sessions:
        table.add_row(
            s["id"],
            s["title"],
            str(s["message_count"]),
            s["updated_at"] or "",
        )

    console.print(table)
    console.print("\n[dim]Resume a session:[/dim] [cyan]codetrace chat --resume <ID>[/cyan]")

@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout)"),
):
    """Export a chat session as a Markdown document."""
    target_dir = get_project_root(".")
    db_dir = target_dir / ".codetrace"
    db_path = db_dir / "chat_history.db"

    if not db_path.exists():
        console.print("[red]No chat history found.[/red]")
        raise typer.Exit(1)

    store = ChatStore(db_path)

    if not store.session_exists(session_id):
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        store.close()
        raise typer.Exit(1)

    md = store.export_session(session_id)
    store.close()

    if output:
        Path(output).write_text(md, encoding="utf-8")
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(Markdown(md))


if __name__ == "__main__":
    app()
@app.command()
def mcp(
    project: str = typer.Argument(".", help="Path to the indexed project"),
    # Kept only so existing scripts don't break: the server speaks MCP over
    # stdio, which has no host or port. Hidden from --help.
    port: Optional[int] = typer.Option(None, "--port", "-p", hidden=True),
    host: Optional[str] = typer.Option(None, "--host", "-h", hidden=True),
):
    """
    Start the Codetrace MCP server (stdio) for IDE integration.

    This exposes code analysis tools to AI-powered IDEs like Cursor, VS Code, Claude Desktop, etc.
    The IDE launches this command and talks to it over stdin/stdout.
    """
    # stdout carries the JSON-RPC stream, so every human-facing line goes to
    # stderr — a banner on stdout would corrupt the protocol for the client.
    err = Console(stderr=True, legacy_windows=False)
    _print_banner(err)

    if port is not None or host is not None:
        err.print(
            "[yellow]Note: --port/--host are ignored. The MCP server uses stdio "
            "and is launched by your IDE, not reached over HTTP.[/yellow]"
        )

    # Check if project is indexed
    project_path = Path(project).resolve()
    db_dir = project_path / ".codetrace"

    if not db_dir.exists():
        err.print(f"[bold red]Error: No .codetrace directory found at {db_dir}[/bold red]")
        err.print("Run 'codetrace index .' on the project first.")
        raise typer.Exit(1)

    err.print(f"[bold cyan]Starting MCP server (stdio) for project: {project_path}[/bold cyan]")
    err.print()

    # Import and start the MCP server
    try:
        from codetrace_mcp.server import main as mcp_main
        import asyncio

        # Run the MCP server
        asyncio.run(mcp_main(project))

    except ImportError as e:
        err.print(f"[bold red]Error: MCP server not available[/bold red]")
        err.print(f"Make sure codetrace_mcp is installed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        err.print(f"[bold red]MCP server error: {e}[/bold red]")
        raise typer.Exit(1)
