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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Windows: Force UTF-8 so Rich Unicode renders correctly in PowerShell or Windows Terminal.
# Two layers are required:
#   1. SetConsoleOutputCP(65001) sets the Win32 console code page to UTF-8.
#   2. sys.stdout.reconfigure() makes Python's stream write UTF-8 bytes.
if sys.platform == "win32":
    # Layer 1: Set Win32 console code page to UTF-8
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    # Layer 2: Reconfigure Python streams
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Fallback for Python versions below 3.7.
    # Layer 3: Disable Rich's LegacyWindowsTerm renderer.
    # Rich defaults to using _win32_console.py on older Windows consoles, which
    # bypasses the UTF-8 fixes. Setting RICH_LEGACY_WINDOWS=0 forces the standard ANSI path.
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
from src.ignore import ALWAYS_IGNORE_DIRS, _is_always_ignored

logger = logging.getLogger(__name__)

# Silence verbose weight-loading logs from HuggingFace and Transformers.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Silence httpx HTTP request logs and FlashRank progress bars.
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

# Set legacy_windows=False to prevent Rich from using the Win32 LegacyWindowsTerm renderer, which causes Unicode garbling.
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


def get_project_root(path: str) -> Path:
    return _get_project_root(path, console)


def ensure_config() -> None:
    _ensure_config(console)


def _run_setup_wizard(config_path: Path, is_reconfigure: bool = False) -> None:
    _run_setup_wizard_impl(config_path, console, is_reconfigure=is_reconfigure)

def collect_source_files(root: Path) -> list[Path]:
    """
    Walk through the directory and collect all source code files,
    while respecting .gitignore rules AND the hardcoded ALWAYS_IGNORE_DIRS.

    Uses os.walk with in-place directory pruning so we never descend into
    node_modules/, venv/, __pycache__/, etc. \u2014 avoiding enumerating 20k+ files.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune always-ignored dirs IN-PLACE so os.walk never enters them.
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for filename in filenames:
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

    # Trigger the interactive setup if config is missing
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
        # Load the databases.
        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)

        graph = CodeGraph()
        graph.db_path = db_dir / "graph_metadata.db"
        graph._init_db()
        graph.load_from_db()

        # Initialize the httpx-powered Agent.
        try:
            agent = AgentOrchestrator(vector_store, graph)
        except ValueError as e:
            # Catch errors from the retriever if the API key is invalid.
            console.print(f"[red]Configuration Error: {e}[/red]")
            raise typer.Exit(1)

    console.print("[bold green]Architect is online! Type 'exit' or 'quit' to stop.[/bold green]")

    # Initialize ChatStore
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

    # 4. The Continuous Chat Loop
    while True:
        # Prompt the user for a question.
        query = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        # Allow the user to exit the loop.
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

        # Guardrail: Catch accidental API key pastes.
        if looks_like_api_key(query):
            console.print(
                "[bold yellow]⚠  That looks like an API key, not a question![/bold yellow]\n"
                "[dim]Your key was NOT sent to the model.\n"
                "To update your configuration, run:[/dim] [cyan]codetrace config[/cyan]"
            )
            continue

        # Execute the agentic pipeline.
        try:
            # Tool label mapping (ASCII-safe).
            tool_labels = {
                "search_codebase":      "Searching codebase",
                "inspect_index":        "Inspecting index",
                "get_symbol_relations": "Tracing symbol relations",
                "read_file":            "Reading file",
                "analyze_impact":       "Analyzing impact",
                "write_file":           "Proposing file change",
                "git_diff":             "Running git diff",
            }

            streaming_started = False
            full_response = ""
            live = None
            # Transient status spinner shown while a tool is executing.
            # Stopped and erased when the tool finishes or streaming begins.
            _tool_status = None

            for event in agent.stream(query, chat_history=chat_store.get_history_for_llm(session_id)):
                evt_type = event["type"]

                if evt_type == "thought":
                    # Show a transient spinner for the current tool; it will be erased automatically.
                    if _tool_status:
                        _tool_status.stop()
                    tool_name = event.get("tool", "")
                    label = tool_labels.get(tool_name, "Working")
                    # Append the tool argument detail from the event message.
                    detail = event.get("message", "")
                    # Provide context detail from the tool message.
                    _tool_status = console.status(
                        f"[dim]{detail}[/dim]",
                        spinner="dots",
                    )
                    _tool_status.start()

                elif evt_type == "tool_end":
                    # Erase the spinner without leaving a permanent line.
                    if _tool_status:
                        _tool_status.stop()
                        _tool_status = None

                elif evt_type == "token":
                    token_text = event.get("content", "")
                    if not isinstance(token_text, str):
                        token_text = str(token_text)
                    if not token_text:
                        continue

                    # Stop any lingering tool spinner before streaming begins.
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

                elif evt_type == "done":
                    if _tool_status:
                        _tool_status.stop()
                        _tool_status = None
                    if live:
                        live.stop()
                    if not streaming_started and not full_response:
                        # The agent finished without streaming tokens.
                        pass

                elif evt_type == "error":
                    # Surface async producer exceptions (API errors, auth failures, etc.)
                    if _tool_status:
                        _tool_status.stop()
                        _tool_status = None
                    if live:
                        live.stop()
                        live = None
                    err_msg = event.get("message", "Unknown error")
                    console.print(f"\n[bold red]Architect Error:[/bold red] {err_msg}")
                    break

                elif evt_type == "usage":
                    # Token counter (this is the only persistent tool-activity line).
                    turn_usage = event.get("turn")
                    if turn_usage:
                        console.print(turn_usage.format())

            # Process pending writes (human-in-the-loop).
            # This runs once per turn, after all events have been processed.
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
                            result = write_file_impl(pw["file_path"], pw["content"])
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

            # Save messages to chat history.
            chat_store.add_message(session_id, "user", query)
            if full_response:
                chat_store.add_message(session_id, "assistant", full_response)

        except Exception as e:
            if _tool_status:
                try:
                    _tool_status.stop()
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
        if llm:
            # Non-interactive: Auto-configure with defaults.
            config_data = {"provider": llm.lower(), "api_key": "", "model_name": "", "base_url": ""}
            if llm.lower() != "ollama":
                api_key = Prompt.ask(f"  [cyan]Enter your {llm.upper()} API Key[/cyan]", password=True)
                config_data["api_key"] = api_key
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
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

    # Discover files using os.walk with in-place pruning.
    # ALWAYS_IGNORE_DIRS are pruned from the walk to avoid enumerating ignored contents.
    # filter_paths then applies all .gitignore rules.
    _raw_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for fn in filenames:
            _raw_files.append(Path(dirpath) / fn)
    _raw_files = orchestrator.graph.filter_paths(_raw_files, repo_root=target_dir)
    all_files = [str(p) for p in _raw_files]
    supported_files = [f for f, _ in orchestrator.parser.iter_supported_files(all_files)]
    supported_set = set(supported_files)

    changed_all_file_pairs = sync_manager.get_changed_files(all_files)
    deleted_files = sync_manager.get_deleted_files(all_files)

    if deleted_files:
        for df in deleted_files:
            if df in supported_set:
                orchestrator.graph.prune_files([df])
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

            # Parse all changed files in parallel.
            MAX_WORKERS = max(4, os.cpu_count() or 4)
            all_symbols = []
            all_calls = []
            all_vs_ids = []
            all_vs_contents = []
            all_vs_metadatas = []

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

            # Batch build the graph.
            name_to_id: dict[str, str] = {}
            qualified_to_id: dict[str, str] = {}
            nodes_batch = []
            edges_batch = []

            for file_path, s in all_symbols:
                qualified_name = s.get("qualified_name") or s["name"]
                symbol_id = f"{file_path}:{qualified_name}"
                nodes_batch.append((symbol_id, s["type"], file_path))
                name_to_id[s["name"]] = symbol_id
                qualified_to_id[qualified_name] = symbol_id

            for file_path, c in all_calls:
                caller_id = f"{file_path}:{c['caller']}"
                callee_id = qualified_to_id.get(c["callee"]) or name_to_id.get(c["callee"])
                if not callee_id:
                    candidates = [
                        sid for qn, sid in qualified_to_id.items()
                        if qn.endswith(f".{c['callee']}")
                    ]
                    if len(candidates) == 1:
                        callee_id = candidates[0]
                edges_batch.append((caller_id, callee_id or c["callee"]))

            orchestrator.graph.add_nodes_batch(nodes_batch)
            orchestrator.graph.add_edges_batch(edges_batch)

            if vector_store and all_vs_ids:
                vector_store.add_symbols_batch(all_vs_ids, all_vs_contents, all_vs_metadatas)

        orchestrator.graph.persist_to_db()
        sync_manager.mark_files_synced_batch(changed_all_file_pairs)
        sync_manager.update_index_manifest(
            target_dir,
            sync_manager.get_all_tracked_file_hashes(),
            supported_file_count=len(supported_files),
        )
        elapsed = time.time() - start_time
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
    mcp_results = _register_mcp(target_dir)
    for msg in mcp_results:
        console.print(f"  {msg}")
    console.print()

    # Final Summary.
    console.print(Panel(
        f"[bold green]✅ Codetrace is ready![/bold green]\n\n"
        f"  Indexed:  [bold]{target_dir}[/bold]\n"
        f"  MCP:      registered for Cursor + Claude Code\n\n"
        f"  [dim]Try:[/dim] [cyan]codetrace chat[/cyan]",
        title="Setup Complete",
        border_style="green"
    ))

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

    # Detect remote repo URL vs local path.
    cloned_dir = None
    repo_info = _parse_github_url(path)

    if repo_info:
        console.print(f"\n[bold blue]Cloning remote repository...[/bold blue]")
        console.print(f"  URL:    [dim]{repo_info['clone_url']}[/dim]")
        if repo_info["branch"]:
            console.print(f"  Branch: [dim]{repo_info['branch']}[/dim]")
        console.print()

        try:
            with console.status("[bold cyan]Running git clone --depth 1...", spinner="dots"):
                cloned_dir = _clone_repo(repo_info["clone_url"], repo_info["branch"])
            console.print(f"  [green]\u2713 Cloned to:[/green] [dim]{cloned_dir}[/dim]\n")
        except RuntimeError as e:
            console.print(f"[bold red]Clone failed:[/bold red] {e}")
            raise typer.Exit(1)

        target_dir = cloned_dir
    else:
        target_dir = get_project_root(path)

    db_dir = target_dir / ".codetrace"

    # Auto-initialize if necessary, especially for cloned repositories.
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        SyncManager(db_dir=str(db_dir))

    console.print(f"[bold blue]Starting Codetrace Indexer[/bold blue] \U0001f680")
    console.print(f"Target: [dim]{target_dir}[/dim]\n")

    # 1. Initialize databases and models.
    with console.status("[bold cyan]Waking up Vector Models and DB connections...", spinner="point"):
        sync_manager = SyncManager(db_dir=str(db_dir))
        orchestrator = GraphOrchestrator()

        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)

        orchestrator.graph.db_path = db_dir / "graph_metadata.db"
        orchestrator.graph._init_db()
        orchestrator.graph.load_from_db()

    # 2. Discover files
    # Discover files using os.walk with in-place pruning.
    # ALWAYS_IGNORE_DIRS are pruned from the walk to avoid enumerating ignored contents.
    # filter_paths then applies all .gitignore rules.
    _raw_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]
        for fn in filenames:
            _raw_files.append(Path(dirpath) / fn)
    _raw_files = orchestrator.graph.filter_paths(_raw_files, repo_root=target_dir)
    all_files = [str(p) for p in _raw_files]
    supported_files = [f for f, _ in orchestrator.parser.iter_supported_files(all_files)]
    supported_set = set(supported_files)

    if not supported_files:
        console.print("[yellow]No supported code files found in this directory.[/yellow]")

    # 3. Calculate Deltas (What changed?)
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
            shutil.rmtree(cloned_dir, ignore_errors=True)
        return

    # 4. Handle Deletions (Pruning)
    if deleted_files:
        with console.status(f"[bold red]Pruning {len(deleted_files)} deleted files..."):
            for df in deleted_files:
                if df in supported_set:
                    orchestrator.graph.prune_files([df])
                sync_manager.remove_file_record(df)
                sync_manager.remove_file_snapshot(df)

    if changed_all_file_pairs:
        with console.status(f"[bold cyan]Updating {len(changed_all_file_pairs)} file snapshot(s) in DB..."):
            for file_path, file_hash in changed_all_file_pairs:
                sync_manager.upsert_file_snapshot_from_disk(file_path, file_hash=file_hash)

    # 5. Ingestion Loop (Parsing & Embedding)
    if changed_supported_file_pairs:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:

            task = progress.add_task(
                f"[cyan]Parsing & Indexing {len(changed_supported_file_pairs)} files...",
                total=len(changed_supported_file_pairs)
            )

            # Parse all changed files in parallel.
            MAX_WORKERS = max(4, os.cpu_count() or 4)
            all_symbols = []
            all_calls = []
            all_vs_ids = []
            all_vs_contents = []
            all_vs_metadatas = []

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

            # Batch build the graph.
            name_to_id: dict[str, str] = {}
            qualified_to_id: dict[str, str] = {}
            nodes_batch = []
            edges_batch = []

            for file_path, s in all_symbols:
                qualified_name = s.get("qualified_name") or s["name"]
                symbol_id = f"{file_path}:{qualified_name}"
                nodes_batch.append((symbol_id, s["type"], file_path))
                name_to_id[s["name"]] = symbol_id
                qualified_to_id[qualified_name] = symbol_id

            for file_path, c in all_calls:
                caller_id = f"{file_path}:{c['caller']}"
                callee_id = qualified_to_id.get(c["callee"]) or name_to_id.get(c["callee"])
                if not callee_id:
                    candidates = [
                        sid for qn, sid in qualified_to_id.items()
                        if qn.endswith(f".{c['callee']}")
                    ]
                    if len(candidates) == 1:
                        callee_id = candidates[0]
                edges_batch.append((caller_id, callee_id or c["callee"]))

            orchestrator.graph.add_nodes_batch(nodes_batch)
            orchestrator.graph.add_edges_batch(edges_batch)

            if vector_store and all_vs_ids:
                vector_store.add_symbols_batch(all_vs_ids, all_vs_contents, all_vs_metadatas)

        # 6. Save State.
        with console.status("[bold magenta]Persisting Graph and Sync states..."):
            orchestrator.graph.persist_to_db()
            sync_manager.mark_files_synced_batch(changed_all_file_pairs)
            sync_manager.update_index_manifest(
                target_dir,
                sync_manager.get_all_tracked_file_hashes(),
                supported_file_count=len(supported_files),
            )

        elapsed = time.time() - start_time
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

    # 7. Cleanup cloned repository.
    if cloned_dir:
        console.print(f"\n[dim]Cloned repo cleaned up. Index stored at: {db_dir}[/dim]")
        # Note: we keep the .codetrace dir within the temp clone for now.
        # Future: move it to a persistent location.

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

    # Show current configuration with masked key.
    with open(config_path) as f:
        cfg = json.load(f)

    console.print(Panel(
        f"[bold]Provider:[/bold]  {cfg.get('provider', 'N/A')}\n"
        f"[bold]API Key:[/bold]   {mask_key(cfg.get('api_key', ''))}\n"
        f"[bold]Model:[/bold]     {cfg.get('model_name') or '(default)'}",
        title="Current Configuration",
        border_style="cyan"
    ))

    # Ask for confirmation before overwriting.
    overwrite = Prompt.ask(
        "\n[yellow]Overwrite this configuration?[/yellow]",
        choices=["y", "n"],
        default="n"
    )

    if overwrite.lower() == "y":
        _run_setup_wizard(config_path, is_reconfigure=True)
    else:
        console.print("[dim]Configuration unchanged.[/dim]")

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
    Extract folder-level architecture data from the CodeGraph.

    Groups all symbols by their parent folder (relative to project root),
    then aggregates cross-folder call edges with the actual function names
    so the visualization can show both the high-level architecture and the
    specific function-to-function connections.
    """
    resolved_root = target_dir.resolve()

    # Pass 1: Group nodes by file and folder.
    files_map: dict[str, dict] = {}   # relative_path -> {folder, name, symbols}
    folders_map: dict[str, dict] = {} # folder_id -> {files, types}

    for node_id, data in graph.direct_graph.nodes(data=True):
        n_type = data.get("type", "unknown")
        n_file = data.get("file", "")
        if not n_file:
            continue

        # Make path relative and normalize separators.
        try:
            rel_path = str(Path(n_file).resolve().relative_to(resolved_root))
        except ValueError:
            rel_path = Path(n_file).name
        rel_path = rel_path.replace("\\", "/")

        # Determine folder.
        parts = rel_path.rsplit("/", 1)
        if len(parts) == 2:
            folder, file_name = parts
        else:
            folder, file_name = "(root)", parts[0]

        # Extract symbol name from the node_id.
        symbol_name = node_id.rsplit(":", 1)[-1] if ":" in node_id else node_id

        # Track file.
        if rel_path not in files_map:
            files_map[rel_path] = {"folder": folder, "name": file_name, "symbols": []}
        files_map[rel_path]["symbols"].append({
            "id": node_id, "name": symbol_name, "type": n_type,
        })

        # Track folder.
        if folder not in folders_map:
            folders_map[folder] = {"files": set(), "types": {}}
        folders_map[folder]["files"].add(rel_path)
        folders_map[folder]["types"][n_type] = folders_map[folder]["types"].get(n_type, 0) + 1

    # Pass 2: Aggregate cross-folder call edges.
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

        # Record individual call details.
        if len(folder_edges_map[key]["calls"]) < 25:
            src_name = src.rsplit(":", 1)[-1] if ":" in src else src
            tgt_name = tgt.rsplit(":", 1)[-1] if ":" in tgt else tgt
            folder_edges_map[key]["calls"].append({
                "sourceName": src_name,
                "targetName": tgt_name,
                "sourceFile": src_parts[-1],
                "targetFile": tgt_parts[-1],
            })

    # Build final JSON structure.
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
