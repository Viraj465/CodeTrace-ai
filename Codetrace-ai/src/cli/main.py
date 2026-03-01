"""
Codetrace-ai: Autonomous entry point to code engine.
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

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

# Silence HuggingFace / Transformers verbose weight-loading spam.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

app = typer.Typer(
    name="codetrace",
    help="CodeTrace-ai: Autonomous System Architect",
    add_completion=False,
)

console = Console()


def print_banner() -> None:
    _print_banner(console)


def get_project_root(path: str) -> Path:
    return _get_project_root(path, console)


def ensure_config() -> None:
    _ensure_config(console)


def _run_setup_wizard(config_path: Path, is_reconfigure: bool = False) -> None:
    _run_setup_wizard_impl(config_path, console, is_reconfigure=is_reconfigure)


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
        # Load the databases
        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)
        
        graph = CodeGraph()
        graph.db_path = db_dir / "graph_metadata.db"
        graph._init_db()
        graph.load_from_db()
        
        # Initialize the LangChain Agent
        try:
            agent = AgentOrchestrator(vector_store, graph)
        except ValueError as e:
            # Catches errors from retriever.py if the API key is invalid
            console.print(f"[red]Configuration Error: {e}[/red]")
            raise typer.Exit(1)
            
    console.print("\n[bold green]✓ Architect is online! Type 'exit' or 'quit' to stop.[/bold green]")

    # ── Initialize ChatStore ──
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

    console.print("─" * 60)

    # 4. The Continuous Chat Loop
    while True:
        # Prompt the user for a question
        query = Prompt.ask("\n[bold blue]❯ You[/bold blue]")
        
        # Allow the user to exit the loop
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

        # ── Guardrail: catch accidental API key paste ──
        if looks_like_api_key(query):
            console.print(
                "[bold yellow]⚠  That looks like an API key, not a question![/bold yellow]\n"
                "[dim]Your key was NOT sent to the model.\n"
                "To update your configuration, run:[/dim] [cyan]codetrace config[/cyan]"
            )
            continue

        # Execute the agentic pipeline
        try:
            # Tool icon mapping
            tool_icons = {
                "search_codebase": "🔍",
                "inspect_index": "🗂️",
                "get_symbol_relations": "🔗",
                "read_file": "📄",
                "analyze_impact": "📊",
                "write_file": "✏️",
                "git_diff": "📝",
            }

            streaming_started = False
            full_response = ""
            live = None

            for event in agent.stream(query, chat_history=chat_store.get_history_for_llm(session_id)):
                evt_type = event["type"]

                if evt_type == "thought":
                    icon = tool_icons.get(event.get("tool", ""), "🔧")
                    console.print(f"  {icon} [dim italic]{event['message']}[/dim italic]")

                elif evt_type == "tool_end":
                    console.print(f"     [dim green]✓ done[/dim green]")

                elif evt_type == "token":
                    token_text = event.get("content", "")
                    if not isinstance(token_text, str):
                        token_text = str(token_text)
                    if not token_text:
                        continue

                    if not streaming_started:
                        console.print("\n[bold dark_orange]Architect:[/bold dark_orange]")
                        streaming_started = True
                        live = Live(Markdown(""), console=console, refresh_per_second=10)
                        live.start()
                    full_response += token_text
                    live.update(Markdown(full_response))

                elif evt_type == "done":
                    if live:
                        live.stop()
                    if not streaming_started and not full_response:
                        # Agent finished without streaming tokens
                        pass

                    # ── Process pending writes (human-in-the-loop) ──
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

                    console.print("─" * 60)

            # ── Save messages to chat history ──
            chat_store.add_message(session_id, "user", query)
            if full_response:
                chat_store.add_message(session_id, "assistant", full_response)

        except Exception as e:
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

    # ── Step 1: Config ──
    console.print("[bold cyan]Step 1/4[/bold cyan] — Configuration")
    global_dir = Path.home() / ".codetrace"
    global_dir.mkdir(parents=True, exist_ok=True)
    config_path = global_dir / "config.json"

    if config_path.exists():
        console.print("  [green]✓ Config already exists — skipping[/green]")
    else:
        if llm:
            # Non-interactive: auto-configure with defaults
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

    # ── Step 2: Download Models ──
    console.print("[bold cyan]Step 2/4[/bold cyan] — Downloading Embedding Models")

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

    # ── Step 3: Index Codebase ──
    console.print("[bold cyan]Step 3/4[/bold cyan] — Indexing Codebase")

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

    all_files = [str(p) for p in target_dir.rglob("*.*")
                 if ".codetrace" not in p.parts and ".git" not in p.parts]
    supported_files = [f for f, _ in orchestrator.parser.iter_supported_files(all_files)]
    supported_set = set(supported_files)

    changed_all_file_pairs = sync_manager.get_changed_files(all_files)
    deleted_files = sync_manager.get_deleted_files(all_files)

    if deleted_files:
        for df in deleted_files:
            if df in supported_set:
                orchestrator.graph.prune_file(df)
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
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"  [cyan]Parsing & Indexing {len(changed_supported_file_pairs)} files...",
                    total=len(changed_supported_file_pairs)
                )
                for file_path, _ in changed_supported_file_pairs:
                    orchestrator.build_from_file(file_path, vector_store=vector_store)
                    progress.advance(task)

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

    # ── Step 4: Register MCP ──
    console.print("[bold cyan]Step 4/4[/bold cyan] — Registering MCP for IDEs")
    mcp_results = _register_mcp(target_dir)
    for msg in mcp_results:
        console.print(f"  {msg}")
    console.print()

    # ── Final Summary ──
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

    # ── Detect remote repo URL vs local path ──
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
            console.print(f"  [green]✓ Cloned to:[/green] [dim]{cloned_dir}[/dim]\n")
        except RuntimeError as e:
            console.print(f"[bold red]Clone failed:[/bold red] {e}")
            raise typer.Exit(1)

        target_dir = cloned_dir
    else:
        target_dir = get_project_root(path)

    db_dir = target_dir / ".codetrace"

    # Auto-init if not yet initialized (especially for cloned repos)
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        SyncManager(db_dir=str(db_dir))

    console.print(f"[bold blue]Starting Codetrace Indexer[/bold blue] \U0001f680")
    console.print(f"Target: [dim]{target_dir}[/dim]\n")

    # 1. Boot up the engines
    with console.status("[bold cyan]Waking up Vector Models and DB connections...", spinner="point"):
        sync_manager = SyncManager(db_dir=str(db_dir))
        orchestrator = GraphOrchestrator()
        
        vs_config = VectorStoreConfig(persist_dir=str(db_dir / "chroma"))
        vector_store = VectorStore(config=vs_config)
        
        orchestrator.graph.db_path = db_dir / "graph_metadata.db"
        orchestrator.graph._init_db()
        orchestrator.graph.load_from_db()

    # 2. Discover files
    all_files = [str(p) for p in target_dir.rglob("*.*")
                 if ".codetrace" not in p.parts and ".git" not in p.parts]
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
        console.print("[bold green]✓ Codebase is fully up to date![/bold green]")
        if cloned_dir:
            shutil.rmtree(cloned_dir, ignore_errors=True)
        return

    # 4. Handle Deletions (Pruning)
    if deleted_files:
        with console.status(f"[bold red]Pruning {len(deleted_files)} deleted files..."):
            for df in deleted_files:
                if df in supported_set:
                    orchestrator.graph.prune_file(df)
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
            
            for file_path, _ in changed_supported_file_pairs:
                orchestrator.build_from_file(file_path, vector_store=vector_store)
                progress.advance(task)

        # 6. Save State
        with console.status("[bold magenta]Persisting Graph and Sync states..."):
            orchestrator.graph.persist_to_db()
            sync_manager.mark_files_synced_batch(changed_all_file_pairs)
            sync_manager.update_index_manifest(
                target_dir,
                sync_manager.get_all_tracked_file_hashes(),
                supported_file_count=len(supported_files),
            )

        elapsed = time.time() - start_time
        console.print(f"\n[bold green]✓ Indexing complete in {elapsed:.2f}s![/bold green]")
    else:
        if changed_all_file_pairs:
            sync_manager.mark_files_synced_batch(changed_all_file_pairs)
        sync_manager.update_index_manifest(
            target_dir,
            sync_manager.get_all_tracked_file_hashes(),
            supported_file_count=len(supported_files),
        )
        if changed_all_file_pairs:
            console.print("\n[bold green]✓ Snapshot update complete (no supported code deltas).[/bold green]")

    # 7. Cleanup cloned repo (database persists in the temp .codetrace/)
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

    # Show current config with masked key
    with open(config_path) as f:
        cfg = json.load(f)

    console.print(Panel(
        f"[bold]Provider:[/bold]  {cfg.get('provider', 'N/A')}\n"
        f"[bold]API Key:[/bold]   {mask_key(cfg.get('api_key', ''))}\n"
        f"[bold]Model:[/bold]     {cfg.get('model_name') or '(default)'}",
        title="Current Configuration",
        border_style="cyan"
    ))

    # Ask for confirmation before overwriting
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
        try:
            from pyvis.network import Network
        except ImportError:
            console.print(
                "[red]pyvis is required for visualization.[/red]\n"
                "Install it with: [cyan]pip install pyvis[/cyan]"
            )
            raise typer.Exit(1)

        net = Network(
            height="100%",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
        )
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.01,
        )

        # Color palette by node type
        type_colors = {
            "function":  "#4fc3f7",   # light blue
            "class":     "#ff8a65",   # orange
            "method":    "#81c784",   # green
            "module":    "#ba68c8",   # purple
            "variable":  "#fff176",   # yellow
            "unknown":   "#90a4ae",   # grey
        }

        for node, data in graph.direct_graph.nodes(data=True):
            n_type = data.get("type", "unknown")
            n_file = data.get("file", "")
            # Use short label (just the symbol name)
            label = node.split(":")[-1] if ":" in node else node
            color = type_colors.get(n_type, type_colors["unknown"])

            net.add_node(
                node,
                label=label,
                title=f"{node}\nType: {n_type}\nFile: {n_file}",
                color=color,
                size=20 if n_type == "class" else 12,
                shape="dot",
            )

        # Edge colors by relation
        edge_colors = {
            "calls":   "#4dd0e1",
            "defines": "#ffb74d",
        }

        for src, tgt, data in graph.direct_graph.edges(data=True):
            relation = data.get("relation", "calls")
            net.add_edge(
                src, tgt,
                title=relation,
                color=edge_colors.get(relation, "#666666"),
                arrows="to",
                width=1.5,
            )

        output_path = db_dir / "graph_visualization.html"
        net.save_graph(str(output_path))

    console.print(Panel(
        f"[green]Graph exported successfully![/green]\n\n"
        f"Nodes: [bold]{node_count}[/bold]  |  Edges: [bold]{edge_count}[/bold]\n\n"
        f"Open in browser: [bold cyan]{output_path}[/bold cyan]\n\n"
        f"[dim]Legend: "
        f"[#4fc3f7]● function[/#4fc3f7]  "
        f"[#ff8a65]● class[/#ff8a65]  "
        f"[#81c784]● method[/#81c784]  "
        f"[#ba68c8]● module[/#ba68c8]  "
        f"[#fff176]● variable[/#fff176][/dim]",
        title="Code Architecture Visualization",
        border_style="cyan"
    ))


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
