import json
import os
import re
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

_API_KEY_PATTERNS = [
    re.compile(r"^sk-[A-Za-z0-9_-]{30,}$"),
    re.compile(r"^sk-ant-[A-Za-z0-9_-]{30,}$"),
    re.compile(r"^gsk_[A-Za-z0-9_-]{30,}$"),
    re.compile(r"^AI[A-Za-z0-9_-]{35,}$"),
    re.compile(r"^[A-Za-z0-9_-]{38,}$"),
]


def enable_offline_mode() -> None:
    """Set environment variables for strict offline mode."""
    from rich.console import Console

    Console().print("[bold yellow]Offline Mode Activated - Telemetry and Network Calls Disabled[/bold yellow]")
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def looks_like_api_key(text: str) -> bool:
    """Return True if input resembles an API key."""
    text = text.strip()
    if " " in text:
        return False
    return any(pat.match(text) for pat in _API_KEY_PATTERNS)


def mask_key(key: str) -> str:
    """Mask API key for display."""
    if len(key) <= 10:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def run_setup_wizard(config_path: Path, console, is_reconfigure: bool = False) -> None:
    """Interactive setup wizard for first-time setup and reconfigure."""
    title = "Reconfigure" if is_reconfigure else "First-Time Setup"
    console.print(Panel("[bold yellow]Let's set up your AI Brain.[/bold yellow]", title=title))

    provider = Prompt.ask(
        "[cyan]1. Choose your LLM Provider[/cyan]",
        choices=["groq", "openai", "anthropic", "gemini", "ollama"],
        default="groq",
    )

    api_key = ""
    base_url = ""

    if provider == "ollama":
        console.print("[green]Ollama selected - no API key needed! Runs 100% locally.[/green]")
        base_url = Prompt.ask("[cyan]2. Ollama base URL[/cyan]", default="http://localhost:11434")
        console.print("[dim]Popular models: llama3.2, codellama, mistral, deepseek-coder[/dim]")
        model_name = Prompt.ask("[cyan]3. Model name[/cyan]", default="llama3.2")
    else:
        api_key = Prompt.ask(f"[cyan]2. Enter your {provider.upper()} API Key[/cyan]", password=True)
        model_name = Prompt.ask("[cyan]3. Enter model name (or press Enter for default)[/cyan]", default="")

    config_data = {
        "provider": provider.lower(),
        "api_key": api_key,
        "model_name": model_name,
        "base_url": base_url,
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    console.print("[bold green]Configuration saved securely![/bold green]\n")


def ensure_config(console) -> None:
    """Ensure global config exists; otherwise run setup wizard."""
    global_dir = Path.home() / ".codetrace"
    global_dir.mkdir(parents=True, exist_ok=True)
    config_path = global_dir / "config.json"

    if config_path.exists():
        return

    run_setup_wizard(config_path, console=console, is_reconfigure=False)


def register_mcp(project_dir: Path) -> list[str]:
    """Auto-register Codetrace MCP in Cursor and Claude Code configs."""
    results = []

    mcp_entry = {
        "command": "python",
        "args": [str(project_dir / "codetrace_mcp" / "server.py"), "--project", str(project_dir)],
    }

    targets = [
        ("Cursor", Path.home() / ".cursor" / "mcp.json"),
        ("Claude Code", Path.home() / ".claude" / "mcp.json"),
    ]

    for ide_name, config_path in targets:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    existing = {}
            else:
                existing = {}

            if "mcpServers" not in existing:
                existing["mcpServers"] = {}

            existing["mcpServers"]["codetrace"] = mcp_entry

            with open(config_path, "w") as f:
                json.dump(existing, f, indent=2)

            results.append(f"[green]✓ {ide_name}:[/green] {config_path}")
        except Exception as e:
            results.append(f"[yellow]⚠ {ide_name}: {e}[/yellow]")

    return results
