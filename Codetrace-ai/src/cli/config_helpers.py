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


# Provider base URLs for model listing
# Mirrors the subset of PROVIDER_REGISTRY from retriever.py needed to hit
# the /models endpoint. Kept here to avoid importing the heavy retriever
# module during interactive config setup.

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
}

# Model IDs that match any of these prefixes are non-chat and should be hidden.
_NON_CHAT_PREFIXES = (
    "text-embedding", "embedding", "whisper", "tts", "dall-e",
    "davinci", "babbage", "ada", "curie", "moderation",
    "text-moderation", "canary", "text-search",
)


def _fetch_provider_models(
    provider: str,
    api_key: str,
    base_url: str = "",
) -> list[str]:
    """
    Fetch the list of available model IDs from the provider's API.

    Uses ``urllib.request`` (stdlib) so there's zero dependency risk during
    first-time setup.  Returns an empty list on any error.

    Supported patterns:
      - OpenAI-compatible: ``GET /models`` → ``{"data": [{"id": "..."}]}``
      - Ollama:            ``GET /api/tags`` → ``{"models": [{"name": "..."}]}``
      - Gemini:            ``GET /v1beta/models?key=...`` → ``{"models": [{"name": "models/..."}]}``
      - Anthropic:         No listing endpoint — returns [].
    """
    import urllib.request
    import urllib.error
    import ssl

    # Create a permissive SSL context (some corporate proxies break cert chains).
    ctx = ssl.create_default_context()

    models: list[str] = []

    try:
        if provider == "anthropic":
            # Anthropic does not expose a /models endpoint.
            return []

        elif provider == "ollama":
            # Strip the /v1 suffix if present; /api/tags lives on the bare host.
            ollama_host = (base_url or "http://localhost:11434").rstrip("/")
            if ollama_host.endswith("/v1"):
                ollama_host = ollama_host[:-3]
            url = f"{ollama_host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
                data = json.loads(resp.read())
            # Ollama returns {"models": [{"name": "llama3.2:latest", ...}]}
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    models.append(name)

        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                data = json.loads(resp.read())
            # Gemini returns {"models": [{"name": "models/gemini-2.0-flash", ...}]}
            for m in data.get("models", []):
                full_name = m.get("name", "")
                # Strip the "models/" prefix.
                model_id = full_name.removeprefix("models/") if full_name else ""
                if model_id and "generateContent" in str(m.get("supportedGenerationMethods", [])):
                    models.append(model_id)

        else:
            # OpenAI-compatible: GET {base_url}/models
            actual_base = base_url or _PROVIDER_BASE_URLS.get(provider, "")
            if not actual_base:
                return []
            url = f"{actual_base}/models"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                data = json.loads(resp.read())
            # OpenAI-compatible returns {"data": [{"id": "model-name", ...}]}
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)

    except Exception:
        # Network error, bad key, timeout, JSON parse error are all handled.
        return []

    # Filter out non-chat models (embeddings, whisper, tts, etc.).
    models = [
        m for m in models
        if not any(m.lower().startswith(p) for p in _NON_CHAT_PREFIXES)
    ]

    models.sort()
    return models


# Known model family prefixes for flat (non-namespaced) model names.
# Used to group models by series when the ID doesn't contain '/'.
_KNOWN_FAMILIES = (
    "gpt-4o", "gpt-4", "gpt-3",  # OpenAI (match longer prefixes first)
    "o1", "o3", "o4",             # OpenAI reasoning
    "claude",                      # Anthropic
    "gemini", "gemma",             # Google
    "llama", "codellama",          # Meta
    "deepseek",                    # DeepSeek
    "mistral", "mixtral", "codestral", "pixtral",  # Mistral AI
    "qwen",                        # Alibaba
    "phi",                         # Microsoft
    "command",                     # Cohere
    "grok",                        # xAI
    "nova",                        # Amazon
    "yi",                          # 01.AI
    "jamba",                       # AI21
    "dbrx",                        # Databricks
)


def _group_by_series(models: list[str]) -> dict[str, list[str]]:
    """
    Group a flat list of model IDs into series/families.

    For namespaced IDs like ``anthropic/claude-sonnet-4.5`` the series key
    is the part before ``/`` (e.g. ``anthropic``).

    For flat IDs like ``llama-3.3-70b-versatile`` the series is matched
    against ``_KNOWN_FAMILIES``; if nothing matches, the first segment
    (split on ``-`` or ``:``) is used.
    """
    series: dict[str, list[str]] = {}

    for model in models:
        if "/" in model:
            key = model.split("/", 1)[0]
        else:
            lower = model.lower()
            key = None
            for fam in _KNOWN_FAMILIES:
                if lower.startswith(fam):
                    key = fam
                    break
            if not key:
                # Fallback: first segment before '-' or ':'.
                key = model.split("-")[0].split(":")[0].split(".")[0]

        series.setdefault(key, []).append(model)

    return series


def _numbered_pick(items: list[str], label: str, console, allow_all: bool = False) -> str | None:
    """
    Display a numbered list and return the chosen item.
    Returns ``None`` if the user types '0' (custom).
    If *allow_all* is True, an 'All' option is prepended.
    """
    offset = 0
    if allow_all:
        console.print(f"   [bold yellow] 1.[/bold yellow] [italic]Show all models[/italic]")
        offset = 1

    for i, item in enumerate(items, start=1 + offset):
        count_suffix = ""
        # Try to show how many models a series key has.
        console.print(f"   [bold white]{i:>2}.[/bold white] {item}")

    console.print(f"   [dim] 0. Enter a custom {label}[/dim]\n")

    max_idx = len(items) + offset
    while True:
        raw = Prompt.ask(
            f"[cyan]Enter number[/cyan] [dim](1–{max_idx}, or 0 for custom)[/dim]",
            default="1",
        ).strip()

        if raw == "0":
            return None  # Caller handles custom input.

        if raw.isdigit() and 1 <= int(raw) <= max_idx:
            idx = int(raw) - 1
            if allow_all and idx == 0:
                return "__ALL__"
            actual_idx = idx - offset if allow_all else idx
            chosen = items[actual_idx]
            console.print(f"   [green]✓ Selected:[/green] {chosen}\n")
            return chosen

        console.print(f"   [red]Please enter a number between 0 and {max_idx}[/red]")


def _pick_model(provider: str, api_key: str, base_url: str, console) -> str:
    """
    Fetch the provider's available models and present a picker.

    When there are ≤20 models, shows a simple flat numbered list.
    When there are >20 models, adds a two-step flow:
      Step 3a → Pick a model series/family (e.g. Claude, GPT, Llama)
      Step 3b → Pick the specific model within that series
    """
    console.print(f"\n[dim]Fetching available models from {provider.upper()}...[/dim]")
    models = _fetch_provider_models(provider, api_key, base_url)

    if not models:
        console.print("[yellow]Could not fetch model list — enter the model name manually.[/yellow]")
        return Prompt.ask("[cyan]3. Enter model name (or press Enter for default)[/cyan]", default="")

    # Small list: flat picker.
    if len(models) <= 20:
        console.print(f"\n[bold cyan]3. Select a model for [white]{provider.upper()}[/white] ({len(models)} available)[/bold cyan]\n")
        chosen = _numbered_pick(models, "model name", console)
        if chosen is None:
            return Prompt.ask("[cyan]Custom model name[/cyan]", default="")
        return chosen

    # Large list: series to model two-step picker.
    grouped = _group_by_series(models)
    series_keys = sorted(grouped.keys(), key=str.lower)

    # Build display labels with counts.
    series_labels = [f"{key}  [dim]({len(grouped[key])} models)[/dim]" for key in series_keys]

    console.print(
        f"\n[bold cyan]3a. Select a model series for [white]{provider.upper()}[/white] "
        f"({len(models)} models across {len(series_keys)} families)[/bold cyan]\n"
    )

    chosen_label = _numbered_pick(series_labels, "model name", console, allow_all=True)

    if chosen_label is None:
        # User typed 0 for a custom model name.
        return Prompt.ask("[cyan]Custom model name[/cyan]", default="")

    if chosen_label == "__ALL__":
        # Show all models flat (capped).
        display = models[:50]
        console.print(f"\n[bold cyan]3b. Select a model ({len(models)} total)[/bold cyan]\n")
        if len(models) > 50:
            console.print(f"   [dim]Showing first 50 of {len(models)}. Enter 0 for custom.[/dim]\n")
        chosen = _numbered_pick(display, "model name", console)
        if chosen is None:
            return Prompt.ask("[cyan]Custom model name[/cyan]", default="")
        return chosen

    # Map the display label back to the series key.
    series_idx = series_labels.index(chosen_label)
    series_key = series_keys[series_idx]
    series_models = sorted(grouped[series_key])

    console.print(f"[bold cyan]3b. Select a model from [white]{series_key}[/white] ({len(series_models)} available)[/bold cyan]\n")
    chosen = _numbered_pick(series_models, "model name", console)
    if chosen is None:
        return Prompt.ask("[cyan]Custom model name[/cyan]", default="")
    return chosen


def run_setup_wizard(config_path: Path, console, is_reconfigure: bool = False) -> None:
    """Interactive setup wizard for first-time setup and reconfigure."""
    title = "Reconfigure" if is_reconfigure else "First-Time Setup"
    console.print(Panel("[bold yellow]Let's set up your AI Brain.[/bold yellow]", title=title))

    provider = Prompt.ask(
        "[cyan]1. Choose your LLM Provider[/cyan]",
        choices=[
            "anthropic","gemini","groq","openai","ollama", "openrouter","custom"
        ],
        default="gemini",
    )

    api_key = ""
    base_url = ""

    if provider == "ollama":
        console.print("[green]Ollama selected - no API key needed! Runs 100% locally.[/green]")
        base_url = Prompt.ask("[cyan]2. Ollama base URL[/cyan]", default="http://localhost:11434")
        # Ensure the /v1 suffix is present for the OpenAI-compatible endpoint. Users often enter the bare host; the Ollama API lives at /v1.
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    else:
        api_key = Prompt.ask(f"[cyan]2. Enter your {provider.upper()} API Key[/cyan]", password=True)

    model_name = _pick_model(provider, api_key, base_url, console)

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
