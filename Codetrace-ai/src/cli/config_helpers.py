import json
import os
import re
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

from ..config_io import write_private_json

# We only match keys by their provider prefix. There used to be a generic
# "long random token" pattern too, but it kept flagging commit SHAs, long
# identifiers and slash-free paths as leaked keys and blocking perfectly good
# one-word queries — so it's gone. If a key has no prefix we recognize, we let
# it through; the occasional miss beats crying wolf constantly.
_API_KEY_PATTERNS = [
    re.compile(r"^sk-[A-Za-z0-9_-]{30,}$"),          # OpenAI / OpenRouter (sk-, sk-proj-, sk-or-)
    re.compile(r"^sk-ant-[A-Za-z0-9_-]{30,}$"),      # Anthropic
    re.compile(r"^gsk_[A-Za-z0-9_-]{30,}$"),         # Groq
    re.compile(r"^AIza[A-Za-z0-9_-]{30,}$"),         # Google / Gemini (AIza prefix)
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


# Just the base URLs we need to hit the /models endpoint — a small slice of
# retriever.py's PROVIDER_REGISTRY, duplicated here so config setup doesn't have
# to import that heavy module.
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
    api_style: str = "openai",
) -> list[str]:
    """
    Fetch the list of available model IDs from the provider's API.

    Uses stdlib ``urllib.request`` so first-time setup doesn't depend on any
    third-party HTTP library. Returns an empty list if anything goes wrong —
    the caller falls back to asking for a model name by hand, which is what
    makes an unknown endpoint usable even when it exposes no listing route.

    Supported patterns:
      - OpenAI-compatible: ``GET {base_url}/models`` → ``{"data": [{"id": "..."}]}``
      - Anthropic-style:   ``GET {base_url}/v1/models`` → ``{"data": [{"id": "..."}]}``
      - Ollama:            ``GET /api/tags`` → ``{"models": [{"name": "..."}]}``
      - Gemini:            ``GET /v1beta/models?key=...`` → ``{"models": [{"name": "models/..."}]}``

    ``api_style`` only matters for the custom provider, where the user picks the
    wire format; the named providers imply their own.
    """
    import urllib.request
    import urllib.error
    import ssl

    # Default SSL context — kept lenient because some corporate proxies mangle
    # the cert chain.
    ctx = ssl.create_default_context()

    models: list[str] = []

    try:
        if provider == "anthropic" or (provider == "custom" and api_style == "anthropic"):
            # Anthropic-style: base_url is the bare host, models live at /v1/models
            # and auth is x-api-key rather than a bearer token.
            actual_base = (base_url or "https://api.anthropic.com").rstrip("/")
            url = f"{actual_base}/v1/models"
            req = urllib.request.Request(url, method="GET")
            req.add_header("x-api-key", api_key)
            req.add_header("anthropic-version", "2023-06-01")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                data = json.loads(resp.read())
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)

        elif provider == "ollama":
            # /api/tags sits on the bare host, so drop any /v1 suffix first.
            ollama_host = (base_url or "http://localhost:11434").rstrip("/")
            if ollama_host.endswith("/v1"):
                ollama_host = ollama_host[:-3]
            url = f"{ollama_host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
                data = json.loads(resp.read())
            # Shape: {"models": [{"name": "llama3.2:latest", ...}]}
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    models.append(name)

        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                data = json.loads(resp.read())
            # Shape: {"models": [{"name": "models/gemini-2.0-flash", ...}]}
            for m in data.get("models", []):
                full_name = m.get("name", "")
                # Drop the "models/" prefix Gemini prepends.
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
            # A self-hosted endpoint may take no auth at all — sending an empty
            # bearer token gets rejected by some gateways, so only send a real one.
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                data = json.loads(resp.read())
            # Shape: {"data": [{"id": "model-name", ...}]}
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)

    except Exception:
        # Network hiccup, bad key, timeout, malformed JSON — swallow them all.
        return []

    # Drop the non-chat models (embeddings, whisper, tts, and so on).
    models = [
        m for m in models
        if not any(m.lower().startswith(p) for p in _NON_CHAT_PREFIXES)
    ]

    models.sort()
    return models


# Family prefixes for flat (non-namespaced) model IDs — used to bucket models by
# series when the ID has no '/' to split on.
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
                # Nothing matched — fall back to the first chunk before -, : or .
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


def _pick_model(
    provider: str,
    api_key: str,
    base_url: str,
    console,
    api_style: str = "openai",
) -> str:
    """
    Fetch the provider's available models and present a picker.

    Up to 20 models: one flat numbered list.
    More than that: a two-step flow — first pick a series/family (Claude, GPT,
    Llama, …), then pick the specific model inside it.

    If the endpoint exposes no listing route, the user just types the model name
    — which is all an unknown provider ever needs.
    """
    console.print(f"\n[dim]Fetching available models from {provider.upper()}...[/dim]")
    models = _fetch_provider_models(provider, api_key, base_url, api_style)

    if not models:
        console.print("[yellow]Could not fetch a model list — enter the model name manually.[/yellow]")
        name = ""
        while not name:
            name = Prompt.ask("[cyan]3. Enter model name[/cyan]", default="").strip()
            if not name:
                console.print("   [red]A model name is required.[/red]")
        return name

    # Short enough to just list everything.
    if len(models) <= 20:
        console.print(f"\n[bold cyan]3. Select a model for [white]{provider.upper()}[/white] ({len(models)} available)[/bold cyan]\n")
        chosen = _numbered_pick(models, "model name", console)
        if chosen is None:
            return Prompt.ask("[cyan]Custom model name[/cyan]", default="")
        return chosen

    # Too many to scroll — group into series first.
    grouped = _group_by_series(models)
    series_keys = sorted(grouped.keys(), key=str.lower)

    # Label each series with its model count.
    series_labels = [f"{key}  [dim]({len(grouped[key])} models)[/dim]" for key in series_keys]

    console.print(
        f"\n[bold cyan]3a. Select a model series for [white]{provider.upper()}[/white] "
        f"({len(models)} models across {len(series_keys)} families)[/bold cyan]\n"
    )

    chosen_label = _numbered_pick(series_labels, "model name", console, allow_all=True)

    if chosen_label is None:
        # They typed 0 — let them enter a model name by hand.
        return Prompt.ask("[cyan]Custom model name[/cyan]", default="")

    if chosen_label == "__ALL__":
        # "Show all" — dump a flat list, capped at 50.
        display = models[:50]
        console.print(f"\n[bold cyan]3b. Select a model ({len(models)} total)[/bold cyan]\n")
        if len(models) > 50:
            console.print(f"   [dim]Showing first 50 of {len(models)}. Enter 0 for custom.[/dim]\n")
        chosen = _numbered_pick(display, "model name", console)
        if chosen is None:
            return Prompt.ask("[cyan]Custom model name[/cyan]", default="")
        return chosen

    # Turn the chosen label back into its series key.
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
    api_style = "openai"

    if provider == "ollama":
        console.print("[green]Ollama selected - no API key needed! Runs 100% locally.[/green]")
        base_url = Prompt.ask("[cyan]2. Ollama base URL[/cyan]", default="http://localhost:11434")
        # People usually type the bare host, but the OpenAI-compatible endpoint
        # lives at /v1, so tack it on if it's missing.
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    elif provider == "custom":
        # No vendor is assumed here. The user names the endpoint and the wire
        # format, which between them cover any provider that speaks either
        # protocol — hosted or self-hosted.
        console.print(
            "[green]Custom provider — point Codetrace at any endpoint that speaks the "
            "OpenAI or Anthropic API.[/green]"
        )
        api_style = Prompt.ask(
            "[cyan]2. API style[/cyan]",
            choices=["openai", "anthropic"],
            default="openai",
        )
        if api_style == "anthropic":
            console.print(
                "   [dim]Requests go to {base_url}/v1/messages — give the bare host, "
                "e.g. https://api.anthropic.com[/dim]"
            )
        else:
            console.print(
                "   [dim]Requests go to {base_url}/chat/completions — include the version "
                "path, e.g. https://api.deepseek.com/v1[/dim]"
            )
        while not base_url:
            base_url = Prompt.ask("[cyan]2a. API base URL[/cyan]", default="").strip().rstrip("/")
            if not base_url:
                console.print("   [red]A base URL is required for a custom provider.[/red]")
        api_key = Prompt.ask(
            "[cyan]2b. API key[/cyan] [dim](leave blank for a local or unauthenticated endpoint)[/dim]",
            password=True,
            default="",
        )
    else:
        api_key = Prompt.ask(f"[cyan]2. Enter your {provider.upper()} API Key[/cyan]", password=True)

    model_name = _pick_model(provider, api_key, base_url, console, api_style)
    context_window = None
    max_output_tokens = None

    if provider != "ollama":
        raw_context = Prompt.ask(
            "[cyan]4. Context window[/cyan] "
            "[dim](optional; leave blank to auto-detect)[/dim]",
            default="",
        ).strip()

        if raw_context:
            if not raw_context.isdigit() or int(raw_context) < 2048:
                console.print("[yellow]Ignoring invalid context window; auto-detection will be used.[/yellow]")
            else:
                context_window = int(raw_context)

        raw_output = Prompt.ask(
            "[cyan]5. Max output tokens[/cyan] "
            "[dim](optional; leave blank to auto-detect)[/dim]",
            default="",
        ).strip()

        if raw_output:
            if not raw_output.isdigit() or int(raw_output) < 128:
                console.print("[yellow]Ignoring invalid output limit; auto-detection will be used.[/yellow]")
            else:
                max_output_tokens = int(raw_output)

    config_data = {
        "provider": provider.lower(),
        "api_key": api_key,
        "model_name": model_name,
        "base_url": base_url,
    }
    if context_window is not None:
        config_data["context_window"] = context_window
    if max_output_tokens is not None:
        config_data["max_output_tokens"] = max_output_tokens

        
    # Only meaningful for the custom provider; the named ones imply their style.
    if provider == "custom":
        config_data["api_style"] = api_style

    write_private_json(config_path, config_data)

    console.print(f"[bold green]Configuration saved to {config_path}[/bold green]\n")


def ensure_config(console) -> None:
    """Ensure global config exists; otherwise run setup wizard."""
    global_dir = Path.home() / ".codetrace"
    global_dir.mkdir(parents=True, exist_ok=True)
    config_path = global_dir / "config.json"

    if config_path.exists():
        return

    run_setup_wizard(config_path, console=console, is_reconfigure=False)


# Registers our MCP server with the IDEs we know about.
def _load_mcp_json(path: Path) -> tuple[dict | None, str | None]:
    """
    Read an existing MCP config. Returns (data, None), or (None, reason) when
    the file can't be safely rewritten — most often VS Code-style JSON with
    comments. We never overwrite a file we couldn't parse: doing so would
    silently erase every other server the user configured there.
    """
    if not path.exists():
        return {}, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return None, f"not plain JSON ({e.__class__.__name__}) — left untouched"
    if not isinstance(data, dict):
        return None, "unexpected top-level JSON type — left untouched"
    return data, None


def _write_mcp_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _is_codetrace_entry(config: object) -> bool:
    """True for a server entry that an older codetrace wrote globally."""
    if not isinstance(config, dict):
        return False
    args = [str(a) for a in config.get("args", [])]
    return "--project" in args and any("codetrace_mcp" in a for a in args)


def _remove_legacy_global_entries(server_names: list[str]) -> list[str]:
    """
    Older versions registered the server in ~/.cursor/mcp.json and
    ~/.claude/mcp.json under a single global name, so every `init` re-pointed
    all IDE sessions at the most recently initialised project (and Claude Code
    never read that second file at all). Drop only entries we recognisably
    wrote ourselves; anything else in those files is left alone.
    """
    results = []
    for ide_name, path in (
        ("Cursor (global)", Path.home() / ".cursor" / "mcp.json"),
        ("Claude Code (legacy)", Path.home() / ".claude" / "mcp.json"),
    ):
        data, error = _load_mcp_json(path)
        if error or not data:
            continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        stale = [n for n in server_names if _is_codetrace_entry(servers.get(n))]
        if not stale:
            continue
        for name in stale:
            del servers[name]
        try:
            _write_mcp_json(path, data)
            results.append(f"[dim]• {ide_name}: removed old global entry from {path}[/dim]")
        except OSError as e:
            results.append(f"[yellow]⚠ {ide_name}: could not clean {path}: {e}[/yellow]")
    return results


def register_mcp(servers: dict[str, dict], workspace_dir: Path | None = None) -> list[str]:
    """Register MCP servers with Claude Code, Cursor and VS Code.

    With ``workspace_dir`` (the normal case — ``init`` / ``register-mcp``) the
    entries go into the project's own config files, so each project gets a
    server bound to *its* index instead of one global entry that the latest
    ``init`` keeps re-pointing:
      - Claude Code  → <project>/.mcp.json          { "mcpServers": {...} }
      - Cursor       → <project>/.cursor/mcp.json   { "mcpServers": {...} }
      - VS Code      → <project>/.vscode/mcp.json   { "servers": { name: { "type": "stdio", ... } } }

    Without ``workspace_dir`` there is no project to scope to, so only Cursor's
    global ~/.cursor/mcp.json is updated.

    Existing files are merged into, never replaced; a file that isn't plain
    JSON (e.g. has comments) is skipped with a warning rather than clobbered.
    """
    if workspace_dir is not None:
        targets = [
            ("Claude Code", workspace_dir / ".mcp.json", "mcpServers", False),
            ("Cursor", workspace_dir / ".cursor" / "mcp.json", "mcpServers", False),
            ("VS Code", workspace_dir / ".vscode" / "mcp.json", "servers", True),
        ]
    else:
        targets = [("Cursor (global)", Path.home() / ".cursor" / "mcp.json", "mcpServers", False)]

    results = []
    for ide_name, config_path, key, needs_type in targets:
        existing, error = _load_mcp_json(config_path)
        if error:
            results.append(
                f"[yellow]⚠ {ide_name}: {config_path} is {error}. "
                f"Add the 'codetrace' server there manually.[/yellow]"
            )
            continue
        try:
            section = existing.setdefault(key, {})
            if not isinstance(section, dict):
                results.append(
                    f"[yellow]⚠ {ide_name}: '{key}' in {config_path} is not an object "
                    f"— left untouched.[/yellow]"
                )
                continue
            for server_name, config in servers.items():
                # VS Code requires an explicit "type": "stdio".
                section[server_name] = {"type": "stdio", **config} if needs_type else config
            _write_mcp_json(config_path, existing)
            results.append(f"[green]✓ {ide_name}:[/green] {config_path}")
        except OSError as e:
            results.append(f"[yellow]⚠ {ide_name}: {e}[/yellow]")

    if workspace_dir is not None:
        results.extend(_remove_legacy_global_entries(list(servers)))
    return results


def discover_mcp_servers() -> dict[str, dict]:
    """Discover MCP servers from existing IDE configurations."""
    discovered = {}
    # Same IDE list as register_mcp, VS Code included, so we pick up servers
    # defined there too.
    targets = [
        Path.home() / ".cursor" / "mcp.json",
        Path.home() / ".claude" / "mcp.json",
        Path.home() / ".vscode" / "mcp.json",
    ]

    for path in targets:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    servers = data.get("mcpServers", {})
                    discovered.update(servers)
            except (json.JSONDecodeError, ValueError):
                continue
    return discovered


def sync_mcp_configs(console) -> None:
    """Interactively discover and sync MCP servers across IDEs."""
    discovered = discover_mcp_servers()
    if not discovered:
        console.print("[yellow]No existing MCP servers discovered to sync.[/yellow]")
        return

    console.print(Panel(f"[bold cyan]Found {len(discovered)} MCP servers in your configs[/bold cyan]"))
    
    server_names = sorted(discovered.keys())
    for i, name in enumerate(server_names, 1):
        console.print(f"  [bold white]{i}.[/bold white] {name}")

    choice = Prompt.ask(
        "\n[cyan]Enter numbers to sync (comma-separated), 'all' for all, or 'none' to skip[/cyan]",
        default="none"
    ).strip().lower()

    if choice == "none" or not choice:
        return
    
    selected_servers = {}
    if choice == "all":
        selected_servers = discovered
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(server_names):
                    name = server_names[idx]
                    selected_servers[name] = discovered[name]
        except ValueError:
            console.print("[red]Invalid input. Skipping sync.[/red]")
            return

    if selected_servers:
        results = register_mcp(selected_servers)
        for res in results:
            console.print(res)
        console.print("[bold green]✓ MCP servers synced successfully![/bold green]")
