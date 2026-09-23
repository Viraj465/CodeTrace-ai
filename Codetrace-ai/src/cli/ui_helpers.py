import colorsys
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

CODETRACE_LOGO = r"""
 ██████  ██████  ██████  ███████ ████████ ██████   █████   ██████ ███████
██      ██    ██ ██   ██ ██         ██    ██   ██ ██   ██ ██      ██
██      ██    ██ ██   ██ █████      ██    ██████  ███████ ██      █████
██      ██    ██ ██   ██ ██         ██    ██   ██ ██   ██ ██      ██
 ██████  ██████  ██████  ███████    ██    ██   ██ ██   ██  ██████ ███████"""


def _installed_version() -> str:
    try:
        return version("codetrace-ai")
    except PackageNotFoundError:
        return "dev"


def print_banner(console) -> None:
    """Print Codetrace ASCII art with a red-to-yellow gradient."""
    lines = [line for line in CODETRACE_LOGO.split("\n") if line.strip()]
    max_len = max(len(line) for line in lines) if lines else 1

    for line in lines:
        rich_text = Text()
        for i, char in enumerate(line):
            progress = i / max_len if max_len > 0 else 0
            hue = 0.02 + (progress * 0.12)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            rich_text.append(char, style=hex_color)
        console.print(rich_text)

    # Tagline block, indented to sit under the wordmark.
    console.print(
        "  [bold dark_orange]⚡ Autonomous System Architect[/bold dark_orange]"
        f"  [dim]·  v{_installed_version()}[/dim]"
    )
    console.print(
        "  [dim]Local-first code intelligence — call graphs · blast-radius · semantic search[/dim]"
    )
    console.print(
        "  [cyan]Shape Codetrace →[/cyan] "
        "[underline blue]https://github.com/Viraj465/CodeTrace-ai/discussions/5[/underline blue]\n"
    )


def show_diff_panel(console, pending: dict) -> bool:
    """Display a diff preview and ask whether to apply it."""
    file_path = pending["file_path"]
    diff_lines = pending["diff"]
    is_new = pending["is_new_file"]

    from rich.text import Text as RichText

    if is_new:
        content_lines = pending["content"].splitlines()
        title = f"New File: {Path(file_path).name} ({len(content_lines)} lines)"
        diff_text = RichText()
        # Print the whole file with 1-based line numbers — don't hide anything.
        width = len(str(len(content_lines))) or 1
        for i, line in enumerate(content_lines, start=1):
            diff_text.append(f"{i:>{width}} ", style="dim")
            diff_text.append(f"+ {line}\n", style="green")
    else:
        title = f"Proposed Edit: {Path(file_path).name}"
        diff_text = RichText()
        # Render the full unified diff, git-diff style. The @@ hunk headers carry
        # the old/new line numbers so you can place every change. Nothing gets
        # truncated; a big diff just scrolls like `git diff` would.
        for line in diff_lines:
            line_clean = line.rstrip("\n").rstrip("\r")
            if line_clean.startswith("+++") or line_clean.startswith("---"):
                diff_text.append(f"{line_clean}\n", style="bold")
            elif line_clean.startswith("+"):
                diff_text.append(f"{line_clean}\n", style="green")
            elif line_clean.startswith("-"):
                diff_text.append(f"{line_clean}\n", style="red")
            elif line_clean.startswith("@@"):
                diff_text.append(f"{line_clean}\n", style="cyan")
            else:
                diff_text.append(f"{line_clean}\n")

    console.print()
    console.print(
        Panel(
            diff_text,
            title=title,
            subtitle=f"[dim]{file_path}[/dim]",
            border_style="yellow",
        )
    )

    choice = Prompt.ask(
        "  [bold yellow]Apply this change?[/bold yellow]",
        choices=["y", "n"],
        default="n",
    )
    return choice.lower() == "y"


def group_pending_writes_by_root_dir(pending_writes: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group proposed edits by top-level directory for batched approvals."""
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for pw in pending_writes:
        p = Path(pw["file_path"])
        parts = list(p.parts)
        if p.is_absolute():
            key = parts[1] if len(parts) > 1 else "<root>"
        else:
            key = parts[0] if parts else "<root>"
        grouped.setdefault(key, []).append(pw)
    return list(grouped.items())
