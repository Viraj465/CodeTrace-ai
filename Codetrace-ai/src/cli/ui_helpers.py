import colorsys
from collections import OrderedDict
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

CODETRACE_LOGO = r"""
                                           ___       ____________
                  ______                  /  /      /____   ____/_  ___________      _______
                /  ____/ ______   _______/  /______    /  /    /  \/____/  ___ `/   /  ____/______
               /  /     /  ___ \ /  ___    /   _   \  /  /    /  /     / /   / /   /  /    /   __ \
              /  /____ /  /__/  /  /__/   /   _____/ /  /    /  /     / /___/   \ /  /____/  _____/ 
              \______/ \_______/\___,____/\_______/ /__/    /__/      \___,__ /\__\_______/\______/
                """


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

    console.print("  [bold dark_orange]Autonomous System Architect v1.0[/bold dark_orange]\n")


def show_diff_panel(console, pending: dict) -> bool:
    """Display a diff preview and ask whether to apply it."""
    file_path = pending["file_path"]
    diff_lines = pending["diff"]
    is_new = pending["is_new_file"]

    from rich.text import Text as RichText

    if is_new:
        title = f"New File: {Path(file_path).name}"
        diff_text = RichText()
        for line in pending["content"].splitlines()[:40]:
            diff_text.append(f"+ {line}\n", style="green")
        if len(pending["content"].splitlines()) > 40:
            diff_text.append(
                f"  ... ({len(pending['content'].splitlines())} total lines)\n",
                style="dim",
            )
    else:
        title = f"Proposed Edit: {Path(file_path).name}"
        diff_text = RichText()
        shown = 0
        for line in diff_lines:
            if shown > 60:
                diff_text.append(f"  ... ({len(diff_lines)} total diff lines)\n", style="dim")
                break
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
            shown += 1

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
