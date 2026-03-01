import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import typer


def get_project_root(path: str, console) -> Path:
    """Resolve and validate the target directory."""
    target = Path(path).resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory '{target}' does not exist.[/red]")
        raise typer.Exit(1)
    return target


def parse_github_url(url: str) -> dict | None:
    """
    Parse GitHub/GitLab URL and extract clone URL and branch.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://", "git@")):
        return None

    branch = None
    match = re.match(r"(https?://[^/]+/[^/]+/[^/]+)/tree/(.+)", url)
    if match:
        clone_url = match.group(1)
        branch = match.group(2)
    elif "/-/tree/" in url:
        parts = url.split("/-/tree/")
        clone_url = parts[0]
        branch = parts[1] if len(parts) > 1 else None
    else:
        clone_url = url

    if not clone_url.endswith(".git"):
        clone_url = clone_url.rstrip("/") + ".git"

    return {"clone_url": clone_url, "branch": branch}


def clone_repo(clone_url: str, branch: str | None = None) -> Path:
    """Shallow-clone repository to a temp directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="codetrace_"))

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(temp_dir)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")
    return temp_dir
