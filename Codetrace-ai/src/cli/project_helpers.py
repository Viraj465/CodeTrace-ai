import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit
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
    if url.startswith(("http://", "https://")):
        parsed = urlsplit(url)
        path = parsed.path.rstrip("/")

        tree_marker = "/-/tree/" if "/-/tree/" in path else "/tree/"
        if tree_marker in path:
            repository_path, branch_path = path.split(tree_marker, 1)
            clone_url = f"{parsed.scheme}://{parsed.netloc}{repository_path}"
            branch = unquote(branch_path) or None
        else:
            clone_url = f"{parsed.scheme}://{parsed.netloc}{path}"
    else:
        # SSH clone URLs use scp-like syntax, so urlsplit cannot parse them.
        clone_url = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")

    if not clone_url.endswith(".git"):
        clone_url += ".git"

    return {"clone_url": clone_url, "branch": branch}


def repo_cache_dir(clone_url: str, branch: str | None = None) -> Path:
    """
    Where a remote repository is kept for indexing:
    ~/.codetrace/repos/<host>/<owner>/<repo>[@<branch>].

    The checkout has to outlive the `index` command — the index lives inside
    it, and `codetrace chat` has to be run from that directory — so it goes
    somewhere stable rather than a temp dir, and re-indexing the same URL
    updates it in place (SHA-256 delta sync then only re-parses what changed).
    """
    if clone_url.startswith(("http://", "https://")):
        parsed = urlsplit(clone_url)
        host, path = parsed.hostname or "unknown-host", parsed.path
    else:
        # scp-like SSH form: git@host:owner/repo.git
        host, _, path = clone_url.partition("@")[2].partition(":")

    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]

    def _safe(part: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", part)
        # Never let a component climb out of the cache dir.
        return cleaned if cleaned.strip(".") else "_"

    parts = [_safe(host)] + [_safe(p) for p in path.split("/") if p]
    if branch:
        parts[-1] = f"{parts[-1]}@{_safe(branch)}"
    return Path.home() / ".codetrace" / "repos" / Path(*parts)


def _run_git(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{result.stderr.strip()}")


def clone_repo(clone_url: str, branch: str | None = None, dest: Path | None = None) -> Path:
    """
    Shallow-clone the repository into ``dest`` (a fresh temp dir if omitted).

    If ``dest`` already holds a clone, it is fast-forwarded to the latest
    remote commit instead. Untracked files — including the .codetrace index —
    survive the update, so the next index run is incremental.
    """
    if branch and branch.startswith("-"):
        raise RuntimeError(f"invalid branch name: {branch!r}")

    if dest is not None and (dest / ".git").exists():
        _run_git(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", branch or "HEAD"])
        _run_git(["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"])
        return dest

    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="codetrace_"))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(dest)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")
    return dest
