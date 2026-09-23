import os
import sys
from pathlib import Path
import warnings
import pathspec

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Safety net: always skip these during indexing even if .gitignore doesn't list them.
# None of it is user source, and most of it is huge, so scanning it just wastes time.

ALWAYS_IGNORE_DIRS: set[str] = {
    # Version control
    ".git",
    ".hg",
    ".svn",
    ".github",
    ".vscode",
    # Codetrace's own data
    ".codetrace",
    # JS / Node
    "node_modules",
    "bower_components",
    ".next",
    ".nuxt",
    # Python virtualenvs and caches
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    # Build and distribution output
    "dist",
    "build",
    "out",
    "_build",
    # Mobile development folders
    ".expo",
    ".kotlin",
    "Pods",
    # Editor / IDE config
    ".idea",
    ".vs",
    # Go / Rust / PHP / Ruby dependency dirs
    "vendor",
    "target",
    # Misc
    "coverage",
    ".cache",
    ".tox",
    ".nox",
}


# Extensions that are never source and should never be scanned, indexed, or
# snapshotted, even if they somehow slip past language detection.
ALWAYS_IGNORE_EXTENSIONS: set[str] = {
    ".pdf",
}


def _is_always_ignored(dir_name: str) -> bool:
    """
    True if ``dir_name`` should be skipped during scanning.

    We skip anything in ``ALWAYS_IGNORE_DIRS``, plus any dot-directory. Dot-dirs
    (.git, .venv, .pytest_cache, .idea, .vscode, .mypy_cache, …) are nearly always
    tooling folders rather than user source, so catching them by prefix saves us
    from having to enumerate every one.
    """
    if dir_name.startswith("."):
        return True
    return dir_name in ALWAYS_IGNORE_DIRS


def _is_always_ignored_file(file_name: str) -> bool:
    """Return True if a file should be skipped during scanning (by extension)."""
    return Path(file_name).suffix.lower() in ALWAYS_IGNORE_EXTENSIONS


def _scope_gitignore_pattern(line: str, rel_dir: str) -> str:
    """
    Rewrite one pattern from the ``.gitignore`` in ``rel_dir`` so it matches
    repo-root-relative paths the way git would apply it.

    Git's rules: a pattern containing a slash (other than a trailing one) is
    anchored to the directory of its ``.gitignore``; a pattern without one
    matches at any depth *below that directory* — never outside it. Root
    patterns are already in that frame, and gitwildmatch anchors a leading
    '/' itself, so they pass through untouched.
    """
    if not rel_dir or rel_dir == ".":
        return line

    negate = line.startswith("!")
    body = line[1:] if negate else line
    if "/" in body.rstrip("/"):
        scoped = f"{rel_dir}/{body.lstrip('/')}"
    else:
        scoped = f"{rel_dir}/**/{body}"
    return f"!{scoped}" if negate else scoped


def read_gitignore(root_dir="."):
    """
    Collect patterns from every ``.gitignore`` in the tree, not just the root one.
    Git works the same way: each nested ``.gitignore`` applies to its own subtree.

    We also inject ``ALWAYS_IGNORE_DIRS`` up front, so common junk folders
    (node_modules, venv, __pycache__, …) stay excluded even when no ``.gitignore``
    mentions them.

    Returns a ``pathspec.PathSpec`` you can query with
    ``spec.match_file(relative_posix_path)``.
    """
    root_path = Path(root_dir).resolve()

    # Seed with the always-ignore dirs. The trailing slash tells gitwildmatch
    # these are directories.
    patterns: list[str] = [f"{d}/" for d in sorted(ALWAYS_IGNORE_DIRS)]

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune ignored dirs in-place so os.walk never descends into them
        # (keeps us out of node_modules/, venv/, and friends).
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]

        if ".gitignore" not in filenames:
            continue

        gitignore_path = Path(dirpath) / ".gitignore"
        # Path from the repo root down to the dir holding this .gitignore.
        try:
            rel_dir = Path(dirpath).relative_to(root_path).as_posix()
        except ValueError:
            continue

        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    # Blank lines and comments don't count.
                    if not line or line.startswith("#"):
                        continue

                    patterns.append(_scope_gitignore_pattern(line, rel_dir))
        except Exception:
            # A .gitignore we can't read just gets skipped.
            pass

    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return spec