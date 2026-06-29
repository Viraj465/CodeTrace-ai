import os
import sys
from pathlib import Path
import warnings
import pathspec

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hardcoded safety net.
# These directories are skipped during indexing, even if not in .gitignore.
# They are not user source code and contain many files that reduce indexing performance.

ALWAYS_IGNORE_DIRS: set[str] = {
    # Version control systems
    ".git",
    ".hg",
    ".svn",
    ".github",
    ".vscode",
    # Codetrace internal data
    ".codetrace",
    # Javascript and Node environments
    "node_modules",
    "bower_components",
    ".next",
    ".nuxt",
    # Python virtual environments and caches
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
    # Editor and IDE configurations
    ".idea",
    ".vs",
    # Dependencies for Go, Rust, PHP, Ruby
    "vendor",
    "target",
    # Miscellaneous files
    "coverage",
    ".cache",
    ".tox",
    ".nox",
}


def _is_always_ignored(dir_name: str) -> bool:
    """Return True if ``dir_name`` is in the hardcoded skip-set."""
    return dir_name in ALWAYS_IGNORE_DIRS


def read_gitignore(root_dir="."):
    """
    Walk the repo tree and collect patterns from **all** ``.gitignore`` files,
    not just the root one.  This mirrors how Git itself works — each nested
    ``.gitignore`` applies to its own subtree.

    Additionally, the hardcoded ``ALWAYS_IGNORE_DIRS`` set is injected so that
    common non-source folders (node_modules, venv, __pycache__, …) are always
    excluded even if no ``.gitignore`` mentions them.

    Returns a ``pathspec.PathSpec`` object that can be used with
    ``spec.match_file(relative_posix_path)``.
    """
    root_path = Path(root_dir).resolve()

    # Seed with hardcoded always-ignore patterns.
    # The trailing slash tells gitwildmatch to match directories.
    patterns: list[str] = [f"{d}/" for d in sorted(ALWAYS_IGNORE_DIRS)]

    # Walk the repository and collect every .gitignore file.
    # We prune ALWAYS_IGNORE_DIRS from the walk so we do not
    # descend into large folders like node_modules/ or venv/.
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune always-ignored directories from the walk in-place
        # so os.walk does not enter them. This improves performance.
        dirnames[:] = [d for d in dirnames if not _is_always_ignored(d)]

        if ".gitignore" not in filenames:
            continue

        gitignore_path = Path(dirpath) / ".gitignore"
        # Relative path from the repo root to the directory containing this .gitignore
        try:
            rel_dir = Path(dirpath).relative_to(root_path).as_posix()
        except ValueError:
            continue

        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    # Skip empty lines and comments in the .gitignore
                    if not line or line.startswith("#"):
                        continue

                    # If this .gitignore is in a subdirectory, prefix anchored patterns
                    # with the subdirectory's path. This ensures they match correctly 
                    # against repo-root-relative paths.
                    if rel_dir and rel_dir != ".":
                        if line.startswith("/"):
                            # Anchored pattern: /foo becomes subdir/foo
                            patterns.append(f"{rel_dir}{line}")
                        else:
                            # Unanchored patterns: match anywhere below the .gitignore's directory. 
                            # We add them as-is, and pathspec gitwildmatch will match them.
                            patterns.append(line)
                    else:
                        # For the root .gitignore, strip optional leading '/' and add it.
                        patterns.append(line.lstrip("/") if line.startswith("/") else line)
        except Exception:
            # Skip unreadable .gitignore files
            pass

    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return spec