"""
Private writes for ~/.codetrace/config.json, which holds the LLM API key.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_private_json(path: str | Path, data: Any, indent: int = 4) -> None:
    """
    Write ``data`` as JSON readable only by the current user.

    The file is written to a temp file created with 0600 permissions and then
    atomically swapped in, so the API key is never world-readable (plain
    ``open(path, "w")`` gets the umask default, typically 0644) and a crash
    mid-write can't leave a truncated config behind. On Windows the mode bits
    are mostly ignored; the file inherits the ACL of the user's profile
    directory, which is already private to that user.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
