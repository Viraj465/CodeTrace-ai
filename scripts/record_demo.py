"""
Record a real CodeTrace session as an asciinema v2 cast, then render it with agg.

VHS can't capture frames on Windows, so this drives the actual CLI through
pipes (Rich still emits colour via FORCE_COLOR), timestamps every chunk of
output, and simulates the typing between commands. Nothing is scripted except
the keystrokes: every line of output is what codetrace really printed.

    python scripts/record_demo.py                # writes Images/demo.cast
    agg --idle-time-limit 2 --speed 1.3 Images/demo.cast Images/demo.gif
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path.home() / ".codetrace" / "repos" / "github.com" / "pallets" / "itsdangerous"
CODETRACE = Path(sys.executable).parent / ("codetrace.exe" if os.name == "nt" else "codetrace")
QUESTION = "What breaks if I change Signer.get_signature?"
COLS, ROWS = 108, 36
PROMPT = "\x1b[1;32m~/itsdangerous\x1b[0m \x1b[1;35m❯\x1b[0m "
OUT = Path(__file__).resolve().parent.parent / "Images" / "demo.cast"

events: list[tuple[float, str]] = []
t0 = time.monotonic()


def emit(text: str) -> None:
    events.append((time.monotonic() - t0, text.replace("\r\n", "\n").replace("\n", "\r\n")))


def type_out(text: str, delay: float = 0.06) -> None:
    for ch in text:
        emit(ch)
        time.sleep(delay)


def pump(stream, q: queue.Queue) -> None:
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            q.put(None)
            return
        q.put(chunk.decode("utf-8", errors="replace"))


def run(args: list[str], answers: list[tuple[str, str]] | None = None, timeout: float = 900) -> None:
    """Run codetrace, streaming its output; when `marker` appears, type `reply`."""
    env = dict(os.environ, FORCE_COLOR="1", COLUMNS=str(COLS), LINES=str(ROWS),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([str(CODETRACE), *args], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    q: queue.Queue = queue.Queue()
    threading.Thread(target=pump, args=(proc.stdout, q), daemon=True).start()

    pending = list(answers or [])
    seen = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            chunk = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None:
            break
        emit(chunk)
        seen += chunk
        if pending and pending[0][0] in seen:
            marker, reply = pending.pop(0)
            seen = seen[seen.index(marker) + len(marker):]
            time.sleep(1.2)
            type_out(reply)
            time.sleep(0.5)
            emit("\n")
            proc.stdin.write((reply + "\n").encode())
            proc.stdin.flush()
    proc.wait(timeout=30)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    emit(PROMPT)
    time.sleep(1.0)
    type_out("codetrace index .")
    time.sleep(0.4)
    emit("\n")
    run(["index", "."])
    time.sleep(2.5)

    emit("\x1b[2J\x1b[H" + PROMPT)
    time.sleep(0.8)
    type_out("codetrace chat")
    time.sleep(0.4)
    emit("\n")
    run(["chat"], answers=[("You:", QUESTION), ("You:", "exit")])
    time.sleep(3.0)
    emit(PROMPT)
    time.sleep(1.5)
    emit("")

    header = {"version": 2, "width": COLS, "height": ROWS,
              "title": "CodeTrace AI — blast radius on a laptop GPU",
              "env": {"TERM": "xterm-256color", "SHELL": "pwsh"}}
    with OUT.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for t, data in events:
            f.write(json.dumps([round(t, 3), "o", data], ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(events)} events, {events[-1][0]:.0f}s)")


if __name__ == "__main__":
    main()
