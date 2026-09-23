"""
SystemInfo: Detect the host environment and return a unified config object.

Detected dimensions:
  - OS / platform (Windows, macOS, Linux)
  - Compute device: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
  - RAM (physical total + available)
  - CPU core count
  - Python version
  - Terminal Unicode capability

Used by:
  - vector_store.py  → pick embedding device + batch size
  - main.py          → display environment summary at startup
"""

from __future__ import annotations

import os
import platform
import sys
import logging
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SystemInfo:
    # OS
    os_name: str          # "Windows" | "macOS" | "Linux" | "Unknown"
    os_version: str       # e.g. "10.0.22631" or "14.5"
    arch: str             # e.g. "AMD64" | "arm64"

    # Compute
    device: str           # "cuda" | "mps" | "cpu"
    gpu_name: str         # e.g. "NVIDIA GeForce RTX 4060" | "Apple M3 Pro" | ""
    cuda_version: str     # e.g. "12.1" | ""

    # Hardware
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float

    # Runtime
    python_version: str   # e.g. "3.11.9"
    unicode_terminal: bool

    # Derived recommendations
    embed_batch_size: int = field(init=False)
    embed_device_label: str = field(init=False)

    def __post_init__(self):
        # Batch size: GPU handles bigger batches; CPU stays conservative.
        if self.device == "cuda":
            self.embed_batch_size = 128
        elif self.device == "mps":
            self.embed_batch_size = 64
        else:
            self.embed_batch_size = 32
        self.embed_device_label = {
            "cuda": f"GPU (CUDA {self.cuda_version})" if self.cuda_version else "GPU (CUDA)",
            "mps":  "GPU (Apple MPS)",
            "cpu":  "CPU",
        }[self.device]

    def display_lines(self) -> list[str]:
        """Return Rich-formatted lines for a startup panel."""
        lines = [
            f"  [bold]OS[/bold]       [cyan]{self.os_name}[/cyan] {self.os_version}  [dim]{self.arch}[/dim]",
            f"  [bold]Compute[/bold]  [green]{self.embed_device_label}[/green]"
            + (f"  [dim]{self.gpu_name}[/dim]" if self.gpu_name else ""),
            f"  [bold]RAM[/bold]      "
            + (
                f"{self.ram_available_gb:.1f} GB free / {self.ram_total_gb:.1f} GB total"
                if self.ram_total_gb > 0
                else "[dim]unavailable[/dim]"
            ),
            f"  [bold]CPU[/bold]      {self.cpu_cores} logical cores",
            f"  [bold]Python[/bold]   {self.python_version}",
        ]
        return lines


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_os() -> tuple[str, str, str]:
    system = platform.system()
    name_map = {"Windows": "Windows", "Darwin": "macOS", "Linux": "Linux"}
    os_name = name_map.get(system, "Unknown")

    if system == "Darwin":
        os_version = platform.mac_ver()[0]
    elif system == "Windows":
        os_version = platform.version()
    else:
        os_version = platform.release()

    arch = platform.machine()
    return os_name, os_version, arch


def _detect_compute() -> tuple[str, str, str]:
    """
    Returns (device, gpu_name, cuda_version).
    Priority order: CUDA → MPS → CPU
    """
    try:
        import torch

        # ── CUDA (NVIDIA / AMD ROCm) ──────────────────────────────────────────
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(idx)
            cuda_ver = torch.version.cuda or ""
            return "cuda", gpu_name, cuda_ver

        # ── MPS (Apple Silicon) ───────────────────────────────────────────────
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple doesn't expose a simple GPU name via torch; use platform.
            chip = _get_apple_chip()
            return "mps", chip, ""

    except ImportError:
        logger.debug("torch not installed — defaulting to CPU.")

    return "cpu", "", ""


def _get_apple_chip() -> str:
    """Try to read the Apple chip name from sysctl."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3
        )
        chip = result.stdout.strip()
        if chip:
            return chip
    except Exception:
        pass
    return "Apple Silicon"


def _detect_ram() -> tuple[float, float]:
    """
    Returns (total_gb, available_gb).

    psutil is the good path, but it isn't a hard dependency — without a fallback
    we reported "0.0 GB free / 0.0 GB total", which is both wrong on screen and
    useless to the callers that want to know whether there's room to load a model.
    So each platform gets a stdlib probe before we give up.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024 ** 3), 2), round(vm.available / (1024 ** 3), 2)
    except ImportError:
        pass

    gib = 1024 ** 3
    system = platform.system()

    if system == "Windows":
        # GlobalMemoryStatusEx — always present, no extra install.
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / gib, 2), round(stat.ullAvailPhys / gib, 2)
        except Exception:
            pass

    elif system == "Linux":
        try:
            fields = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts and parts[0].isdigit():
                        fields[key] = int(parts[0]) * 1024  # kB → bytes
            total = fields.get("MemTotal", 0)
            # MemAvailable is the honest number; fall back to free + cache.
            avail = fields.get("MemAvailable") or (
                fields.get("MemFree", 0) + fields.get("Cached", 0)
            )
            if total:
                return round(total / gib, 2), round(avail / gib, 2)
        except Exception:
            pass

    else:  # macOS / BSD
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            # No cheap "available" equivalent here; report total for both rather
            # than claiming zero free.
            return round(total / gib, 2), round(total / gib, 2)
        except Exception:
            pass

    logger.debug("Could not determine RAM on this platform.")
    return 0.0, 0.0


def _detect_unicode_terminal() -> bool:
    """
    Returns True if the terminal is likely capable of rendering Unicode.
    On Windows this checks if stdout encoding is UTF-8 (after our fix).
    """
    enc = getattr(sys.stdout, "encoding", "") or ""
    return enc.lower().replace("-", "") in {"utf8", "utf32"}


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_system_info() -> SystemInfo:
    """
    Detect and cache the full system configuration.
    Thread-safe after the first call (lru_cache is GIL-protected).
    """
    os_name, os_version, arch = _detect_os()
    device, gpu_name, cuda_version = _detect_compute()
    ram_total, ram_available = _detect_ram()

    info = SystemInfo(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        device=device,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        cpu_cores=os.cpu_count() or 1,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        python_version=platform.python_version(),
        unicode_terminal=_detect_unicode_terminal(),
    )

    logger.info(
        "System detected: OS=%s, device=%s, RAM=%.1f GB, cores=%d",
        os_name, device, ram_total, info.cpu_cores,
    )
    return info
