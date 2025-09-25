import os
import sys
from pathlib import Path

def _ensure_dll_dir(rel_subpath: str) -> None:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    target = base / rel_subpath
    if not target.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(target))
    else:
        os.environ["PATH"] = str(target) + os.pathsep + os.environ.get("PATH", "")

_ensure_dll_dir("llvmlite")
_ensure_dll_dir("llvmlite/binding")
