import ctypes
import os
import sys
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _add_search_dir(rel_subpath: str) -> Path | None:
    target = BASE / rel_subpath
    if not target.is_dir():
        return None
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(target))
    else:
        os.environ["PATH"] = str(target) + os.pathsep + os.environ.get("PATH", "")
    return target


def _preload(rel_dll: str) -> None:
    dll_path = BASE / rel_dll
    if dll_path.is_file():
        try:
            ctypes.WinDLL(str(dll_path))
        except OSError:
            # Let llvmlite raise a proper error later; we just tried eagerly.
            pass


_add_search_dir("llvmlite")
_add_search_dir("llvmlite/binding")
_preload("llvmlite/binding/llvmlite.dll")
