import ctypes
import os
import sys
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _add_search_dir(rel_subpath: str) -> None:
    target = BASE / rel_subpath
    if not target.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(target))
    else:
        os.environ["PATH"] = str(target) + os.pathsep + os.environ.get("PATH", "")


def _preload(rel_dll: str) -> None:
    dll_path = BASE / rel_dll
    if not dll_path.is_file():
        return
    try:
        ctypes.CDLL(str(dll_path))
    except OSError:
        pass


_add_search_dir("llvmlite")
_add_search_dir("llvmlite/binding")
_preload("llvmlite/binding/llvmlite.dll")
