# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

BLOCK_CIPHER = None

packages = [
    'minian',
    'distributed',
    'skimage',
    'h5py'
]

excludes = [
    "IPython", 
    "PyQt5", 
    "Markdown",
    "jupyter", 
    "panel", 
    "matplotlib", 
    "notebook", 
    "bokeh"
]

datas           = []
binaries        = []
hiddenimports   = []
for package in packages:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

binaries += collect_dynamic_libs('llvmlite',destdir='.\\Library\\bin')

binaries += collect_dynamic_libs('h5py', destdir='h5py')
_conda_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
_conda_bin_dir = _conda_prefix / "Library" / "bin"
print(f"[spec] HDF5 lookup using CONDA_PREFIX={_conda_prefix}")
if _conda_bin_dir.is_dir():
    print(f"[spec] Scanning {_conda_bin_dir} for runtime DLLs")
    _dll_patterns = ["*.dll"]
    _existing = {Path(src).resolve(): dest for src, dest in binaries}
    _added = 0
    for _pattern in _dll_patterns:
        for _dll in _conda_bin_dir.glob(_pattern):
            _dll = _dll.resolve()
            if _dll in _existing:
                continue
            binaries.append((str(_dll), os.path.join('h5py', _dll.name)))
            _existing[_dll] = os.path.join('h5py', _dll.name)
            _added += 1
            print(f"[spec]   + bundled {_dll.name}")
    if _added == 0:
        print(f"[spec] WARNING: No extra HDF5 DLLs matched in {_conda_bin_dir}")
else:
    print(f"[spec] WARNING: Could not locate conda Library/bin under {_conda_prefix}")

a_minian = Analysis(
    ['../MiniAn/minian_run.py'],
    pathex=['../'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

pyz_minian = PYZ(
    a_minian.pure, 
    a_minian.zipped_data, 
    cipher=BLOCK_CIPHER
)

exe_minian = EXE(
    pyz_minian,
    a_minian.scripts,
    [],
    name='minian',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_minian,
    a_minian.binaries,
    a_minian.zipfiles,
    a_minian.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='minian',
)
