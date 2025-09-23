# -*- mode: python ; coding: utf-8 -*-

import os, importlib
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.building.datastruct import Tree

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

block_cipher = None

datas         = []
binaries      = []
hiddenimports = []
excludes      = []

# Where caiman is installed:
cmn = importlib.import_module("caiman")
CAIMAN_PKG = Path(cmn.__file__).parent

# Likely places for models/data created by 'caimanmanager install'
HOME = Path.home()
CANDIDATE_DIRS = [
    CAIMAN_PKG / "caiman_data" / "model",
    CAIMAN_PKG / "data" / "model",
    HOME / ".caimanmanager" / "caiman_data" / "model",
    HOME / ".caimanmanager" / "data" / "model",
    HOME / ".caimanmanager" / "model",
]

def first_existing_dir(paths):
    for p in paths:
        if p.is_dir():
            return p
    return None

MODELS_DIR = first_existing_dir(CANDIDATE_DIRS)

# Build datas list safely (don’t hard fail if missing)
datas = []
if MODELS_DIR:
    # Include the whole 'model' dir under 'caiman_data/model' in the bundle
    datas += Tree(str(MODELS_DIR), prefix=os.path.join("caiman_data", "model")).toc
    print(f"[spec] Including Caiman models from: {MODELS_DIR}")
else:
    print("[spec] WARNING: could not find Caiman 'model' directory; skipping bundle of models")

tmp_ret = collect_all('caiman')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('hdmf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pynwb')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += copy_metadata('param', recursive=True)
tmp_ret = collect_all('param')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('skimage')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# datas += [( '../CaImAn/caiman_data/model', 'caiman_data/model')]

excludes = ["PyQt5", "Markdown", "jupyter", "panel", "matplotlib", "bokeh", "IPython", "ipyparallel", "ipywidgets", "tensorflow", "pyqtgraph",
            "torch", "torchvision", "scipy._lib.array_api_compat.torch"]

a_caimAn = Analysis(
    ['../CaImAn/caiman_run.py'],
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
    cipher=block_cipher,
    noarchive=False,
)

pyz_caimAn = PYZ(a_caimAn.pure, a_caimAn.zipped_data, cipher=block_cipher)

exe_caimAn = EXE(
    pyz_caimAn,
    a_caimAn.scripts,
    [],
    name='caiman',
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
    exe_caimAn,
    a_caimAn.binaries,
    a_caimAn.zipfiles,
    a_caimAn.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='caiman',
)
