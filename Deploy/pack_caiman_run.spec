# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.building.datastruct import Tree

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

CAIMAN_DATA_DIR = os.environ.get("CAIMAN_DATA_DIR")

BLOCK_CIPHER = None

packages = [
    'caiman',
    'hdmf',
    'pynwb',
    'param',
    'skimage',
    'scipy'
]

excludes = [
    "PyQt5", 
    "Markdown",
    "jupyter",
    "panel",
    "matplotlib",
    "bokeh",
    "IPython",
    "ipyparallel",
    "ipywidgets",
    "tensorflow",
    "pyqtgraph",
    "torch",
    "torchvision",
    "scipy._lib.array_api_compat.torch"
]

datas         = []
binaries      = []
hiddenimports = []
for package in packages:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

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
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

pyz_caimAn = PYZ(a_caimAn.pure, a_caimAn.zipped_data, cipher=BLOCK_CIPHER)

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

extra_nodes = []
if CAIMAN_DATA_DIR and os.path.isdir(CAIMAN_DATA_DIR):
    extra_nodes.append(Tree(CAIMAN_DATA_DIR, prefix="caiman_data"))
else:
    print("[spec] WARNING: CAIMAN_DATA_DIR not set or not a directory; models/configs not bundled.")

coll = COLLECT(
    exe_caimAn,
    a_caimAn.binaries,
    a_caimAn.zipfiles,
    a_caimAn.datas,
    *extra_nodes,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='caiman',
)
