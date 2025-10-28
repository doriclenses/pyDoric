# -*- mode: python ; coding: utf-8 -*-

import os

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
    'scipy',
]

excludes = [
    "bokeh",
    "IPython",
    "jupyter",
    "Markdown",
    "matplotlib",
    "notebook",
    "panel",
    "PyQt5",
]

datas           = []
binaries        = []
hiddenimports   = []
for package in packages:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

for dyn_pkg, dest in [
    ('llvmlite', os.path.join('Library', 'bin')),
    ('scipy', os.path.join('scipy', '.libs')),
    ('h5py', '.'),
    ('SimpleITK', '.'),
    ('cv2', '.'),
]:
    try:
        binaries += collect_dynamic_libs(dyn_pkg, destdir=dest)
    except ImportError:
        pass

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
