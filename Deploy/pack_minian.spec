# -*- mode: python ; coding: utf-8 -*-

import os
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

conda_prefix = os.environ.get('MINIAN_CONDA_PREFIX') or os.environ.get('CONDA_PREFIX')
if conda_prefix:
    library_bin = Path(conda_prefix) / 'Library' / 'bin'
    if library_bin.is_dir():
        def need(name):
            target = name.lower()
            keep = {
                'hdf.dll',
                'hdf5.dll',
                'hdf5_hl.dll',
                'hdf5_cpp.dll',
                'hdf5_hl_cpp.dll',
                'hdf5_tools.dll',
                'libaec.dll',
                'szip.dll',
            }
            return target in keep

        existing = {os.path.basename(src).lower() for src, _ in binaries}
        for dll in library_bin.glob('*.dll'):
            name = dll.name.lower()
            if name in existing or not need(name):
                continue
            binaries.append((str(dll), '.'))
            existing.add(name)

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
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name='minian',
)
