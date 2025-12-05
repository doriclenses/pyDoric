# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

BLOCK_CIPHER = None

PACKAGES = [
    'minian',
    'distributed',
    'skimage',
    'scipy',
    'pyviz_comms',
]

EXCLUDES = [
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
for package in PACKAGES:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

binaries += collect_dynamic_libs('llvmlite', destdir=os.path.join('Library', 'bin'))

DLL_PREFIXES = {
    "ff",
    "hdf",
    "icu",
    "itk",
    "lib",
    "open",
    "qt5",
}

DLL_EXACT = {
    "freetype.dll",
    "lerc.dll",
    "mfhdf.dll",
    "mkl_pgi_thread.2.dll",
    "netcdf.dll",
    "pthreadvse2.dll",
    "szip.dll",
    "tiff.dll",
    "xdr.dll",
    "yaml.dll",
    "zip.dll",
    "zstd.dll",
}

conda_prefix = os.environ.get('MINIAN_CONDA_PREFIX') or os.environ.get('CONDA_PREFIX')
library_bin = Path(conda_prefix) / 'Library' / 'bin'
if library_bin.is_dir():
    existing = {os.path.basename(src).lower() for src, _ in binaries}
    selected = []
    for dll in sorted(library_bin.glob('*.dll'), key=lambda p: p.name.lower()):
        name = dll.name.lower()
        if name not in DLL_EXACT and not any(name.startswith(prefix) for prefix in DLL_PREFIXES):
            continue
        if name in existing:
            continue
        selected.append((str(dll), '.'))
        existing.add(name)
    binaries.extend(selected)
    
a_minian = Analysis(
    ['../MiniAn/minian_run.py'],
    pathex=['../'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
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
