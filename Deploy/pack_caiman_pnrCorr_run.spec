# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

datas = []
binaries = []
hiddenimports = []

tmp_ret = collect_all('hdmf')
datas += tmp_ret[0]

a = Analysis(
    ['caiman_pnrCorr_run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='caiman_pnrCorr',
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
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='caiman_pnrCorr',
)
