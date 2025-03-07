# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs

#
# for main MiniAn python script
#

block_cipher = None

datas           = []
binaries        = []
hiddenimports   = []
excludes        = []

# datas += copy_metadata('suite2p', recursive=True)
tmp_ret = collect_all('suite2p')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('ScanImageTiffReader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# binaries += collect_dynamic_libs('ScanImageTiffReaderAPI', destdir='.\\Library\\bin')

excludes = ["IPython", "PyQt6", "PyQt5", "Markdown", "jupyter"]

a_suite2p = Analysis(
    ['../Suite2p/suite2p_run.py'],
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
pyz_suite2p = PYZ(a_suite2p.pure, a_suite2p.zipped_data, cipher=block_cipher)

exe_suite2p = EXE(
    pyz_suite2p,
    a_suite2p.scripts,
    [],
    name='suite2p',
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
    exe_suite2p,
    a_suite2p.binaries,
    a_suite2p.zipfiles,
    a_suite2p.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='suite2p',
)