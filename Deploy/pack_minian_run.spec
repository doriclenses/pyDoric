# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs

#
# for main MiniAn python script
#

block_cipher = None

datas = []
binaries = []
hiddenimports = []

tmp_ret = collect_all('minian')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('distributed')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

binaries += collect_dynamic_libs('llvmlite',destdir='.\\Library\\bin')

a_minian = Analysis(
    ['../MiniAn/minian_run.py'],
    pathex=['../'],
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
pyz_minian = PYZ(a_minian.pure, a_minian.zipped_data, cipher=block_cipher)

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

#
# for MiniAn Preview python script
#

block_cipher = None

datas = []
binaries = []
hiddenimports = []

tmp_ret = collect_all('minian')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('distributed')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

binaries += collect_dynamic_libs('llvmlite',destdir='.\\Library\\bin')

a_preview = Analysis(
    ['../MiniAn/minian_preview.py'],
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

pyz_preview = PYZ(a_preview.pure, a_preview.zipped_data, cipher=block_cipher)

exe_preview = EXE(
    pyz_preview,
    a_preview.scripts,
    [],
    name='minian_preview',
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
    exe_preview,
    a_preview.binaries,
    a_preview.zipfiles,
    a_preview.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='minian',
)