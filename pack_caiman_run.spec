# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

datas = []
binaries = []
hiddenimports = []

tmp_ret = collect_all('hdmf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pynwb')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += copy_metadata('param', recursive=True)
tmp_ret = collect_all('param')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('skimage')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += [( './caiman_data/model', 'caiman_data/model')]

#binaries += collect_dynamic_libs('llvmlite',destdir='.\\Library\\bin')

a_caimAn = Analysis(
    ['caiman_run.py'],
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

block_cipher = None

datas = []
binaries = []
hiddenimports = []

tmp_ret = collect_all('hdmf')
datas += tmp_ret[0]

a_pnrCorr = Analysis(
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

pyz_pnrCorr = PYZ(a_pnrCorr.pure, a_pnrCorr.zipped_data, cipher=block_cipher)

exe_pnrCorr = EXE(
    pyz_pnrCorr,
    a_pnrCorr.scripts,
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
    exe_caimAn,
    a_caimAn.binaries,
    a_caimAn.zipfiles,
    a_caimAn.datas,
    exe_pnrCorr,
    a_pnrCorr.binaries,
    a_pnrCorr.zipfiles,
    a_pnrCorr.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='caiman',
)
