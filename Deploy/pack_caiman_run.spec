# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs

#
# for maim CaimAn python script
#

block_cipher = None

datas         = []
binaries      = []
hiddenimports = []
excludes      = []

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

tmp_ret = collect_all('ipyparallel')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += [( '../CaImAn/caiman_data/model', 'caiman_data/model')]

#excludes = ["IPython", "PyQt5", "Markdown", "jupyter", "panel", "matplotlib"]
#excludes = ["PyQt5", "Markdown", "jupyter", "panel"]

#binaries += collect_dynamic_libs('llvmlite',destdir='.\\Library\\bin')

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
