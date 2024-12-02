# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs


block_cipher = None

datas           = []
binaries        = []
hiddenimports   = []
excludes        = []

# datas += copy_metadata('deeplabcut', recursive=True)
tmp_ret = collect_all('deeplabcut')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

excludes = ["tensorflow","PySide6", "PyQT6", "PyQT5", "IPython", "Markdown", "jupyter"]

a_poseEstimation = Analysis(
    ['../PoseEstimation/poseEstimation_run.py'],
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
pyz_poseEstimation = PYZ(a_poseEstimation.pure, a_poseEstimation.zipped_data, cipher=block_cipher)

exe_poseEstimation = EXE(
    pyz_poseEstimation,
    a_poseEstimation.scripts,
    [],
    name='poseEstimation',
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
    exe_poseEstimation,
    a_poseEstimation.binaries,
    a_poseEstimation.zipfiles,
    a_poseEstimation.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='poseEstimation',
)



