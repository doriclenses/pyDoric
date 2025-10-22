# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.util
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, TOC
from PyInstaller.utils.hooks import collect_all

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

BLOCK_CIPHER = None

required_packages = [
    'deeplabcut'
]
optional_packages = [
    'charset_normalizer',
    'dateutil',
    'safetensors',
    'shapely',
    'tables',
    'tkinter',
    'pyarrow',
]

datas = []
binaries = []
hiddenimports = []

for package in required_packages:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

for package in optional_packages:
    if importlib.util.find_spec(package) is not None:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

if importlib.util.find_spec('pyarrow') is not None:
    hiddenimports += ['pyarrow._generated_version']

excludes = [
    "matplotlib"
    ]

a_deeplabcut = Analysis(
    ['../DeepLabCut/deeplabcut_run.py'],
    pathex=['../'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['constraints/torch_openmp_env.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

exclude_binaries = [
    'cublasLt64_12.dll',
    'cublas64_12.dll',
    'cudart64_12.dll',
    'cufft64_11.dll',
    'cusolver64_11.dll',
    'cusparse64_12.dll',
    'nvJitLink_120_0.dll',
    'nvjpeg64_12.dll',
    'nvrtc64_120_0.dll',
    'cudnn_adv64_9.dll',
    'cudnn_cnn64_9.dll',
    'cudnn_engines_precompiled64_9.dll',
    'cudnn_engines_runtime_compiled64_9.dll',
    'cudnn_graph64_9.dll',
    'cudnn_heuristic64_9.dll',
    'cudnn_ops64_9.dll',
    'cudnn64_9.dll'
]

a_deeplabcut.binaries= TOC([
    x for x in a_deeplabcut.binaries if not any(exclude in x[0] for exclude in exclude_binaries)
])

pyz_deeplabcut = PYZ(
    a_deeplabcut.pure,
    a_deeplabcut.zipped_data,
    cipher=BLOCK_CIPHER
)

exe_deeplabcut = EXE(
    pyz_deeplabcut,
    a_deeplabcut.scripts,
    [],
    name='deeplabcut',
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
    exe_deeplabcut,
    a_deeplabcut.binaries,
    a_deeplabcut.zipfiles,
    a_deeplabcut.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='deeplabcut',
)



