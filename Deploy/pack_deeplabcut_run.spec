# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, collect_data_files

# Force output folders next to this spec file (i.e., in Deploy/)
_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")   # -> Deploy/dist
workpath  = os.path.join(_specdir, "build")  # -> Deploy/build

BLOCK_CIPHER = None

datas         = collect_data_files('deeplabcut')
datas         += collect_data_files('llvmlite')

binaries      = collect_dynamic_libs('deeplabcut')
binaries      += collect_dynamic_libs('llvmlite')

hiddenimports = collect_submodules('deeplabcut')
hiddenimports += collect_submodules('llvmlite')

excludes = [
    "tensorflow",
    "PySide6", 
    "PyQT6", 
    "PyQT5", 
    "IPython", 
    "Markdown", 
    "jupyter", 
    "napari",
    "napari_deeplabcut", 
    "napari_console", 
    "npe2", 
    "napari_plugin_engine", 
    "napari_svg", 
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
    runtime_hooks=[],
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

a_deeplabcut.binaries= TOC([x for x in a_deeplabcut.binaries if not any(exclude in x[0] for exclude in exclude_binaries)])

pyz_deeplabcut = PYZ(a_deeplabcut.pure, a_deeplabcut.zipped_data, cipher=BLOCK_CIPHER)

exe_deeplabcut = EXE(
    pyz_deeplabcut,
    a_deeplabcut.scripts,
    [],
    name='deeplabcut',
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
    exe_deeplabcut,
    a_deeplabcut.binaries,
    a_deeplabcut.zipfiles,
    a_deeplabcut.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='deeplabcut',
)
