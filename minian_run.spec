# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['minian_run.py',
    'minian_utilities.py',
    'utilities.py'],
    pathex=[],
    binaries=[
        ('C:/Users/ING55/Anaconda3/Lib/site-packages/llvmlite/binding/llvmlite.dll','./Library/bin'),
        ('C:/Users/ING55/Anaconda3/Lib/site-packages/distributed/dashboard/theme.yaml','./distributed/dashboard')
    ],
    datas=[
        ('C:/Users/ING55/Anaconda3/Lib/site-packages/dask/dask.yaml', './dask'), 
        ('C:/Users/ING55/Anaconda3/Lib/site-packages/distributed/distributed.yaml', './distributed')
    ],
    hiddenimports=[
        'distributed.http.scheduler',
        'distributed.http.scheduler.prometheus',
        'distributed.http.scheduler.info',
        'distributed.http.scheduler.json',
        'distributed.http.health',
        'distributed.http.proxy',
        'distributed.http.statics'
    ],
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
    a.binaries,
    a.zipfiles,
    a.datas,
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
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='minian',
)
