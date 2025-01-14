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

datas += [( '../CaImAn/caiman_data/model', 'caiman_data/model')]

excludes = ["PyQt5", "Markdown", "jupyter", "jupyterlab", "panel", "matplotlib", "bokeh", "IPython", "ipyparallel", "ipywidgets", "tensorflow", "pyqtgraph"]

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

exclude_binaries = ["api-ms-win-core-debug-l1-1-0.dll",
                    "api-ms-win-core-errorhandling-l1-1-0.dll",
                    "api-ms-win-core-fibers-l1-1-0.dll",
                    "api-ms-win-core-file-l1-1-0.dll",
                    "api-ms-win-core-file-l1-2-0.dll",
                    "api-ms-win-core-file-l2-1-0.dll",
                    "api-ms-win-core-handle-l1-1-0.dll",
                    "api-ms-win-core-heap-l1-1-0.dll",
                    "api-ms-win-core-interlocked-l1-1-0.dll",
                    "api-ms-win-core-libraryloader-l1-1-0.dll",
                    "api-ms-win-core-localization-l1-2-0.dll",
                    "api-ms-win-core-memory-l1-1-0.dll",
                    "api-ms-win-core-namedpipe-l1-1-0.dll",
                    "api-ms-win-core-processenvironment-l1-1-0.dll",
                    "api-ms-win-core-processthreads-l1-1-0.dll",
                    "api-ms-win-core-processthreads-l1-1-1.dll",
                    "api-ms-win-core-profile-l1-1-0.dll",
                    "api-ms-win-core-rtlsupport-l1-1-0.dll",
                    "api-ms-win-core-string-l1-1-0.dll",
                    "api-ms-win-core-synch-l1-1-0.dll",
                    "api-ms-win-core-synch-l1-2-0.dll",
                    "api-ms-win-core-sysinfo-l1-1-0.dll",
                    "api-ms-win-core-timezone-l1-1-0.dll",
                    "api-ms-win-core-util-l1-1-0.dll",
                    "api-ms-win-crt-multibyte-l1-1-0.dll",
                    "api-ms-win-crt-private-l1-1-0.dll",
                    "api-ms-win-crt-utility-l1-1-0.dll",
                    "charset.dll",
                    "concrt140.dll",
                    "ffi-8.dll",
                    "freetype.dll",
                    "iconv.dll",
                    "libblas.dll",
                    "libbz2.dll",
                    "libcblas.dll",
                    "libcrypto-1_1-x64.dll",
                    "libimalloc.dll",
                    "libiomp5md.dll",
                    "libiomp5md_db.dll",
                    "libiompstubs5md.dll",
                    "liblapack.dll",
                    "liblzma.dll",
                    "libpng16.dll",
                    "libsharpyuv.dll",
                    "libssh2.dll",
                    "libssl-1_1-x64.dll",
                    "libwebp.dll",
                    "libwebpdemux.dll",
                    "libwebpmux.dll",
                    "mkl_avx2.2.dll",
                    "mkl_avx512.2.dll",
                    "mkl_blacs_ilp64.2.dll",
                    "mkl_blacs_intelmpi_ilp64.2.dll",
                    "mkl_blacs_intelmpi_lp64.2.dll",
                    "mkl_blacs_lp64.2.dll",
                    "mkl_blacs_msmpi_ilp64.2.dll",
                    "mkl_blacs_msmpi_lp64.2.dll",
                    "mkl_cdft_core.2.dll",
                    "mkl_core.2.dll",
                    "mkl_def.2.dll",
                    "mkl_intel_thread.2.dll",
                    "mkl_mc3.2.dll",
                    "mkl_pgi_thread.2.dll",
                    "mkl_rt.2.dll",
                    "mkl_scalapack_ilp64.2.dll",
                    "mkl_scalapack_lp64.2.dll",
                    "mkl_sequential.2.dll",
                    "mkl_tbb_thread.2.dll",
                    "mkl_vml_avx2.2.dll",
                    "mkl_vml_avx512.2.dll",
                    "mkl_vml_cmpt.2.dll",
                    "mkl_vml_def.2.dll",
                    "mkl_vml_mc3.2.dll",
                    "msvcp140.dll",
                    "msvcp140_1.dll",
                    "msvcp140_2.dll",
                    "msvcp140_atomic_wait.dll",
                    "msvcp140_codecvt_ids.dll",
                    "omptarget.dll",
                    "omptarget.rtl.level0.dll",
                    "omptarget.rtl.opencl.dll",
                    "omptarget.rtl.unified_runtime.dll",
                    "omptarget.sycl.wrap.dll",
                    "pthreadVSE2.dll",
                    "sqlite3.dll",
                    "tcl86t.dll",
                    "tk86t.dll",
                    "ucrtbase.dll",
                    "vcamp140.dll",
                    "vccorlib140.dll",
                    "vcomp140.dll",
                    "vcruntime140.dll",
                    "vcruntime140_1.dll",
                    "vcruntime140_threads.dll",
                    "yaml.dll",
                    "zlib1.dll",
                    "zstd.dll",
                    "api-ms-win-core-console-l1-1-0.dll",
                    "api-ms-win-core-console-l1-2-0.dll",
                    "api-ms-win-core-datetime-l1-1-0.dll"]

a_caimAn.binaries= TOC([x for x in a_caimAn.binaries if not any(exclude in x[0] for exclude in exclude_binaries)])

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
