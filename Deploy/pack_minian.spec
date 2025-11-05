# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

_specdir  = os.path.abspath(os.path.dirname(SPEC))
distpath  = os.path.join(_specdir, "dist")
workpath  = os.path.join(_specdir, "build")

BLOCK_CIPHER = None

packages = [
    'minian',
    'distributed',
    'skimage',
    'scipy',
    'pyviz_comms',
]

REQUIRED_DLL_NAMES = {
    "ffi-7.dll",
    "ffi.dll",
    "fftw3.dll",
    "fftw3f.dll",
    "freetype.dll",
    "hdf.dll",
    "hdf5.dll",
    "hdf5_cpp.dll",
    "hdf5_hl.dll",
    "hdf5_hl_cpp.dll",
    "hdf5_tools.dll",
    "icudt69.dll",
    "icuin69.dll",
    "icuuc69.dll",
    "itkcommon-5.1.dll",
    "itkdenoising-5.1.dll",
    "itkfastmarching-5.1.dll",
    "itkiobiorad-5.1.dll",
    "itkiobmp-5.1.dll",
    "itkiobruker-5.1.dll",
    "itkiogdcm-5.1.dll",
    "itkioge-5.1.dll",
    "itkiogipl-5.1.dll",
    "itkiohdf5-5.1.dll",
    "itkioimagebase-5.1.dll",
    "itkioipl-5.1.dll",
    "itkiojpeg-5.1.dll",
    "itkiojpeg2000-5.1.dll",
    "itkiolsm-5.1.dll",
    "itkiometa-5.1.dll",
    "itkiominc-5.1.dll",
    "itkiomrc-5.1.dll",
    "itkionifti-5.1.dll",
    "itkionrrd-5.1.dll",
    "itkiopng-5.1.dll",
    "itkiostimulate-5.1.dll",
    "itkiotiff-5.1.dll",
    "itkiotransformbase-5.1.dll",
    "itkiotransformhdf5-5.1.dll",
    "itkiotransforminsightlegacy-5.1.dll",
    "itkiotransformmatlab-5.1.dll",
    "itkiovtk-5.1.dll",
    "itklabelmap-5.1.dll",
    "itkoptimizersv4-5.1.dll",
    "itkpdedeformableregistration-5.1.dll",
    "itkregiongrowing-5.1.dll",
    "itkregistrationmethodsv4-5.1.dll",
    "itksimpleitkfilters-5.1.dll",
    "itksmoothing-5.1.dll",
    "itkspatialobjects-5.1.dll",
    "itkstatistics-5.1.dll",
    "itktransform-5.1.dll",
    "itkwatersheds-5.1.dll",
    "lerc.dll",
    "libaec.dll",
    "libcurl.dll",
    "libdeflate.dll",
    "libpng16.dll",
    "libsharpyuv.dll",
    "libsodium.dll",
    "libssh2.dll",
    "libwebp.dll",
    "libwebpdemux.dll",
    "libwebpmux.dll",
    "libzmq-mt-4_3_5.dll",
    "mfhdf.dll",
    "mkl_pgi_thread.2.dll",
    "netcdf.dll",
    "opencv_aruco420.dll",
    "opencv_bgsegm420.dll",
    "opencv_calib3d420.dll",
    "opencv_ccalib420.dll",
    "opencv_core420.dll",
    "opencv_dnn420.dll",
    "opencv_face420.dll",
    "opencv_features2d420.dll",
    "opencv_flann420.dll",
    "opencv_fuzzy420.dll",
    "opencv_hfs420.dll",
    "opencv_highgui420.dll",
    "opencv_img_hash420.dll",
    "opencv_imgcodecs420.dll",
    "opencv_imgproc420.dll",
    "opencv_line_descriptor420.dll",
    "opencv_ml420.dll",
    "opencv_objdetect420.dll",
    "opencv_optflow420.dll",
    "opencv_phase_unwrapping420.dll",
    "opencv_photo420.dll",
    "opencv_plot420.dll",
    "opencv_quality420.dll",
    "opencv_reg420.dll",
    "opencv_rgbd420.dll",
    "opencv_saliency420.dll",
    "opencv_shape420.dll",
    "opencv_stitching420.dll",
    "opencv_structured_light420.dll",
    "opencv_surface_matching420.dll",
    "opencv_text420.dll",
    "opencv_tracking420.dll",
    "opencv_video420.dll",
    "opencv_videoio420.dll",
    "opencv_xfeatures2d420.dll",
    "opencv_ximgproc420.dll",
    "opencv_xphoto420.dll",
    "openjp2.dll",
    "pthreadvse2.dll",
    "qt5core_conda.dll",
    "qt5gui_conda.dll",
    "qt5test_conda.dll",
    "qt5widgets_conda.dll",
    "szip.dll",
    "tiff.dll",
    "xdr.dll",
    "yaml.dll",
    "zip.dll",
    "zstd.dll",
}

excludes = [
    "bokeh",
    "IPython",
    "jupyter",
    "Markdown",
    "matplotlib",
    "notebook",
    "panel",
    "PyQt5",
]

datas           = []
binaries        = []
hiddenimports   = []
for package in packages:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

for dyn_pkg, dest in [
    ('llvmlite', os.path.join('Library', 'bin')),
    ('scipy', os.path.join('scipy', '.libs')),
    ('h5py', '.'),
    ('SimpleITK', '.'),
    ('cv2', '.'),
]:
    try:
        binaries += collect_dynamic_libs(dyn_pkg, destdir=dest)
    except ImportError:
        pass

conda_prefix = os.environ.get('MINIAN_CONDA_PREFIX') or os.environ.get('CONDA_PREFIX')
if conda_prefix:
    library_bin = Path(conda_prefix) / 'Library' / 'bin'
    if library_bin.is_dir():
        existing = {os.path.basename(src).lower() for src, _ in binaries}
        existing_dest = {dest.replace("\\", "/").lower() for _, dest in binaries}
        for dll in library_bin.glob('*.dll'):
            name = dll.name.lower()
            if name not in REQUIRED_DLL_NAMES:
                continue
            dest_dir = '.'
            normalized_dest = dest_dir.replace("\\", "/")
            dest_key = name if normalized_dest in ('', '.') else f"{normalized_dest}/{name}"
            if name in existing and dest_key in existing_dest:
                continue
            binaries.append((str(dll), dest_dir))
            existing.add(name)
            existing_dest.add(dest_key)


a_minian = Analysis(
    ['../MiniAn/minian_run.py'],
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

pyz_minian = PYZ(
    a_minian.pure,
    a_minian.zipped_data,
    cipher=BLOCK_CIPHER
)

exe_minian = EXE(
    pyz_minian,
    a_minian.scripts,
    [],
    name='minian',
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
    exe_minian,
    a_minian.binaries,
    a_minian.zipfiles,
    a_minian.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='minian',
)
