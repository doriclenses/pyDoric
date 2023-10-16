# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import inspect
import tempfile
import dask as da
import numpy as np
import xarray as xr
import pandas as pand
import functools as fct
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager
from typing import Optional, Callable
import functools as fct

sys.path.append('..')
import utilities as utils
import definitions as defs
import minian_parameters as mn_params
import minian_definitions as mn_defs

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

def main(minian_parameters):
    """minian_main.py
    """

    # Start cluster
    print(mn_defs.Messages.START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_parameters.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_parameters.paths[defs.Parameters.Path.TMP_DIR], mn_defs.FolderName.INTERMEDIATE)
    subset = {"frame": slice(0, None)}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_parameters)

    varr_ref = preprocess(varr_ref, intpath, minian_parameters)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_parameters)

    seeds_final, _ = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters)

    A, C, C_chk, f, b = initialize_components(Y_hw_chk, Y_fm_chk, seeds_final, intpath, chk, minian_parameters)

    A, C, C_chk, sn_spatial = cnmf1(Y_hw_chk, intpath, A, C, C_chk, Y_fm_chk, chk, minian_parameters)

    A, C, AC, S, c0, b0 = cnmf2(Y_hw_chk, A, C, sn_spatial, intpath, C_chk, Y_fm_chk, chk, minian_parameters)

    ### Save final results ###
    print(mn_defs.Messages.SAVING_FINAL, flush=True)
    A = save_minian(A.rename("A"), **minian_parameters.params_save_minian)
    C = save_minian(C.rename("C"), **minian_parameters.params_save_minian)
    AC = save_minian(AC.rename("AC"), **minian_parameters.params_save_minian)
    S = save_minian(S.rename("S"), **minian_parameters.params_save_minian)
    c0 = save_minian(c0.rename("c0"), **minian_parameters.params_save_minian)
    b0 = save_minian(b0.rename("b0"), **minian_parameters.params_save_minian)
    b = save_minian(b.rename("b"), **minian_parameters.params_save_minian)
    f = save_minian(f.rename("f"), **minian_parameters.params_save_minian)

    ### Save results to doric file ###
    print(mn_defs.Messages.SAVING_TO_DORIC, flush=True)
    # Get the path from the source data
    data, driver, operation, series, sensor = minian_parameters.get_h5path_names()

    # Get paramaters of the operation on source data
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")
    # Get the attributes of the images stack
    attrs = utils.load_attributes(file_, f"{minian_parameters.clean_h5path()}/{defs.DoricFile.Dataset.IMAGE_STACK}")
    file_.close()

    # Parameters
    if minian_parameters.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE] > 1:
        minian_parameters.parameters[defs.DoricFile.Attribute.BINNING_FACTOR] = minian_parameters.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE]

    save_minian_to_doric(
        Y, A, C, AC, S,
        fr = minian_parameters.fr,
        bit_count = attrs[defs.DoricFile.Attribute.BIT_COUNT],
        qt_format = attrs[defs.DoricFile.Attribute.FORMAT],
        username = attrs.get(defs.DoricFile.Attribute.USERNAME, sensor),
        vname = minian_parameters.params_load_doric['fname'],
        vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}/",
        vdataset = f"{series}/{sensor}/",
        params_doric = minian_parameters.parameters,
        params_source = params_source_data,
        saveimages = True,
        saveresiduals = True,
        savespikes = True
    )

    # Close cluster
    client.close()
    cluster.close()



def preview(minian_parameters):
    """minian_preview
    """

    # Start cluster
    print(mn_defs.Messages.START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_parameters.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_parameters.paths[defs.Parameters.Path.TMP_DIR], mn_defs.FolderName.INTERMEDIATE)
    subset = {"frame": slice(*minian_parameters.preview_parameters[defs.Parameters.Preview.RANGE])}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_parameters)

    varr_ref = preprocess(varr_ref, intpath, minian_parameters)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_parameters)

    seeds, max_proj = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters, True)

    # Save data for preview to hdf5 file
    try:
        with h5py.File(minian_parameters.preview_parameters[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:

            if minian_parameters.preview_parameters[mn_defs.DictionaryKeys.Preview.MAX_PROJ_DATASET_NAME] in hdf5_file:
                del hdf5_file[minian_parameters.preview_parameters[mn_defs.DictionaryKeys.Preview.MAX_PROJ_DATASET_NAME]]

            hdf5_file.create_dataset(minian_parameters.preview_parameters[mn_defs.DictionaryKeys.Preview.MAX_PROJ_DATASET_NAME], data = max_proj.values, dtype = 'float', chunks = True)

            seeds_dataset = hdf5_file.create_dataset(mn_defs.DictionaryKeys.Preview.SEEDS, data = seeds[['width', 'height']], dtype='int', chunks = True)
            seeds_dataset.attrs[mn_defs.DictionaryKeys.Preview.MERGED]    = seeds.index[seeds['mask_mrg'] == True].tolist()
            seeds_dataset.attrs[mn_defs.DictionaryKeys.Preview.REFINED]   = seeds.index[(seeds['mask_ks'] == True) & (seeds['mask_pnr'] == True)].tolist()

    except Exception as error:
        utils.print_error(error, mn_defs.Messages.SAVE_TO_HDF5)


    file_.close()

    # Close cluster
    client.close()
    cluster.close()

################### Functions defintion ###################
def load_chunk(intpath, subset, minian_parameters):
    ### Load and chunk the data ###
    print(mn_defs.Messages.LOAD_DATA, flush=True)
    varr, file_ = load_doric_to_xarray(**minian_parameters.params_load_doric)
    chk, _ = get_optimal_chk(varr, **minian_parameters.params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                       intpath, overwrite=True)
    varr_ref = varr.sel(subset)

    return file_, chk, varr_ref

def preprocess(varr_ref, intpath, minian_parameters):
    ### Pre-process data ###
    print(mn_defs.Messages.PREPROCESS, flush=True)
    # 1. Glow removal
    print(mn_defs.Messages.PREPROC_REMOVE_GLOW, flush=True)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    # 2. Denoise
    print(mn_defs.Messages.PREPROC_DENOISING, flush=True)
    with except_type_error("denoise"):
        varr_ref = denoise(varr_ref, **minian_parameters.params_denoise)

    # 3. Background removal
    print(mn_defs.Messages.PREPROC_REMOV_BACKG, flush=True)
    with except_type_error("remove_background"):
        varr_ref = remove_background(varr_ref, **minian_parameters.params_remove_background)

    # Save
    print(mn_defs.Messages.PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    return varr_ref

def correct_motion(varr_ref, intpath, chk, minian_parameters):
    ### Motion correction ###
    if minian_parameters.parameters[defs.Parameters.danse.CORRECT_MOTION]:
        print(mn_defs.Messages.CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        with except_type_error("estimate_motion"):
            motion = estimate_motion(varr_ref, **minian_parameters.params_estimate_motion)

        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **minian_parameters.params_save_minian)
        print(mn_defs.Messages.CORRECT_MOTION_APPLY_SHIFT, flush=True)
        Y = apply_transform(varr_ref, motion, **minian_parameters.params_apply_transform)

    else:
        Y = varr_ref

    print(mn_defs.Messages.PREP_DATA_INIT, flush=True)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    return Y, Y_fm_chk, Y_hw_chk

def initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters, return_all_seeds = False):
    ### Seed initialization ###
    print(mn_defs.Messages.INIT_SEEDS, flush=True)
    with except_print_error_no_cells(mn_defs.Messages.INIT_SEEDS):
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_parameters.params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        with except_type_error("seeds_init"):
            seeds = seeds_init(Y_fm_chk, **minian_parameters.params_seeds_init)

        # 3. Peak-Noise-Ratio refine
        print(mn_defs.Messages.INIT_SEEDS_PNR_REFI, flush=True)
        with except_type_error("pnr_refine"):
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_parameters.params_pnr_refine)

        # 4. Kolmogorov-Smirnov refine
        print(mn_defs.Messages.INIT_SEEDS_KOLSM_REF, flush=True)
        with except_type_error("ks_refine"):
            seeds = ks_refine(Y_hw_chk, seeds, **minian_parameters.params_ks_refine)

        # 5. Merge seeds
        print(mn_defs.Messages.INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        with except_type_error("seeds_merge"):
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_parameters.params_seeds_merge)

        if return_all_seeds:
            return pand.merge(seeds, seeds_final, how="outer").fillna(False), max_proj
        else:
            return seeds_final, max_proj


def initialize_components(Y_hw_chk, Y_fm_chk, seeds_final, intpath, chk, minian_parameters):
    ### Component initialization ###
    print(mn_defs.Messages.INIT_COMP, flush=True)
    with except_print_error_no_cells(mn_defs.Messages.INIT_COMP):
        # 1. Initialize spatial
        print(mn_defs.Messages.INIT_COMP_SPATIAL, flush=True)
        with except_type_error("initA"):
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **minian_parameters.params_initA)

        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
        # 2. Initialize temporal
        print(mn_defs.Messages.INIT_COMP_TEMP, flush=True)
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True,
                            chunks={"unit_id": 1, "frame": -1})
        # 3. Merge components
        print(mn_defs.Messages.INIT_COMP_MERG, flush=True)
        with except_type_error("unit_merge"):
            A, C = unit_merge(A_init, C_init, **minian_parameters.params_unit_merge)

        A = save_minian(A.rename("A"), intpath, overwrite=True)
        C = save_minian(C.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        # 4. Initialize background
        print(mn_defs.Messages.INIT_COMP_BACKG, flush=True)
        b, f = update_background(Y_fm_chk, A, C_chk)
        f = save_minian(f.rename("f"), intpath, overwrite=True)
        b = save_minian(b.rename("b"), intpath, overwrite=True)

    return A, C, C_chk, f, b

def cnmf1(Y_hw_chk, intpath, A, C, C_chk, Y_fm_chk, chk, minian_parameters):
    ### CNMF 1st itteration ###
    with except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("1st")):
        # 1. Estimate spatial noise
        print(mn_defs.Messages.CNMF_ESTIM_NOISE.format("1st"), flush=True)
        with except_type_error("get_noise_fft"):
            sn_spatial = get_noise_fft(Y_hw_chk, **minian_parameters.params_get_noise_fft)

        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
        # 2. First spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        with except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)

        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 3. Update background
        print(mn_defs.Messages.CNMF_UPDAT_BACKG.format("1st"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 4. First temporal update
        print(mn_defs.Messages.CNMF_UPDAT_TEMP.format("1st"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        with except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)

        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)
        # 5. Merge components
        print(mn_defs.Messages.CNMF_MERG_COMP.format("1st"), flush=True)
        with except_type_error("unit_merge"):
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **minian_parameters.params_unit_merge)

        # Save
        print(mn_defs.Messages.CNMF_SAVE_INTERMED.format("1st"), flush=True)
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    return A, C, C_chk, sn_spatial


def cnmf2(Y_hw_chk, A, C, sn_spatial, intpath, C_chk, Y_fm_chk, chk, minian_parameters):
    ### CNMF 2nd itteration ###
    with except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("2nd")):
        # 5. Second spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("2nd"), flush=True)
        with except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)

        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 6. Second background update
        print(mn_defs.Messages.CNMF_UPDAT_BACKG.format("2nd"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 7. Second temporal update
        print(mn_defs.Messages.CNMF_UPDAT_TEMP.format("2nd"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        with except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)

        # Save
        print(mn_defs.Messages.CNMF_SAVE_INTERMED.format("2nd"), flush=True)
        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)

        AC = compute_AtC(A, C_chk)

    return A, C, AC, S, c0, b0

#################################################################### Utilities functions ###################################################################
def load_doric_to_xarray(
    fname: str,
    h5path: str,
    dtype: type = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Loads images stack from HDF as xarray

    Args:
        fname: full path to the file


    Returns:


    Raises:
    """

    file_ = h5py.File(fname, 'r')
    varr = da.array.from_array(file_[h5path+defs.DoricFile.Dataset.IMAGE_STACK])
    varr = xr.DataArray(
            varr,
            dims = ["height", "width", "frame"],
            coords = {"height" : np.arange(varr.shape[0]),
                    "width" : np.arange(varr.shape[1]),
                    "frame" : np.arange(varr.shape[2]) + 1, #Frame number start a 1 not 0
                    },
            )
    varr = varr.transpose('frame', 'height', 'width')

    if dtype != varr.dtype:
        if dtype == np.uint8:
            #varr = (varr - varr.values.min()) / (varr.values.max() - varr.values.min()) * 2**8 + 1
            bit_count = file_[h5path+defs.DoricFile.Dataset.IMAGE_STACK].attrs[defs.DoricFile.Attribute.BIT_COUNT]
            varr = varr / 2**bit_count * 2**8

        varr = varr.astype(dtype)

    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError(mn_defs.Messages.UNREC_DOWNSAMPLING_STRAT)
    varr = varr.rename("fluorescence")

    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)

    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])

    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]

    varr = varr.assign_coords({"height" : np.arange(varr.sizes["height"]),
                                "width" : np.arange(varr.sizes["width"]),
                                "frame" : np.arange(varr.sizes["frame"]) + 1, #Frame number start a 1 not 0
                                })

    return varr, file_


def save_minian_to_doric(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    AC: xr.DataArray,
    S: xr.DataArray,
    fr: int,
    bit_count: int,
    qt_format: int,
    username:str,
    vname: str = "minian.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = 'Series1/Sensor1/',
    params_doric: Optional[dict] = {},
    params_source: Optional[dict] = {},
    saveimages: bool = True,
    saveresiduals: bool = True,
    savespikes: bool = True
) -> str:
    """
    Save MiniAn results to .doric file:
    MiniAnImages - `AC` representing cellular activities as computed by :func:`minian.cnmf.compute_AtC`
    MiniAnResidualImages - residule movie computed as the difference between `Y` and `AC`
    MiniAnSignals - `C` with coordinates from `A`
    MiniAnSpikes - `S`
    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing
    scalar is computed with least square using a subset of frames from `Y` and `AC`
    such that their numerical values matches.

    Parameters
    ----------
    varr : xr.DataArray
        Input reference movie data. Should have dimensions ("frame", "height",
        "width"), and should only be chunked along "frame" dimension.
    Y : xr.DataArray
        Movie data representing input to CNMF algorithm. Should have dimensions
        ("frame", "height", "width"), and should only be chunked along "frame"
        dimension.
    A : xr.DataArray, optional
        Spatial footprints of cells. Only used if `AC` is `None`. By default
        `None`.
    C : xr.DataArray, optional
        Temporal activities of cells. Only used if `AC` is `None`. By default
        `None`.
    AC : xr.DataArray, optional
        Spatial-temporal activities of cells. Should have dimensions ("frame",
        "height", "width"), and should only be chunked along "frame" dimension.
        If `None` then both `A` and `C` should be supplied and
        :func:`minian.cnmf.compute_AtC` will be used to compute this variable.
        By default `None`.
    nfm_norm : int, optional
        Number of frames to randomly draw from `Y` and `AC` to compute the
        normalizing factor with least square. By default `None`.
    gain : float, optional
        A gain factor multiplied to `Y`. Useful to make the results visually
        brighter. By default `1.5`.
    vpath : str, optional
        Desired folder containing the resulting video. By default `"."`.
    vname : str, optional
        Desired name of the video. By default `"minian.mp4"`.
    Returns
    -------
    fname : str
        Absolute path of the resulting video.
    """



    res = Y - AC # residual images

    duration = Y.shape[0]
    time_ = np.arange(0, duration/fr, 1/fr, dtype='float64')

    print(mn_defs.Messages.GEN_ROI_NAMES, flush = True)
    names = []
    usernames = []
    for i in range(len(C)):
        names.append('ROI'+str(i+1).zfill(4))
        usernames.append('ROI {0}'.format(i+1))

    with h5py.File(vname, 'a') as f:

        # Check if MiniAn results already exist
        operationCount = ''
        if vpath in f:
            operations = [ name for name in f[vpath] if mn_defs.DoricFile.Group.ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = utils.load_attributes(f, vpath+operation)
                    if utils.merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(mn_defs.DoricFile.Group.ROISIGNALS)):
                            operationCount = ''
                        else:
                            operationCount = operation[-1]

                        break


        if vpath[-1] != '/':
            vpath += '/'

        if vdataset[-1] != '/':
            vdataset += '/'

        params_doric[defs.DoricFile.Attribute.OPERATIONS] += operationCount

        print(mn_defs.Messages.SAVE_ROI_SIG, flush=True)
        pathROIs = vpath+mn_defs.DoricFile.Group.ROISIGNALS+operationCount+'/'
        utils.save_roi_signals(C.values, A.values, time_, f, pathROIs+vdataset, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "AU"})
        utils.print_group_path_for_DANSE(pathROIs+vdataset)
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathROIs)

        if saveimages:
            print(mn_defs.Messages.SAVE_IMAGES, flush=True)
            pathImages = vpath+mn_defs.DoricFile.Group.IMAGES+operationCount+'/'
            utils.save_images(AC.values, time_, f, pathImages+vdataset, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(pathImages+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.OPERATIONS] + "(Images)"), f, pathImages)

        if saveresiduals:
            print(mn_defs.Messages.SAVE_RES_IMAGES, flush=True)
            pathResiduals = vpath+mn_defs.DoricFile.Group.RESIDUALS+operationCount+'/'
            utils.save_images(res.values, time_, f, pathResiduals+vdataset, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(pathResiduals+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source,  params_doric[defs.DoricFile.Attribute.OPERATIONS] + "(Residuals)"), f, pathResiduals)

        if savespikes:
            print(mn_defs.Messages.SAVE_SPIKES, flush=True)
            pathSpikes = vpath+mn_defs.DoricFile.Group.SPIKES+operationCount+'/'
            utils.save_signals(S.values > 0, time_, f, pathSpikes+vdataset, names, usernames, range_min=0, range_max=1)
            utils.print_group_path_for_DANSE(pathSpikes+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathSpikes)

    print(mn_defs.Messages.SAVE_TO.format(vname))

#################### definition of try: expect:
@contextmanager
def except_type_error(function_name: str):
    """
    conext try except to show specific message
    """

    try:
        yield
    except TypeError:
        utils.print_to_intercept(mn_defs.Messages.ONE_PARM_WRONG_TYPE.format(function_name))
        sys.exit()

@contextmanager
def except_print_error_no_cells(position: str):
    """
    conext try except to show no cells found
    """

    try:
        yield
    except Exception as error:
        utils.print_error(error, position)
        utils.print_to_intercept(mn_defs.Messages.NO_CELLS_FOUND)
        sys.exit()
