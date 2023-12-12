# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import inspect
import tempfile
import dask as da
import numpy as np
import pandas as pd
import xarray as xr
import cv2
import functools as fct
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager
from typing import Optional, Callable

sys.path.append("..")
import utilities as utils
import definitions as defs
import minian_parameters as mn_params
import minian_definitions as mn_defs

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC, smooth_sig
from minian.motion_correction import apply_transform, estimate_motion
from minian.cross_registration import calculate_centroids, calculate_centroid_distance, calculate_mapping, resolve_mapping, fill_mapping, group_by_session

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()


def main(minian_params):

    """
    MiniAn CNMF algorithm
    """

    # Start cluster
    print(mn_defs.Messages.START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_params.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_params.paths[defs.Parameters.Path.TMP_DIR], mn_defs.Folder.INTERMEDIATE)
    subset = {"frame": slice(0, None)}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_params)

    varr_ref = preprocess(varr_ref, intpath, minian_params)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_params)

    seeds, _ = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_params)

    A, C, C_chk, f, b = initialize_components(Y_hw_chk, Y_fm_chk, seeds, intpath, chk, minian_params)

    A, C, C_chk, sn_spatial = cnmf1(Y_hw_chk, intpath, A, C, C_chk, Y_fm_chk, chk, minian_params)

    A, C, AC, S, c0, b0 = cnmf2(Y_hw_chk, A, C, sn_spatial, intpath, C_chk, Y_fm_chk, chk, minian_params)

    # Cross registration
    A = cross_register(AC, A, minian_params)

    # Save final MiniAn results
    print(mn_defs.Messages.SAVING_FINAL, flush=True)
    A = save_minian(A.rename("A"), **minian_params.params_save_minian)
    C = save_minian(C.rename("C"), **minian_params.params_save_minian)
    AC = save_minian(AC.rename("AC"), **minian_params.params_save_minian)
    S = save_minian(S.rename("S"), **minian_params.params_save_minian)
    c0 = save_minian(c0.rename("c0"), **minian_params.params_save_minian)
    b0 = save_minian(b0.rename("b0"), **minian_params.params_save_minian)
    b = save_minian(b.rename("b"), **minian_params.params_save_minian)
    f = save_minian(f.rename("f"), **minian_params.params_save_minian)

    # Save results to .doric file
    print(mn_defs.Messages.SAVING_TO_DORIC, flush=True)

    # Get all operation parameters and dataset attributes
    data, driver, operation, series, sensor = minian_params.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")

    if defs.DoricFile.Dataset.IMAGE_STACK in file_[minian_params.paths[defs.Parameters.Path.H5PATH]]:
        attrs = utils.load_attributes(file_, f"{minian_params.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.IMAGE_STACK}")
    else:
        attrs = utils.load_attributes(file_, f"{minian_params.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}")

    if minian_params.params[defs.Parameters.danse.SPATIAL_DOWNSAMPLE] > 1:
        minian_params.params[defs.DoricFile.Attribute.Group.BINNING_FACTOR] = minian_params.params[defs.Parameters.danse.SPATIAL_DOWNSAMPLE]

    time_ = np.array(file_[f"{minian_params.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}"])

    file_.close()

    save_minian_to_doric(
        Y, A, C, AC, S,
        time_ = time_,
        bit_count = attrs[defs.DoricFile.Attribute.Image.BIT_COUNT],
        qt_format = attrs[defs.DoricFile.Attribute.Image.FORMAT],
        username = attrs.get(defs.DoricFile.Attribute.Dataset.USERNAME, sensor),
        vname = minian_params.paths[defs.Parameters.Path.FILEPATH],
        vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
        vdataset = f"{series}/{sensor}",
        params_doric = minian_params.params,
        params_source = params_source_data,
        saveimages = True,
        saveresiduals = True,
        savespikes = True
    )

    # Close cluster
    client.close()
    cluster.close()


def preview(minian_params):
    """
    ...
    """

    if minian_params.preview_params[defs.Parameters.Preview.PREVIEW_TYPE] == mn_defs.Preview.Type.INIT_PREVIEW:
        init_preview(minian_params)
    elif minian_params.preview_params[defs.Parameters.Preview.PREVIEW_TYPE] == mn_defs.Preview.Type.PENALTIES_PREVIEW:
        penalties_preview(minian_params)

    return 1


def init_preview(minian_params):
    """
    Saves max projection image and seeds in HDF5 file
    """

    # Start cluster
    print(mn_defs.Messages.START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_params.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_params.paths[defs.Parameters.Path.TMP_DIR], mn_defs.Folder.INTERMEDIATE)
    subset = {"frame": slice(*minian_params.preview_params[defs.Parameters.Preview.RANGE])}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_params)

    varr_ref = preprocess(varr_ref, intpath, minian_params)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_params)

    time_ = np.array(file_[f"{minian_params.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}"])

    seeds, max_proj = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_params, True)

    example_trace = Y_hw_chk.sel(height=seeds["height"].to_xarray(),
                                 width=seeds["width"].to_xarray(),
                                 ).rename(**{"index": "seed"})

    trace_smth_low = smooth_sig(example_trace, minian_params.params[defs.Parameters.danse.NOISE_FREQ])
    trace_smth_high = smooth_sig(example_trace, minian_params.params[defs.Parameters.danse.NOISE_FREQ], btype="high")
    trace_smth_low = trace_smth_low.compute()
    trace_smth_high = trace_smth_high.compute()

    # Save data for preview to hdf5 file
    try:
        with h5py.File(minian_params.preview_params[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:
            initialization_group = hdf5_file.create_group(mn_defs.Preview.Group.INITIALIZATION)
            initialization_group.create_dataset(mn_defs.Preview.Dataset.MAX_PROJECTION, data = max_proj.values, dtype = "float64", chunks = True)

            seeds_dataset = initialization_group.create_dataset(mn_defs.Preview.Dataset.SEEDS, data = seeds[["width", "height"]], dtype="int", chunks = True)
            seeds_dataset.attrs[mn_defs.Preview.Attribute.SEED_COUNT]  = seeds.shape[0]
            seeds_dataset.attrs[mn_defs.Preview.Attribute.MERGED]      = seeds.index[seeds["mask_mrg"] == True].tolist()
            seeds_dataset.attrs[mn_defs.Preview.Attribute.REFINED]     = seeds.index[(seeds["mask_ks"] == True) & (seeds["mask_pnr"] == True)].tolist()

            noise_freq_group = hdf5_file.create_group(mn_defs.Preview.Group.NOISE_FREQ)
            noise_freq_group.create_dataset(defs.DoricFile.Dataset.TIME, data = time_, dtype='float', chunks = True)

            signal_group = noise_freq_group.create_group(mn_defs.Preview.Group.SIGNAL)
            noise_group  = noise_freq_group.create_group(mn_defs.Preview.Group.NOISE)
            for seed in range(trace_smth_high.shape[0]):
                signal_group.create_dataset(mn_defs.Preview.Dataset.SEED.format(idx=str(seed).zfill(4)), data = trace_smth_low[seed], dtype='float64', chunks = True)
                noise_group.create_dataset(mn_defs.Preview.Dataset.SEED.format(idx=str(seed).zfill(4)), data = trace_smth_high[seed], dtype='float64', chunks = True)

    except Exception as error:
        utils.print_error(error, mn_defs.Messages.SAVE_TO_HDF5)

    file_.close()

    # Close cluster
    client.close()
    cluster.close()


def penalties_preview(minian_params):
    """
    Saves penalties_preview
    """

    return 0


def load_chunk(intpath, subset, minian_params):

    print(mn_defs.Messages.LOAD_DATA, flush=True)
    varr, file_ = load_doric_to_xarray(**minian_params.params_load_doric)
    chk, _ = get_optimal_chk(varr, **minian_params.params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                       intpath, overwrite=True)
    varr_ref = varr.sel(subset)

    return file_, chk, varr_ref


def preprocess(varr_ref, intpath, minian_params):

    print(mn_defs.Messages.PREPROCESS, flush=True)

    # 1. Glow removal
    print(mn_defs.Messages.PREPROC_REMOVE_GLOW, flush=True)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min

    # 2. Denoise
    print(mn_defs.Messages.PREPROC_DENOISING, flush=True)
    with except_type_error("denoise"):
        varr_ref = denoise(varr_ref, **minian_params.params_denoise)

    # 3. Background removal
    print(mn_defs.Messages.PREPROC_REMOV_BACKG, flush=True)
    with except_type_error("remove_background"):
        varr_ref = remove_background(varr_ref, **minian_params.params_remove_background)

    # Save
    print(mn_defs.Messages.PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    return varr_ref


def correct_motion(varr_ref, intpath, chk, minian_params):

    if minian_params.params[defs.Parameters.danse.CORRECT_MOTION]:
        print(mn_defs.Messages.CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        with except_type_error("estimate_motion"):
            motion = estimate_motion(varr_ref, **minian_params.params_estimate_motion)

        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **minian_params.params_save_minian)
        print(mn_defs.Messages.CORRECT_MOTION_APPLY_SHIFT, flush=True)
        Y = apply_transform(varr_ref, motion, **minian_params.params_apply_transform)

    else:
        Y = varr_ref

    print(mn_defs.Messages.PREP_DATA_INIT, flush=True)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    return Y, Y_fm_chk, Y_hw_chk


def initialize_seeds(Y_fm_chk, Y_hw_chk, minian_params, return_all_seeds = False):

    print(mn_defs.Messages.INIT_SEEDS, flush=True)
    with except_print_error_no_cells(mn_defs.Messages.INIT_SEEDS):
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_params.params_save_minian).compute()

        # 2. Generating over-complete set of seeds
        with except_type_error("seeds_init"):
            seeds = seeds_init(Y_fm_chk, **minian_params.params_seeds_init)

        # 3. Peak-Noise-Ratio refine
        print(mn_defs.Messages.INIT_SEEDS_PNR_REFI, flush=True)
        with except_type_error("pnr_refine"):
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_params.params_pnr_refine)

        # 4. Kolmogorov-Smirnov refine
        print(mn_defs.Messages.INIT_SEEDS_KOLSM_REF, flush=True)
        with except_type_error("ks_refine"):
            seeds = ks_refine(Y_hw_chk, seeds, **minian_params.params_ks_refine)

        # 5. Merge seeds
        print(mn_defs.Messages.INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        with except_type_error("seeds_merge"):
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_params.params_seeds_merge)

        if return_all_seeds:
            return pd.merge(seeds, seeds_final, how="outer").fillna(False), max_proj
        else:
            return seeds_final, max_proj


def initialize_components(Y_hw_chk, Y_fm_chk, seeds_final, intpath, chk, minian_params):

    print(mn_defs.Messages.INIT_COMP, flush=True)
    with except_print_error_no_cells(mn_defs.Messages.INIT_COMP):
        # 1. Initialize spatial
        print(mn_defs.Messages.INIT_COMP_SPATIAL, flush=True)
        with except_type_error("initA"):
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **minian_params.params_initA)
        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

        # 2. Initialize temporal
        print(mn_defs.Messages.INIT_COMP_TEMP, flush=True)
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True,
                            chunks={"unit_id": 1, "frame": -1})

        # 3. Merge components
        print(mn_defs.Messages.INIT_COMP_MERG, flush=True)
        with except_type_error("unit_merge"):
            A, C = unit_merge(A_init, C_init, **minian_params.params_unit_merge)

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


def cnmf1(Y_hw_chk, intpath, A, C, C_chk, Y_fm_chk, chk, minian_params):

    with except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("1st")):
        # 1. Estimate spatial noise
        print(mn_defs.Messages.CNMF_ESTIM_NOISE.format("1st"), flush=True)
        with except_type_error("get_noise_fft"):
            sn_spatial = get_noise_fft(Y_hw_chk, **minian_params.params_get_noise_fft)

        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

        # 2. First spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        with except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_params.params_update_spatial)

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
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_params.params_update_temporal)

        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)

        # 5. Merge components
        print(mn_defs.Messages.CNMF_MERG_COMP.format("1st"), flush=True)
        with except_type_error("unit_merge"):
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **minian_params.params_unit_merge)

        # Save
        print(mn_defs.Messages.CNMF_SAVE_INTERMED.format("1st"), flush=True)
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # Renumber unit_ids to start from 1 instead of 0
    ids = A["unit_id"].values + 1
    A["unit_id"] = ids
    C["unit_id"] = ids
    C_chk["unit_id"] = ids

    return A, C, C_chk, sn_spatial


def cnmf2(Y_hw_chk, A, C, sn_spatial, intpath, C_chk, Y_fm_chk, chk, minian_params):

    with except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("2nd")):
        # 1. Second spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("2nd"), flush=True)
        with except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_params.params_update_spatial)

        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)

        # 2. Second background update
        print(mn_defs.Messages.CNMF_UPDAT_BACKG.format("2nd"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

        # 3. Second temporal update
        print(mn_defs.Messages.CNMF_UPDAT_TEMP.format("2nd"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        with except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_params.params_update_temporal)

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
    """

    file_ = h5py.File(fname, 'r')

    h5path = utils.clean_path(h5path)

    if defs.DoricFile.Dataset.IMAGE_STACK in file_[h5path]:
        file_image_stack = file_[f"{h5path}/{defs.DoricFile.Dataset.IMAGE_STACK}"]
    else:
        file_image_stack= file_[f"{h5path}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}"]

    varr = da.array.from_array(file_image_stack)

    varr = xr.DataArray(
            varr,
            dims = ["height", "width", "frame"],
            coords = {"height" : np.arange(varr.shape[0]),
                    "width" : np.arange(varr.shape[1]),
                    "frame" : np.arange(varr.shape[2]) + 1, #Frame number start a 1 not 0
                    },
            )
    varr = varr.transpose("frame", "height", "width")

    if dtype != varr.dtype:
        if dtype == np.uint8:
            #varr = (varr - varr.values.min()) / (varr.values.max() - varr.values.min()) * 2**8 + 1
            bit_count = file_image_stack.attrs[defs.DoricFile.Attribute.Image.BIT_COUNT]
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


def cross_register(AC, A, minian_params):

    if not minian_params.params_cross_reg:
        return A

    print(mn_defs.Messages.CROSS_REGISTRATING , flush=True)

    # Load AC componenets from the reference file
    ref_filepath = minian_params.params_cross_reg["fname"]
    ref_images   = minian_params.params_cross_reg["h5path_images"]
    AC_ref, file_ref = load_doric_to_xarray(ref_filepath, ref_images, np.float64, None, "subset", None)

    # Concatenate max proj of both results
    AC_ref_max = AC_ref.max("frame")
    AC_max  = AC.max("frame")
    AC_max_concat = xr.concat([AC_ref_max, AC_max], pd.Index(["reference", "current"], name = "session"))

    # Estimate a translational shift along the session dimension using the max projection for each dataset.
    # Combine the shifts, original templates temps, and shifted templates temps_sh into a single dataset shiftds to use later
    shifts = estimate_motion(AC_max_concat, dim = "session").compute().rename("shifts")
    temps_sh = apply_transform(AC_max_concat, shifts).compute().rename("temps_shifted")
    shiftds = xr.merge([AC_max_concat, shifts, temps_sh])

    # Load A componenets from the reference file
    ref_rois_path  = minian_params.params_cross_reg["h5path_roi"]
    A_ref = get_footprints(ref_filepath, ref_rois_path, AC_ref.coords)
    A_concat = xr.concat([A_ref, A], pd.Index(["reference", "current"], name="session"))
    file_ref.close()

    # Apply shifts to spatial footprints of each session
    A_shifted = apply_transform(A_concat.chunk(dict(height = -1, width = -1)), shiftds["shifts"])

    window = shiftds['temps_shifted'].isnull().sum('session')
    window, _ = xr.broadcast(window, shiftds['temps_shifted'])
    def set_window(wnd):
        return wnd == wnd.min()
    window = xr.apply_ufunc(set_window, window, input_core_dims=[["height", "width"]],
                            output_core_dims=[["height", "width"]], vectorize=True)

    # Calculate centroids of spatial footprints for cells inside a window.
    cents = calculate_centroids(A_shifted, window)

    # Calculate pairwise distance between cells in all pairs of sessions.
    # Note that at this stage, since we are computing something along the session dimension,
    # it is no longer considered as a metadata dimension, so we remove it
    dist = calculate_centroid_distance(cents, "session", [])

    # Threshold centroid distances, keeping only cell pairs with distance less than param_dist.
    param_dist = minian_params.params_cross_reg["param_dist"]

    dist_ft = dist[dist["variable", "distance"] < param_dist].copy()
    dist_ft = group_by_session(dist_ft)

    # Generate mappings for ids of the current and reference sessions
    mappings = calculate_mapping(dist_ft)
    mappings_meta = resolve_mapping(mappings)
    mappings_meta_fill = fill_mapping(mappings_meta, cents)

    # Update unit ids of the current spatial componenets A
    ids        = list(A["unit_id"].values)
    new_ids    = [0]*len(ids)
    ref_id_max = int(A_ref.coords["unit_id"].values.max()) + 1

    for i in range(len(mappings_meta_fill)):
        # Matching ids between the sessions
        group = mappings_meta_fill.iloc[i]["group"][0]
        if "current" in group and "reference" in group:
            index = ids.index(mappings_meta_fill.iloc[i]["session"]["current"])
            new_ids[index] = int(mappings_meta_fill.iloc[i]["session"]["reference"])
        # Unique ids for the current session
        elif "current" in group:
            index = ids.index(mappings_meta_fill.iloc[i]["session"]["current"])
            new_ids[index] = ref_id_max
            ref_id_max += 1

    A["unit_id"] = new_ids

    return A


def get_footprints(filename, rois_h5path, dims):

    with h5py.File(filename, 'r') as file_:

        roi_names =  list(file_.get(rois_h5path))

        roi_ids = np.zeros((len(roi_names) - 1))
        footprints = np.zeros(((len(roi_names) - 1), dims["height"].size, dims["width"].size), np.float64)
        for i in range(len(roi_names) - 1):
            attrs = utils.load_attributes(file_, f"{rois_h5path}/{roi_names[i]}")

            roi_ids[i] = int(attrs["ID"])

            coords = np.array(attrs["Coordinates"])
            mask = np.zeros((dims["height"].size, dims["width"].size), np.float64)
            cv2.drawContours(mask, [coords], -1, 255, cv2.FILLED)
            footprints[i, :, :] = mask

        footprints_xr = xr.DataArray(footprints,
                                     coords = {"unit_id": roi_ids, "height": dims["height"], "width": dims["width"]},
                                     dims   = ["unit_id", "height", "width"])

    return footprints_xr


def save_minian_to_doric(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    AC: xr.DataArray,
    S: xr.DataArray,
    time_: np.array,
    bit_count: int,
    qt_format: int,
    username:str,
    vname: str = "minian.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C",
    vdataset: str = "Series1/Sensor1",
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

    vpath    = utils.clean_path(vpath)
    vdataset = utils.clean_path(vdataset)

    res = Y - AC # residual images

    print(mn_defs.Messages.GEN_ROI_NAMES, flush = True)
    names = []
    usernames = []
    for i in range(len(C)):
        names.append("ROI"+str(i+1).zfill(4))
        usernames.append("ROI {0}".format(i+1))

    with h5py.File(vname, 'a') as f:

        # Check if MiniAn results already exist
        operationCount = ""
        if vpath in f:
            operations = [ name for name in f[vpath] if mn_defs.DoricFile.Group.ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = utils.load_attributes(f, f"{vpath}/{operation}")
                    if utils.merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(mn_defs.DoricFile.Group.ROISIGNALS)):
                            operationCount = ""
                        else:
                            operationCount = operation[-1]

                        break

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operationCount

        print(mn_defs.Messages.SAVE_ROI_SIG, flush=True)
        rois_grouppath = f"{vpath}/{mn_defs.DoricFile.Group.ROISIGNALS+operationCount}"
        rois_datapath  = f"{rois_grouppath}/{vdataset}"
        utils.save_roi_signals(C.values, A.values, time_, f, rois_datapath, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "AU"}, roi_ids=A.coords["unit_id"].values)
        utils.print_group_path_for_DANSE(rois_datapath)
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, rois_grouppath)

        if saveimages:
            print(mn_defs.Messages.SAVE_IMAGES, flush=True)
            images_grouppath = f"{vpath}/{mn_defs.DoricFile.Group.IMAGES+operationCount}"
            images_datapath  = f"{images_grouppath}/{vdataset}"
            utils.save_images(AC.values, time_, f, images_datapath, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(images_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Images)"), f, images_grouppath)

        if saveresiduals:
            print(mn_defs.Messages.SAVE_RES_IMAGES, flush=True)
            residuals_grouppath = f"{vpath}/{mn_defs.DoricFile.Group.RESIDUALS+operationCount}"
            residuals_datapath  = f"{residuals_grouppath}/{vdataset}"
            utils.save_images(res.values, time_, f, residuals_datapath, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(residuals_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source,  params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Residuals)"), f, residuals_grouppath)

        if savespikes:
            print(mn_defs.Messages.SAVE_SPIKES, flush=True)
            spikes_grouppath = f"{vpath}/{mn_defs.DoricFile.Group.SPIKES+operationCount}"
            spikes_datapath  = f"{spikes_grouppath}/{vdataset}"
            utils.save_signals(S.values > 0, time_, f, spikes_datapath, names, usernames, range_min=0, range_max=1)
            utils.print_group_path_for_DANSE(spikes_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, spikes_grouppath)

    print(mn_defs.Messages.SAVE_TO.format(path = vname), flush = True)


@contextmanager
def except_type_error(function_name: str):
    """
    conext try except to show specific message
    """

    try:
        yield
    except TypeError:
        utils.print_to_intercept(mn_defs.Messages.ONE_PARM_WRONG_TYPE.format(func = function_name))
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
