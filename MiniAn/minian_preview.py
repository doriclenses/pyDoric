
# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import json
import tempfile
import dask as da
import numpy as np
import xarray as xr
import functools as fct
from PIL import Image
from typing import Tuple, Optional, Callable
from dask.distributed import Client, LocalCluster
sys.path.append('..')
from utilities import get_frequency, load_attributes, print_to_intercept
import minian_parameter as mn_param

import minian_utilities as mn_utils

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion



def minian_preview(minian_parameters):

    # Start cluster
    print("Starting cluster...", flush=True)
    cluster = LocalCluster(**minian_parameters.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_parameters.paths["tmpDir"], "intermediate")
    #subset = dict(frame=slice(0, None))
    subset = dict(frame = slice(minian_parameters.preview_parameters["VideoStartFrame"], minian_parameters.preview_parameters["VideoStopFrame"]))

    ### Load and chunk the data ###
    print("Loading dataset to MiniAn...", flush=True)
    varr, file_ = mn_utils.load_doric_to_xarray(**minian_parameters.params_load_doric)
    varr = varr.sel(subset)
    chk, _ = get_optimal_chk(varr, **minian_parameters.params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                       intpath, overwrite=True)
    varr_ref = varr

    ### Pre-process data ###
    print("Pre-processing...", flush=True)
    # 1. Glow removal
    print("Pre-processing: removing glow...", flush=True)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    # 2. Denoise
    print("Pre-processing: denoising...", flush=True)
    try:
        varr_ref = denoise(varr_ref, **minian_parameters.params_denoise)
    except TypeError:
        print("[intercept] One parameter of denoise function is of the wrong type  [end]", flush=True)
        sys.exit()
    # 3. Background removal
    print("Pre-processing: removing background...", flush=True)
    try:
        varr_ref = remove_background(varr_ref, **minian_parameters.params_remove_background)
    except TypeError:
        print("[intercept] One parameter of remove_background function is of the wrong type  [end]", flush=True)
        sys.exit()
    # Save
    print("Pre-processing: saving...", flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if minian_parameters.parameters["CorrectMotion"]:
        print("Correcting motion: estimating shifts...", flush=True)
        try:
            motion = estimate_motion(varr_ref, **minian_parameters.params_estimate_motion)
        except TypeError:
            print("[intercept] One parameter of estimate_motion function is of the wrong type  [end]", flush=True)
            sys.exit()
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **minian_parameters.params_save_minian)
        print("Correcting motion: applying shifts...", flush=True)
        Y = apply_transform(varr_ref, motion, **minian_parameters.params_apply_transform)

    else:
        Y = varr_ref

    print("Preparing data for initialization...", flush=True)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    ### Seed initialization ###
    print("Initializing seeds...", flush=True)
    try:
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_parameters.params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        try:
            seeds = seeds_init(Y_fm_chk, **minian_parameters.params_seeds_init)
        except TypeError:
            print("[intercept] One parameter of seeds_init function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 3. Peak-Noise-Ratio refine
        print("Initializing seeds: PNR refinement...", flush=True)
        try:
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_parameters.params_pnr_refine)
        except TypeError:
            print("[intercept] One parameter of pnr_refine function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 4. Kolmogorov-Smirnov refine
        print("Initializing seeds: Kolmogorov-Smirnov refinement...", flush=True)
        try:
            seeds = ks_refine(Y_hw_chk, seeds, **minian_parameters.params_ks_refine)
        except TypeError:
            print("[intercept] One parameter of ks_refine function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 5. Merge seeds
        print("Initializing seeds: merging...", flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        try:
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_parameters.params_seeds_merge)
        except TypeError:
            print("[intercept] One parameter of seeds_merge function is of the wrong type  [end]", flush=True)
            sys.exit()
    except:
        print("[intercept] No cells where found [end]", flush=True)
        sys.exit()

    max_proj.values[np.isnan(max_proj.values)] = 0
    max_proj_image = Image.fromarray(max_proj.values)
    max_proj_image.save(minian_parameters.preview_parameters["fnameMaxProjection"])

    #save seed that was keeped after merging
    seeds_final[seeds_final.mask_mrg].to_json(minian_parameters.preview_parameters["fnameSeed"], orient="split", indent=4)

    # Close cluster
    client.close()
    cluster.close()
