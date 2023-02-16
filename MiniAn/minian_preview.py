
# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import tempfile
import dask as da
import numpy as np
import xarray as xr
import functools as fct
from tifffile import imread, imwrite
from typing import Tuple, Optional, Callable
from dask.distributed import Client, LocalCluster
sys.path.append('..')
from utilities import get_frequency, load_attributes, save_attributes
from minian_utilities import load_doric_to_xarray, save_minian_to_doric, round_up_to_odd, round_down_to_odd , set_advanced_parameters_for_func_params

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

try:
    for arg in sys.argv[1:]:
        exec(arg)
except SyntaxError:
    print("[intercept] One of the advanced settings is not of a python type [end]", flush=True)
    sys.exit()

tmpDir = tempfile.TemporaryDirectory(prefix="minian_")
dpath = tmpDir.name
fr = get_frequency(kwargs["fname"], kwargs['h5path']+'Time')

os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["MINIAN_INTERMEDIATE"]   = os.path.join(dpath, "intermediate")

params = params_doric

neuron_diameter             = tuple([params_doric["NeuronDiameterMin"], params_doric["NeuronDiameterMax"]])
noise_freq: float           = params["NoiseFreq"]
thres_corr: float           = params["ThresCorr"]
spatial_penalty: float      = params["SpatialPenalty"]
temporal_penalty: float     = params["TemporalPenalty"]
spatial_downsample: int     = params["SpatialDownsample"]
temporal_downsample: int    = params["TemporalDownsample"]
json_path: str               = kwargs["fnameSeed"]
max_projection_path: str    = kwargs["fnameMaxProjection"]
video_start_frame           = params["videoStartFrame"]
video_stop_frame            = params["videoStopFrame"]

advanced_settings = {}
if "AdvancedSettings" in params_doric:
    advanced_settings = params_doric["AdvancedSettings"]
    del params_doric["AdvancedSettings"]

# removing advanced_sesttings function keys that are not in the minian functions list
minian_functions_list = ["TaskAnnotation", "get_optimal_chk", "custom_arr_optimize", "save_minian", "open_minian", "denoise",
                        "remove_background", "seeds_init", "pnr_refine", "ks_refine", "seeds_merge", "initA", "initC",
                        "compute_trace", "get_noise_fft", "update_spatial", "update_temporal", "unit_merge", "update_background", "compute_AtC",
                        "apply_transform", "estimate_motion"] + ["LocalCluster"]

advanced_settings = {key: advanced_settings[key] for key in advanced_settings if key in minian_functions_list}

for params_, dict_ in kwargs.items():
    if type(dict_) is dict:
        for key, value in dict_.items():
            params[params_.replace('params_','')+'-'+key] = value

params_LocalCluster = dict(
    n_workers=int(os.getenv("MINIAN_NWORKERS", 4)),
    memory_limit="3GB",
    resources={"MEM": 1},
    threads_per_worker=2,
    dashboard_address=":8787",
    local_directory=dpath
)
if "LocalCluster" in advanced_settings:
    params_LocalCluster, advanced_settings["LocalCluster"] = set_advanced_parameters_for_func_params(params_LocalCluster, advanced_settings["LocalCluster"], LocalCluster)


params_load_doric = {
    "fname": kwargs["fname"],
    "h5path": kwargs['h5path'],
    "dtype": np.uint8,
    "downsample": dict(frame=temporal_downsample, 
                       height=spatial_downsample, 
                       width=spatial_downsample),
    "downsample_strategy": "subset",
}

params_save_minian = {
    "dpath": os.path.join(dpath, "final"),
    "meta_dict": dict(session=-1, animal=-2),
    "overwrite": True,
}

params_get_optimal_chk = {
    "dtype": float
}
if "get_optimal_chk" in advanced_settings:
    params_get_optimal_chk, advanced_settings["get_optimal_chk"] = set_advanced_parameters_for_func_params(params_get_optimal_chk, advanced_settings["get_optimal_chk"], get_optimal_chk)


params_denoise = {
    'method': 'median',
    'ksize': round_down_to_odd((neuron_diameter[0]+neuron_diameter[-1])/4.0) # half of average size
}
if "denoise" in advanced_settings:
    params_denoise, advanced_settings["denoise"] = set_advanced_parameters_for_func_params(params_denoise, advanced_settings["denoise"], denoise)


params_remove_background = {
    'method': 'tophat',
    'wnd': np.ceil(neuron_diameter[-1]) # largest neuron diameter
}
if "remove_background" in advanced_settings:
    params_remove_background, advanced_settings["remove_background"] = set_advanced_parameters_for_func_params(params_remove_background, advanced_settings["remove_background"], remove_background)


params_estimate_motion = {
    'dim': 'frame'
}
if "estimate_motion" in advanced_settings:
    params_estimate_motion, advanced_settings["estimate_motion"] = set_advanced_parameters_for_func_params(params_estimate_motion, advanced_settings["estimate_motion"], estimate_motion)


params_apply_transform = {
    'fill': 0
}
if "apply_transform" in advanced_settings:
    params_apply_transform, advanced_settings["apply_transform"] = set_advanced_parameters_for_func_params(params_apply_transform, advanced_settings["apply_transform"], apply_transform)


wnd = 60 # time window of 60 seconds
params_seeds_init = {
        'wnd_size': fr*wnd,
        'method': 'rolling',
        'stp_size': fr*wnd / 2,
        'max_wnd': neuron_diameter[-1],
        'diff_thres': 3
}
if "seeds_init" in advanced_settings:
    params_seeds_init, advanced_settings["seeds_init"] = set_advanced_parameters_for_func_params(params_seeds_init, advanced_settings["seeds_init"], seeds_init)


params_pnr_refine = {
    "noise_freq": noise_freq,
    "thres": 1
}
if "pnr_refine" in advanced_settings:
    params_pnr_refine, advanced_settings["pnr_refine"] = set_advanced_parameters_for_func_params(params_pnr_refine, advanced_settings["pnr_refine"], pnr_refine)


params_ks_refine = {
    "sig": 0.05
}
if "ks_refine" in advanced_settings:
    params_ks_refine, advanced_settings["ks_refine"] = set_advanced_parameters_for_func_params(params_ks_refine, advanced_settings["ks_refine"], ks_refine)


params_seeds_merge = {
    'thres_dist': neuron_diameter[0],
    'thres_corr': thres_corr,
    'noise_freq': noise_freq
}
if "seeds_merge" in advanced_settings:
    params_seeds_merge, advanced_settings["seeds_merge"] = set_advanced_parameters_for_func_params(params_seeds_merge, advanced_settings["seeds_merge"], seeds_merge)

params_unit_merge = {
    'thres_corr': thres_corr
}
if "unit_merge" in advanced_settings:
    params_unit_merge, advanced_settings["unit_merge"] = set_advanced_parameters_for_func_params(params_unit_merge, advanced_settings["unit_merge"], unit_merge)


if __name__ == "__main__":

    # Start cluster
    print("Starting cluster...", flush=True)
    cluster = LocalCluster(**params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(dpath, "intermediate")
    #subset = dict(frame=slice(0, None))
    subset = dict(frame = slice(video_start_frame, video_stop_frame))

    ### Load and chunk the data ###
    print("Loading dataset to MiniAn...", flush=True)
    varr, file_ = load_doric_to_xarray(**params_load_doric)
    print(varr.coords)
    varr = varr.sel(subset)
    print(varr.coords)
    chk, _ = get_optimal_chk(varr, **params_get_optimal_chk)
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
        varr_ref = denoise(varr_ref, **params_denoise)
    except TypeError:
        print("[intercept] One parameter of denoise function is of the wrong type  [end]", flush=True)
        sys.exit()
    # 3. Background removal
    print("Pre-processing: removing background...", flush=True)
    try:
        varr_ref = remove_background(varr_ref, **params_remove_background)
    except TypeError:
        print("[intercept] One parameter of remove_background function is of the wrong type  [end]", flush=True)
        sys.exit()
    # Save
    print("Pre-processing: saving...", flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if params["CorrectMotion"]:
        print("Correcting motion: estimating shifts...", flush=True)
        try:
            motion = estimate_motion(varr_ref, **params_estimate_motion)
        except TypeError:
            print("[intercept] One parameter of estimate_motion function is of the wrong type  [end]", flush=True)
            sys.exit()   
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **params_save_minian)
        print("Correcting motion: applying shifts...", flush=True)
        Y = apply_transform(varr_ref, motion, **params_apply_transform)

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
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        try:
            seeds = seeds_init(Y_fm_chk, **params_seeds_init)
        except TypeError:
            print("[intercept] One parameter of seeds_init function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 3. Peak-Noise-Ratio refine
        print("Initializing seeds: PNR refinement...", flush=True)
        try:
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **params_pnr_refine)
        except TypeError:
            print("[intercept] One parameter of pnr_refine function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 4. Kolmogorov-Smirnov refine
        print("Initializing seeds: Kolmogorov-Smirnov refinement...", flush=True)
        try:
            seeds = ks_refine(Y_hw_chk, seeds, **params_ks_refine)
        except TypeError:
            print("[intercept] One parameter of ks_refine function is of the wrong type  [end]", flush=True)
            sys.exit()
        # 5. Merge seeds
        print("Initializing seeds: merging...", flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        try:
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **params_seeds_merge)
        except TypeError:
            print("[intercept] One parameter of seeds_merge function is of the wrong type  [end]", flush=True)
            sys.exit()
    except:
        print("[intercept] No cells where found [end]", flush=True)
        sys.exit()

    imwrite(max_projection_path, max_proj.values)
    seeds_final.to_json(json_path, orient="split", indent=4)


    # Close cluster
    client.close()
    cluster.close()