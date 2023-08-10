# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np
from dask.distributed import Client, LocalCluster

# needed but not directly used
import h5py
import xarray as xr

sys.path.append('..')
import utilities as utils
import minian_utilities as mn_utils

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

# Text definitions
START_CLUSTER       = "Starting cluster..."
ADVANCED_BAD_TYPE   = "One of the advanced settings is not of a python type"
LOAD_DATA           = "Loading dataset to MiniAn..."
ONE_PARM_WRONG_TYPE = "One parameter of {0} function is of the wrong type"
NO_CELLS_FOUND      = "No cells where found"
PREPROCESS          = "Pre-processing..."
PREPROC_REMOVE_GLOW = "Pre-processing: removing glow..."
PREPROC_DENOISING   = "Pre-processing: denoising..."
PREPROC_REMOV_BACKG = "Pre-processing: removing background..."
PREPROC_SAVE        = "Pre-processing: saving..."
CORRECT_MOTION_ESTIM_SHIFT  = "Correcting motion: estimating shifts..."
CORRECT_MOTION_APPLY_SHIFT  = "Correcting motion: applying shifts..."
PREP_DATA_INIT              = "Preparing data for initialization..."
INIT_SEEDS                  = "Initializing seeds..."
INIT_SEEDS_PNR_REFI         = "Initializing seeds: PNR refinement..."
INIT_SEEDS_KOLSM_REF        = "Initializing seeds: Kolmogorov-Smirnov refinement..."
INIT_SEEDS_MERG             = "Initializing seeds: merging..."
INIT_COMP                   = "Initializing components..."
INIT_COMP_SPATIAL           = "Initializing components: spatial..."
INIT_COMP_TEMP              = "Initializing components: temporal..."
INIT_COMP_MERG              = "Initializing components: merging..."
INIT_COMP_BACKG             = "Initializing components: background..."
RUN_CNMF_ITT                = "Running CNMF {0} itteration: "
RUN_CNMF_ESTIM_NOISE        = RUN_CNMF_ITT + "estimating noise..."
RUN_CNMF_UPDAT_SPATIAL      = RUN_CNMF_ITT + "updating spatial components..."
RUN_CNMF_UPDAT_BACKG        = RUN_CNMF_ITT + "updating background components..."
RUN_CNMF_UPDAT_TEMP         = RUN_CNMF_ITT + "updating temporal components..."
RUN_CNMF_MERG_COMP          = RUN_CNMF_ITT + "merging components..."
RUN_CNMF_SAVE_INTERMED      = RUN_CNMF_ITT + "saving intermediate results..."
SAVING_FINAL                = "Saving final results..."
SAVING_TO_DORIC             = "Saving data to doric file..."

kwargs = {}
params_doric = {}
danse_parameters = {}

try:
    for arg in sys.argv[1:]:
        exec(arg)
except SyntaxError:
    utils.print_to_intercept(ADVANCED_BAD_TYPE)
    sys.exit()
except Exception as error:
    mn_utils.print_error(error)
    sys.exit()

if not danse_parameters: # for backwards compatibility
    danse_parameters = {"paths": kwargs , "parameters": params_doric}

paths   = danse_parameters.get("paths", {})
parameters  = danse_parameters.get("parameters", {})

if "tmpDir" in paths:
    dpath   = paths["tmpDir"]
else : # for backwards compatibility
    tmpDir  = tempfile.TemporaryDirectory(prefix="minian_")
    dpath   = tmpDir.name

fr      = utils.get_frequency(paths["fname"], paths['h5path']+'Time')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = os.path.join(dpath, "intermediate")

neuron_diameter     = tuple((np.array([parameters["NeuronDiameterMin"], parameters["NeuronDiameterMax"]])/parameters["SpatialDownsample"]).round().astype('int'))
noise_freq: float   = parameters["NoiseFreq"]
thres_corr: float   = parameters["ThresCorr"]

advanced_settings = parameters.get("AdvancedSettings", {})

# removing advanced_sesttings function keys that are not in the minian functions list
minian_functions_list = ["TaskAnnotation", "get_optimal_chk", "custom_arr_optimize",
                         "save_minian", "open_minian", "denoise", "remove_background",
                         "seeds_init", "pnr_refine", "ks_refine", "seeds_merge", "initA", "initC",
                         "compute_trace", "get_noise_fft", "update_spatial", "update_temporal",
                         "unit_merge", "update_background", "compute_AtC", "apply_transform",
                         "estimate_motion"] + ["LocalCluster"]

advanced_settings = {key: advanced_settings[key] for key in advanced_settings if key in minian_functions_list}

params_LocalCluster = {
    "n_workers": 4,
    "memory_limit": "auto",
    "resources": {"MEM": 1}, # constrain the number of tasks that can be concurrently in memory for each worker
    "threads_per_worker": 2,
    "dashboard_address": ":8787",
    "local_directory": dpath
}
if "LocalCluster" in advanced_settings:
    advanced_settings["LocalCluster"] = {key: advanced_settings["LocalCluster"][key] for key in advanced_settings["LocalCluster"] if key in params_LocalCluster}
    params_LocalCluster.update(advanced_settings["LocalCluster"])


params_load_doric = {
    "fname": paths["fname"],
    "h5path": paths['h5path'],
    "dtype": np.uint8,
    "downsample": {"frame": parameters["TemporalDownsample"],
                    "height": parameters["SpatialDownsample"],
                    "width": parameters["SpatialDownsample"]},
    "downsample_strategy": "subset",
}

params_save_minian = {
    "dpath": os.path.join(dpath, "final"),
    "meta_dict": {"session": -1, "animal": -2},
    "overwrite": True,
}

params_get_optimal_chk = {
    "dtype": float
}
if "get_optimal_chk" in advanced_settings:
    params_get_optimal_chk, advanced_settings["get_optimal_chk"] = mn_utils.set_advanced_parameters_for_func_params(params_get_optimal_chk, advanced_settings["get_optimal_chk"], get_optimal_chk)


params_denoise = {
    'method': 'median',
    'ksize': mn_utils.round_down_to_odd(neuron_diameter[-1]/2.0) # half of the maximum diameter
}
if "denoise" in advanced_settings:
    params_denoise, advanced_settings["denoise"] = mn_utils.set_advanced_parameters_for_denoise(params_denoise, advanced_settings["denoise"], denoise)


params_remove_background = {
    'method': 'tophat',
    'wnd': np.ceil(neuron_diameter[-1]) # largest neuron diameter
}
if "remove_background" in advanced_settings:
    params_remove_background, advanced_settings["remove_background"] = mn_utils.set_advanced_parameters_for_func_params(params_remove_background, advanced_settings["remove_background"], remove_background)


params_estimate_motion = {
    'dim': 'frame'
}
if "estimate_motion" in advanced_settings:
    params_estimate_motion, advanced_settings["estimate_motion"] = mn_utils.set_advanced_parameters_for_estimate_motion(params_estimate_motion, advanced_settings["estimate_motion"], estimate_motion)


params_apply_transform = {
    'fill': 0
}
if "apply_transform" in advanced_settings:
    params_apply_transform, advanced_settings["apply_transform"] = mn_utils.set_advanced_parameters_for_func_params(params_apply_transform, advanced_settings["apply_transform"], apply_transform)


wnd = 60 # time window of 60 seconds
params_seeds_init = {
    'wnd_size': fr*wnd,
    'method': 'rolling',
    'stp_size': fr*wnd / 2,
    'max_wnd': neuron_diameter[-1],
    'diff_thres': 3
}
if "seeds_init" in advanced_settings:
    params_seeds_init, advanced_settings["seeds_init"] = mn_utils.set_advanced_parameters_for_func_params(params_seeds_init, advanced_settings["seeds_init"], seeds_init)


params_pnr_refine = {
    "noise_freq": noise_freq,
    "thres": 1
}
if "pnr_refine" in advanced_settings:
    params_pnr_refine, advanced_settings["pnr_refine"] = mn_utils.set_advanced_parameters_for_func_params(params_pnr_refine, advanced_settings["pnr_refine"], pnr_refine)


params_ks_refine = {
    "sig": 0.05
}
if "ks_refine" in advanced_settings:
    params_ks_refine, advanced_settings["ks_refine"] = mn_utils.set_advanced_parameters_for_func_params(params_ks_refine, advanced_settings["ks_refine"], ks_refine)


params_seeds_merge = {
    'thres_dist': neuron_diameter[0],
    'thres_corr': thres_corr,
    'noise_freq': noise_freq
}
if "seeds_merge" in advanced_settings:
    params_seeds_merge, advanced_settings["seeds_merge"] = mn_utils.set_advanced_parameters_for_func_params(params_seeds_merge, advanced_settings["seeds_merge"], seeds_merge)


params_initA = {
    'thres_corr': thres_corr,
    'wnd': neuron_diameter[-1],
    'noise_freq': noise_freq
}
if "initA" in advanced_settings:
    params_initA, advanced_settings["initA"] = mn_utils.set_advanced_parameters_for_func_params(params_initA, advanced_settings["initA"], initA)


params_unit_merge = {
    'thres_corr': thres_corr
}
if "unit_merge" in advanced_settings:
    params_unit_merge, advanced_settings["unit_merge"] = mn_utils.set_advanced_parameters_for_func_params(params_unit_merge, advanced_settings["unit_merge"], unit_merge)


params_get_noise_fft = {
    'noise_range': (noise_freq, 0.5)
}
if "get_noise_fft" in advanced_settings:
    params_get_noise_fft, advanced_settings["get_noise_fft"] = mn_utils.set_advanced_parameters_for_func_params(params_get_noise_fft, advanced_settings["get_noise_fft"], get_noise_fft)


params_update_spatial = {
    'dl_wnd': neuron_diameter[-1],
    'sparse_penal': parameters["SpatialPenalty"],
    'size_thres': (np.ceil(0.9*(np.pi*neuron_diameter[0]/2)**2), np.ceil(1.1*(np.pi*neuron_diameter[-1]/2)**2))
}
if "update_spatial" in advanced_settings:
    params_update_spatial, advanced_settings["update_spatial"] = mn_utils.set_advanced_parameters_for_func_params(params_update_spatial, advanced_settings["update_spatial"], update_spatial)


params_update_temporal = {
    'noise_freq': noise_freq,
    'sparse_penal': parameters["TemporalPenalty"],
    'p': 1,
    'add_lag': 20,
    'jac_thres': 0.2
}
if "update_temporal" in advanced_settings:
    params_update_temporal, advanced_settings["update_temporal"] = mn_utils.set_advanced_parameters_for_func_params(params_update_temporal, advanced_settings["update_temporal"], update_temporal)

# Update AdvancedSettings in params_doric
parameters["AdvancedSettings"] = advanced_settings.copy()

if __name__ == "__main__":

    # Start cluster
    print(START_CLUSTER, flush=True)
    cluster = LocalCluster(**params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(dpath, "intermediate")
    subset = {"frame": slice(0, None)}

    ### Load and chunk the data ###
    print(LOAD_DATA, flush=True)
    varr, file_ = mn_utils.load_doric_to_xarray(**params_load_doric)
    chk, _ = get_optimal_chk(varr, **params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                       intpath, overwrite=True)
    varr_ref = varr.sel(subset)

    ### Pre-process data ###
    print(PREPROCESS, flush=True)
    # 1. Glow removal
    print(PREPROC_REMOVE_GLOW, flush=True)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    # 2. Denoise
    print(PREPROC_DENOISING, flush=True)
    try:
        varr_ref = denoise(varr_ref, **params_denoise)
    except TypeError:
        utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("denoise"))
        sys.exit()
    # 3. Background removal
    print(PREPROC_REMOV_BACKG, flush=True)
    try:
        varr_ref = remove_background(varr_ref, **params_remove_background)
    except TypeError:
        utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("remove_background"))
        sys.exit()
    # Save
    print(PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if parameters["CorrectMotion"]:
        print(CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        try:
            motion = estimate_motion(varr_ref, **params_estimate_motion)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("estimate_motion"))
            sys.exit()
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **params_save_minian)
        print(CORRECT_MOTION_APPLY_SHIFT, flush=True)
        Y = apply_transform(varr_ref, motion, **params_apply_transform)

    else:
        Y = varr_ref

    print(PREP_DATA_INIT, flush=True)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    ### Seed initialization ###
    print(INIT_SEEDS, flush=True)
    try:
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        try:
            seeds = seeds_init(Y_fm_chk, **params_seeds_init)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("seeds_init"))
            sys.exit()
        # 3. Peak-Noise-Ratio refine
        print(INIT_SEEDS_PNR_REFI, flush=True)
        try:
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **params_pnr_refine)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("pnr_refine"))
            sys.exit()
        # 4. Kolmogorov-Smirnov refine
        print(INIT_SEEDS_KOLSM_REF, flush=True)
        try:
            seeds = ks_refine(Y_hw_chk, seeds, **params_ks_refine)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("ks_refine"))
            sys.exit()
        # 5. Merge seeds
        print(INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        try:
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **params_seeds_merge)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("seeds_merge"))
            sys.exit()
    except Exception as error:
        mn_utils.print_error(error)
        utils.print_to_intercept(NO_CELLS_FOUND)
        sys.exit()

    ### Component initialization ###
    print(INIT_COMP, flush=True)
    try:
        # 1. Initialize spatial
        print(INIT_COMP_SPATIAL, flush=True)
        try:
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **params_initA)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("initA"))
            sys.exit()
        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
        # 2. Initialize temporal
        print(INIT_COMP_TEMP, flush=True)
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True,
                            chunks={"unit_id": 1, "frame": -1})
        # 3. Merge components
        print(INIT_COMP_MERG, flush=True)
        try:
            A, C = unit_merge(A_init, C_init, **params_unit_merge)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("unit_merge"))
            sys.exit()
        A = save_minian(A.rename("A"), intpath, overwrite=True)
        C = save_minian(C.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        # 4. Initialize background
        print(INIT_COMP_BACKG, flush=True)
        b, f = update_background(Y_fm_chk, A, C_chk)
        f = save_minian(f.rename("f"), intpath, overwrite=True)
        b = save_minian(b.rename("b"), intpath, overwrite=True)

    except Exception as error:
        mn_utils.print_error(error)
        utils.print_to_intercept(NO_CELLS_FOUND)
        sys.exit()


    ### CNMF 1st itteration ###
    try:
        # 1. Estimate spatial noise
        print(RUN_CNMF_ESTIM_NOISE.format("1st"), flush=True)
        try:
            sn_spatial = get_noise_fft(Y_hw_chk, **params_get_noise_fft)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("get_noise_fft"))
            sys.exit()
        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
        # 2. First spatial update
        print(RUN_CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        try:
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **params_update_spatial)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("update_spatial"))
            sys.exit()
        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 3. Update background
        print(RUN_CNMF_UPDAT_BACKG.format("1st"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 4. First temporal update
        print(RUN_CNMF_UPDAT_TEMP.format("1st"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        try:
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **params_update_temporal)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("update_temporal"))
            sys.exit()
        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)
        # 5. Merge components
        print(RUN_CNMF_MERG_COMP.format("1st"), flush=True)
        try:
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **params_unit_merge)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("unit_merge"))
            sys.exit()
        # Save
        print(RUN_CNMF_SAVE_INTERMED.format("1st"), flush=True)
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    except Exception as error:
        mn_utils.print_error(error)
        utils.print_to_intercept(NO_CELLS_FOUND)
        sys.exit()


    ### CNMF 2nd itteration ###
    try:
        # 5. Second spatial update
        print(RUN_CNMF_UPDAT_SPATIAL.format("2nd"), flush=True)
        try:
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **params_update_spatial)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("update_spatial"))
            sys.exit()
        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 6. Second background update
        print(RUN_CNMF_UPDAT_BACKG.format("2nd"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 7. Second temporal update
        print(RUN_CNMF_UPDAT_TEMP.format("2nd"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        try:
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **params_update_temporal)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("update_temporal"))
            sys.exit()
        # Save
        print(RUN_CNMF_SAVE_INTERMED.format("2nd"), flush=True)
        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)

        AC = compute_AtC(A, C_chk)

    except Exception as error:
        mn_utils.print_error(error)
        utils.print_to_intercept(NO_CELLS_FOUND)
        sys.exit()

    ### Save final results ###
    print(SAVING_FINAL, flush=True)
    A = save_minian(A.rename("A"), **params_save_minian)
    C = save_minian(C.rename("C"), **params_save_minian)
    AC = save_minian(AC.rename("AC"), **params_save_minian)
    S = save_minian(S.rename("S"), **params_save_minian)
    c0 = save_minian(c0.rename("c0"), **params_save_minian)
    b0 = save_minian(b0.rename("b0"), **params_save_minian)
    b = save_minian(b.rename("b"), **params_save_minian)
    f = save_minian(f.rename("f"), **params_save_minian)

    ### Save results to doric file ###
    print(SAVING_TO_DORIC, flush=True)
    # Get the path from the source data
    h5path = params_load_doric['h5path']
    if h5path[0] == '/':
        h5path = h5path[1:]
    if h5path[-1] == '/':
        h5path = h5path[:-1]
    h5path_names = h5path.split('/')
    data = h5path_names[0]
    driver = h5path_names[1]
    operation = h5path_names[2]
    series = h5path_names[-2]
    sensor = h5path_names[-1]
    # Get paramaters of the operation on source data
    params_source_data = utils.load_attributes(file_, data+'/'+driver+'/'+operation)
    # Get the attributes of the images stack
    attrs = utils.load_attributes(file_, h5path+'/ImagesStack')
    file_.close()

    # Parameters
    # Set only "Operations" for params_srouce_data
    if "OperationName" in params_source_data:
        if "Operations" not in params_source_data:
            params_source_data["Operations"] = params_source_data["OperationName"]

        del params_source_data["OperationName"]

    if parameters["SpatialDownsample"] > 1:
        parameters["BinningFactor"] = parameters["SpatialDownsample"]

    mn_utils.save_minian_to_doric(
        Y, A, C, AC, S,
        fr=fr,
        bits_count=attrs['BitsCount'],
        qt_format=attrs['Format'],
        imagesStackUsername=attrs['Username'] if 'Username' in attrs else sensor,
        vname=params_load_doric['fname'],
        vpath='DataProcessed/'+driver+'/',
        vdataset=series+'/'+sensor+'/',
        params_doric = parameters,
        params_source = params_source_data,
        saveimages=True,
        saveresiduals=True,
        savespikes=True
    )

    # Close cluster
    client.close()
    cluster.close()
