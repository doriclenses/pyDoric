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
import minian_parameter as mn_param

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

def minian_main(minian_parameters):
    # Start cluster
    print(START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_parameters.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_parameters.paths["tmpDir"], "intermediate")
    subset = {"frame": slice(0, None)}

    ### Load and chunk the data ###
    print(LOAD_DATA, flush=True)
    varr, file_ = mn_utils.load_doric_to_xarray(**minian_parameters.params_load_doric)
    chk, _ = get_optimal_chk(varr, **minian_parameters.params_get_optimal_chk)
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
        varr_ref = denoise(varr_ref, **minian_parameters.params_denoise)
    except TypeError:
        utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("denoise"))
        sys.exit()
    # 3. Background removal
    print(PREPROC_REMOV_BACKG, flush=True)
    try:
        varr_ref = remove_background(varr_ref, **minian_parameters.params_remove_background)
    except TypeError:
        utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("remove_background"))
        sys.exit()
    # Save
    print(PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if minian_parameters.parameters["CorrectMotion"]:
        print(CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        try:
            motion = estimate_motion(varr_ref, **minian_parameters.params_estimate_motion)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("estimate_motion"))
            sys.exit()
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **minian_parameters.params_save_minian)
        print(CORRECT_MOTION_APPLY_SHIFT, flush=True)
        Y = apply_transform(varr_ref, motion, **minian_parameters.params_apply_transform)

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
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_parameters.params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        try:
            seeds = seeds_init(Y_fm_chk, **minian_parameters.params_seeds_init)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("seeds_init"))
            sys.exit()
        # 3. Peak-Noise-Ratio refine
        print(INIT_SEEDS_PNR_REFI, flush=True)
        try:
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_parameters.params_pnr_refine)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("pnr_refine"))
            sys.exit()
        # 4. Kolmogorov-Smirnov refine
        print(INIT_SEEDS_KOLSM_REF, flush=True)
        try:
            seeds = ks_refine(Y_hw_chk, seeds, **minian_parameters.params_ks_refine)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("ks_refine"))
            sys.exit()
        # 5. Merge seeds
        print(INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        try:
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_parameters.params_seeds_merge)
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
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **minian_parameters.params_initA)
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
            A, C = unit_merge(A_init, C_init, **minian_parameters.params_unit_merge)
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
            sn_spatial = get_noise_fft(Y_hw_chk, **minian_parameters.params_get_noise_fft)
        except TypeError:
            utils.print_to_intercept(ONE_PARM_WRONG_TYPE.format("get_noise_fft"))
            sys.exit()
        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
        # 2. First spatial update
        print(RUN_CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        try:
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)
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
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)
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
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **minian_parameters.params_unit_merge)
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
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)
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
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)
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
    A = save_minian(A.rename("A"), **minian_parameters.params_save_minian)
    C = save_minian(C.rename("C"), **minian_parameters.params_save_minian)
    AC = save_minian(AC.rename("AC"), **minian_parameters.params_save_minian)
    S = save_minian(S.rename("S"), **minian_parameters.params_save_minian)
    c0 = save_minian(c0.rename("c0"), **minian_parameters.params_save_minian)
    b0 = save_minian(b0.rename("b0"), **minian_parameters.params_save_minian)
    b = save_minian(b.rename("b"), **minian_parameters.params_save_minian)
    f = save_minian(f.rename("f"), **minian_parameters.params_save_minian)

    ### Save results to doric file ###
    print(SAVING_TO_DORIC, flush=True)
    # Get the path from the source data
    h5path = minian_parameters.params_load_doric['h5path']
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

    if minian_parameters.parameters["SpatialDownsample"] > 1:
        minian_parameters.parameters["BinningFactor"] = minian_parameters.parameters["SpatialDownsample"]

    mn_utils.save_minian_to_doric(
        Y, A, C, AC, S,
        fr=minian_parameters.fr,
        bits_count=attrs['BitsCount'],
        qt_format=attrs['Format'],
        imagesStackUsername=attrs['Username'] if 'Username' in attrs else sensor,
        vname=minian_parameters.params_load_doric['fname'],
        vpath='DataProcessed/'+driver+'/',
        vdataset=series+'/'+sensor+'/',
        params_doric = minian_parameters.parameters,
        params_source = params_source_data,
        saveimages=True,
        saveresiduals=True,
        savespikes=True
    )

    # Close cluster
    client.close()
    cluster.close()