# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager

# needed but not directly used
import h5py
import xarray as xr

sys.path.append('..')
import utilities as utils
import minian_utilities as mn_utils
import minian_parameter as mn_param
import minian_text_def as mn_txt

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()


def minian_main(minian_parameters):
    """minian_main.py
    """

    # Start cluster
    print(mn_txt.START_CLUSTER, flush=True)
    cluster = LocalCluster(**minian_parameters.params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(minian_parameters.paths["tmpDir"], "intermediate")
    subset = {"frame": slice(0, None)}

    ### Load and chunk the data ###
    print(mn_txt.LOAD_DATA, flush=True)
    varr, file_ = mn_utils.load_doric_to_xarray(**minian_parameters.params_load_doric)
    chk, _ = get_optimal_chk(varr, **minian_parameters.params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                       intpath, overwrite=True)
    varr_ref = varr.sel(subset)

    ### Pre-process data ###
    print(mn_txt.PREPROCESS, flush=True)
    # 1. Glow removal
    print(mn_txt.PREPROC_REMOVE_GLOW, flush=True)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    # 2. Denoise
    print(mn_txt.PREPROC_DENOISING, flush=True)
    with mn_utils.except_type_error("denoise"):
        varr_ref = denoise(varr_ref, **minian_parameters.params_denoise)

    # 3. Background removal
    print(mn_txt.PREPROC_REMOV_BACKG, flush=True)
    with mn_utils.except_type_error("remove_background"):
        varr_ref = remove_background(varr_ref, **minian_parameters.params_remove_background)

    # Save
    print(mn_txt.PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if minian_parameters.parameters["CorrectMotion"]:
        print(mn_txt.CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        with mn_utils.except_type_error("estimate_motion"):
            motion = estimate_motion(varr_ref, **minian_parameters.params_estimate_motion)

        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **minian_parameters.params_save_minian)
        print(mn_txt.CORRECT_MOTION_APPLY_SHIFT, flush=True)
        Y = apply_transform(varr_ref, motion, **minian_parameters.params_apply_transform)

    else:
        Y = varr_ref

    print(mn_txt.PREP_DATA_INIT, flush=True)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    ### Seed initialization ###
    print(mn_txt.INIT_SEEDS, flush=True)
    with mn_utils.except_print_error_no_cells(mn_txt.INIT_SEEDS):
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_parameters.params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        with mn_utils.except_type_error("seeds_init"):
            seeds = seeds_init(Y_fm_chk, **minian_parameters.params_seeds_init)

        # 3. Peak-Noise-Ratio refine
        print(mn_txt.INIT_SEEDS_PNR_REFI, flush=True)
        with mn_utils.except_type_error("pnr_refine"):
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_parameters.params_pnr_refine)

        # 4. Kolmogorov-Smirnov refine
        print(mn_txt.INIT_SEEDS_KOLSM_REF, flush=True)
        with mn_utils.except_type_error("ks_refine"):
            seeds = ks_refine(Y_hw_chk, seeds, **minian_parameters.params_ks_refine)

        # 5. Merge seeds
        print(mn_txt.INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        with mn_utils.except_type_error("seeds_merge"):
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_parameters.params_seeds_merge)

    ### Component initialization ###
    print(mn_txt.INIT_COMP, flush=True)
    with mn_utils.except_print_error_no_cells(mn_txt.INIT_COMP):
        # 1. Initialize spatial
        print(mn_txt.INIT_COMP_SPATIAL, flush=True)
        with mn_utils.except_type_error("initA"):
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **minian_parameters.params_initA)

        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
        # 2. Initialize temporal
        print(mn_txt.INIT_COMP_TEMP, flush=True)
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True,
                            chunks={"unit_id": 1, "frame": -1})
        # 3. Merge components
        print(mn_txt.INIT_COMP_MERG, flush=True)
        with mn_utils.except_type_error("unit_merge"):
            A, C = unit_merge(A_init, C_init, **minian_parameters.params_unit_merge)

        A = save_minian(A.rename("A"), intpath, overwrite=True)
        C = save_minian(C.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        # 4. Initialize background
        print(mn_txt.INIT_COMP_BACKG, flush=True)
        b, f = update_background(Y_fm_chk, A, C_chk)
        f = save_minian(f.rename("f"), intpath, overwrite=True)
        b = save_minian(b.rename("b"), intpath, overwrite=True)


    ### CNMF 1st itteration ###
    with mn_utils.except_print_error_no_cells(mn_txt.CNMF_IT.format("1st")):
        # 1. Estimate spatial noise
        print(mn_txt.CNMF_ESTIM_NOISE.format("1st"), flush=True)
        with mn_utils.except_type_error("get_noise_fft"):
            sn_spatial = get_noise_fft(Y_hw_chk, **minian_parameters.params_get_noise_fft)

        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
        # 2. First spatial update
        print(mn_txt.CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        with mn_utils.except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)

        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 3. Update background
        print(mn_txt.CNMF_UPDAT_BACKG.format("1st"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 4. First temporal update
        print(mn_txt.CNMF_UPDAT_TEMP.format("1st"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        with mn_utils.except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)

        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)
        # 5. Merge components
        print(mn_txt.CNMF_MERG_COMP.format("1st"), flush=True)
        with mn_utils.except_type_error("unit_merge"):
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **minian_parameters.params_unit_merge)

        # Save
        print(mn_txt.CNMF_SAVE_INTERMED.format("1st"), flush=True)
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)


    ### CNMF 2nd itteration ###
    with mn_utils.except_print_error_no_cells(mn_txt.CNMF_IT.format("2nd")):
        # 5. Second spatial update
        print(mn_txt.CNMF_UPDAT_SPATIAL.format("2nd"), flush=True)
        with mn_utils.except_type_error("update_spatial"):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **minian_parameters.params_update_spatial)

        C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
        C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
        # 6. Second background update
        print(mn_txt.CNMF_UPDAT_BACKG.format("2nd"), flush=True)
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
        A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
        # 7. Second temporal update
        print(mn_txt.CNMF_UPDAT_TEMP.format("2nd"), flush=True)
        YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                        chunks={"unit_id": 1, "frame": -1})
        with mn_utils.except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)

        # Save
        print(mn_txt.CNMF_SAVE_INTERMED.format("2nd"), flush=True)
        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)

        AC = compute_AtC(A, C_chk)

    ### Save final results ###
    print(mn_txt.SAVING_FINAL, flush=True)
    A = save_minian(A.rename("A"), **minian_parameters.params_save_minian)
    C = save_minian(C.rename("C"), **minian_parameters.params_save_minian)
    AC = save_minian(AC.rename("AC"), **minian_parameters.params_save_minian)
    S = save_minian(S.rename("S"), **minian_parameters.params_save_minian)
    c0 = save_minian(c0.rename("c0"), **minian_parameters.params_save_minian)
    b0 = save_minian(b0.rename("b0"), **minian_parameters.params_save_minian)
    b = save_minian(b.rename("b"), **minian_parameters.params_save_minian)
    f = save_minian(f.rename("f"), **minian_parameters.params_save_minian)

    ### Save results to doric file ###
    print(mn_txt.SAVING_TO_DORIC, flush=True)
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