# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import cv2
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager

# needed but not directly used
import h5py
import xarray as xr

sys.path.append('..')
import utilities as utils
import definitions as defs
import minian_utilities as mn_utils
import minian_parameters as mn_params
import minian_definitions as mn_defs

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion
from minian.cross_registration import calculate_centroids, calculate_centroid_distance, calculate_mapping, resolve_mapping, fill_mapping, group_by_session

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
    intpath = os.path.join(minian_parameters.paths[defs.Parameters.Path.TMP_DIR], "intermediate")
    subset = {"frame": slice(0, None)}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_parameters)

    varr_ref = preprocess(varr_ref, intpath, minian_parameters)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_parameters)

    seeds_final, _ = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters)

    A, C, C_chk, f, b = initialize_components(Y_hw_chk, Y_fm_chk, seeds_final, intpath, chk, minian_parameters)

    ### CNMF 1st itteration ###
    with mn_utils.except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("1st")):
        # 1. Estimate spatial noise
        print(mn_defs.Messages.CNMF_ESTIM_NOISE.format("1st"), flush=True)
        with mn_utils.except_type_error("get_noise_fft"):
            sn_spatial = get_noise_fft(Y_hw_chk, **minian_parameters.params_get_noise_fft)

        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
        # 2. First spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("1st"), flush=True)
        with mn_utils.except_type_error("update_spatial"):
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
        with mn_utils.except_type_error("update_temporal"):
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **minian_parameters.params_update_temporal)

        C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
        S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
        A = A.sel(unit_id=C.coords["unit_id"].values)
        # 5. Merge components
        print(mn_defs.Messages.CNMF_MERG_COMP.format("1st"), flush=True)
        with mn_utils.except_type_error("unit_merge"):
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **minian_parameters.params_unit_merge)

        # Save
        print(mn_defs.Messages.CNMF_SAVE_INTERMED.format("1st"), flush=True)
        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                            chunks={"unit_id": -1, "frame": chk["frame"]})
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)


    ### CNMF 2nd itteration ###
    with mn_utils.except_print_error_no_cells(mn_defs.Messages.CNMF_IT.format("2nd")):
        # 5. Second spatial update
        print(mn_defs.Messages.CNMF_UPDAT_SPATIAL.format("2nd"), flush=True)
        with mn_utils.except_type_error("update_spatial"):
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
        with mn_utils.except_type_error("update_temporal"):
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

    # Cross registration
    cross_register(minian_parameters, AC, A)

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
    attrs = utils.load_attributes(file_, f"{minian_parameters.clean_h5path()}/ImageStack")
    file_.close()

    # Parameters
    # Set only "Operations" for params_srouce_data
    if defs.DoricFile.Attribute.OPERATION_NAME in params_source_data:
        if defs.DoricFile.Attribute.OPERATIONS not in params_source_data:
            params_source_data[defs.DoricFile.Attribute.OPERATIONS] = params_source_data[defs.DoricFile.Attribute.OPERATION_NAME]

        del params_source_data[defs.DoricFile.Attribute.OPERATION_NAME]

    if minian_parameters.parameters[defs.Parameters.danse.SPATIAL_DOWN_SAMP] > 1:
        minian_parameters.parameters[defs.DoricFile.Attribute.BINNING_FACTOR] = minian_parameters.parameters[defs.Parameters.danse.SPATIAL_DOWN_SAMP]

    mn_utils.save_minian_to_doric(
        Y, A, C, AC, S,
        fr = minian_parameters.fr,
        bits_count = attrs[defs.DoricFile.Attribute.BIT_COUNT],
        qt_format = attrs['Format'],
        imagesStackUsername = attrs['Username'] if 'Username' in attrs else sensor,
        vname = minian_parameters.params_load_doric['fname'],
        vpath = f"DataProcessed/{driver}/",
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
    intpath = os.path.join(minian_parameters.paths[defs.Parameters.Path.TMP_DIR], "intermediate")
    subset = {"frame": slice(*minian_parameters.preview_parameters[defs.Parameters.Preview.RANGE])}

    file_, chk, varr_ref = load_chunk(intpath, subset, minian_parameters)

    varr_ref = preprocess(varr_ref, intpath, minian_parameters)

    Y, Y_fm_chk, Y_hw_chk = correct_motion(varr_ref, intpath, chk, minian_parameters)

    seeds_final, max_proj = initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters)

    # Save data for preview to hdf5 file
    try:
        with h5py.File(minian_parameters.preview_parameters[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:

            if minian_parameters.preview_parameters[mn_defs.Preview.MAX_PROJ_DATASET_NAME] in hdf5_file:
                del hdf5_file[minian_parameters.preview_parameters[mn_defs.Preview.MAX_PROJ_DATASET_NAME]]

            hdf5_file.create_dataset(minian_parameters.preview_parameters[mn_defs.Preview.MAX_PROJ_DATASET_NAME], data = max_proj.values, dtype='float', chunks = True)

            groupseed = hdf5_file.create_group(minian_parameters.preview_parameters[mn_defs.Preview.SEED_GROUP_NAME])
            for key in seeds_final:
                groupseed.create_dataset(key, data = seeds_final[key], dtype = 'float',chunks = True)

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
    varr, file_ = mn_utils.load_doric_to_xarray(**minian_parameters.params_load_doric)
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
    with mn_utils.except_type_error("denoise"):
        varr_ref = denoise(varr_ref, **minian_parameters.params_denoise)

    # 3. Background removal
    print(mn_defs.Messages.PREPROC_REMOV_BACKG, flush=True)
    with mn_utils.except_type_error("remove_background"):
        varr_ref = remove_background(varr_ref, **minian_parameters.params_remove_background)

    # Save
    print(mn_defs.Messages.PREPROC_SAVE, flush=True)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    return varr_ref

def correct_motion(varr_ref, intpath, chk, minian_parameters):
    ### Motion correction ###
    if minian_parameters.parameters[defs.Parameters.danse.CORRECT_MOTION]:
        print(mn_defs.Messages.CORRECT_MOTION_ESTIM_SHIFT, flush=True)
        with mn_utils.except_type_error("estimate_motion"):
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

def initialize_seeds(Y_fm_chk, Y_hw_chk, minian_parameters):
    ### Seed initialization ###
    print(mn_defs.Messages.INIT_SEEDS, flush=True)
    with mn_utils.except_print_error_no_cells(mn_defs.Messages.INIT_SEEDS):
        # 1. Compute max projection
        max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **minian_parameters.params_save_minian).compute()
        # 2. Generating over-complete set of seeds
        with mn_utils.except_type_error("seeds_init"):
            seeds = seeds_init(Y_fm_chk, **minian_parameters.params_seeds_init)

        # 3. Peak-Noise-Ratio refine
        print(mn_defs.Messages.INIT_SEEDS_PNR_REFI, flush=True)
        with mn_utils.except_type_error("pnr_refine"):
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **minian_parameters.params_pnr_refine)

        # 4. Kolmogorov-Smirnov refine
        print(mn_defs.Messages.INIT_SEEDS_KOLSM_REF, flush=True)
        with mn_utils.except_type_error("ks_refine"):
            seeds = ks_refine(Y_hw_chk, seeds, **minian_parameters.params_ks_refine)

        # 5. Merge seeds
        print(mn_defs.Messages.INIT_SEEDS_MERG, flush=True)
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        with mn_utils.except_type_error("seeds_merge"):
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **minian_parameters.params_seeds_merge)

        return seeds_final, max_proj

def initialize_components(Y_hw_chk, Y_fm_chk, seeds_final, intpath, chk, minian_parameters):
    ### Component initialization ###
    print(mn_defs.Messages.INIT_COMP, flush=True)
    with mn_utils.except_print_error_no_cells(mn_defs.Messages.INIT_COMP):
        # 1. Initialize spatial
        print(mn_defs.Messages.INIT_COMP_SPATIAL, flush=True)
        with mn_utils.except_type_error("initA"):
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **minian_parameters.params_initA)

        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
        # 2. Initialize temporal
        print(mn_defs.Messages.INIT_COMP_TEMP, flush=True)
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True,
                            chunks={"unit_id": 1, "frame": -1})
        # 3. Merge components
        print(mn_defs.Messages.INIT_COMP_MERG, flush=True)
        with mn_utils.except_type_error("unit_merge"):
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

def cross_register(minian_parameters, currentFile_AC, currentFile_A):
    performCrossReg = minian_parameters.params_crossRegister["crossReg"]
    if performCrossReg:
        refFileName = minian_parameters.params_crossRegister["fname_crossReg"]
        refImages   = minian_parameters.params_crossRegister["h5path_images"]
        refRois     = minian_parameters.params_crossRegister["h5path_roi"]

        # Load base/reference file to xarray
        refCR, fileCR_ =  mn_utils.load_doric_to_xarray(refFileName, refImages, np.float64, None, "subset", None)

        # max project of the miniAn images for base file and current file
        ref_max     = refCR.max("frame")
        current_max = currentFile_AC.max("frame")

        result = xr.concat([ref_max, current_max], pd.Index(['session1', 'session2'], name="frame"))
        temps = result.rename('temps')

        chk, _ = get_optimal_chk(temps, **minian_parameters.params_get_optimal_chk)
        temps = temps.chunk({"frame": 1, "height": -1, "width": -1}).rename("temps")

        # estimate shift 
        shifts = estimate_motion(temps, dim='frame').compute().rename('shifts')

        # Apply Shifts
        temps_sh = apply_transform(temps, shifts).compute().rename('temps_shifted')
        shiftds = xr.merge([temps, shifts, temps_sh])
        # All Ok till here

        # function to get ROI footprints ('A') for the base (reference) file
        refFootprints = getRefFileFootprints(refFileName, refRois)
        mergedFootprints  = xr.merge(refFootprints, currentFile_A)

        # Apply shifts to spatial footprint of each session
        A_shifted = apply_transform(mergedFootprints.chunk(dict(height=-1, width=-1)), shiftds['shifts'])
        
        # Calculate centroids
        cents = calculate_centroids(A_shifted, window)

        # Calculate centroid distance
        id_dims.remove("session")
        dist = calculate_centroid_distance(cents, index_dim=id_dims)
        # Threshold centroid distances
        dist_ft = dist[dist['variable', 'distance'] < param_dist].copy()
        dist_ft = group_by_session(dist_ft)

        # Generate mappings
        mappings = calculate_mapping(dist_ft)
        mappings_meta = resolve_mapping(mappings)
        mappings_meta_fill = fill_mapping(mappings_meta, cents)
        mappings_meta_fill.head()

    return refFile

def getRefFileFootprints(refFileName, refRois):

    return refFootprints