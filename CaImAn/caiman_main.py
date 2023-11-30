# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import cv2
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

sys.path.append("..")
# Import CaimAn related utilities libraries
import utilities as utils
import definitions as defs
import caiman_definitions as cm_defs
import caiman_parameters  as cm_params
from caiman.base.rois import register_multisession

# Import for CaimAn lib
import caiman as cm

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()


def main(caiman_parameters):
    """
    CaImAn CNMF algorithm
    """

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        cv2.setNumThreads(0)
    except:
        pass

    # start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend="local", n_processes=None, single_thread=False)

    fname_new = motion_correction(dview, caiman_parameters)

    dims, T, cnm, images = cnmf(n_processes, dview, fname_new, caiman_parameters)

    # CaimAn Cross register
    A = cnm.estimates.A[:,cnm.estimates.idx_components]
    C = cnm.estimates.C[cnm.estimates.idx_components,:]
    shape = (dims[0],dims[1],T)
    AC = (A.dot(C)).reshape(shape, order='F').transpose((-1, 0, 1))
    A = A.toarray()
    A = A.reshape((shape[0], shape[1], -1), order='F').transpose((-1, 0, 1))
    registered_ids = cross_register(A, AC, caiman_parameters)

    # Save results to .doric file
    print(cm_defs.Messages.SAVING_DATA, flush=True)

    file_ = h5py.File(caiman_parameters.paths[defs.Parameters.Path.FILEPATH], 'r')
    # Get all operation parameters and dataset attributes
    data, driver, operation, series, sensor = caiman_parameters.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")

    attrs = utils.load_attributes(file_, f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{caiman_parameters.dataname}")
    time_ = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}"])

    file_.close()

    if len(cnm.estimates.C[cnm.estimates.idx_components,:]) == 0 :
        utils.print_to_intercept(cm_defs.Messages.NO_CELLS_FOUND)
    else :
        save_caiman_to_doric(
            images,
            cnm.estimates.A[:,cnm.estimates.idx_components],
            cnm.estimates.C[cnm.estimates.idx_components,:],
            cnm.estimates.S[cnm.estimates.idx_components,:],
            time_= time_,
            shape = (dims[0], dims[1], T),
            bit_count = attrs[defs.DoricFile.Attribute.Image.BIT_COUNT],
            qt_format = attrs[defs.DoricFile.Attribute.Image.FORMAT],
            username = attrs.get(defs.DoricFile.Attribute.Dataset.USERNAME, sensor),
            vname = caiman_parameters.paths[defs.Parameters.Path.FILEPATH],
            vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
            vdataset = f"{series}/{sensor}",
            params_doric = caiman_parameters.parameters,
            params_source = params_source_data,
            saveimages = True,
            saveresiduals = True,
            savespikes = True,
            ids = registered_ids)

    cm.stop_server(dview=dview)


def preview(caiman_parameters: cm_params.CaimanParameters):
    """
    Save Correlation and PNR in HDF5 file
    """

    # Import for CaimAn lib
    #from summary_images import correlation_pnr

    video_start_frame, video_stop_frame   = caiman_parameters.preview_parameters[defs.Parameters.Preview.RANGE]

    with h5py.File(caiman_parameters.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
        if defs.DoricFile.Dataset.IMAGE_STACK in file_[caiman_parameters.paths[defs.Parameters.Path.H5PATH]]:
            images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.IMAGE_STACK}"])
        else:
            images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}"])

    images = images[:, :, (video_start_frame-1):video_stop_frame]
    images = images[:, :, ::caiman_parameters.preview_parameters[defs.Parameters.Preview.TEMPORAL_DOWNSAMPLE]]

    images = images.transpose(2, 0, 1)

    cr, pnr = cm.summary_images.correlation_pnr(images, swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0

    try:
        with h5py.File(caiman_parameters.preview_parameters[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.LOCALCORR, data = cr, dtype = "float64", chunks = True)
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.PNR, data = pnr, dtype = "float64", chunks = True)

    except Exception as error:
        utils.print_error(error, cm_defs.Messages.SAVE_TO_HDF5)


def motion_correction(dview, caiman_parameters):
    """
    Perform the motion correction
    """

    if bool(caiman_parameters.parameters[defs.Parameters.danse.CORRECT_MOTION]):
        # MOTION CORRECTION
        print(cm_defs.Messages.MOTION_CORREC,  flush=True)
        # do motion correction rigid
        try:
            mc = cm.motion_correction.MotionCorrect(caiman_parameters.params_caiman["fnames"], dview=dview, **caiman_parameters.cnmf_params.get_group("motion"))
        except TypeError:
            utils.print_to_intercept(cm_defs.Messages.PARAM_WRONG_TYPE)
            sys.exit()
        except Exception as error:
            utils.print_error(error, cm_defs.Messages.MOTION_CORREC)
            sys.exit()

        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if caiman_parameters.params_caiman["pw_rigid"] else mc.fname_tot_rig

        bord_px = 0 if caiman_parameters.params_caiman["border_nan"] == "copy" else caiman_parameters.params_caiman["bord_px"]
        fname_new = cm.save_memmap(fname_mc, base_name="memmap_", order='C', border_to_0=bord_px)

    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap([caiman_parameters.params_caiman["fnames"]], base_name="memmap_", order='C', border_to_0=0)

    return fname_new


def cnmf(n_processes, dview, fname_new, caiman_parameters):
    """
    Peform CNMF operation
    """

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    try:
        print(cm_defs.Messages.START_CNMF, flush=True)
        cnm = cm.source_extraction.cnmf.CNMF(n_processes = n_processes, dview = dview, params = caiman_parameters.cnmf_params)
        print(cm_defs.Messages.FITTING, flush=True)
        cnm.fit(images)

        print(cm_defs.Messages.EVA_COMPO, flush=True)
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    except TypeError:
        utils.print_to_intercept(cm_defs.Messages.PARAM_WRONG_TYPE)
        sys.exit()
    except Exception as error:
        utils.print_error(error, cm_defs.Messages.START_CNMF)
        sys.exit()

    return dims, T, cnm, images


def save_caiman_to_doric(
    Y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    shape,
    time_: np.array,
    bit_count: int,
    qt_format: int,
    username: str,
    vname: str = "caiman.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C",
    vdataset: str = "Series1/Sensor1",
    params_doric: Optional[dict] = {},
    params_source: Optional[dict] = {},
    saveimages: bool = True,
    saveresiduals: bool = True,
    savespikes: bool = True,
    ids = None
) -> str:
    """
    Save CaImAn results to .doric file:

    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing
    scalar is computed with least square using a subset of frames from `Y` and `AC`
    such that their numerical values matches.

    """

    vpath    = utils.clean_path(vpath)
    vdataset = utils.clean_path(vdataset)

    AC = (A.dot(C)).reshape(shape, order='F').transpose((-1, 0, 1))
    Y  = Y.reshape(shape, order='F').transpose((-1, 0, 1))
    A = A.toarray()
    A = A.reshape((shape[0], shape[1], -1), order='F').transpose((-1, 0, 1))

    res = Y - AC

    print(cm_defs.Messages.GEN_ROI_NAMES, flush = True)
    names = []
    usernames = []
    for i in range(len(C)):
        names.append("ROI"+str(i+1).zfill(4))
        usernames.append(f"ROI {i+1}")

    with h5py.File(vname, 'a') as f:

        # Check if CaImAn results already exist
        operationCount = ""
        if vpath in f:
            operations = [ name for name in f[vpath] if cm_defs.DoricFile.Group.ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = utils.load_attributes(f, f"{vpath}/{operation}")
                    if utils.merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(cm_defs.DoricFile.Group.ROISIGNALS)):
                            operationCount = ""
                        else:
                            operationCount = operation[-1]

                        break

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operationCount

        print(cm_defs.Messages.SAVE_ROI_SIG, flush = True)
        rois_grouppath = f"{vpath}/{cm_defs.DoricFile.Group.ROISIGNALS+operationCount}"
        rois_datapath  = f"{rois_grouppath}/{vdataset}"
        utils.save_roi_signals(C, A, time_, f, rois_datapath, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "Intensity"}, roi_ids = ids)
        utils.print_group_path_for_DANSE(rois_datapath)
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, rois_grouppath)

        if saveimages:
            print(cm_defs.Messages.SAVE_IMAGES, flush = True)
            images_grouppath = f"{vpath}/{cm_defs.DoricFile.Group.IMAGES+operationCount}"
            images_datapath  = f"{images_grouppath}/{vdataset}"
            utils.save_images(AC, time_, f, images_datapath, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(images_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Images)"), f, images_grouppath)

        if saveresiduals:
            print(cm_defs.Messages.SAVE_RES_IMAGES, flush = True)
            residuals_grouppath = f"{vpath}/{cm_defs.DoricFile.Group.RESIDUALS+operationCount}"
            residuals_datapath  = f"{residuals_grouppath}/{vdataset}"
            utils.save_images(res, time_, f, residuals_datapath, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(residuals_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Residuals)"), f, residuals_grouppath)

        if savespikes:
            print(cm_defs.Messages.SAVE_SPIKES, flush = True)
            spikes_grouppath = f"{vpath}/{cm_defs.DoricFile.Group.SPIKES+operationCount}"
            spikes_datapath  = f"{spikes_grouppath}/{vdataset}"
            utils.save_signals(S > 0, time_, f, spikes_datapath, names, usernames, range_min=0, range_max=1)
            utils.print_group_path_for_DANSE(spikes_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, spikes_grouppath)

    print(cm_defs.Messages.SAVE_TO.format(path = vname), flush = True)


def cross_register(A, AC, caiman_parameters):

    if not caiman_parameters.params_cross_reg:
        return

    print(cm_defs.Messages.CROSS_REGISTRATING , flush=True)

    # Load AC componenets from the reference file
    ref_filepath = caiman_parameters.params_cross_reg["fname"]
    ref_images   = caiman_parameters.params_cross_reg["h5path_images"]
    file_ = h5py.File(ref_filepath, 'r')
    ref_images = utils.clean_path(ref_images)

    if defs.DoricFile.Dataset.IMAGE_STACK in file_[ref_images]:
        file_image_stack = file_[f"{ref_images}/{defs.DoricFile.Dataset.IMAGE_STACK}"]
    else:
        file_image_stack= file_[f"{ref_images}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}"]

    AC_ref = np.array(file_image_stack)
    AC_ref = AC_ref.astype(float)

    AC_ref_max = AC_ref.max(axis = 2)
    AC_max  = AC.max(axis = 0)

    templates = [AC_ref_max, AC_max]
    dims = AC_max.shape

    # Load A componenets from the reference file
    ref_rois_path  = caiman_parameters.params_cross_reg["h5path_roi"]
    A_ref = get_footprints(ref_filepath, ref_rois_path, AC_ref.shape)

    A = A.transpose(1, 2, 0)
    A_ref = A_ref.transpose(1, 2, 0)

    A = A.reshape(-1, A.shape[2])
    A_ref = A_ref.reshape(-1, A_ref.shape[2])
    spatial = [A, A_ref]

    spatial_union, assignments, matchings = register_multisession(A=spatial, dims=dims, templates=templates)

    # Update unit ids of the current spatial componenets A
    ids = [0] * A.shape[1]
    ref_id_max = A_ref.shape[1] + 1

    for i in range(spatial_union.shape[1]):
        current = assignments[i,0]
        ref = assignments[i,1]
        if np.isfinite(current) and np.isfinite(ref):
            print(current, ref)
            ids[i] = ref
        elif np.isfinite(current):
            ids[i] = ref_id_max
            ref_id_max += 1

    print(A.shape, flush = True)
    print(ids, flush = True)

    return ids


def get_footprints(filename, rois_h5path, refShape):

    with h5py.File(filename, 'r') as file_:
        roi_names =  list(file_.get(rois_h5path))
        roi_ids = np.zeros((len(roi_names) - 1))
        footprints = np.zeros(((len(roi_names) - 1), refShape[0], refShape[0]), np.float64)

        for i in range(len(roi_names) - 1):
            attrs = utils.load_attributes(file_, f"{rois_h5path}/{roi_names[i]}")
            roi_ids[i] = int(attrs["ID"])
            coords = np.array(attrs["Coordinates"])
            mask = np.zeros((refShape[0], refShape[0]), np.float64)
            cv2.drawContours(mask, [coords], -1, 255, cv2.FILLED)
            footprints[i, :, :] = mask
            
    return footprints