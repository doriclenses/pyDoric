# Import miscellaneous and utilities lybraries
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


def main(caiman_params):
    """
    CaImAn CNMF algorithm
    """

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        cv2.setNumThreads(0)
    except:
        pass

    # start the cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend="multiprocessing", n_processes = None, single_thread = False)

    fname_new = motion_correction(dview, caiman_params)

    estimates, images = cnmf(n_processes, dview, fname_new, caiman_params)

    # CaimAn Cross register
    registered_ids = cross_register(
                        estimates.A[:,estimates.idx_components],
                        estimates.C[estimates.idx_components,:],
                        np.array(images).shape,
                        caiman_params)

    print(cm_defs.Messages.SAVING_DATA, flush=True)
    file_ = h5py.File(caiman_params.paths[defs.Parameters.Path.FILEPATH], 'r')

    data, driver, operation, series, sensor = caiman_params.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")
    attrs = utils.load_attributes(file_, caiman_params.paths[defs.Parameters.Path.H5PATH])
    time_path = caiman_params.paths[defs.Parameters.Path.H5PATH].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)
    time_ = np.array(file_[time_path])

    file_.close()

    if len(estimates.C[estimates.idx_components,:]) == 0 :
        utils.print_to_intercept(cm_defs.Messages.NO_CELLS_FOUND)
    else :
        save_caiman_to_doric(
            images,
            estimates.A[:,estimates.idx_components],
            estimates.C[estimates.idx_components,:],
            estimates.S[estimates.idx_components,:],
            time_= time_,
            bit_count = attrs[defs.DoricFile.Attribute.Image.BIT_COUNT],
            qt_format = attrs[defs.DoricFile.Attribute.Image.FORMAT],
            username = attrs.get(defs.DoricFile.Attribute.Dataset.USERNAME, sensor),
            vname = caiman_params.paths[defs.Parameters.Path.FILEPATH],
            vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
            vdataset = f"{series}/{sensor}",
            params_doric = caiman_params.params,
            params_source = params_source_data,
            saveimages = True,
            saveresiduals = True,
            savespikes = True,
            ids = registered_ids)

    cm.stop_server(dview=dview)


def preview(caiman_params: cm_params.CaimanParameters):
    """
    Save Correlation and PNR in HDF5 file
    """

    video_start_frame, video_stop_frame   = caiman_params.preview_params[defs.Parameters.Preview.RANGE]

    with h5py.File(caiman_params.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
        images = np.array(file_[caiman_params.paths[defs.Parameters.Path.H5PATH]])

    images = images[:, :, (video_start_frame-1):video_stop_frame]
    images = images[:, :, ::caiman_params.preview_params[defs.Parameters.Preview.TEMPORAL_DOWNSAMPLE]]

    images = images.transpose(2, 0, 1)

    corr, pnr = cm.summary_images.correlation_pnr(images, swap_dim = False)

    corr[np.isnan(corr)] = 0
    pnr[np.isnan(pnr)] = 0

    try:
        with h5py.File(caiman_params.preview_params[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.LOCALCORR, data = corr, dtype = "float64", chunks = True)
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.PNR, data = pnr, dtype = "float64", chunks = True)

    except Exception as error:
        utils.print_error(error, cm_defs.Messages.SAVE_TO_HDF5)


def motion_correction(dview, caiman_params):
    """
    Perform the motion correction
    """

    if bool(caiman_params.params[defs.Parameters.danse.CORRECT_MOTION]):
        print(cm_defs.Messages.MOTION_CORREC,  flush=True)
        # do motion correction rigid
        try:
            mc = cm.motion_correction.MotionCorrect(caiman_params.cnmf_params.data["fnames"], dview=dview, **caiman_params.cnmf_params.get_group("motion"))
        except TypeError:
            utils.print_to_intercept(cm_defs.Messages.PARAM_WRONG_TYPE)
            sys.exit()
        except Exception as error:
            utils.print_error(error, cm_defs.Messages.MOTION_CORREC)
            sys.exit()

        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if caiman_params.cnmf_params.motion["pw_rigid"] else mc.fname_tot_rig

        bord_px = 0 if caiman_params.cnmf_params.motion["border_nan"] == "copy" else caiman_params.cnmf_params.patch["border_pix"]
        fname_new = cm.save_memmap(fname_mc, base_name="memmap_", order='C', border_to_0 = bord_px)

    else:
        fname_new = cm.save_memmap(caiman_params.cnmf_params.data["fnames"], base_name="memmap_", order='C', border_to_0=0)

    return fname_new


def cnmf(n_processes, dview, fname_new, caiman_params):
    """
    Peform CNMF operation
    """

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    try:
        print(cm_defs.Messages.START_CNMF, flush=True)
        cnm = cm.source_extraction.cnmf.CNMF(n_processes = n_processes, dview = dview, params = caiman_params.cnmf_params)
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

    return cnm.estimates, images


def cross_register(
    A: np.ndarray,
    C: np.ndarray,
    shape,
    caiman_parameters
) -> List[int]:

    if not caiman_parameters.params_cross_reg:
        return
    print(cm_defs.Messages.CROSS_REGISTRATING , flush=True)

    # Load reference images (AC_ref) and ROI footprints (A_ref)
    ref_filepath = caiman_parameters.params_cross_reg["fname"]
    ref_rois_path  = caiman_parameters.params_cross_reg["h5path_roi"]
    ref_images_path = utils.clean_path(caiman_parameters.params_cross_reg["h5path_images"])
    with h5py.File(ref_filepath, 'r') as file_:
        AC_ref = np.array(file_[f"{ref_images_path}/{caiman_parameters.dataname}"]).astype(float)
        A_ref, roi_ids_ref = get_footprints(file_, ref_rois_path, AC_ref.shape)

    # Concatenate max proj of images (AC and AC_ref)
    AC = (A.dot(C)).reshape((shape[1], shape[2], shape[0]), order='F')  # reshaping from caiman shape (2D) to image shape (3D)
    AC_max  = AC.max(axis = 2)
    AC_ref_max = AC_ref.max(axis = 2)
    templates = [AC_ref_max, AC_max]

    # Concatenate footprints (A and A_ref)
    A_ref = A_ref.transpose(1, 2, 0).reshape((-1, A_ref.shape[0]), order='F')  # reshape to caiman shape (2D)
    A  = A.toarray()
    spatial = [A, A_ref]

    _, assignments, _ = register_multisession(A=spatial, dims=AC_max.shape, templates=templates)

    # Update unit ids of the current spatial componenets A
    ids = [0] * A.shape[1]
    ref_id_max = int(max(roi_ids_ref)) + 1
    for i in range(assignments.shape[0]):
        current = assignments[i,0]
        ref = assignments[i,1]
        if np.isfinite(current) and np.isfinite(ref):
            ids[i] = int(roi_ids_ref[int(ref)])
        elif np.isfinite(current):
            ids[i] = ref_id_max
            ref_id_max += 1

    return ids


def get_footprints(file_, rois_h5path, refShape):

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

    return footprints, roi_ids


def save_caiman_to_doric(
    Y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
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
):
    """
    Save CaImAn results to .doric file:

    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing
    scalar is computed with least square using a subset of frames from `Y` and `AC`
    such that their numerical values matches.

    """

    vpath    = utils.clean_path(vpath)
    vdataset = utils.clean_path(vdataset)

    shape = np.array(Y).shape
    shape = (shape[1],shape[2], shape[0])

    # Before saving, the arrays have to be reshaped with Fortran order and shape(Height, Width, Time), then transposed to (Time, Height, Width)
    AC = (A.dot(C)).reshape(shape, order = 'F').transpose((2, 0, 1))
    A = A.toarray()
    A = A.reshape((shape[0], shape[1], -1), order = 'F').transpose((2, 0, 1))

    res = Y - AC

    print(cm_defs.Messages.GEN_ROI_NAMES, flush = True)
    ids           = ids if ids is not None else [i+1 for i in range(len(C))]
    dataset_names = [defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4)) for id_ in ids]
    usernames     = [defs.DoricFile.Dataset.ROI.format(id_) for id_ in ids]

    with h5py.File(vname, 'a') as f:

        operationCount = utils.operation_count(vpath, f, cm_defs.DoricFile.Group.ROISIGNALS, params_doric, params_source)

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operationCount

        print(cm_defs.Messages.SAVE_ROI_SIG, flush = True)
        rois_grouppath = f"{vpath}/{cm_defs.DoricFile.Group.ROISIGNALS+operationCount}"
        rois_datapath  = f"{rois_grouppath}/{vdataset}"
        attrs={"RangeMin": 0, "RangeMax": 0, "Unit": "Intensity"}
        utils.save_roi_signals(C, A, time_, f, rois_datapath,
                                ids            = ids,
                                dataset_names  = dataset_names,
                                usernames      = usernames,
                                common_attrs   = attrs)
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
            attrs = {"RangeMin": 0, "RangeMax": 0, "Unit": "AU"}
            utils.save_signals(S, time_, f, spikes_datapath,
                                dataset_names  = dataset_names,
                                usernames      = usernames,
                                attrs          = attrs)
            utils.print_group_path_for_DANSE(spikes_datapath)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, spikes_grouppath)

    print(cm_defs.Messages.SAVE_TO.format(path = vname), flush = True)