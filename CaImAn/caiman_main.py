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
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend="local", n_processes=None, single_thread=False)

    fname_new = motion_correction(dview, caiman_params)

    estimates, images = cnmf(n_processes, dview, fname_new, caiman_params)

    print(cm_defs.Messages.SAVING_DATA, flush=True)
    file_ = h5py.File(caiman_params.paths[defs.Parameters.Path.FILEPATH], 'r')

    data, driver, operation, series, sensor = caiman_params.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")
    attrs = utils.load_attributes(file_, f"{caiman_params.paths[defs.Parameters.Path.H5PATH]}/{caiman_params.dataname}")
    time_ = np.array(file_[f"{caiman_params.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}"])

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
            savespikes = True)

    cm.stop_server(dview=dview)



def preview(caiman_params: cm_params.CaimanParameters):
    """
    Save Correlation and PNR in HDF5 file
    """

    video_start_frame, video_stop_frame   = caiman_params.preview_params[defs.Parameters.Preview.RANGE]

    with h5py.File(caiman_params.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
        images = np.array(file_[f"{caiman_params.paths[defs.Parameters.Path.H5PATH]}/{caiman_params.dataname}"])


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
        fname_new = cm.save_memmap([caiman_params.cnmf_params.data["fnames"]], base_name="memmap_", order='C', border_to_0=0)

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
    savespikes: bool = True
) -> str:
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
        utils.save_roi_signals(C, A, time_, f, rois_datapath, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "Intensity"})
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


