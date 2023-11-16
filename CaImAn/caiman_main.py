# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import cv2
import numpy as np
import tempfile
from tifffile import imwrite
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

sys.path.append("..")
# Import CaimAn related utilities libraries
import utilities as utils
import definitions as defs
import caiman_definitions as cm_defs
import caiman_parameters  as cm_params

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()


def main(caiman_parameters):

        # Import for CaimAn lib
    import caiman as cm
    from caiman.cluster import setup_cluster
    from caiman.source_extraction import cnmf
    from caiman.source_extraction.cnmf import params
    from caiman.motion_correction import MotionCorrect

    """
    MiniAn CNMF algorithm
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

    c, dview, n_processes = setup_cluster(backend="local", n_processes=None, single_thread=False)

    file_ = h5py.File(caiman_parameters.paths[defs.Parameters.Path.FILEPATH], 'r')
    if defs.DoricFile.Dataset.IMAGE_STACK in file_[caiman_parameters.paths[defs.Parameters.Path.H5PATH]]:
        images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.IMAGE_STACK}"])
    else:
        images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}"])

    logging.debug(images.shape)

    print(cm_defs.Messages.WRITE_IMAGE_TIFF, flush=True)
    imwrite(caiman_parameters.params_caiman["fnames"], images.transpose(2, 0, 1))
    del images

    opts_dict, advanced_settings = set_advanced_parameters(params.CNMFParams(params_dict=caiman_parameters.params_caiman),
                                                            caiman_parameters.parameters[defs.Parameters.danse.ADVANCED_SETTINGS])
    #Update parameters and Advanced Setting
    opts = params.CNMFParams(params_dict = opts_dict)
    caiman_parameters.parameters[defs.Parameters.danse.ADVANCED_SETTINGS] = advanced_settings.copy()

    if bool(caiman_parameters.parameters[defs.Parameters.danse.CORRECT_MOTION]):
        # MOTION CORRECTION
        print(cm_defs.Messages.MOTION_CORREC,  flush=True)
        # do motion correction rigid
        try:
            mc = MotionCorrect(caiman_parameters.params_caiman["fnames"], dview=dview, **opts.get_group("motion"))
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
        fname_new = cm.save_memmap(caiman_parameters.params_caiman["fnames"], base_name="memmap_", order='C', border_to_0=0)


    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    try:
        print(cm_defs.Messages.START_CNMF, flush=True)
        cnm = cnmf.CNMF(n_processes = n_processes, dview=dview, params=opts)
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

    ### Save results to doric file ###
    print(cm_defs.Messages.SAVING_DATA, flush=True)
    # Get paramaters of the operation on source data
    data, driver, operation, series, sensor = caiman_parameters.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")
    # Get the attributes of the images stack
    if defs.DoricFile.Dataset.IMAGE_STACK in file_[caiman_parameters.paths[defs.Parameters.Path.H5PATH]]:
        attrs = utils.load_attributes(file_, f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.IMAGE_STACK}")
    else:
        attrs = utils.load_attributes(file_, f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}")

    time_ = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}"])

    file_.close()

    Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))

    if len(cnm.estimates.C[cnm.estimates.idx_components,:]) == 0 :
        utils.print_to_intercept(cm_defs.Messages.NO_CELLS_FOUND)
    else :
        save_caiman_to_doric(
            Yr,
            cnm.estimates.A[:,cnm.estimates.idx_components],
            cnm.estimates.C[cnm.estimates.idx_components,:],
            cnm.estimates.S[cnm.estimates.idx_components,:],
            time_= time_,
            shape = (dims[0],dims[1],T),
            bit_count = attrs[defs.DoricFile.Attribute.Image.BIT_COUNT],
            qt_format = attrs[defs.DoricFile.Attribute.Image.FORMAT],
            username = attrs.get(defs.DoricFile.Attribute.Dataset.USERNAME, sensor),
            vname = caiman_parameters.paths[defs.Parameters.Path.FILEPATH],
            vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}/",
            vdataset = f"{series}/{sensor}/",
            params_doric = caiman_parameters.parameters,
            params_source = params_source_data,
            saveimages = True,
            saveresiduals = True,
            savespikes = True)

    cm.stop_server(dview=dview)



def preview(caiman_parameters: cm_params.CaimanParameters):
    """
    in HDF5 file
    """

    # Import for CaimAn lib
    from summary_images import correlation_pnr

    video_start_frame, video_stop_frame   = caiman_parameters.preview_parameters[defs.Parameters.Preview.RANGE]

    with h5py.File(caiman_parameters.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
        if defs.DoricFile.Dataset.IMAGE_STACK in file_[caiman_parameters.paths[defs.Parameters.Path.H5PATH]]:
            images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.IMAGE_STACK}"])
        else:
            images = np.array(file_[f"{caiman_parameters.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}"])

    images = images[:, :, (video_start_frame-1):video_stop_frame]
    images = images[:, :, ::caiman_parameters.preview_parameters[defs.Parameters.Preview.TEMPORAL_DOWNSAMPLE]]

    images = images.transpose(2, 0, 1)

    cr, pnr = correlation_pnr(images, swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0

    try:
        with h5py.File(caiman_parameters.preview_parameters[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.LOCALCORR, data = cr, dtype = "float64", chunks = True)
            hdf5_file.create_dataset(cm_defs.Preview.Dataset.PN, data = pnr, dtype = "float64", chunks = True)

    except Exception as error:
        utils.print_error(error, cm_defs.Messages.SAVE_TO_HDF5)


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
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = "Series1/Sensor1/",
    params_doric: Optional[dict] = {},
    params_source: Optional[dict] = {},
    saveimages: bool = True,
    saveresiduals: bool = True,
    savespikes: bool = True
) -> str:
    """
    Save CaImAn results to .doric file:
    MiniAnImages - `AC` representing cellular activities as computed by :func:`minian.cnmf.compute_AtC`
    MiniAnResidualImages - residule movie computed as the difference between `Y` and `AC`
    MiniAnSignals - `C` with coordinates from `A`
    MiniAnSpikes - `S`
    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing
    scalar is computed with least square using a subset of frames from `Y` and `AC`
    such that their numerical values matches.

    """

    print("generating traces")
    AC = A.dot(C)
    res = Y - AC

    AC = AC.reshape(shape, order='F').transpose((-1, 0, 1))
    res = res.reshape(shape, order='F').transpose((-1, 0, 1))
    A = A.toarray()
    A = A.reshape(shape[0],shape[1],A.shape[1], order='F').transpose((-1, 0, 1))

    print(cm_defs.Messages.GEN_ROI_NAMES)
    names = []
    roiUsernames = []
    for i in range(len(C)):
        names.append("ROI"+str(i+1).zfill(4))
        roiUsernames.append(f"ROI {i+1}")

    with h5py.File(vname, 'a') as f:

        # Check if CaImAn results already exist
        operationCount = ""
        if vpath in f:
            operations = [ name for name in f[vpath] if cm_defs.DoricFile.Group.ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = utils.load_attributes(f, vpath+operation)
                    if utils.merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(cm_defs.DoricFile.Group.ROISIGNALS)):
                            operationCount = ""
                        else:
                            operationCount = operation[-1]

                        break


        if vpath[-1] != '/':
            vpath += '/'

        if vdataset[-1] != '/':
            vdataset += '/'

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operationCount

        print(cm_defs.Messages.SAVE_ROI_SIG)
        pathROIs = vpath+cm_defs.DoricFile.Group.ROISIGNALS+operationCount+'/'
        utils.save_roi_signals(C, A, time_, f, pathROIs+vdataset, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "Intensity"})
        utils.print_group_path_for_DANSE(pathROIs+vdataset)
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathROIs)

        if saveimages:
            print(cm_defs.Messages.SAVE_IMAGES)
            pathImages = vpath+cm_defs.DoricFile.Group.IMAGES+operationCount+'/'
            utils.save_images(AC, time_, f, pathImages+vdataset, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(pathImages+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Images)"), f, pathImages)

        if saveresiduals:
            print(cm_defs.Messages.SAVE_RES_IMAGES)
            pathResiduals = vpath+cm_defs.DoricFile.Group.RESIDUALS+operationCount+'/'
            utils.save_images(res, time_, f, pathResiduals+vdataset, bit_count=bit_count, qt_format=qt_format, username=username)
            utils.print_group_path_for_DANSE(pathResiduals+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] + "(Residuals)"), f, pathResiduals)

        if savespikes:
            print(cm_defs.Messages.SAVE_SPIKES)
            pathSpikes = vpath+cm_defs.DoricFile.Group.SPIKES+operationCount+'/'
            utils.save_signals(S > 0, time_, f, pathSpikes+vdataset, names, roiUsernames, range_min=0, range_max=1)
            utils.print_group_path_for_DANSE(pathSpikes+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathSpikes)

    print(cm_defs.Messages.SAVE_TO.format(path = vname))


def set_advanced_parameters(
    param,
    advanced_parameters
):

    """
    input:
    param: is a class CNMFParams
    advanced_parameters: dict

    ouput:
    new param: is a class CNMFParams
    new advanced_parameters: dict
    """
    param_dict = param.to_dict()
    advan_param_keys_used = []
    for param_part_key, param_part_value in param_dict.items():
        for part_key, part_value in param_part_value.items():
            if part_key in advanced_parameters:
               param_dict[param_part_key][part_key] = advanced_parameters[part_key]
               advan_param_keys_used.append(part_key)

    #keep only used keys in andvanced parameters
    used_advanced_parameters = {key: advanced_parameters[key] for key in advan_param_keys_used}

    return [param_dict, used_advanced_parameters]
