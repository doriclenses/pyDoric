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


def main(caiman_params):

        # Import for CaimAn lib
    import caiman as cm
    from caiman.cluster import setup_cluster
    from caiman.source_extraction import cnmf
    from caiman.source_extraction.cnmf import params
    from caiman.motion_correction import MotionCorrect

    """
    MiniAn CNMF algorithm
    """
    paths             = caiman_params.paths
    parameters        = caiman_params.parameters
    tmpDirName        = caiman_params.tmpDirName
    params_caiman     = caiman_params.params_caiman
    advanced_settings = caiman_params.advanced_settings
    fr                = caiman_params.fr

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

    c, dview, n_processes = setup_cluster(backend='local', n_processes=None, single_thread=False)

    file_ = h5py.File(paths["fname"], 'r')
    if defs.DoricFile.Dataset.IMAGE_STACK in file_[paths["h5path"]]:
        images = np.array(file_[paths['h5path']+defs.DoricFile.Dataset.IMAGE_STACK])
    else:
        images = np.array(file_[paths['h5path']+defs.DoricFile.Deprecated.Dataset.IMAGES_STACK])

    logging.debug(images.shape)

    images = images.transpose(2, 0, 1)
    h5path_list = paths['h5path'].split('/')
    fname_tif = os.path.join(tmpDirName, 'tiff' + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif')
    print(cm_defs.Messages.WRITE_IMAGE_TIFF, flush=True)
    imwrite(fname_tif, images)
    del images

    params_caiman['fnames'] = [fname_tif]

    opts = params.CNMFParams(params_dict=params_caiman)
    opts, advanced_settings = set_advanced_parameters(opts, advanced_settings)
    #Update AdvancedSettings
    parameters["AdvancedSettings"] = advanced_settings.copy()

    if bool(parameters["CorrectMotion"]):
        # MOTION CORRECTION
        print(cm_defs.Messages.MOTION_CORREC,  flush=True)
        # do motion correction rigid
        try:
            mc = MotionCorrect(params_caiman['fnames'], dview=dview, **opts.get_group('motion'))
        except TypeError:
            utils.print_to_intercept(cm_defs.Messages.PARAM_WRONG_TYPE)
            sys.exit()
        except Exception as error:
            utils.print_error(error, cm_defs.Messages.MOTION_CORREC)
            sys.exit()

        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if params_caiman['pw_rigid'] else mc.fname_tot_rig

        bord_px = 0 if params_caiman['border_nan'] == 'copy' else params_caiman['bord_px']
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)

    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(params_caiman['fnames'], base_name='memmap_', order='C', border_to_0=0)


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
    # Get the path from the source data
    h5path = paths['h5path']
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
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")
    # Get the attributes of the images stack
    if defs.DoricFile.Dataset.IMAGE_STACK in file_[caiman_params.paths["h5path"]]:
        attrs = utils.load_attributes(file_, f"{caiman_params.paths["h5path"]}/{defs.DoricFile.Dataset.IMAGE_STACK}")
    else:
        attrs = utils.load_attributes(file_, f"{caiman_params.paths["h5path"]}/{defs.DoricFile.Deprecated.Dataset.IMAGES_STACK}")

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
            fr=fr,
            shape=(dims[0],dims[1],T),
            bit_count=attrs['BitCount'],
            qt_format=attrs['Format'],
            imagesStackUsername=attrs['Username'] if 'Username' in attrs else sensor,
            vname=paths['fname'],
            vpath='DataProcessed/'+driver+'/',
            vdataset=series+'/'+sensor+'/',
            params_doric = parameters,
            params_source = params_source_data,
            saveimages=True,
            saveresiduals=True,
            savespikes=True)

    cm.stop_server(dview=dview)
    file_.close()


def preview(caiman_params: cm_params.CaimanParameters):
    """
    in HDF5 file
    """

    # Import for CaimAn lib
    from summary_images import correlation_pnr

    #To be deprecated
    kwargs       = caiman_params.paths
    params_doric = caiman_params.parameters

    #********************

    video_start_frame   = caiman_params.preview_parameters[defs.Parameters.Preview.RANGE][0]
    video_stop_frame    = caiman_params.preview_parameters[defs.Parameters.Preview.RANGE][1]


    with h5py.File(kwargs["fname"], 'r') as f:
        if defs.DoricFile.Dataset.IMAGE_STACK in f[kwargs["h5path"]]:
            images = np.array(f[kwargs['h5path']+defs.DoricFile.Dataset.IMAGE_STACK])
        else:
            images = np.array(f[kwargs['h5path']+defs.DoricFile.Deprecated.Dataset.IMAGES_STACK])

    images = images[:, :, (video_start_frame-1):video_stop_frame]
    images = images[:, :, ::params_doric['TemporalDownsample']]

    images = images.transpose(2, 0, 1)

    cr, pnr = correlation_pnr(images, swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0

    try:
        with h5py.File(caiman_params.preview_parameters[defs.Parameters.Preview.FILEPATH], 'w') as hdf5_file:
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
    fr: int,
    bit_count: int,
    qt_format: int,
    imagesStackUsername: str,
    vname: str = "caiman.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = 'Series1/Sensor1/',
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
    ROISIGNALS = 'CaImAnROISignals'
    IMAGES = 'CaImAnImages'
    RESIDUALS = 'CaImAnResidualImages'
    SPIKES = 'CaImAnSpikes'

    print("generating traces")
    AC = A.dot(C)
    res = Y - AC

    AC = AC.reshape(shape, order='F').transpose((-1, 0, 1))
    res = res.reshape(shape, order='F').transpose((-1, 0, 1))
    A = A.toarray()
    A = A.reshape(shape[0],shape[1],A.shape[1], order='F').transpose((-1, 0, 1))

    time_ = np.arange(0, shape[2]/fr, 1/fr, dtype='float64')

    print("generating ROI names")
    names = []
    roiUsernames = []
    for i in range(len(C)):
        names.append('ROI'+str(i+1).zfill(4))
        roiUsernames.append('ROI {}'.format(i+1))

    with h5py.File(vname, 'a') as f:

        # Check if CaImAn results already exist
        operationCount = ''
        if vpath in f:
            operations = [ name for name in f[vpath] if ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = utils.load_attributes(f, vpath+operation)
                    if utils.merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(ROISIGNALS)):
                            operationCount = ''
                        else:
                            operationCount = operation[-1]

                        break


        if vpath[-1] != '/':
            vpath += '/'

        if vdataset[-1] != '/':
            vdataset += '/'

        params_doric["Operations"] += operationCount

        print("saving ROI signals")
        pathROIs = vpath+ROISIGNALS+operationCount+'/'
        utils.save_roi_signals(C, A, time_, f, pathROIs+vdataset, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "Intensity"})
        utils.print_group_path_for_DANSE(pathROIs+vdataset)
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathROIs)

        if saveimages:
            print("saving images")
            pathImages = vpath+IMAGES+operationCount+'/'
            utils.save_images(AC, time_, f, pathImages+vdataset, bit_count=bit_count, qt_format=qt_format, username=imagesStackUsername)
            utils.print_group_path_for_DANSE(pathImages+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric["Operations"] + "(Images)"), f, pathImages)

        if saveresiduals:
            print("saving residual images")
            pathResiduals = vpath+RESIDUALS+operationCount+'/'
            utils.save_images(res, time_, f, pathResiduals+vdataset, bit_count=bit_count, qt_format=qt_format, username=imagesStackUsername)
            utils.print_group_path_for_DANSE(pathResiduals+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source, params_doric["Operations"] + "(Residuals)"), f, pathResiduals)

        if savespikes:
            print("saving spikes")
            pathSpikes = vpath+SPIKES+operationCount+'/'
            utils.save_signals(S > 0, time_, f, pathSpikes+vdataset, names, roiUsernames, range_min=0, range_max=1)
            utils.print_group_path_for_DANSE(pathSpikes+vdataset)
            utils.save_attributes(utils.merge_params(params_doric, params_source), f, pathSpikes)

    print("Saved to {}".format(vname))

def set_advanced_parameters(
    param,
    advanced_parameters
):

    '''
    input:
    param: is a class CNMFParams
    advanced_parameters: dict

    ouput:
    new param: is a class CNMFParams
    new advanced_parameters: dict
    '''
    param_dict = param.to_dict()
    advan_param_keys_used = []
    for param_part_key, param_part_value in param_dict.items():
        for part_key, part_value in param_part_value.items():
            if part_key in advanced_parameters:
               param_dict[param_part_key][part_key] = advanced_parameters[part_key]
               advan_param_keys_used.append(part_key)

    #keep only used keys in andvanced parameters
    advanced_parameters = {key: advanced_parameters[key] for key in advan_param_keys_used}

    return [param, advanced_parameters]
