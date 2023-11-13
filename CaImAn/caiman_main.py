# Import miscellaneous and utilities librarys
import os
import sys
import h5py

import cv2
import h5py
import numpy as np

# Import for CaimAn lib
import caiman as cm
from caiman.cluster import setup_cluster
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params
from caiman.motion_correction import MotionCorrect

sys.path.append("..")
# Import CaimAn related utilities libraries
import utilities as utils
import caiman_utilities   as cm_utils
import caiman_definitions as cm_defs
import caiman_parameters  as cm_params

# Import miscellaneous libraries
import tempfile
from tifffile import imwrite
import logging

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

def main(caiman_params):

    """
    MiniAn CNMF algorithm
    """
    paths             = caiman_params.paths
    parameters        = caiman_params.parameters
    tmpDirName        = caiman_params.tmpDirName
    params_caiman     = caiman_params.params_caiman
    advanced_settings = caiman_params.advanced_settings
    IMAGE_STACK       = caiman_params.IMAGE_STACK
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

    with h5py.File(paths["fname"], 'r') as f:
        images = np.array(f[paths['h5path']+IMAGE_STACK])

    logging.debug(images.shape)

    images = images.transpose(2, 0, 1)
    h5path_list = paths['h5path'].split('/')
    fname_tif = os.path.join(tmpDirName, 'tiff' + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif')
    print(cm_defs.Messages.WRITE_IMAGE_TIFF, flush=True)
    imwrite(fname_tif, images)
    del images

    params_caiman['fnames'] = [fname_tif]

    opts = params.CNMFParams(params_dict=params_caiman)
    opts, advanced_settings = cm_utils.set_advanced_parameters(opts, advanced_settings)
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
    params_source_data = utils.load_attributes(paths['fname'], data+'/'+driver+'/'+operation)
    # Get the attributes of the images stack
    attrs = utils.load_attributes(paths['fname'], paths['h5path']+'/'+IMAGE_STACK)

    # Parameters
    # Set only "Operations" for params_srouce_data
    if "OperationName" in params_source_data:
        if "Operations" not in params_source_data:
            params_source_data["Operations"] = params_source_data["OperationName"]

        del params_source_data["OperationName"]


    Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))

    if len(cnm.estimates.C[cnm.estimates.idx_components,:]) == 0 :
        utils.print_to_intercept(cm_defs.Messages.NO_CELLS_FOUND)
    else :
        cm_utils.save_caiman_to_doric(
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
