# Import system related libraries
import os
import sys

# Edit system variables and path
# /!\ The change of environment variable CAIMAN_DATA need to be done before all caiman related imports
os.environ["CAIMAN_DATA"] = os.path.dirname(os.path.abspath(__file__))+"\\caiman_data"
sys.path.append('..')

# Import for CaimAn lib
import caiman as cm
from caiman import stop_server
from caiman.cluster import setup_cluster
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params
from caiman.motion_correction import MotionCorrect

# Import CaimAn related utilities libraries
import utilities as utils
from utilities import print_to_intercept
import caiman_utilities as cm_utils

# Import for PyInstaller
from multiprocessing import freeze_support

# Import miscellaneous libraries
import cv2
import h5py
import logging
import tempfile
import numpy as np
from tifffile import imwrite

# Miscellaneous configuration for CaImAn
logging.basicConfig(level=logging.DEBUG)
freeze_support()


kwargs = {}
params_doric = {}
danse_parameters = {}

try:
    for arg in sys.argv[1:]:
        exec(arg)
except SyntaxError:
    print_to_intercept("One of the advanced settings is not of a python type")
    sys.exit()

if not danse_parameters:
    danse_parameters = {"file_path": kwargs , "parameters": params_doric}

file_path   = danse_parameters.get("file_path", {})
parameters  = danse_parameters.get("parameters", {})

tmpDir = tempfile.TemporaryDirectory(prefix="caiman_")
tmpDirName = tmpDir.name
fr = utils.get_frequency(file_path['fname'], file_path['h5path']+'Time')
dims, T = utils.get_dims(file_path['fname'], file_path['h5path']+'ImagesStack')

neuron_diameter             = tuple([parameters["NeuronDiameterMin"], parameters["NeuronDiameterMax"]])

params_caiman = {
    'fr': fr,
    'dims': dims,
    'decay_time': 0.4,
    'pw_rigid': False,
    'max_shifts': (neuron_diameter[0], neuron_diameter[0]),
    'gSig_filt': (neuron_diameter[0], neuron_diameter[0]),
    'strides': (neuron_diameter[-1]*4, neuron_diameter[-1]*4),
    'overlaps': (neuron_diameter[-1]*2, neuron_diameter[-1]*2),
    'max_deviation_rigid': neuron_diameter[0],
    'border_nan': 'copy',
    'method_init': 'corr_pnr',  # use this for 1 photon
    'K': None,
    'gSig': (neuron_diameter[0], neuron_diameter[0]),
    'merge_thr': 0.8,
    'p': 1,
    'tsub': parameters["TemporalDownsample"],
    'ssub': parameters["SpatialDownsample"],
    'rf': neuron_diameter[-1]*4,
    'only_init': True,    # set it to True to run CNMF-E
    'nb': 0,
    'nb_patch': 0,
    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
    'low_rank_background': None,
    'update_background_components': True,  # sometimes setting to False improve the results
    'min_corr': parameters["CorrelationThreshold"],
    'min_pnr': parameters["PNRThreshold"],
    'normalize_init': False,               # just leave as is
    'center_psf': True,                    # leave as is for 1 photon
    'ssub_B': 2,
    'ring_size_factor': 1.4,
    'del_duplicates': True,
    'use_cnn': False
}

advanced_settings = parameters.get("AdvancedSettings", {})

if __name__ == "__main__":

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

    with h5py.File(kwargs["fname"], 'r') as f:
        images = np.array(f[kwargs['h5path']+'ImagesStack'])

    logging.debug(images.shape)

    images = images.transpose(2, 0, 1)
    h5path_list = kwargs['h5path'].split('/')
    fname_tif = os.path.join(tmpDirName, 'tiff' + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif')
    print("Write image in tiff...", flush=True)
    imwrite(fname_tif, images)

    params_caiman['fnames'] = [fname_tif]

    opts = params.CNMFParams(params_dict=params_caiman)
    opts, advanced_settings = cm_utils.set_advanced_parameters(opts, advanced_settings)
    #Update AdvancedSettings
    parameters["AdvancedSettings"] = advanced_settings.copy()

    if bool(parameters["CorrectMotion"]):
        # MOTION CORRECTION
        print("MOTION CORRECTION",  flush=True)
        # do motion correction rigid
        try:
            mc = MotionCorrect(params_caiman['fnames'], dview=dview, **opts.get_group('motion'))
        except TypeError:
            print_to_intercept("One parameter is of the wrong type")
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
        print("Starting CNMF...", flush=True)
        cnm = cnmf.CNMF(n_processes = n_processes, dview=dview, params=opts)
        print("Fitting...", flush=True)
        cnm.fit(images)

        print("evaluate_components...", flush=True)
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    except TypeError:
        print_to_intercept("One parameter is of the wrong type")
        sys.exit()

    ### Save results to doric file ###
    print("Saving data to doric file...", flush=True)
    # Get the path from the source data
    h5path = file_path['h5path']
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
    params_source_data = utils.load_attributes(file_path['fname'], data+'/'+driver+'/'+operation)
    # Get the attributes of the images stack
    attrs = utils.load_attributes(file_path['fname'], h5path+'/ImagesStack')

    # Parameters
    # Set only "Operations" for params_srouce_data
    if "OperationName" in params_source_data:
        if "Operations" not in params_source_data:
            params_source_data["Operations"] = params_source_data["OperationName"]

        del params_source_data["OperationName"]


    Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))

    if len(cnm.estimates.C[cnm.estimates.idx_components,:]) == 0 :
        print_to_intercept("No cells where found")
    else :
        cm_utils.save_caiman_to_doric(
            Yr,
            cnm.estimates.A[:,cnm.estimates.idx_components],
            cnm.estimates.C[cnm.estimates.idx_components,:],
            cnm.estimates.S[cnm.estimates.idx_components,:],
            fr=fr,
            shape=(dims[0],dims[1],T),
            bits_count=attrs['BitsCount'],
            qt_format=attrs['Format'],
            imagesStackUsername=attrs['Username'] if 'Username' in attrs else sensor,
            vname=file_path['fname'],
            vpath='DataProcessed/'+driver+'/',
            vdataset=series+'/'+sensor+'/',
            params_doric = parameters,
            params_source = params_source_data,
            saveimages=True,
            saveresiduals=True,
            savespikes=True)

    stop_server(dview=dview)
