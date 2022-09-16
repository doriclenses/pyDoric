import os
import h5py
import psutil
import logging
import numpy as np
from io import StringIO
from dotenv import load_dotenv

config = StringIO("CAIMAN_DATA = "+os.path.dirname(os.path.abspath(__file__))+"\\caiman_data")
load_dotenv(stream=config)

from tifffile import imwrite
from caiman import stop_server
from caiman.cluster import setup_cluster
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params
from caiman_utilities import save_caiman_to_doric
from utilities import get_frequency, get_dims, load_attributes, save_attributes
logging.basicConfig(level=logging.DEBUG)

# Read parameters
#kwargs = eval(input("Enter paramaters:"))
kwargs = {
    "fname": "C:/Users/ING55/data/sampleDG.doric",
    "h5path": "/DataProcessed/MicroscopeDriver-1stGen1C/ProcessedImages/Series1/Sensor1/",
}

correct_motion: bool = False
neuron_diameter: str = (5, 15)
pnr_thres: float = 10
corr_thres: float = 0.8
spatial_downsample: int = 1
temporal_downsample: int = 1

params_doric = {
    "CorrectMotion": correct_motion,
    "NeuronDiameter": neuron_diameter,
    "PNRThreshold": pnr_thres,
    "CorrelationThreshold": corr_thres,
    "SpatialDownsample": spatial_downsample,
    "TemporalDownsample": temporal_downsample,
}
fname = kwargs['fname']
h5path = kwargs['h5path']
fr = get_frequency(kwargs['fname'], kwargs['h5path']+'Time')
T, dims = get_dims(kwargs['fname'], kwargs['h5path']+'ImagesStack')
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
    'tsub': 1,
    'ssub': 1,
    'rf': neuron_diameter[-1]*4,
    'only_init': True,    # set it to True to run CNMF-E
    'nb': 0,
    'nb_patch': 0,
    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
    'low_rank_background': None,
    'update_background_components': True,  # sometimes setting to False improve the results
    'min_corr': 0.8,
    'min_pnr': 10,
    'normalize_init': False,               # just leave as is
    'center_psf': True,                    # leave as is for 1 photon
    'ssub_B': 2,
    'ring_size_factor': 1.4,
    'del_duplicates': True
}
for params_, dict_ in kwargs.items():
    if type(dict_) is dict:
        for key, value in dict_.items():
            params_doric[params_+'-'+key] = value
            params_caiman[key] = value


if __name__ == "__main__":

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    try:
        cv2.setNumThreads(0)
    except:
        pass

    if 'dview' in locals():
        stop_server(dview=dview)
    c, dview, n_processes = setup_cluster(backend='local', n_processes=None, single_thread=False)

    with h5py.File(kwargs["fname"], 'r') as f:
        images = np.array(f[kwargs['h5path']+'ImagesStack'])
    
    logging.debug(images.shape)

    images = images.transpose(2, 0, 1)
    h5path_list = kwargs['h5path'].split('/')
    fname_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif'
    logging.info(fname_tif)
    imwrite(fname_tif, images)
    
    params_caiman['fnames'] = fname_tif
    opts = params.CNMFParams(params_dict=params_caiman)
    cnm = cnmf.CNMF(n_processes, dview=dview, params=opts)
    cnm = cnm.fit_file(motion_correct=correct_motion, include_eval=True)

    ### Save results to doric file ###
    print("Saving data to doric file...")
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
    params_source_data = load_attributes(fname, data+'/'+driver+'/'+operation)
    # Get the attributes of the images stack
    attrs = load_attributes(fname, h5path+'/ImagesStack')

    # Parameters
    if "OperationName" in params_source_data:
        if "Operations" in params_source_data:
            params_doric["Operations"] = params_source_data["Operations"] + " > CaImAn"
            del params_source_data["Operations"]
        else:
            params_doric["Operations"] = params_source_data["OperationName"] + " > CaImAn"
        del params_source_data["OperationName"]
    params = {**params_doric, **params_source_data}
    params["OperationName"] = "CaImAn"

    Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
    logging.info(Yr.shape,cnm.estimates.A.shape, cnm.estimates.C.shape, cnm.estimates.S.shape)
    logging.info(cnm.estimates.idx_components)
    save_caiman_to_doric(
        Yr, 
        cnm.estimates.A[:,cnm.estimates.idx_components], 
        cnm.estimates.C[cnm.estimates.idx_components,:],
        cnm.estimates.S[cnm.estimates.idx_components,:], 
        fr=fr,
        shape=(dims[0],dims[1],T),
        bits_count=attrs['BitsCount'],
        qt_format=attrs['Format'],
        vname=fname, 
        vpath='DataProcessed/'+driver+'/',
        vdataset=series+'/'+sensor+'/',
        attrs=params, 
        saveimages=True, 
        saveresiduals=True, 
        savespikes=True)

    stop_server(dview=dview)