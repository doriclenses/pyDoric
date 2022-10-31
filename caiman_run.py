
# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import psutil
import logging
import numpy as np
from tifffile import imwrite
from utilities import get_frequency, get_dims, load_attributes, save_attributes

# Import for CaimAn lib
from caiman import stop_server
from caiman.cluster import setup_cluster
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params
from caiman_utilities import save_caiman_to_doric

# Import for PyInstaller
from io import StringIO
from dotenv import load_dotenv
from multiprocessing import freeze_support

logging.basicConfig(level=logging.DEBUG)

config = StringIO("CAIMAN_DATA = "+os.path.dirname(os.path.abspath(__file__))+"\\caiman_data")
load_dotenv(stream=config)

freeze_support()


for arg in sys.argv[1:]:
    exec(arg)

#params_doric = {
#    "CorrectMotion": bool(kwargs["CorrectMotion"]),
#    "NeuronDiameter": eval(kwargs["NeuronDiameter"]),
#    "PNRThreshold": kwargs["PNRThreshold"],
#    "CorrelationThreshold": kwargs["CorrelationThreshold"],
#    "SpatialDownsample": kwargs["SpatialDownsample"],
#    "TemporalDownsample": kwargs["TemporalDownsample"],
#}

correct_motion: bool        = bool(params_doric["CorrectMotion"])
neuron_diameter             = tuple([params_doric["NeuronDiameterMin"], params_doric["NeuronDiameterMax"]])
pnr_thres: float            = params_doric["PNRThreshold"]
corr_thres: float           = params_doric["CorrelationThreshold"]
spatial_downsample: int     = params_doric["SpatialDownsample"]
temporal_downsample: int    = params_doric["TemporalDownsample"]

fname = kwargs['fname']
h5path = kwargs['h5path']
fr = get_frequency(kwargs['fname'], kwargs['h5path']+'Time')
dims, T = get_dims(kwargs['fname'], kwargs['h5path']+'ImagesStack')

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
    print("Write image in tiff...", flush=True)
    imwrite(fname_tif, images)
    
    params_caiman['fnames'] = fname_tif
    opts = params.CNMFParams(params_dict=params_caiman)
    print("Starting CNMF...", flush=True)
    cnm = cnmf.CNMF(n_processes, dview=dview, params=opts)
    print("Fitting...", flush=True)
    cnm = cnm.fit_file(motion_correct=correct_motion, include_eval=True)

    ### Save results to doric file ###
    print("Saving data to doric file...", flush=True)
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
