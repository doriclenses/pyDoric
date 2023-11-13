
import h5py
import numpy as np
import utilities as utils


class CaimanParameters:

    """
    Parameters used in Caiman library
    """

    def __init__(self, danse_parameters):

        self.danse_parameters = danse_parameters

        self.paths   = danse_parameters.get("paths", {})
        self.parameters  = danse_parameters.get("parameters", {})

        paths = self.paths
        parameters = self.parameters

        self.tmpDirName   = paths["tmpDir"]

        self.IMAGE_STACK = 'ImageStack'
        IMAGE_STACK = self.IMAGE_STACK
        with h5py.File(paths["fname"], 'r') as f:
            if IMAGE_STACK not in f[paths['h5path']]:
                IMAGE_STACK = "ImagesStack"

        self.fr = utils.get_frequency(paths['fname'], paths['h5path']+'Time')
        fr = self.fr
        dims, T = utils.get_dims(paths['fname'], paths['h5path']+IMAGE_STACK)


        neuron_diameter = tuple([parameters["NeuronDiameterMin"], parameters["NeuronDiameterMax"]])

        self.params_caiman = {
            'fr': fr,
            'dims': dims,
            'decay_time': 0.4,
            'pw_rigid': True,
            'max_shifts': (neuron_diameter[0], neuron_diameter[0]),
            'gSig_filt': (neuron_diameter[0], neuron_diameter[0]),
            'strides': (neuron_diameter[-1]*4, neuron_diameter[-1]*4),
            'overlaps': (neuron_diameter[-1]*2, neuron_diameter[-1]*2),
            'max_deviation_rigid': neuron_diameter[0]/2,
            'border_nan': 'copy',
            'method_init': 'corr_pnr',  # use this for 1 photon
            'K': None,
            'gSig': (neuron_diameter[0], neuron_diameter[0]),
            'merge_thr': 0.8,
            'p': 1,
            'tsub': parameters["TemporalDownsample"],
            'ssub': parameters["SpatialDownsample"],
            'rf': neuron_diameter[-1]*4,
            'stride': neuron_diameter[-1]*2,
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

        self.advanced_settings = parameters.get("AdvancedSettings", {})

