
import h5py
import sys
import numpy as np

sys.path.append("..")
import utilities as utils
import definitions as defs
import caiman_definitions as cm_defs

class CaimanParameters:

    """
    Parameters used in Caiman library
    """

    def __init__(self, danse_parameters):

        self.paths   = danse_parameters.get(defs.Parameters.Main.PATHS, {})
        self.parameters  = danse_parameters.get(defs.Parameters.Main.PARAMETERS, {})

        # to deprecated use function in utils when there
        h5path = self.paths[defs.Parameters.Path.H5PATH]
        if h5path[0] == '/':
            h5path = h5path[1:]
        if h5path[-1] == '/':
            h5path = h5path[:-1]
        #*****************
        self.paths[defs.Parameters.Path.H5PATH] = h5path

        self.preview = False
        if defs.Parameters.Main.PREVIEW in danse_parameters:
            self.preview = True
            self.preview_parameters = danse_parameters[defs.Parameters.Main.PREVIEW]

        # To be deprecated
        self.danse_parameters = danse_parameters

        self.IMAGE_STACK = 'ImageStack'
        IMAGE_STACK = self.IMAGE_STACK
        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as f:
            if IMAGE_STACK not in f[self.paths[defs.Parameters.Path.H5PATH]]:
                IMAGE_STACK = "ImagesStack"

        freq    = utils.get_frequency(self.paths[defs.Parameters.Path.FILEPATH], f"{self.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}")
        dims, T = utils.get_dims(self.paths[defs.Parameters.Path.FILEPATH], f"{self.paths[defs.Parameters.Path.H5PATH]}/{IMAGE_STACK}")

        neuron_diameter = tuple([self.parameters[defs.Parameters.danse.NEURO_DIAM_MIN], self.parameters[defs.Parameters.danse.NEURO_DIAM_MAX]])

        self.params_caiman = {
            'fr': freq,
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
            'tsub': self.parameters[defs.Parameters.danse.TEMPORAL_DOWNSAMPLE],
            'ssub': self.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE],
            'rf': neuron_diameter[-1]*4,
            'stride': neuron_diameter[-1]*2,
            'only_init': True,    # set it to True to run CNMF-E
            'nb': 0,
            'nb_patch': 0,
            'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
            'low_rank_background': None,
            'update_background_components': True,  # sometimes setting to False improve the results
            'min_corr': self.parameters[defs.Parameters.danse.LOCAL_CORR_THRESHOLD],
            'min_pnr': self.parameters[defs.Parameters.danse.PNR_THRESHOLD],
            'normalize_init': False,               # just leave as is
            'center_psf': True,                    # leave as is for 1 photon
            'ssub_B': 2,
            'ring_size_factor': 1.4,
            'del_duplicates': True,
            'use_cnn': False
            }

        self.advanced_settings = self.parameters.get(defs.Parameters.danse.ADVANCED_SETTINGS, {})



    def get_h5path_names(self):

        """
        Split the path to dataset into relevant names
        """

        h5path_names = self.paths[defs.Parameters.Path.H5PATH].split('/')

        data = h5path_names[0]
        driver = h5path_names[1]
        operation = h5path_names[2]
        series = h5path_names[-2]
        sensor = h5path_names[-1]

        return [data, driver, operation, series, sensor]

