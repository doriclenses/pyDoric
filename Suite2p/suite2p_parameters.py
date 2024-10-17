import os
import sys
import inspect
import numpy as np

sys.path.append("..")
import utilities as utils
import definitions as defs
import suite2p_definitions as s2p_defs

import suite2p

class Suite2pParameters:

    """
    Parameters used in suite2p library
    """

    def __init__(self, danse_params: dict):
        
        self.paths: dict          = danse_params.get(defs.Parameters.Main.PATHS, {})
        self.params: dict         = danse_params.get(defs.Parameters.Main.PARAMETERS, {})
        self.preview_params: dict = danse_params.get(defs.Parameters.Main.PREVIEW, {})
        
        self.ops = suite2p.default_ops()
        self.ops['batch_size'] = 50 # we will decrease the batch_size in case low RAM on computer
        self.ops['threshold_scaling'] = 1.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
        self.ops['fs'] = 20 # sampling rate of recording, determines binning for cell detection
        self.ops['tau'] = 1.25 # timescale of gcamp to use for deconvolution
        self.ops['nplanes'] = len(self.paths[defs.Parameters.Path.H5PATH])

        self.db = {
            'data_path': [self.paths[defs.Parameters.Path.TMP_DIR]]
        }

        
    def get_h5path_names(self):
        """
        Split the path to dataset into relevant names
        """
        
        h5path_names = utils.clean_path(self.paths[defs.Parameters.Path.H5PATH][0]).split('/')

        data = h5path_names[0]
        driver = h5path_names[1]
        operation = h5path_names[2]
        series = h5path_names[-3]
        sensor = h5path_names[-2]

        return [data, driver, operation, series, sensor]
