#import os
import sys
#import inspect
import h5py
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
        self.timeLength: int      = self.get_time_length()
        
        self.ops = suite2p.default_ops()
        self.ops['batch_size'] = 50 # we will decrease the batch_size in case low RAM on computer
        self.ops['threshold_scaling'] = self.params['threshold_scaling'] # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
        self.ops['fs'] = self.params['fs'] # sampling rate of recording, determines binning for cell detection
        self.ops['tau'] = self.params['tau'] # timescale of gcamp to use for deconvolution
        self.ops['nplanes'] = len(self.paths[defs.Parameters.Path.H5PATH])
        self.ops['data_path'] = [self.paths[defs.Parameters.Path.TMP_DIR]]

        # self.db = {'data_path': [self.paths[defs.Parameters.Path.TMP_DIR]]}
        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            time_ = np.array(file_[self.paths[defs.Parameters.Path.H5PATH][0].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])
            frequency = len(time_)/time_[-1]

        self.ops['fs'] = frequency

        # Remove advanced_sesttings function keys that are not in the minian functions list
        self.advanced_settings = self.params.get(defs.Parameters.danse.ADVANCED_SETTINGS, {})
        self.advanced_settings = {key: self.advanced_settings[key] for key in self.advanced_settings if key in self.ops}

        self.params[defs.Parameters.danse.ADVANCED_SETTINGS] = self.advanced_settings.copy()

        # self.db = {**self.db, **self.advanced_settings}
        self.db = self.advanced_settings
        
    def get_h5path_names(self):
        """
        Split the path to dataset into relevant names
        """
        
        h5path_names = utils.clean_path(self.paths[defs.Parameters.Path.H5PATH][0]).split('/')

        data      = h5path_names[0]
        driver    = h5path_names[1]
        operation = h5path_names[2]
        series    = h5path_names[-3]
        sensor    = h5path_names[-2]

        return [data, driver, operation, series, sensor]

    def get_time_length(self):
        nTime = -1
        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            for datapath in self.paths[defs.Parameters.Path.H5PATH]:
                _, _, timeCount = file_[datapath].shape
                
                if nTime == -1:
                    nTime = timeCount
                else:
                    nTime = min(nTime, timeCount)

        return nTime