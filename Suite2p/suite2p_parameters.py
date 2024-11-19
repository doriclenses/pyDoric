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
        self.time_length: int     = self.get_time_length()
        self.is_microscope: bool  = danse_params.pop(defs.Parameters.Main.IS_MICROSCOPE, False)
        
        self.ops = suite2p.default_ops() # https://suite2p.readthedocs.io/en/latest/settings.html#
        # Suite2p Main Settings
        self.ops['data_path']         = [self.paths[defs.Parameters.Path.TMP_DIR]]
        self.ops['nplanes']           = len(self.paths[defs.Parameters.Path.H5PATH])
        self.ops['tau']               = self.params['DecayTime'] # Timescale of GCaMP to use for deconvolution

        # Suite2p Registration Settings
        self.ops['batch_size']        = 100 # Decrease the batch_size in case low RAM on computer
        self.ops['smooth_sigma']      = 4   # STD in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered
    
        # Suite2p 1P registration
        self.ops['1Preg']             = True # High-pass spatial filtering and tapering, which help with 1P registration
        self.ops['spatial_hp_reg']    = 42   # Window in pixels for spatial high-pass filtering before registration
        self.ops['pre_smooth']        = self.params['CellDiameter'] # STD of Gaussian smoothing, which is applied before spatial high-pass filtering

        # Suite2p ROI Detection Settings
        self.ops['threshold_scaling'] = self.params['CellThreshold'] # Threshold for ROIs detection

        # Suite2p Cellpose Detection
        self.ops['anatomical_only']   = 3   # Sets to use Cellpose algorithm and find masks on enhanced mean image
        self.ops['diameter']          = self.params['CellDiameter'] # Diameter that will be used for Cellpose
        self.ops['flow_threshold']    = 0.4 # Flow threshold that will be used for cellpose

        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            time_ = np.array(file_[self.paths[defs.Parameters.Path.H5PATH][0].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])
            frequency = len(time_)/(time_[-1] - time_[0])

        self.ops['fs'] = frequency

        # Remove advanced_settings function keys that are not in the minian functions list
        self.advanced_settings = self.params.get(defs.Parameters.danse.ADVANCED_SETTINGS, {})
        self.advanced_settings = {key: self.advanced_settings[key] for key in self.advanced_settings if key in self.ops}

        self.params[defs.Parameters.danse.ADVANCED_SETTINGS] = self.advanced_settings.copy()

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
        time_count = -1
        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            for datapath in self.paths[defs.Parameters.Path.H5PATH]:
                _, _, time_count = file_[datapath].shape
                
                if time_count == -1:
                    time_count = time_count
                else:
                    time_count = min(time_count, time_count)

        return time_count
