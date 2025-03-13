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
        self.is_microscope: bool  = danse_params.pop(defs.Parameters.Main.IS_MICROSCOPE, False)
        height, width, self.time_length = self.get_data_shape()
        
        self.ops = suite2p.default_ops() # https://suite2p.readthedocs.io/en/latest/settings.html#
        # Suite2p Main Settings
        self.ops['data_path']           = [self.paths[defs.Parameters.Path.TMP_DIR]]
        self.ops['nplanes']             = len(self.paths[defs.Parameters.Path.H5PATH])
        self.ops['tau']                 = self.params['BiosensorDecayTime'] # Timescale of GCaMP to use for deconvolution
        self.ops['force_sktiff']        = True # Whether or not to use scikit-image (tifffile package) for tiff reading, default is False (uses scanimage tiff-reader)
        self.ops['nchannels']           = 1 # Each tiff has these many channels per plane
        self.ops['functional_chan']     = 1 # This channel is used to extract functional ROIs (1-based)
        self.ops['frames_include']      = -1 # If greater than zero, only frames_include frames are processed
        self.ops['multiplane_parallel'] = False # Whether or not to run on server
        self.ops['ignore_flyback']      = [] # Specifies which planes should be ignored as flyback planes during processing
        
        # Bidirectional phase offset, applies to 2P recordings only
        self.ops['do_bidiphase']   = False
        self.ops['bidiphase']      = 0 # If set to any value besides 0, then this offset is used
        self.ops['bidi_corrected'] = False

        # Suite2p Registration Settings
        self.ops['batch_size']        = 100 # Decrease the batch_size in case low RAM on computer
        self.ops['smooth_sigma']      = 4   # STD in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered
        self.ops['nimg_init']         = int(0.1*self.time_length) # How many frames to use to compute reference image for registration
    
        # Suite2p 1P registration
        self.ops['1Preg']             = self.params["1PRegistration"] # High-pass spatial filtering and tapering, which help with 1P registration
        if self.ops['1Preg']:
            spatial_hp_reg = 2.8 * self.params['CellDiameter']
            self.ops['spatial_hp_reg']    = (spatial_hp_reg - spatial_hp_reg%2) # Window in pixels (even number) for spatial high-pass filtering before registration
            self.ops['spatial_taper']     = int(0.03 * min(height, width)) # How many pixels to ignore on edges - they are set to zero

        # Suite2p ROI Detection Settings
        self.ops['threshold_scaling'] = self.params['CellThreshold'] # Threshold for ROIs detection

        # Suite2p Cellpose Detection
        self.ops['anatomical_only']    = 3   # Sets to use Cellpose algorithm and find masks on enhanced mean image
        self.ops['diameter']           = self.params['CellDiameter'] # Diameter that will be used for Cellpose
        self.ops['flow_threshold']     = 0.4 # Maximum allowed error of the flows. Increase this threshold if cellpose is not returning as many ROIs as you’d expect.
        self.ops['cellprob_threshold'] = 0.0 # -6 to +6, Pixels > cellprob_threshold are used to run dynamics & determine ROIs. Decrease it, if cellpose is not returning as many ROIs as you’d expect
        self.ops['pretrained_model']   = "cyto"  # 'nuclei'
        self.ops['spatial_hp_cp']      = int(self.params['CellDiameter']/4) # Window for spatial high-pass filtering of image to be used for cellpose. Recommended: 1/4-1/8 diameter in px

        # Classification Settings
        self.ops['use_builtin_classifier'] = True # Specifies whether or not to use built-in classifier for cell detection.

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

    def get_data_shape(self):
        height = -1
        width = -1
        time_count = -1
        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            for datapath in self.paths[defs.Parameters.Path.H5PATH]:
                height, width, time_count = file_[datapath].shape
                
                if time_count == -1:
                    time_count = time_count
                else:
                    time_count = min(time_count, time_count)

        return [height, width, time_count]
