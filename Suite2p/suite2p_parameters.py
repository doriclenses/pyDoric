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
        self.ops['do_registration']       = True # whether or not to run registration
        self.ops['align_by_chan']         = 1
        self.ops['nimg_init']             = int(0.1*self.time_length) # How many frames to use to compute reference image for registration
        self.ops['batch_size']            = 100 # Decrease the batch_size in case low RAM on computer
        self.ops['maxregshift']           = 0.1 # maximum shift size - for rigid registration. 0.1 = 10% of the size of the FOV      
        self.ops['smooth_sigma_time']     = 0 # Temporal smoothing, standard deviation for Gaussian kernel; Might need this to be set to 1 or 2 for low SNR data
        self.ops['smooth_sigma']          = 4 # STD in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered
        self.ops['keep_movie_raw']        = False
        self.ops['two_step_registration'] = False # Whether or not to run registration twice (for low SNR data). keep_movie_raw must be True for this to work.
        self.ops['subpixel']              = 10 # Precision of Subpixel Registration (1/subpixel steps; default is 10 = 0.1 pixel accuracy)
        self.ops['th_badframes']          = 1.0 # Determines the frames that are excluded when determining the cropping region. Decrease th_badframes and more frames will be set as bad frames
        self.ops['norm_frames']           = True # Normalize frames when detecting shifts
        self.ops['force_refImg']          = False # if True, use refImg stored in ops['refImg'] 
        self.ops['pad_fft']               = False

        # Suite2p 1P registration
        self.ops['1Preg'] = self.params["1PRegistration"] # High-pass spatial filtering and tapering, which help with 1P registration
        if self.ops['1Preg']:
            self.ops['pre_smooth'] = False
            spatial_hp_reg = 2.8 * self.params['CellDiameter']
            self.ops['spatial_hp_reg'] = (spatial_hp_reg - spatial_hp_reg%2) # Window in pixels (even number) for spatial high-pass filtering before registration
            self.ops['spatial_taper']  = int(0.03 * min(height, width)) # How many pixels to ignore on edges - they are set to zero

        # Suite2p Non-Rigid registration (optional) will approx double the time
        self.ops['nonrigid']      = True
        self.ops['block_size']    = [128,128]  # Can be [64,64], [256,256]. Recommend keeping this a power of 2 and/or 3
        self.ops['maxregshiftNR'] = int(self.params['CellDiameter']/3)
        self.ops['snr_thresh']    = 1.5 # default: 1.2, How big the phase correlation peak has to be relative to the noise in the phase correlation map for the block shift to be accepted.

        # Suite2p ROI Detection Settings
        self.ops['roidetect']         = True
        self.ops['sparse_mode']       = True
        self.ops['spatial_scale']     = 0 # if anatomical_only = 0, then diameter is ignored and spatial_scale matters. 0 means multi_scale (automatic detection of the cell diameter).
                                          # 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        self.ops['threshold_scaling'] = self.params['CellThreshold'] # Threshold for ROIs detection
        self.ops['spatial_hp_detect'] = 40 # default: 25, window for spatial high-pass filtering for neuropil subtracation before ROI detection takes place.
        self.ops['high_pass']         = 40 # default:100, but suggest less than 10 value for 1p images. Temporal high pass filter, running mean subtraction across time with window of size
        self.ops['max_overlap']       = 0.75 # 1 means, no cells are discarded
        self.ops['max_iterations']    = 20 # at most ops[‘max_iterations’], but usually stops before
        self.ops['denoise']           = True # default:False

        # Suite2p Cellpose Detection
        self.ops['anatomical_only']    = 3   # Sets to use Cellpose algorithm and find masks on enhanced mean image
        self.ops['diameter']           = self.params['CellDiameter'] # Diameter that will be used for Cellpose
        self.ops['flow_threshold']     = 0.4 # Maximum allowed error of the flows. Increase this threshold if cellpose is not returning as many ROIs as you’d expect.
        self.ops['cellprob_threshold'] = 0.0 # -6 to +6, Pixels > cellprob_threshold are used to run dynamics & determine ROIs. Decrease it, if cellpose is not returning as many ROIs as you’d expect
        self.ops['pretrained_model']   = "cyto"  # 'nuclei'
        self.ops['spatial_hp_cp']      = int(self.params['CellDiameter']/4) # Window for spatial high-pass filtering of image to be used for cellpose. Recommended: 1/4-1/8 diameter in px

        # Suite2p Spike Deconvolution settings
        self.ops['spikedetect']      = True # Whether or not to run spike_deconvolution
        self.ops['neucoeff']         = 0.7 # neuropil coefficient for all ROIs
        self.ops['baseline']         = "maximin" # method for computing baseline. maxmin: gaussian filter->min Filter->max filter; constant: gaussian filter->min of the trace; 
                                                 # constant_percentile: computes a constant baseline by taking the ops['prctile_baseline'] percentile of the trace 
        self.ops['win_baseline']     = 60 # window for maximin filter in seconds
        self.ops['sig_baseline']     = 10 # width of Gaussian filter in frames
        self.ops['prctile_baseline'] = 8 # percentile of trace to use as baseline

        # Classification Settings
        self.ops['use_builtin_classifier'] = True # Specifies whether or not to use built-in classifier for cell detection.
        self.ops['preclassify'] = 0.5 # default:0, apply classifier before signal extraction with probability 0.5 (turn off with value 0), does not affect the detected 'cells' but removes some of the 'non-cells'

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
