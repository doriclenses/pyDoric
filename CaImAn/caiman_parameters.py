import h5py
import sys
import numpy as np
from tifffile import imwrite

sys.path.append("..")
import utilities as utils
import definitions as defs
import caiman_definitions as cm_defs

from caiman.source_extraction.cnmf.params import CNMFParams

class CaimanParameters:

    """
    Parameters used in Caiman library
    """

    def __init__(self, danse_params):

        self.paths  = danse_params.get(defs.Parameters.Main.PATHS, {})
        self.params = danse_params.get(defs.Parameters.Main.PARAMETERS, {})
        self.preview_params = danse_params.get(defs.Parameters.Main.PREVIEW, {})

        self.paths[defs.Parameters.Path.H5PATH] = utils.clean_path(self.paths[defs.Parameters.Path.H5PATH][0])

        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            images = np.array(file_[self.paths[defs.Parameters.Path.H5PATH]])

            time_path = self.paths[defs.Parameters.Path.H5PATH].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)
            freq    = utils.get_frequency(file_, time_path)
            dims, T = utils.get_dims(file_, self.paths[defs.Parameters.Path.H5PATH])

        neuron_diameter = tuple([self.params[defs.Parameters.danse.NEURO_DIAM_MIN], self.params[defs.Parameters.danse.NEURO_DIAM_MAX]])

        print(cm_defs.Messages.WRITE_IMAGE_TIFF, flush=True)
        fnames = f"{self.paths[defs.Parameters.Path.TMP_DIR]}/tiff_{'_'.join(self.get_h5path_names()[2:4])}.tif"
        imwrite(fnames, images.transpose(2, 0, 1))
        del images

        self.cnmf_params = CNMFParams(params_dict = {
            "fr": freq,
            "dims": dims,
            "decay_time": 0.4,
            "pw_rigid": True,
            "max_shifts": (neuron_diameter[0], neuron_diameter[0]),
            "gSig_filt": (neuron_diameter[0], neuron_diameter[0]),
            "strides": (neuron_diameter[-1]*4, neuron_diameter[-1]*4),
            "overlaps": (neuron_diameter[-1]*2, neuron_diameter[-1]*2),
            "max_deviation_rigid": neuron_diameter[0]/2,
            "border_nan": "copy",
            "method_init": "corr_pnr",  # use this for 1 photon
            "K": None,
            "gSig": (neuron_diameter[0], neuron_diameter[0]),
            "merge_thr": 0.8,
            "p": 1,
            "tsub": self.params[defs.Parameters.danse.TEMPORAL_DOWNSAMPLE],
            "ssub": self.params[defs.Parameters.danse.SPATIAL_DOWNSAMPLE],
            "rf": neuron_diameter[-1]*4,
            "stride": neuron_diameter[-1]*2,
            "only_init": True,    # set it to True to run CNMF-E
            "nb": 0,
            "nb_patch": 0,
            "method_deconvolution": "oasis",       # could use "cvxpy" alternatively
            "low_rank_background": None,
            "update_background_components": True,  # sometimes setting to False improve the results
            "min_corr": self.params[defs.Parameters.danse.LOCAL_CORR_THRESHOLD],
            "min_pnr": self.params[defs.Parameters.danse.PNR_THRESHOLD],
            "normalize_init": False,               # just leave as is
            "center_psf": True,                    # leave as is for 1 photon
            "ssub_B": 2,
            "ring_size_factor": 1.4,
            "del_duplicates": True,
            "use_cnn": False,
            "fnames": fnames
        })

        self.params_cross_reg = {}
        if self.params.get(defs.Parameters.danse.CROSS_REG, False):
            self.params_cross_reg = {
                "fname"         : self.params[defs.Parameters.danse.REF_FILEPATH],
                "h5path_images" : self.params[defs.Parameters.danse.REF_IMAGES_PATH],
                "h5path_roi"    : self.params[defs.Parameters.danse.REF_ROIS_PATH]
            }

        advanced_settings = self.remove_wrong_keys(self.cnmf_params.to_dict(), self.params.get(defs.Parameters.danse.ADVANCED_SETTINGS, {}))
        self.cnmf_params.change_params(advanced_settings)
        self.params[defs.Parameters.danse.ADVANCED_SETTINGS] = advanced_settings.copy()



    def get_h5path_names(self):

        """
        Split the path to dataset into relevant names
        """

        h5path_names = utils.clean_path(self.paths[defs.Parameters.Path.H5PATH]).split('/')

        data      = h5path_names[0]
        driver    = h5path_names[1]
        operation = h5path_names[2]
        series    = h5path_names[-3]
        sensor    = h5path_names[-2]

        return [data, driver, operation, series, sensor]


    def remove_wrong_keys(self, cnmf_dict, advanced_params):

        """
        Remove wrong keys from advanced parameters
        """

        cnmf_keys = []
        for params_type, params_dict in cnmf_dict.items():
            for key, value in params_dict.items():
                if key in advanced_params:
                    cnmf_keys.append(key)

        return {key: advanced_params[key] for key in cnmf_keys}