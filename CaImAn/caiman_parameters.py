import h5py
import sys
import numpy as np
from tifffile import imwrite

sys.path.append("..")
import utilities as utils
import definitions as defs
import caiman_definitions as cm_defs

from caiman.source_extraction.cnmf import params

class CaimanParameters:

    """
    Parameters used in Caiman library
    """

    def __init__(self, danse_parameters):

        self.paths      = danse_parameters.get(defs.Parameters.Main.PATHS, {})
        self.parameters = danse_parameters.get(defs.Parameters.Main.PARAMETERS, {})

        self.paths[defs.Parameters.Path.H5PATH] = utils.clean_path(self.paths[defs.Parameters.Path.H5PATH])

        self.preview_parameters = danse_parameters.get(defs.Parameters.Main.PREVIEW, {})

        with h5py.File(self.paths[defs.Parameters.Path.FILEPATH], 'r') as file_:
            self.dataname  = defs.DoricFile.Dataset.IMAGE_STACK if defs.DoricFile.Dataset.IMAGE_STACK in file_[self.paths[defs.Parameters.Path.H5PATH]] else defs.DoricFile.Deprecated.Dataset.IMAGES_STACK

            images = np.array(file_[f"{self.paths[defs.Parameters.Path.H5PATH]}/{self.dataname}"])

            freq    = utils.get_frequency(file_, f"{self.paths[defs.Parameters.Path.H5PATH]}/{defs.DoricFile.Dataset.TIME}")
            dims, T = utils.get_dims(file_, f"{self.paths[defs.Parameters.Path.H5PATH]}/{self.dataname}")

        neuron_diameter = tuple([self.parameters[defs.Parameters.danse.NEURO_DIAM_MIN], self.parameters[defs.Parameters.danse.NEURO_DIAM_MAX]])

        self.params_caiman = {
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
            "tsub": self.parameters[defs.Parameters.danse.TEMPORAL_DOWNSAMPLE],
            "ssub": self.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE],
            "rf": neuron_diameter[-1]*4,
            "stride": neuron_diameter[-1]*2,
            "only_init": True,    # set it to True to run CNMF-E
            "nb": 0,
            "nb_patch": 0,
            "method_deconvolution": "oasis",       # could use "cvxpy" alternatively
            "low_rank_background": None,
            "update_background_components": True,  # sometimes setting to False improve the results
            "min_corr": self.parameters[defs.Parameters.danse.LOCAL_CORR_THRESHOLD],
            "min_pnr": self.parameters[defs.Parameters.danse.PNR_THRESHOLD],
            "normalize_init": False,               # just leave as is
            "center_psf": True,                    # leave as is for 1 photon
            "ssub_B": 2,
            "ring_size_factor": 1.4,
            "del_duplicates": True,
            "use_cnn": False,
            "fnames": self.paths[defs.Parameters.Path.TMP_DIR] + '/' + f"tiff_{'_'.join(self.get_h5path_names()[2:4])}.tif"
            }

        print(cm_defs.Messages.WRITE_IMAGE_TIFF, flush=True)
        imwrite(self.params_caiman["fnames"], images.transpose(2, 0, 1))
        del images

        self.cnmf_params = params.CNMFParams(params_dict = self.params_caiman)
        advanced_settings = self.remove_wrong_keys(self.cnmf_params, self.parameters.get(defs.Parameters.danse.ADVANCED_SETTINGS, {}))

        # Update cnmf parameters and Advanced Setting
        self.cnmf_params.change_params(advanced_settings, True)
        self.parameters[defs.Parameters.danse.ADVANCED_SETTINGS] = advanced_settings.copy()


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


    def remove_wrong_keys(self, cnmf_params, advanced_parameters):

        """
        remove bad keys
        """

        cnmf_dict = cnmf_params.to_dict()
        caiman_keys = []
        for key1, value1 in cnmf_dict.items():
            for key2, value2 in value1.items():
                if key2 in advanced_parameters:
                    caiman_keys.append(key2)

        return {key: advanced_parameters[key] for key in caiman_keys}