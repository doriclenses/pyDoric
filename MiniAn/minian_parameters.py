import os
import sys
import inspect
import numpy as np

# Import for MiniAn lib
import minian.utilities as mnUtils
import minian.preprocessing as mnPreproc
import minian.initialization as mnInit
import minian.cnmf as mnCnmf
import minian.motion_correction as mnMotcorr

sys.path.append('..')
import utilities as utils
import definitions as defs
import minian_definitions as mn_defs

class MinianParameters:
    
    """
    Parameters used in minian library
    """

    def __init__(self, danse_parameters):
        self.paths   = danse_parameters.get(defs.Parameters.Main.PATHS, {})
        self.parameters  = danse_parameters.get(defs.Parameters.Main.PARAMETERS, {})
        
        self.preview = False
        if defs.Parameters.Main.PREVIEW in danse_parameters:
            self.preview = True
            self.preview_parameters = danse_parameters[defs.Parameters.Main.PREVIEW]

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MINIAN_INTERMEDIATE"] = os.path.join(self.paths[defs.Parameters.Path.TMP_DIR], "intermediate")

        self.fr = utils.get_frequency(self.paths[defs.Parameters.Path.FILEPATH], self.paths[defs.Parameters.Path.H5PATH]+'Time')

        neuron_diameter = np.array([self.parameters[defs.Parameters.danse.NEURO_DIAM_MIN], self.parameters[defs.Parameters.danse.NEURO_DIAM_MAX]])
        neuron_diameter = neuron_diameter / self.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE]
        neuron_diameter = tuple(neuron_diameter.round().astype('int'))

        noise_freq: float   = self.parameters[defs.Parameters.danse.NOISE_FREQ]
        thres_corr: float   = self.parameters[defs.Parameters.danse.THRES_CORR]

        self.params_LocalCluster = {
            "n_workers": 4,
            "memory_limit": "auto",
            "resources": {"MEM": 1}, # constrain the number of tasks that can be concurrently in memory for each worker
            "threads_per_worker": 2,
            "dashboard_address": ":8787",
            "local_directory": self.paths[defs.Parameters.Path.TMP_DIR]
        }

        self.params_load_doric = {
            "fname": self.paths[defs.Parameters.Path.FILEPATH],
            "h5path": self.paths[defs.Parameters.Path.H5PATH],
            "dtype": np.uint8,
            "downsample": {"frame": self.parameters[defs.Parameters.danse.TEMPORAL_DOWNSAMPLE] if not self.preview else self.preview_parameters[defs.Parameters.Preview.TEMPORAL_DOWNSAMPLE],
                            "height": self.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE],
                            "width": self.parameters[defs.Parameters.danse.SPATIAL_DOWNSAMPLE]},
            "downsample_strategy": "subset",
        }

        self.params_save_minian = {
            "dpath": os.path.join(self.paths[defs.Parameters.Path.TMP_DIR], "final"),
            "meta_dict": {"session": -1, "animal": -2},
            "overwrite": True,
        }

        self.params_get_optimal_chk = {
            "dtype": float
        }

        self.params_denoise = {
            'method': 'median',
            'ksize': self.round_down_to_odd(neuron_diameter[-1]/2.0) # half of the maximum diameter
        }

        self.params_remove_background = {
            'method': 'tophat',
            'wnd': np.ceil(neuron_diameter[-1]) # largest neuron diameter
        }

        self.params_estimate_motion = {
            'dim': 'frame'
        }

        self.params_apply_transform = {
            'fill': 0
        }

        wnd = 60 # time window of 60 seconds
        self.params_seeds_init = {
            'wnd_size': self.fr*wnd,
            'method': 'rolling',
            'stp_size': self.fr*wnd / 2,
            'max_wnd': neuron_diameter[-1],
            'diff_thres': 3
        }

        self.params_pnr_refine = {
            "noise_freq": noise_freq,
            "thres": 1
        }

        self.params_ks_refine = {
            "sig": 0.05
        }

        self.params_seeds_merge = {
            'thres_dist': neuron_diameter[0],
            'thres_corr': thres_corr,
            'noise_freq': noise_freq
        }

        self.params_initA = {
            'thres_corr': thres_corr,
            'wnd': neuron_diameter[-1],
            'noise_freq': noise_freq
        }

        self.params_unit_merge = {
            'thres_corr': thres_corr
        }

        self.params_get_noise_fft = {
            'noise_range': (noise_freq, fr/2) # the range max has to be Nyquist frequency (sample rate / 2)
        }

        self.params_update_spatial = {
            'dl_wnd': neuron_diameter[-1],
            'sparse_penal': self.parameters[defs.Parameters.danse.SPATIAL_PENALTY],
            'size_thres': (np.ceil(0.9*(np.pi*neuron_diameter[0]/2)**2), np.ceil(1.1*(np.pi*neuron_diameter[-1]/2)**2))
        }

        self.params_update_temporal = {
            'noise_freq': noise_freq,
            'sparse_penal': self.parameters[defs.Parameters.danse.TEMPORAL_PENALTY],
            'p': 1,
            'add_lag': 20,
            'jac_thres': 0.2
        }

        # Remove advanced_sesttings function keys that are not in the minian functions list
        self.advanced_settings = self.parameters.get(defs.Parameters.danse.ADVANCED_SETTINGS, {})
        self.advanced_settings = {key: self.advanced_settings[key] for key in self.advanced_settings
                            if (hasattr(mnUtils, key) or hasattr(mnPreproc, key) or hasattr(mnInit, key)
                                or hasattr(mnCnmf, key) or hasattr(mnMotcorr, key) or key == "LocalCluster")}

        self.update_all_func_params()
        self.parameters[defs.Parameters.danse.ADVANCED_SETTINGS] = self.advanced_settings.copy()


    def update_all_func_params(self):
        for func_name in self.advanced_settings:
            self.update_func_advanced_params(func_name)


    def update_func_advanced_params(self,
        func_name
        ):

        params = self.advanced_settings[func_name]

        if not (hasattr(self, "params_"+func_name)):
            return

        if func_name == "LocalCluster":
            new_params = {key: params[key] for key in params if key in self.params_LocalCluster}
            self.params_LocalCluster.update(new_params)
            self.advanced_settings[func_name] = new_params
            return

        if func_name == "denoise":
            self.set_denoise_advanced_params()
            return

        if func_name == "estimate_motion":
            self.set_estimate_motion_advanced_params()
            return

        # Check if the package contains the function name
        if hasattr(mnUtils, func_name):  # hasattr(object, "name") give true if object.name is possible and false if not
            mn_package = mnUtils
        elif hasattr(mnPreproc, func_name):
            mn_package = mnPreproc
        elif hasattr(mnInit, func_name):
            mn_package = mnInit
        elif hasattr(mnCnmf, func_name):
            mn_package = mnCnmf
        elif hasattr(mnMotcorr, func_name):
            mn_package = mnMotcorr
        else:
            return

        func_arguments = inspect.getfullargspec(getattr(mn_package, func_name)).args # getattr(object, "name") is like doing object.name
        new_params = {key: params[key] for key in params if key in func_arguments}

        getattr(self, "params_" + func_name).update(new_params)
        self.advanced_settings[func_name] = new_params


    def set_denoise_advanced_params(self):
        
        """
        denoise() function has specific parameters that update_func_advanced_params() can not see
        Here, the parameters of the denoise() are manually checked and updated

        denoise() -- https://minian.readthedocs.io/en/stable/_modules/minian/preprocessing.html#denoise
        openCV -- https://docs.opencv.org/4.7.0/index.html
        anisotropic() from medpy - https://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html
        """

        if 'method' in self.advanced_settings["denoise"]:
            self.params_denoise['method'] = self.advanced_settings["denoise"]['method']

        if self.params_denoise['method'] != 'median':
            del self.params_denoise['ksize']

        method = self.params_denoise['method']
        method_keys = []

        if method == "gaussian":
            method_keys = ["ksize", "sigmaX", "dst", "sigmaY", "borderType"]
        elif method == "anisotropic":
            method_keys = ["niter", "kappa", "gamma", "voxelspacing", "option"]
        elif method == "median":
            method_keys = ["ksize", "dst"]
        elif method == "bilateral":
            method_keys = ["d", "sigmaColor", "sigmaSpace", "dst", "borderType"]

        denoise_method_parameters = {key: self.advanced_settings["denoise"][key] for key in method_keys if key in self.advanced_settings["denoise"]}

        self.params_denoise, self.advanced_settings["denoise"] = self.remove_unused_keys(self.params_denoise, self.advanced_settings["denoise"], mnPreproc.denoise)

        self.params_denoise.update(denoise_method_parameters)
        self.advanced_settings["denoise"].update(denoise_method_parameters)


    def set_estimate_motion_advanced_params(self):
        
        """
        estimate_motion() function has specific parameters that update_func_advanced_params() can not see
        Here, the parameters of the estimate_motion() are manually checked and updated

        https://minian.readthedocs.io/en/stable/_modules/minian/motion_correction.html#estimate_motion
        """

        keys = ["mesh_size"]
        # est_motion_part()
        keys += ["alt_error"]

        # For est_motion_chunk -> est_motion_part (dask delayed)
        keys += ["varr", "sh_org", "npart", "alt_error", "aggregation", "upsample", "max_sh", "circ_thres",
                "mesh_size", "niter", "bin_thres"]

        special_parameters = {}
        for key in keys:
            if key in self.advanced_settings["estimate_motion"]:
                special_parameters[key] = self.advanced_settings["estimate_motion"][key]

        self.params_estimate_motion, self.advanced_settings["estimate_motion"] = self.remove_unused_keys(self.params_estimate_motion, self.advanced_settings["estimate_motion"], mnMotcorr.estimate_motion)

        self.params_estimate_motion.update(special_parameters)
        self.advanced_settings["estimate_motion"].update(special_parameters)


    def remove_unused_keys(self,
        all_params: dict, #old_param
        advanced_params: dict, #new_params
        func
        ) -> [dict, dict]:

        """
        Remove the keys from advanced_params that don't exist in the func arguments.
        Update all_params with advanced_params, and return both
        """

        func_arguments = inspect.getfullargspec(func).args
        advanced_params = {key: advanced_params[key] for key in advanced_params if key in func_arguments}

        all_params.update(advanced_params)

        return [all_params, advanced_params]


    def clean_h5path(self):
        
        """
        Correct the path for hdf5 file
        """

        h5path = self.paths[defs.Parameters.Path.H5PATH]

        if h5path[0] == '/':
            h5path = h5path[1:]
        if h5path[-1] == '/':
            h5path = h5path[:-1]

        return h5path


    def get_h5path_names(self):
        
        """
        Split the path to dataset into relevant names
        """

        h5path_names = self.clean_h5path().split('/')

        data = h5path_names[0]
        driver = h5path_names[1]
        operation = h5path_names[2]
        series = h5path_names[-2]
        sensor = h5path_names[-1]

        return [data, driver, operation, series, sensor]


    def round_up_to_odd(self, x):
        
        """
        Round up to odd
        """

        x = int(np.ceil(x))
        return x + 1 if x % 2 == 0 else x

    def round_down_to_odd(self, x):
        
        """
        Round down to odd
        """

        x = int(np.ceil(x))
        return x - 1 if x % 2 == 0 else x