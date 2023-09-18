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
import minian_utilities as mn_utils
import minian_definitions as mn_defs

class MinianParameters:
    '''
    MinianParameters
    '''

    def __init__(self, danse_parameters):
        self.parameters  = danse_parameters.get("parameters", {})
        self.paths   = danse_parameters.get(mn_defs.ParametersKeys.PATHS, {})
        self.preview = False
        if mn_defs.ParametersKeys.PREVIEW in danse_parameters:
            self.preview = True
            self.preview_parameters = danse_parameters[mn_defs.ParametersKeys.PREVIEW]

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MINIAN_INTERMEDIATE"] = os.path.join(self.paths["tmpDir"], "intermediate")

        parameters = self.parameters

        self.fr = utils.get_frequency(self.paths["fname"], self.paths['h5path']+'Time')

        neuron_diameter     = tuple((np.array([self.parameters["NeuronDiameterMin"], self.parameters["NeuronDiameterMax"]])/self.parameters["SpatialDownsample"]).round().astype('int'))
        noise_freq: float   = parameters["NoiseFreq"]
        thres_corr: float   = parameters["ThresCorr"]

        advanced_settings = parameters.get("AdvancedSettings", {})

        # removing advanced_sesttings function keys that are not in the minian functions list
        minian_functions_list = ["TaskAnnotation", "get_optimal_chk", "custom_arr_optimize",
                                "save_minian", "open_minian", "denoise", "remove_background",
                                "seeds_init", "pnr_refine", "ks_refine", "seeds_merge", "initA", "initC",
                                "compute_trace", "get_noise_fft", "update_spatial", "update_temporal",
                                "unit_merge", "update_background", "compute_AtC", "apply_transform",
                                "estimate_motion"] + ["LocalCluster"]

        advanced_settings = {key: advanced_settings[key] for key in advanced_settings if key in minian_functions_list}

        self.params_LocalCluster = {
            "n_workers": 4,
            "memory_limit": "auto",
            "resources": {"MEM": 1}, # constrain the number of tasks that can be concurrently in memory for each worker
            "threads_per_worker": 2,
            "dashboard_address": ":8787",
            "local_directory": self.paths["tmpDir"]
        }
        if "LocalCluster" in advanced_settings:
            advanced_settings["LocalCluster"] = {key: advanced_settings["LocalCluster"][key] for key in advanced_settings["LocalCluster"] if key in self.params_LocalCluster}
            self.params_LocalCluster.update(advanced_settings["LocalCluster"])

        self.params_load_doric = {
            "fname": self.paths["fname"],
            "h5path": self.paths['h5path'],
            "dtype": np.uint8,
            "downsample": {"frame": parameters["TemporalDownsample"],
                            "height": parameters["SpatialDownsample"],
                            "width": parameters["SpatialDownsample"]},
            "downsample_strategy": "subset",
        }

        self.params_save_minian = {
            "dpath": os.path.join(self.paths["tmpDir"], "final"),
            "meta_dict": {"session": -1, "animal": -2},
            "overwrite": True,
        }

        self.params_get_optimal_chk = {
            "dtype": float
        }
        if "get_optimal_chk" in advanced_settings:
            self.params_get_optimal_chk, advanced_settings["get_optimal_chk"] = self.update_func_advanced_param(self.params_get_optimal_chk, advanced_settings["get_optimal_chk"], mnUtils.get_optimal_chk)

        self.params_denoise = {
            'method': 'median',
            'ksize': mn_utils.round_down_to_odd(neuron_diameter[-1]/2.0) # half of the maximum diameter
        }
        if "denoise" in advanced_settings:
            self.params_denoise, advanced_settings["denoise"] = self.set_denoise_advanced_params(self.params_denoise, advanced_settings["denoise"], mnPreproc.denoise)

        self.params_remove_background = {
            'method': 'tophat',
            'wnd': np.ceil(neuron_diameter[-1]) # largest neuron diameter
        }
        if "remove_background" in advanced_settings:
            self.params_remove_background, advanced_settings["remove_background"] = self.update_func_advanced_param(self.params_remove_background, advanced_settings["remove_background"], mnPreproc.remove_background)

        self.params_estimate_motion = {
            'dim': 'frame'
        }
        if "estimate_motion" in advanced_settings:
            self.params_estimate_motion, advanced_settings["estimate_motion"] = self.set_estimate_motion_advanced_params(self.params_estimate_motion, advanced_settings["estimate_motion"], mnMotcorr.estimate_motion)

        self.params_apply_transform = {
            'fill': 0
        }
        if "apply_transform" in advanced_settings:
            self.params_apply_transform, advanced_settings["apply_transform"] = self.update_func_advanced_param(self.params_apply_transform, advanced_settings["apply_transform"], mnMotcorr.apply_transform)

        wnd = 60 # time window of 60 seconds
        self.params_seeds_init = {
            'wnd_size': self.fr*wnd,
            'method': 'rolling',
            'stp_size': self.fr*wnd / 2,
            'max_wnd': neuron_diameter[-1],
            'diff_thres': 3
        }
        if "seeds_init" in advanced_settings:
            self.params_seeds_init, advanced_settings["seeds_init"] = self.update_func_advanced_param(self.params_seeds_init, advanced_settings["seeds_init"], mnInit.seeds_init)

        self.params_pnr_refine = {
            "noise_freq": noise_freq,
            "thres": 1
        }
        if "pnr_refine" in advanced_settings:
            self.params_pnr_refine, advanced_settings["pnr_refine"] = self.update_func_advanced_param(self.params_pnr_refine, advanced_settings["pnr_refine"], mnInit.pnr_refine)

        self.params_ks_refine = {
            "sig": 0.05
        }
        if "ks_refine" in advanced_settings:
            self.params_ks_refine, advanced_settings["ks_refine"] = self.update_func_advanced_param(self.params_ks_refine, advanced_settings["ks_refine"], mnInit.ks_refine)

        self.params_seeds_merge = {
            'thres_dist': neuron_diameter[0],
            'thres_corr': thres_corr,
            'noise_freq': noise_freq
        }
        if "seeds_merge" in advanced_settings:
            self.params_seeds_merge, advanced_settings["seeds_merge"] = self.update_func_advanced_param(self.params_seeds_merge, advanced_settings["seeds_merge"], mnInit.seeds_merge)

        self.params_initA = {
            'thres_corr': thres_corr,
            'wnd': neuron_diameter[-1],
            'noise_freq': noise_freq
        }
        if "initA" in advanced_settings:
            self.params_initA, advanced_settings["initA"] = self.update_func_advanced_param(self.params_initA, advanced_settings["initA"], mnInit.initA)

        self.params_unit_merge = {
            'thres_corr': thres_corr
        }
        if "unit_merge" in advanced_settings:
            self.params_unit_merge, advanced_settings["unit_merge"] = self.update_func_advanced_param(self.params_unit_merge, advanced_settings["unit_merge"], mnCnmf.unit_merge)

        self.params_get_noise_fft = {
            'noise_range': (noise_freq, 0.5)
        }
        if "get_noise_fft" in advanced_settings:
            self.params_get_noise_fft, advanced_settings["get_noise_fft"] = self.update_func_advanced_param(self.params_get_noise_fft, advanced_settings["get_noise_fft"], mnCnmf.get_noise_fft)

        self.params_update_spatial = {
            'dl_wnd': neuron_diameter[-1],
            'sparse_penal': parameters["SpatialPenalty"],
            'size_thres': (np.ceil(0.9*(np.pi*neuron_diameter[0]/2)**2), np.ceil(1.1*(np.pi*neuron_diameter[-1]/2)**2))
        }
        if "update_spatial" in advanced_settings:
            self.params_update_spatial, advanced_settings["update_spatial"] = self.update_func_advanced_param(self.params_update_spatial, advanced_settings["update_spatial"], mnCnmf.update_spatial)

        self.params_update_temporal = {
            'noise_freq': noise_freq,
            'sparse_penal': parameters["TemporalPenalty"],
            'p': 1,
            'add_lag': 20,
            'jac_thres': 0.2
        }
        if "update_temporal" in advanced_settings:
            self.params_update_temporal, advanced_settings["update_temporal"] = self.update_func_advanced_param(self.params_update_temporal, advanced_settings["update_temporal"], mnCnmf.update_temporal)

        # Update AdvancedSettings for self.parameters
        self.parameters["AdvancedSettings"] = advanced_settings.copy()


    #--------------------------------------------- functions for advanced parameters -------------------------------------------------------------------------
    def remove_keys_not_in_fun_arguments(self,
        input_dic:dict, func
    ) -> dict:
        '''
        This function while keep the keys in input_dic that are not use in the function func

        Returns
        -------
        The new dictionary new_dictionary
        '''
        func_arguments = inspect.getfullargspec(func).args
        new_dictionary = {key: input_dic[key] for key in input_dic if key in func_arguments}
        return new_dictionary

    def update_func_advanced_param(self,
        param_func,
        advanced_parameters,
        func
        ):
        '''
        This function while change the value of the key from the dictionary param with the value of the key from the dictionary advanced_parameters.
        It while also remove the keys from the dictionary advanced_parameters that are not used in the function func

        Returns
        -------
        it while return the dictionary param_func with the new values and also the new advanced_parameters with only the used keys

        '''
        advanced_parameters = self.remove_keys_not_in_fun_arguments(advanced_parameters, func)
        for key, value in advanced_parameters.items():
            param_func[key] = value

        return [param_func, advanced_parameters]

    def set_denoise_advanced_params(self,
        param_func,
        advanced_parameters,
        func):

        '''
        Denoise function have some specific parameters that update_func_advanced_param() can not see therefor
        the function is to update the value of the advanced parameter for the denoise function.

        '''

        if 'method' in advanced_parameters:
            param_func['method'] = advanced_parameters['method']

        if param_func['method'] != 'median':
            del param_func['ksize']


        # Doc for denoise function
        # https://minian.readthedocs.io/en/stable/_modules/minian/preprocessing.html#denoise
        # opencv functions
        # https://docs.opencv.org/4.7.0/index.html
        # anisotropic is function from medpy
        # https://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html

        method = param_func['method']

        if method == "gaussian":
            keys = ["ksize", "sigmaX", "dst", "sigmaY", "borderType"]
        elif method == "anisotropic":
            keys = ["niter", "kappa", "gamma", "voxelspacing", "option"]
        elif method == "median":
            keys = ["ksize", "dst"]
        elif method == "bilateral":
            keys = ["d", "sigmaColor", "sigmaSpace", "dst", "borderType"]

        denoise_method_parameters = {}
        for key in keys:
            if key in advanced_parameters:
                denoise_method_parameters[key] = advanced_parameters[key]

        param_func, advanced_parameters = self.update_func_advanced_param(param_func, advanced_parameters, func)

        for key, value in denoise_method_parameters.items():
            param_func[key] = value
            advanced_parameters[key] = value

        return [param_func, advanced_parameters]

    def set_estimate_motion_advanced_params(self,
        param_func,
        advanced_parameters,
        func):

        '''
        Denoise function have some specific parameters that update_func_advanced_param() can not see therefor
        the function is to update the value of the advanced parameter for the estimate_motion function

        '''

        # Doc for estimate motion function
        # https://minian.readthedocs.io/en/stable/_modules/minian/motion_correction.html#estimate_motion

        keys = ["mesh_size"]
        # est_motion_part()
        keys += ["alt_error"]

        # For est_motion_chunk -> est_motion_part (dask delayed)
        keys += ["varr", "sh_org", "npart", "alt_error", "aggregation", "upsample", "max_sh", "circ_thres",
                "mesh_size", "niter", "bin_thres"]

        special_parameters = {}
        for key in keys:
            if key in advanced_parameters:
                special_parameters[key] = advanced_parameters[key]

        param_func, advanced_parameters = self.update_func_advanced_param(param_func, advanced_parameters, func)

        for key, value in special_parameters.items():
            param_func[key] = value
            advanced_parameters[key] = value

        return [param_func, advanced_parameters]
