# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np
from dask.distributed import Client, LocalCluster

# needed but not directly used
import h5py
import xarray as xr

sys.path.append('..')
import utilities as utils
import minian_utilities as mn_utils
import minian_parameter as mn_param
import minian_main      as mn_main
import minian_preview   as mn_preview

# Import for MiniAn lib
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian.motion_correction import apply_transform, estimate_motion

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

ADVANCED_BAD_TYPE   = "One of the advanced settings is not of a python type"

kwargs = {}
params_doric = {}
danse_parameters = {}

try:
    for arg in sys.argv[1:]:
        exec(arg)
except SyntaxError:
    utils.print_to_intercept(ADVANCED_BAD_TYPE)
    sys.exit()
except Exception as error:
    mn_utils.print_error(error)
    sys.exit()

if not danse_parameters: # for backwards compatibility
    danse_parameters = {"paths": kwargs , "parameters": params_doric}

if __name__ == "__main__":
    minian_params = mn_param.MinianParameters(danse_parameters)
    if minian_params.preview:
        mn_preview.minian_preview(minian_params)
    else:
        mn_main.minian_main(minian_params)
