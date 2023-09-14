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
import minian_parameters as mn_params
import minian_main      as mn_main
import minian_definitions  as mn_defs

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

danse_parameters = {}

try:
    for arg in sys.argv[1:]:
        exec(arg)
except SyntaxError:
    utils.print_to_intercept(mn_defs.ADVANCED_BAD_TYPE)
    sys.exit()
except Exception as error:
    utils.print_error(error, mn_defs.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    minian_params = mn_params.MinianParameters(danse_parameters)

    if minian_params.preview:
        mn_main.minian_preview(minian_params)
    else:
        mn_main.minian_main(minian_params)
