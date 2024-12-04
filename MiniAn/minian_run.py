# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np
from dask.distributed import Client, LocalCluster

# Make unneeded package a dummy package so they can be removing during pyinstaller
from unittest.mock import Mock
sys.modules['panel']      = Mock()
sys.modules['matplotlib'] = Mock()

# needed but not directly used
import h5py
import xarray as xr

sys.path.append("..")
import utilities as utils
import minian_main        as mn_main
import minian_parameters  as mn_params
import minian_definitions as mn_defs

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

danse_params = {}

try:
    for arg in sys.argv[1:]:
        danse_params = eval(arg)

except SyntaxError:
    utils.print_to_intercept(mn_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, mn_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    minian_params = mn_params.MinianParameters(danse_params)

    if minian_params.preview_params:
        mn_main.preview(minian_params)
    else:
        mn_main.main(minian_params)

    print(mn_defs.Messages.PROCESS_DONE, flush=True)
