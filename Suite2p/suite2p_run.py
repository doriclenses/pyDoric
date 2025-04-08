# Import miscellaneous and utilities librarys
import os
import sys

sys.path.append("..")
path_dev = os.path.abspath(f"{os.getcwd()}\\..\\..\\..\\DNAMainApp\\Libraries\\pythondlls")
path_re  = os.path.abspath(f"{os.getcwd()}\\libraries\\pythondlls")

if os.path.exists(path_dev):
    os.add_dll_directory(path_dev)
elif os.path.exists(path_re):
    os.add_dll_directory(path_re)

import tempfile
import numpy as np

# needed but not directly used
import h5py

import utilities as utils
import suite2p_main        as s2p_main
import suite2p_parameters  as s2p_params
import suite2p_definitions as s2p_defs

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

danse_params: dict = {}

try:
    for arg in sys.argv[1:]:
        danse_params = eval(arg)

except SyntaxError:
    utils.print_to_intercept(s2p_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, s2p_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    suite2p_params = s2p_params.Suite2pParameters(danse_params)

    if suite2p_params.preview_params:
        s2p_main.preview(suite2p_params)
    else:
        s2p_main.main(suite2p_params)

    print(s2p_defs.Messages.PROCESS_DONE, flush=True)