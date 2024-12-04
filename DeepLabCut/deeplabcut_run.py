# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np

# needed but not directly used
import h5py

sys.path.append("..")
import utilities                  as utils
import DeepLabCut.deeplabcut_main        as dlc_main
import DeepLabCut.deeplabcut_parameters  as dlc_params
import DeepLabCut.deeplabcut_definitions as dlc_defs

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

danse_params: dict = {}

try:
    for arg in sys.argv[1:]:
        danse_params = eval(arg)

except SyntaxError:
    utils.print_to_intercept(dlc_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, dlc_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    deeplabcut_params = dlc_params.DeepLabCutParameters(danse_params)

    # if deeplabcut_params.preview_params:
    # dlc_main.preview(deeplabcut_params)
    # else:
    dlc_main.main(deeplabcut_params)

    print(dlc_defs.Messages.PROCESS_DONE, flush=True)