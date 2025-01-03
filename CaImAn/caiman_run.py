# Import system related libraries
import os
import sys

# Make unneeded package a dummy package so they can be removing during pyinstaller
from unittest.mock import Mock
packages_name  = ["panel", "bokeh", "ipywidgets", "ipyparallel"]
packages_name += ["matplotlib", "matplotlib.pyplot", "matplotlib.animation", "matplotlib.patches",  "matplotlib.widgets"]
packages_name += ["IPython", "IPython.display"]
packages_name += ["tensorflow", "tensorflow.keras", "tensorflow.keras.layers", "tensorflow.keras.models", "tensorflow.keras.optimizers",
                  "tensorflow.keras.backend", "tensorflow.keras.callbacks", "tensorflow.keras.initializers", "tensorflow.keras.utils"]

for pack_name in packages_name:
    sys.modules[pack_name] = Mock()

# Edit system variables and path
# /!\ The change of environment variable CAIMAN_DATA need to be done before all caiman related imports
os.environ["CAIMAN_DATA"] = f"{os.path.dirname(os.path.abspath(__file__))}\\caiman_data"

sys.path.append("..")
# Import CaimAn related utilities libraries
import utilities as utils
import caiman_definitions as cm_defs
import caiman_parameters  as cm_params
import caiman_main        as cm_main

# Import for PyInstaller
from multiprocessing import freeze_support

# Miscellaneous configuration for CaImAn
import logging
logging.basicConfig(level=logging.DEBUG)
freeze_support()

danse_parameters = {}

try:
    for arg in sys.argv[1:]:
        danse_parameters = eval(arg)

except SyntaxError:
    utils.print_to_intercept(cm_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, cm_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()


if __name__ == "__main__":
    caiman_params = cm_params.CaimanParameters(danse_parameters)

    if caiman_params.preview_params:
        cm_main.preview(caiman_params)
    else:
        cm_main.main(caiman_params)

    print(cm_defs.Messages.PROCESS_DONE, flush=True)
