# Import system related libraries
import os
import sys

# Edit system variables and path
# /!\ The change of environment variable CAIMAN_DATA need to be done before all caiman related imports
os.environ["CAIMAN_DATA"] = os.path.dirname(os.path.abspath(__file__))+"\\caiman_data"

sys.path.append('..')
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
    caiman_params = cm_params.CaimanParametersParameters(danse_parameters)

    cm_main.main(caiman_params)
