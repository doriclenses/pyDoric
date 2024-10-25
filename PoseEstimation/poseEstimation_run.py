# Import miscellaneous and utilities librarys
import os
import sys
import tempfile
import numpy as np

# needed but not directly used
import h5py

sys.path.append("..")
import utilities                  as utils
import poseEstimation_main        as poseEst_main
import poseEstimation_parameters  as poseEst_params
import poseEstimation_definitions as poseEst_defs

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

danse_params: dict = {}

try:
    for arg in sys.argv[1:]:
        danse_params = eval(arg)

except SyntaxError:
    utils.print_to_intercept(poseEst_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, poseEst_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    poseEstimation_params = poseEst_params.PoseEstimation(danse_params)

    if poseEstimation_params.preview_params:
        poseEst_main.preview(poseEstimation_params)
    else:
        poseEst_main.main(poseEstimation_params)

    print(poseEst_defs.Messages.PROCESS_DONE, flush=True)