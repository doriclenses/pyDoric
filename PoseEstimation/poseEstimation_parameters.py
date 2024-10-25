import os
import sys
import inspect
import numpy as np

sys.path.append("..")
import utilities as utils
import definitions as defs
import poseEstimation_definitions as poseEst_defs

import deeplabcut
print("Imported DLC!")

class PoseEstimationParameters:

    """
    Parameters used in deepLabCut library
    """

    def __init__(self, danse_params: dict):
        
        self.paths: dict          = danse_params.get(defs.Parameters.Main.PATHS, {})
        self.params: dict         = danse_params.get(defs.Parameters.Main.PARAMETERS, {})
        self.preview_params: dict = danse_params.get(defs.Parameters.Main.PREVIEW, {})
