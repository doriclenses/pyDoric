"""Parameters used in DeepLabCut library"""

import sys

sys.path.append("..")
import definitions as defs

class DeepLabCutParameters:

    """
    Parameters used in deepLabCut library
    """

    def __init__(self, danse_params: dict):

        self.paths: dict  = danse_params.get(defs.Parameters.Main.PATHS, {})
        self.params: dict = danse_params.get(defs.Parameters.Main.PARAMETERS, {})
        self.stage: str   = danse_params.get(defs.Parameters.Main.STAGE)

    def get_h5path_names(self, datapath: str) -> tuple[str]:

        """
        Split the path to dataset into relevant names
        """
        h5path_names = datapath.split('/')

        data      = h5path_names[0]
        driver    = h5path_names[1]
        data_type = h5path_names[-4]
        series    = h5path_names[-3]
        group     = h5path_names[-2]

        return data, driver, data_type, series, group
    