import sys
sys.path.append("..")
import utilities as utils
import definitions as defs
import DeepLabCut.deeplabcut_definitions as poseEst_defs

class DeepLabCutParameters:

    """
    Parameters used in deepLabCut library
    """

    def __init__(self, danse_params: dict):
        
        self.paths: dict  = danse_params.get(defs.Parameters.Main.PATHS, {})
        self.params: dict = danse_params.get(defs.Parameters.Main.PARAMETERS, {})
        
    def get_h5path_names(self):

        """
        Split the path to dataset into relevant names
        """
        h5path_names = self.paths[defs.Parameters.Path.H5PATH].split('/')

        data = h5path_names[0]
        driver = h5path_names[1]
        operation = h5path_names[2]
        series = h5path_names[-2]
        sensor = h5path_names[-1]

        return [data, driver, operation, series, sensor]