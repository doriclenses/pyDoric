"""
Quick driver to run MiniAn on the provided msCam_raw.doric dataset using the
pipeline defined in minian_main.py.
"""

import os
import sys

sys.path.append("..")
import definitions as defs
import minian_main as mn_main
import minian_parameters as mn_params


# Source data
DATA_FILE = r"C:\Users\ING55\dataFiles\Clients\MiniAnData\msCam_raw.doric"
H5_DATASET = "DataAcquisition/FMD/Images/Series0001/Sensor1/ImageStack"

# Temporary folder for MiniAn intermediate files
TMP_DIR = os.path.join(os.path.dirname(DATA_FILE), "minian_tmp")

# Parameters passed to MinianParameters / minian_main.main
danse_params = {
    defs.Parameters.Main.PARAMETERS: {
        defs.Parameters.danse.NEURO_DIAM_MIN: 5,
        defs.Parameters.danse.NEURO_DIAM_MAX: 15,
        defs.Parameters.danse.TEMPORAL_DOWNSAMPLE: 1,
        defs.Parameters.danse.SPATIAL_DOWNSAMPLE: 1,
        defs.Parameters.danse.NOISE_FREQ: 0.06,
        defs.Parameters.danse.THRES_CORR: 0.8,
        defs.Parameters.danse.SPATIAL_PENALTY: 0.01,
        defs.Parameters.danse.TEMPORAL_PENALTY: 1.0,
        defs.Parameters.danse.CORRECT_MOTION: True,
        defs.Parameters.danse.ADVANCED_SETTINGS: {},
        defs.DoricFile.Attribute.Group.OPERATIONS: "MiniAn CNMF",
    },
    defs.Parameters.Main.PATHS: {
        defs.Parameters.Path.FILEPATH: DATA_FILE,
        defs.Parameters.Path.H5PATH: [H5_DATASET],
        defs.Parameters.Path.TMP_DIR: TMP_DIR,
    },
    defs.Parameters.Main.PREVIEW: {},
}


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    minian_params = mn_params.MinianParameters(danse_params)
    mn_main.main(minian_params)
