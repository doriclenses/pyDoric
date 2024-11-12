import os, requests
import sys
import numpy as np
import pandas as pd
import h5py
import typing
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imwrite, TiffWriter, TiffFile

sys.path.append("..")
import utilities as utils
import definitions as defs
import poseEstimation_parameters as poseEst_params
import poseEstimation_definitions as poseEst_defs

import deeplabcut
print("Imported DLC!", flush=True)

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

# Figure Style settings for notebook.
import matplotlib as mpl
mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')


def main(poseEstimation_params: poseEst_params.PoseEstimationParameters):

    """
    DeepLabCut algorithm
    """
    #filePath: str = poseEstimation_params.paths[defs.Parameters.Path.FILEPATH]
    #doricFile = h5py.File(filePath, 'r')
    tempDir: str    = poseEstimation_params.paths[defs.Parameters.Path.TMP_DIR]
    positions: dict = poseEstimation_params.params[poseEst_defs.Parameters.danse.POSITIONS]

    # --------------- Create hdf file for labeled data ---------------
    scorer = "Doric"
    cols = []
    rows = len(positions[list(positions.keys())[0]])
    data:list = [[] for _ in range(rows)]

    for pose in positions:
        cols.extend([(scorer, pose, 'x'),(scorer, pose, 'y')])
        for i in range(len(positions[pose])):
            data[i] += positions[pose][list(positions[pose].keys())[i]]

    columns = pd.MultiIndex.from_tuples(cols
    , names = ['scorer','bodypart', 'coords'])

    axisLeft = []
    for img in positions[next(iter(positions))].keys():
        axisLeft.extend([('labeled-data', 'LHA86_avoidance', img)])
    Axis1 = pd.MultiIndex.from_tuples(axisLeft
                                      ,)
    df = pd.DataFrame(data, columns=columns)
    df.index = Axis1

    file_path = tempDir + "/CollectedData.h5"
    df.to_hdf(file_path, key='df', mode='w')

    # --------------- Read the config file ---------------
    path_config_file: str = poseEstimation_params.paths[defs.Parameters.Path.TMP_DIR] + "/config.yaml"
    utils.print_to_intercept(path_config_file)

    # --------------- Create a training dataset ---------------
    deeplabcut.create_training_dataset(path_config_file)

    # --------------- Start training ---------------
    deeplabcut.train_network(path_config_file)

    # --------------- Start evaluating ---------------
    deeplabcut.evaluate_network(path_config_file, plotting=True)

    # --------------- Start Analyzing videos ---------------
    videofile_path = 'C:/Users/MARK05/Desktop/DLC/test'
    deeplabcut.analyze_videos(path_config_file, videofile_path, videotype='.avi')

    # --------------- Create labeled video ---------------
    deeplabcut.create_labeled_video(path_config_file,videofile_path)

    # --------------- Plot the trajectories of the analyzed videos ---------------
    deeplabcut.plot_trajectories(path_config_file,videofile_path)

def preview(poseEstimation_params: poseEst_params.PoseEstimationParameters):
    print("hello preview")