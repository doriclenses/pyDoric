import os, requests
import sys
import numpy as np
import pandas as pd
import h5py
import typing
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imwrite, TiffWriter, TiffFile
import yaml
from datetime import datetime

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
    tempDir: str       = poseEstimation_params.paths[defs.Parameters.Path.TMP_DIR]
    positions: dict    = poseEstimation_params.params[poseEst_defs.Parameters.danse.POSITIONS]
    ProjectPath        = poseEstimation_params.params[poseEst_defs.Parameters.danse.PROJECT_PATH]
    Task               = poseEstimation_params.params[poseEst_defs.Parameters.danse.PROJECT_NAME]
    scorer             = poseEstimation_params.params[poseEst_defs.Parameters.danse.SCORER]
    Project_folderName = Task + '-' + scorer + '-' + datetime.now().strftime("%Y-%m-%d")
    Project_fullPath   = os.path.join(ProjectPath, Project_folderName)

    os.makedirs(Project_fullPath)

    # --------------- Read the config file ---------------
    path_config_file: str = createConfigFile(scorer, Task, positions, Project_fullPath)
    utils.print_to_intercept(path_config_file)

    # --------------- Create hdf file for labeled data ---------------
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
    
    labeledDataPath  = os.path.join(Project_fullPath, "labeled-data")
    if not os.path.exists(labeledDataPath):
        os.makedirs(labeledDataPath)

    file_path = labeledDataPath + "/CollectedData.h5"
    df.to_hdf(file_path, key='df', mode='w')

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

def createConfigFile(Scorer, Task, positions, project_fullPath):
    filepath        = project_fullPath + "/config.yaml"

    yaml_content = {
        # Project definitions (do not edit)
        'Task'  : Task,
        'Scorer': Scorer,
        'Date'  : datetime.now().strftime("%b%d"),
        'multianimalproject': 'false',
        'identity': '',

        # Project path (change when moving around)
        'project_path': project_fullPath,

        # Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
        'engine': 'pytorch',

        # Annotation data set configuration (and individual video cropping parameters)
        'video_sets': {
            'C:/Users/MARK05/Desktop/DLC/Testingkapil-KS-2024-10-17/videos/LHA86_avoidance.mp4': {'crop': '0, 640, 0, 480'},
            'C:/Users/MARK05/Desktop/DLC/Testingkapil-KS-2024-10-17/videos/m3v1mp4.mp4': {'crop': '0, 640, 0, 480'}
        },
        'bodyparts': list(positions.keys()),
        # Fraction of video to start/stop when extracting frames for labeling/refinement
        'start': 0,
        'stop': 1,
        'numframes2pick': 20,

        # Plotting configuration
        'skeleton': list(positions.keys()),

        'skeleton_color': 'black',
        'pcutoff': 0.6,
        'dotsize': 12,
        'alphavalue': 0.7,
        'colormap': 'rainbow',

        # Training,Evaluation and Analysis configuration
        'TrainingFraction':
        - 0.95,
        'iteration': 0,
        'default_net_type': 'resnet_50',
        'default_augmenter': 'default',
        'snapshotindex': -1,
        'detector_snapshotindex': -1,
        'batch_size': 8,
        'detector_batch_size': 1,

        # Cropping Parameters (for analysis and outlier frame detection)
        'cropping': False,
        #f cropping is true for analysis, then set the values here:
        'x1': 0,
        'x2': 640,
        'y1': 277,
        'y2': 624,

        # Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
        'corner2move2': [50,50],
        'move2corner': True,

        # Conversion tables to fine-tune SuperAnimal weights
        'SuperAnimalConversionTables': ''
    }

    # Create and write to the .yaml file 
    with open(filepath, 'w') as file: 
        yaml.dump(yaml_content, file, default_flow_style=False)
        print("YAML file created successfully at", filepath)

    return filepath