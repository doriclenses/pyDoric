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
import cv2
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

def main(poseEstimation_params: poseEst_params.PoseEstimationParameters):

    """
    DeepLabCut algorithm
    """
    #filePath: str = poseEstimation_params.paths[defs.Parameters.Path.FILEPATH]
    #doricFile = h5py.File(filePath, 'r')
    projectFolder   = poseEstimation_params.params[poseEst_defs.Parameters.danse.PROJECT_FOLDER]
    bodyPartNames   = poseEstimation_params.params[poseEst_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    extractedFrames = poseEstimation_params.params[poseEst_defs.Parameters.danse.EXTRACTED_FRAMES]
    
    # --------------- Create Project folder and config file ---------------
    task: str        = "DLC" 
    experimenter:str = "doric"
    video: str       = poseEstimation_params.paths[poseEst_defs.Parameters.danse.VIDEO_PATH]

    path_config_file: str = deeplabcut.create_new_project(task, experimenter, [video], projectFolder, copy_videos = False)
    updateConfigFile(path_config_file, bodyPartNames)

    # --------------- Create hdf file for labeled data ---------------
    videoName = os.path.splitext(video)[0]
    createlabeledDataHDF(path_config_file, extractedFrames, bodyPartNames, experimenter, poseEstimation_params, videoName.rsplit("/", 1)[1])

    # --------------- Create a training dataset ---------------
    deeplabcut.create_training_dataset(path_config_file)

    # --------------- Start training ---------------
    deeplabcut.train_network(path_config_file)

    # --------------- Start evaluating ---------------
    deeplabcut.evaluate_network(path_config_file, plotting=True)

    # --------------- Start Analyzing videos ---------------
    deeplabcut.analyze_videos(path_config_file, [video])

    # --------------- Create labeled video ---------------
    deeplabcut.create_labeled_video(path_config_file, [video])

    # --------------- Plot the trajectories of the analyzed videos ---------------
    deeplabcut.plot_trajectories(path_config_file, [video])

def preview(poseEstimation_params: poseEst_params.PoseEstimationParameters):
    print("hello preview")

def createlabeledDataHDF(path_config_file, extractedFrames, bodyPartNames, experimenter, poseEstimation_params, videoName):
    cols = []
    rows = len(extractedFrames)
    data:list = [[] for _ in range(rows)]

    for pose in bodyPartNames:
        name = pose + poseEst_defs.Parameters.danse.BODY_PART_COORDS
        cols.extend([(experimenter, pose, 'x'),(experimenter, pose, 'y')])
        for i in range(len(extractedFrames)):
            data[i] += poseEstimation_params.params[name][i]

    columns = pd.MultiIndex.from_tuples(cols
     , names = ['scorer','bodyparts', 'coords'])

    axisLeft = []
    for frameNum in [int(item) for item in extractedFrames]:
        axisLeft.extend([('labeled-data', videoName, f'img{frameNum}.png')])
    Axis1 = pd.MultiIndex.from_tuples(axisLeft
                                       ,)
    df = pd.DataFrame(data, columns=columns)
    df.index = Axis1

    pathParts = path_config_file.rsplit("\\", 1)
    labeledDataPath = os.path.join(pathParts[0], "labeled-data", videoName)
    if not os.path.exists(labeledDataPath):
        os.makedirs(labeledDataPath)

    path: str = poseEstimation_params.paths[poseEst_defs.Parameters.danse.VIDEO_PATH]
    cap = cv2.VideoCapture(path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        if frame_count in [int(item) for item in extractedFrames]:
            # Save the frame as a PNG file
            frame_filename = f'{labeledDataPath}/img{frame_count}.png'
            cv2.imwrite(frame_filename, frame)
        frame_count += 1 
    cap.release() 
    cv2.destroyAllWindows()

    file_path = labeledDataPath + "/CollectedData_"  + experimenter + ".h5"
    df.to_hdf(file_path, key='keypoints', mode='w')

def updateConfigFile(path_config_file, bodyPartNames):
    # Load the YAML file
    with open(path_config_file, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the specific text   
    data['bodyparts'] = bodyPartNames
    data['skeleton']  = [bodyPartNames]

    # Save the modified YAML back to the file
    with open(path_config_file, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

