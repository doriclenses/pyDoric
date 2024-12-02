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
import copy
from datetime import datetime
import glob

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
    # --------------- Read danse parameters ---------------
    Operations: str    = poseEstimation_params.params[defs.DoricFile.Attribute.Group.OPERATIONS]
    filePath: str      = poseEstimation_params.paths[defs.Parameters.Path.FILEPATH]
    datapath: str      = poseEstimation_params.paths[defs.Parameters.Path.H5PATH]
    projectFolder: str = poseEstimation_params.params[poseEst_defs.Parameters.danse.PROJECT_FOLDER]
    bodyPartNames      = poseEstimation_params.params[poseEst_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodyPartColors     = poseEstimation_params.params[poseEst_defs.Parameters.danse.BODY_PART_COLORS].split(', ')
    extractedFrames    = poseEstimation_params.params[poseEst_defs.Parameters.danse.EXTRACTED_FRAMES]
    trainingCoordinates = {}
    for bodyPart in bodyPartNames:
        label = bodyPart + poseEst_defs.Parameters.danse.COORDINATES
        trainingCoordinates[label] = poseEstimation_params.params[label]

    # --------------- Create Project folder and config file ---------------
    task: str        = "DLC" 
    experimenter:str = "doric"

    file_ = h5py.File(filePath, 'a')
    attributes = utils.load_attributes(file_, datapath)
    key = [key for key in attributes if poseEst_defs.Parameters.danse.VIDEO_FILEPATH in key]   
    videoPath: str = attributes[key[0]]
    # file_.close

    path_config_file: str = deeplabcut.create_new_project(task, experimenter, [videoPath], projectFolder, copy_videos = False)
    updateConfigFile(path_config_file, bodyPartNames)

    # --------------- Create hdf file for labeled data ---------------
    createlabeledDataHDF(path_config_file, extractedFrames, bodyPartNames, experimenter, poseEstimation_params, videoPath)

    # --------------- Create a training dataset ---------------
    deeplabcut.create_training_dataset(path_config_file)

    # --------------- Start training ---------------
    deeplabcut.train_network(path_config_file)

    # --------------- Start evaluating ---------------
    deeplabcut.evaluate_network(path_config_file, plotting=True)

    # --------------- Start Analyzing videos ---------------
    path_output = path_config_file.rsplit("\\", 1)[0]
    deeplabcut.analyze_videos(path_config_file, [videoPath], destfolder = path_output)

    # --------------- Saving data ---------------
    saveCoordsToDoric(file_, datapath, path_output, extractedFrames, bodyPartNames, projectFolder, bodyPartColors, Operations, 
                      trainingCoordinates, groupNames = poseEstimation_params.get_h5path_names())
    
    msg = ("************Process Completed*************************")
    utils.print_to_intercept(msg)

def preview(poseEstimation_params: poseEst_params.PoseEstimationParameters):
    print("hello preview")

def createlabeledDataHDF(path_config_file, extractedFrames, bodyPartNames, experimenter, poseEstimation_params, videoPath):
    cols = []
    rows = len(extractedFrames)
    data:list = [[] for _ in range(rows)]
    path_withoutExt = os.path.splitext(videoPath)[0]
    videoName = path_withoutExt.rsplit("/", 1)[1]

    for pose in bodyPartNames:
        name = pose + poseEst_defs.Parameters.danse.COORDINATES
        cols.extend([(experimenter, pose, 'x'),(experimenter, pose, 'y')])
        for i in range(len(extractedFrames)):
            data[i] += poseEstimation_params.params[name][i]

    columns  = pd.MultiIndex.from_tuples(cols, names = ['scorer','bodyparts', 'coords'])
    axisLeft = []
    for frameNum in [int(item) for item in extractedFrames]:
        axisLeft.extend([('labeled-data', videoName, f'img{frameNum}.png')])
    Axis1 = pd.MultiIndex.from_tuples(axisLeft)
    df = pd.DataFrame(data, columns=columns)
    df.index = Axis1

    pathParts = path_config_file.rsplit("\\", 1)
    labeledDataPath = os.path.join(pathParts[0], "labeled-data", videoName)
    if not os.path.exists(labeledDataPath):
        os.makedirs(labeledDataPath)

    cap = cv2.VideoCapture(videoPath)
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

    file_path = f'{labeledDataPath}/CollectedData_{experimenter}.h5'
    df.to_hdf(file_path, key='keypoints', mode='w')

def updateConfigFile(path_config_file, bodyPartNames):
    # Load the YAML file
    with open(path_config_file, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the specific text   
    data['bodyparts'] = copy.deepcopy(bodyPartNames)
    data['skeleton']  = copy.deepcopy([bodyPartNames])

    # Save the modified YAML back to the file
    with open(path_config_file, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def saveCoordsToDoric(file_, datapath, path_output, extractedFrames, bodyPartNames, projectFolder, bodyPartColors, Operations, trainingCoordinates, groupNames):
    data, driver, operation, series, sensor = groupNames
    groupAttrs: dict = {}
    groupAttrs[defs.DoricFile.Attribute.Group.OPERATIONS]      = Operations
    groupAttrs['VideoDatapath']                                = datapath
    groupAttrs[poseEst_defs.Parameters.danse.EXTRACTED_FRAMES] = extractedFrames
    groupAttrs[poseEst_defs.Parameters.danse.PROJECT_FOLDER]   = projectFolder
    for key in trainingCoordinates:
        groupAttrs[key] = trainingCoordinates[key]

    time_ = np.array(file_[f"{datapath}/{defs.DoricFile.Dataset.TIME}"])

    h5_files = glob.glob(os.path.join(path_output, '*.h5'))
    file_path = h5_files[0]
    data_coords = pd.read_hdf(file_path)
    operation_path = f'{driver}/Coordinates/{series}/{sensor}PoseEstimation'

    for i in range(len(bodyPartNames)):
        coords = data_coords.loc[:, pd.IndexSlice[:, bodyPartNames[i],['x','y']]]
        coords_df = pd.DataFrame(coords.values)
        dataset_path = f'{operation_path}/{bodyPartNames[i]}'

        if dataset_path in file_: 
            del file_[dataset_path] # Remove existing dataset if it exists 
        file_.create_dataset(dataset_path, data=coords_df, dtype = 'int32', chunks=utils.def_chunk_size(coords_df.shape), maxshape=(h5py.UNLIMITED, 2))
        attrs = {
            "Username": bodyPartNames[i],
            "Color":    bodyPartColors[i]
        }

        timePath = f'{driver}/Coordinates/{series}/{sensor}PoseEstimation/Time'
        if timePath not in file_:
            file_.create_dataset(timePath, data=time_)

        utils.save_attributes(attrs, file_, dataset_path)

    utils.save_attributes(utils.merge_params(params_current = groupAttrs), file_, operation_path)
    file_.close()