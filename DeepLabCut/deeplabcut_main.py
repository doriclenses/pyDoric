import os
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import yaml
import cv2
import copy
import glob

sys.path.append("..")
import utilities as utils
import definitions as defs
import DeepLabCut.deeplabcut_parameters as dlc_params
import DeepLabCut.deeplabcut_definitions as dlc_defs

import deeplabcut

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

def main(deeplabcut_params: dlc_params.DeepLabCutParameters):

    """
    DeepLabCut algorithm
    """
    # --------------- Read danse parameters ---------------
    operations: str      = deeplabcut_params.params[defs.DoricFile.Attribute.Group.OPERATIONS]
    filepath: str        = deeplabcut_params.paths[defs.Parameters.Path.FILEPATH]
    datapath: str        = deeplabcut_params.paths[defs.Parameters.Path.H5PATH]
    project_folder: str  = deeplabcut_params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    bodypart_names       = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodypart_colors      = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_COLORS].split(', ')
    extracted_frames     = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES]

    file_, video_path, path_config_file = create_project(filepath, datapath, project_folder, bodypart_names, extracted_frames, deeplabcut_params)
    deeplabcut.create_training_dataset(path_config_file)
    deeplabcut.train_network(path_config_file)
    deeplabcut.evaluate_network(path_config_file)
 
    path_output = path_config_file.rsplit("\\", 1)[0]
    deeplabcut.analyze_videos(path_config_file, [video_path], destfolder = path_output)

    save_coords_to_doric(file_, datapath, path_output, deeplabcut_params,
                         group_names = deeplabcut_params.get_h5path_names()) 

def preview(deeplabcut_params: dlc_params.DeepLabCutParameters):
    print("hello preview")

def create_project(filepath, datapath, project_folder, bodypart_names, extracted_frames, deeplabcut_params):
    task          = os.path.splitext(os.path.basename(filepath))[0] 
    experimenter  = "danse"
    file_         = h5py.File(filepath, 'a')
    attributes    = utils.load_attributes(file_, datapath)
    relative_path = attributes[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]
    dir           = os.path.dirname(filepath)
    video_path    = os.path.join(dir, relative_path)
    video_path    = video_path.replace("\\", "/")

    path_config_file: str = deeplabcut.create_new_project(task, experimenter, [video_path], project_folder, copy_videos = False)
    update_config_file(path_config_file, bodypart_names)

    create_labeled_data(path_config_file, extracted_frames, bodypart_names, experimenter, deeplabcut_params, video_path)

    return file_, video_path, path_config_file

def create_labeled_data(path_config_file, extracted_frames, bodypart_names, experimenter, deeplabcut_params, video_path):
    #--------------- Create labeled-data folder ---------------
    path_withoutExt = os.path.splitext(video_path)[0]
    video_name      = path_withoutExt.rsplit("/", 1)[1]
    pathParts = path_config_file.rsplit("\\", 1)
    labeled_datapath = os.path.join(pathParts[0], "labeled-data", video_name)
    if not os.path.exists(labeled_datapath):
        os.makedirs(labeled_datapath)

    #--------------- Create a pandas DataFrame with body part coordinates (x, y) ---------------
    header    = []
    rows      = len(extracted_frames)
    data:list = [[] for _ in range(rows)]
    for pose in bodypart_names:
        header.extend([(experimenter, pose, 'x'),(experimenter, pose, 'y')])
        label = pose + dlc_defs.Parameters.danse.COORDINATES
        for i in range(len(extracted_frames)):
            data[i] += deeplabcut_params.params[label][i]

    columns  = pd.MultiIndex.from_tuples(header, names = ['scorer', 'bodyparts', 'coords'])
    axis_left = []
    for frameNum in [int(item) for item in extracted_frames]:
        axis_left.extend([('labeled-data', video_name, f'img{frameNum}.png')])
    axis1 = pd.MultiIndex.from_tuples(axis_left)
    df = pd.DataFrame(data, columns=columns)
    df.index = axis1

    #--------------- Save dataframe as a hdf file ---------------
    file_path = f'{labeled_datapath}/CollectedData_{experimenter}.h5'
    df.to_hdf(file_path, key='keypoints', mode='w')

    #--------------- Save extracted images (used for labeling bodyparts) as .png ---------------
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        if frame_count in [int(item) for item in extracted_frames]:
            # Save the frame as a PNG file
            frame_filename = f'{labeled_datapath}/img{frame_count}.png'
            cv2.imwrite(frame_filename, frame)
        frame_count += 1 
    cap.release() 
    cv2.destroyAllWindows()

def update_config_file(path_config_file, bodypart_names):
    # Load the YAML file
    with open(path_config_file, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the specific text   
    data['bodyparts'] = bodypart_names

    # Save the modified YAML back to the file
    with open(path_config_file, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def save_coords_to_doric(file_, datapath, path_output, deeplabcut_params, group_names):
    bodypart_names  = deeplabcut_params.params.pop(dlc_defs.Parameters.danse.BODY_PART_NAMES).split(', ')
    bodypart_colors = deeplabcut_params.params.pop(dlc_defs.Parameters.danse.BODY_PART_COLORS).split(', ')
    deeplabcut_params.params[dlc_defs.Parameters.danse.VIDEO_DATAPATH] = datapath

    data, group, operation, series, videoName = group_names
    time_ = np.array(file_[f"{datapath}/{defs.DoricFile.Dataset.TIME}"])
    h5_files = glob.glob(os.path.join(path_output, '*.h5'))
    file_path = h5_files[0]
    data_coords = pd.read_hdf(file_path)
    operation_path = f'{group}/{dlc_defs.Parameters.danse.COORDINATES}/{series}/{videoName}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}'

    for index, bodypart in enumerate(bodypart_names):
        coords = data_coords.loc[:, pd.IndexSlice[:, bodypart, ['x','y']]]
        coords_df = pd.DataFrame(coords.values)
        dataset_path = f'{operation_path}/{bodypart}'

        if dataset_path in file_: 
            del file_[dataset_path] # Remove existing dataset if it exists 
        file_.create_dataset(dataset_path, data=coords_df, dtype = 'int32', chunks=utils.def_chunk_size(coords_df.shape), maxshape=(h5py.UNLIMITED, 2))
        attrs = {
            defs.DoricFile.Attribute.Dataset.USERNAME: bodypart,
            defs.DoricFile.Attribute.ROI.COLOR       : bodypart_colors[index]
        }

        time_path = f'{group}/{dlc_defs.Parameters.danse.COORDINATES}/{series}/{videoName}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}/{defs.DoricFile.Dataset.TIME}'
        if time_path not in file_:
            file_.create_dataset(time_path, data=time_)

        utils.save_attributes(attrs, file_, dataset_path)

    utils.save_attributes(utils.merge_params(params_current = deeplabcut_params.params), file_, operation_path)
    file_.close()