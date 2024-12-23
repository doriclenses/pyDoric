import os
import sys
import numpy as np
import pandas as pd
import h5py
import yaml
import cv2
import glob

sys.path.append("..")
import utilities as utils
import definitions as defs
import DeepLabCut.deeplabcut_parameters as dlc_params
import DeepLabCut.deeplabcut_definitions as dlc_defs

import deeplabcut

from multiprocessing import freeze_support
freeze_support()

def main(deeplabcut_params: dlc_params.DeepLabCutParameters):

    """
    DeepLabCut algorithm
    """
    # Read danse parameters 
    filepath: str = deeplabcut_params.paths[defs.Parameters.Path.FILEPATH]
    datapath: str = deeplabcut_params.paths[defs.Parameters.Path.H5PATH]

    project_folder: str    = deeplabcut_params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    bodypart_names: list   = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    extracted_frames: list = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES]

    # Create project and train network
    video_path, config_file_path = create_project(filepath, datapath, project_folder, bodypart_names, extracted_frames, deeplabcut_params.params)
    deeplabcut.create_training_dataset(config_file_path)
    deeplabcut.train_network(config_file_path)
    update_Pytorch_config_file(config_file_path)
    deeplabcut.evaluate_network(config_file_path)
 
    # Analyze video and save the result
    deeplabcut.analyze_videos(config_file_path, [video_path], destfolder = os.path.dirname(config_file_path))
    save_coords_to_doric(filepath, datapath, os.path.dirname(config_file_path), deeplabcut_params.params,
                         group_names = deeplabcut_params.get_h5path_names()) 


def preview(deeplabcut_params: dlc_params.DeepLabCutParameters):
    print("hello preview")


def create_project(filepath, datapath, project_folder, bodypart_names, extracted_frames, params):
    """
    Create DeepLabCut project folder with config file and labeled data
    """

    task = os.path.splitext(os.path.basename(filepath))[0]
    user = "danse"

    with h5py.File(filepath, 'r') as file_:
        relative_path = file_[datapath].attrs[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]

    full_path = os.path.join(os.path.dirname(filepath), relative_path.lstrip('/'))

    config_file_path = deeplabcut.create_new_project(task, user, [full_path], project_folder, copy_videos = False)
    
    update_config_file(config_file_path, bodypart_names)

    create_labeled_data(config_file_path, extracted_frames, bodypart_names, user, params, full_path)

    return full_path, config_file_path


def update_config_file(config_file_path, bodypart_names):
    """
    Update label names in the config file
    """

    with open(config_file_path, 'r') as file:
        data = yaml.safe_load(file)
  
    data['bodyparts'] = bodypart_names

    with open(config_file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def update_Pytorch_config_file(config_file_path):
    """
    Update label names in the pytorch config file
    """
    root_path   = os.path.dirname(config_file_path)
    target_file = 'pytorch_config.yaml'
    for dir_path, dirnames, filenames in os.walk(root_path): 
        if target_file in filenames:
            print(dir_path)
            pytorch_config_file_path = os.path.join(dir_path, target_file)

    with open(pytorch_config_file_path, 'r') as file:
        data = yaml.safe_load(file)
  
    data['model']['backbone']['freeze_bn_stats']    = False
    data['train_settings']['dataloader_pin_memory'] = True

    with open(pytorch_config_file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def create_labeled_data(config_file_path, extracted_frames, bodypart_names, experimenter, params, video_path):
    """
    Create labeled data in DeepLabCut format
    """

    # Create folder for labeled data
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    dir_path   = os.path.dirname(config_file_path)

    labeled_data_path = os.path.join(dir_path, "labeled-data", video_name)
    if not os.path.exists(labeled_data_path):
        os.makedirs(labeled_data_path)

    # Create pandas dataframe with body part coordinates
    data = [] # [[bp1_x1, bp1_x2, ..., bp1_xn], [bp1_y1, bp1_y2, ..., bp1_yn], [bp2_x1, bp2_x2, ..., bp2_xn], [bp2_y1, bp2_y2, ..., bp2_yn], ...]
    for name in bodypart_names:
        data.append([x for x, y in params[name+dlc_defs.Parameters.danse.COORDINATES]])
        data.append([y for x, y in params[name+dlc_defs.Parameters.danse.COORDINATES]]) 
    dataT = list(map(list, zip(*data))) # [[bp1_x1, bp1_y1, bp2_x1, bp2_y1, ...], ..., [bp1_xn, bp1_yn, bp2_xn, bp2_yn, ...]]

    indices = [('labeled-data', video_name, f'img{frame}.png') for frame in extracted_frames]
    header1 = []
    for bodypart_name in bodypart_names:
        header1 += [(experimenter, bodypart_name, 'x'),(experimenter, bodypart_name, 'y')]
    header2  = pd.MultiIndex.from_tuples(header1, names = ['scorer', 'bodyparts', 'coords'])
    
    df = pd.DataFrame(dataT, columns = header2, index = pd.MultiIndex.from_tuples(indices))

    # Save dataframe as hdf file 
    file_path = f'{labeled_data_path}/CollectedData_{experimenter}.h5'
    df.to_hdf(file_path, key='keypoints', mode='w')

    # Save extracted frames used for labeling as .png
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        if frame_index in extracted_frames: 
            # Save the frame as a PNG file
            frame_filename = f'{labeled_data_path}/img{frame_index}.png'
            cv2.imwrite(frame_filename, frame)
        frame_index += 1

    cap.release() 
    cv2.destroyAllWindows()

def save_coords_to_doric(filepath, datapath, output_path, params, group_names):
    """
    Save DeepLabCut analyzed video labels in doric file
    """
    print(dlc_defs.Messages.SAVING_TO_DORIC, flush=True)

    file_ = h5py.File(filepath, 'a')
    bodypart_names  = params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodypart_colors = params[dlc_defs.Parameters.danse.BODY_PART_COLORS].split(', ')

    # Define correct path for saving operaion results
    _, _, _, series, video_name = group_names
    group_path = f"{defs.DoricFile.Group.DATA_BEHAVIOR}/{dlc_defs.Parameters.danse.COORDINATES}/{series}"
    
    operation_name  = f"{video_name}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}"
    operation_count = utils.operation_count(group_path, file_, operation_name, params, {})    
    operation_path  = f'{group_path}/{operation_name+operation_count}'
    
    # Save time
    time_ = np.array(file_[f"{datapath}/{defs.DoricFile.Dataset.TIME}"])
    time_path = f'{operation_path}/{defs.DoricFile.Dataset.TIME}'
    if time_path not in file_:
        file_.create_dataset(time_path, data=time_)

    # Save operation attributes
    params[dlc_defs.Parameters.danse.VIDEO_DATAPATH] = datapath
    utils.save_attributes(utils.merge_params(params_current = params), file_, operation_path)

    # Save coordinates for each body part
    h5_file1 = glob.glob(os.path.join(output_path, '*.h5'))[0]
    df_coords = pd.read_hdf(h5_file1)

    for index, bodypart_name in enumerate(bodypart_names):
        coords = np.array(df_coords.loc[:, pd.IndexSlice[:, bodypart_name, ['x','y']]])

        datapath = f'{operation_path}/{defs.DoricFile.Dataset.COORDINATES.format(str(index+1).zfill(2))}'
        if datapath in file_: 
            del file_[datapath] # Remove existing dataset if it exists 

        file_.create_dataset(datapath, data=coords, dtype = 'int32', chunks=utils.def_chunk_size(coords.shape), maxshape=(h5py.UNLIMITED, 2))
        attrs = {
            defs.DoricFile.Attribute.Dataset.USERNAME: bodypart_name,
            defs.DoricFile.Attribute.Dataset.COLOR   : bodypart_colors[index]
        }
        utils.save_attributes(attrs, file_, datapath)

    utils.print_group_path_for_DANSE(operation_path)
    
    file_.close()