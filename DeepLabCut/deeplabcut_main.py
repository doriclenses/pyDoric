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
    filepaths: str = deeplabcut_params.paths[defs.Parameters.Path.FILEPATH]
    datapath: str  = deeplabcut_params.paths[defs.Parameters.Path.H5PATH]
    expFile:  str  = deeplabcut_params.params[defs.Parameters.Path.EXP_FILE]

    project_folder: str    = deeplabcut_params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    bodypart_names: list   = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    extracted_frames: list = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES]
    frames_to_extract: int = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES_COUNT]

    # Create project and train network
    video_paths, config_file_path = create_project(filepaths, datapath, expFile, project_folder, bodypart_names, frames_to_extract, extracted_frames, deeplabcut_params.params)
    training_dataset_info = deeplabcut.create_training_dataset(config_file_path)
    
    shuffle = training_dataset_info[0][1]
    update_pytorch_config_file(config_file_path, shuffle) 
    deeplabcut.train_network(config_file_path, batch_size=8, shuffle = shuffle)
    deeplabcut.evaluate_network(config_file_path, Shuffles = [shuffle])

    # Analyze video and save the result
    deeplabcut.analyze_videos(config_file_path, video_paths, destfolder = os.path.dirname(config_file_path), shuffle = shuffle)
    save_coords_to_doric(filepaths, datapath, deeplabcut_params, config_file_path, shuffle)

def preview(deeplabcut_params: dlc_params.DeepLabCutParameters):
    print("hello preview")


def create_project(filepaths, datapath, expFile, project_folder, bodypart_names, frames_to_extract, extracted_frames, params):
    """
    Create DeepLabCut project folder with config file and labeled data
    """

    filename = expFile if len(filepaths) > 1 else filepaths[0]
    task = os.path.splitext(os.path.basename(filename))[0]
    user = "danse"
    video_paths = []
    for i, filepath in enumerate(filepaths):
        with h5py.File(filepath, 'r') as file_:
            relative_path = file_[datapath].attrs[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]
            video_paths.append(os.path.join(os.path.dirname(filepath), relative_path.lstrip('/')))

    config_file_path = deeplabcut.create_new_project(task, user, video_paths, project_folder, copy_videos = False)
    
    update_config_file(config_file_path, bodypart_names)

    create_labeled_data(config_file_path, frames_to_extract, extracted_frames, bodypart_names, user, params, video_paths)

    return video_paths, config_file_path


def update_config_file(config_file_path, bodypart_names):
    """
    Update label names in the config file
    """

    with open(config_file_path, 'r') as cfg:
        data = yaml.safe_load(cfg)
  
    data['bodyparts'] = bodypart_names

    with open(config_file_path, 'w') as cfg:
        yaml.safe_dump(data, cfg, default_flow_style=False)

def update_pytorch_config_file(config_file_path, shuffle_index):
    """
    Update label names in the pytorch config file
    """
    root_path   = os.path.dirname(config_file_path)
    target_file = 'pytorch_config.yaml'

    task, date, trainset, iterations = get_info_config_file(config_file_path)
    folderName = f'{task}{date}-trainset{trainset}shuffle{shuffle_index}'
    dir_path   = os.path.join(root_path, 'dlc-models-pytorch', f'iteration-{iterations}', folderName, 'train')
    pytorch_config_file_path = os.path.join(dir_path, target_file)

    with open(pytorch_config_file_path, 'r') as file:
        data = yaml.safe_load(file)
    data['model']['backbone']['freeze_bn_stats']    = False
    data['train_settings']['dataloader_pin_memory'] = True

    with open(pytorch_config_file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def create_labeled_data(config_file_path, frames_to_extract, extracted_frames, bodypart_names, experimenter, params, video_paths):
    """
    Create labeled data in DeepLabCut format
    """

    # Create folder for labeled data
    for i, video_path in enumerate(video_paths):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        dir_path   = os.path.dirname(config_file_path)

        labeled_data_path = os.path.join(dir_path, "labeled-data", video_name)
        if not os.path.exists(labeled_data_path):
            os.makedirs(labeled_data_path)

        frames_range = [i*(frames_to_extract*i), frames_to_extract*(i+1)]

        # Create pandas dataframe with body part coordinates
        data = [] # [[bp1_x1, bp1_x2, ..., bp1_xn], [bp1_y1, bp1_y2, ..., bp1_yn], [bp2_x1, bp2_x2, ..., bp2_xn], [bp2_y1, bp2_y2, ..., bp2_yn], ...]
        for name in bodypart_names:
            data.append([x for x, y in params[name+dlc_defs.Parameters.danse.COORDINATES][frames_range[0]:frames_range[1]]])
            data.append([y for x, y in params[name+dlc_defs.Parameters.danse.COORDINATES][frames_range[0]:frames_range[1]]]) 
        dataT = list(map(list, zip(*data))) # [[bp1_x1, bp1_y1, bp2_x1, bp2_y1, ...], ..., [bp1_xn, bp1_yn, bp2_xn, bp2_yn, ...]]

        indices = [('labeled-data', video_name, f'img{frame}.png') for frame in extracted_frames[i]]
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
        for frame in extracted_frames[i]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            frame_filename = f'{labeled_data_path}/img{frame}.png'
            cv2.imwrite(frame_filename, img)

        cap.release() 
        cv2.destroyAllWindows()

def save_coords_to_doric(filepaths, datapath, deeplabcut_params, config_file_path, shuffle):
    """
    Save DeepLabCut analyzed video labels in doric file
    """

    print(dlc_defs.Messages.SAVING_TO_DORIC, flush=True)
    
    bodypart_names  = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodypart_colors = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_COLORS].split(', ')

    for file_name in filepaths:
        file_ = h5py.File(file_name, 'a')
        # Define correct path for saving operaion results
        group_names = deeplabcut_params.get_h5path_names()
        _, _, _, series, video_group_name = group_names
        group_path = f"{defs.DoricFile.Group.DATA_BEHAVIOR}/{dlc_defs.Parameters.danse.COORDINATES}/{series}"
    
        operation_name  = f"{video_group_name}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}"
        operation_count = utils.operation_count(group_path, file_, operation_name, deeplabcut_params.params, {})    
        operation_path  = f'{group_path}/{operation_name+operation_count}'
    
        # Save time
        time_ = np.array(file_[f"{datapath}/{defs.DoricFile.Dataset.TIME}"])
        time_path = f'{operation_path}/{defs.DoricFile.Dataset.TIME}'
        if time_path not in file_:
            file_.create_dataset(time_path, data=time_)

        # Save operation attributes
        deeplabcut_params.params[dlc_defs.Parameters.danse.VIDEO_DATAPATH] = datapath
        utils.save_attributes(utils.merge_params(params_current = deeplabcut_params.params), file_, operation_path)

        root_path   = os.path.dirname(config_file_path)
        target_file = 'pytorch_config.yaml'

        task, date, trainset, iterations = get_info_config_file(config_file_path)
        folderName = f'{task}{date}-trainset{trainset}shuffle{shuffle}'
        dir_path   = os.path.join(root_path, 'dlc-models-pytorch', f'iteration-{iterations}', folderName, 'train')
        pytorch_config_file_path = os.path.join(dir_path, target_file)

        # get info from Pytorch configFile
        with open(pytorch_config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        model = data['net_type'].replace("_", "").capitalize()
        epochs = data['train_settings']['epochs']

        relative_path = file_[datapath].attrs[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]
        video_path = os.path.join(os.path.dirname(file_name), relative_path.lstrip('/'))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # get coords data from hdf file using info (above) from config and pytorch config file
        hdf_data_file = f'{video_name}DLC_{model}_{task}{date}shuffle{shuffle}_snapshot_{epochs}.h5'
        hdf_data_file = os.path.join(root_path, hdf_data_file)
        df_coords = pd.read_hdf(hdf_data_file)

        # Save coordinates for each body part
        for index, bodypart_name in enumerate(bodypart_names):
            coords = np.array(df_coords.loc[:, pd.IndexSlice[:, bodypart_name, ['x','y']]])

            coord_datapath = f'{operation_path}/{defs.DoricFile.Dataset.COORDINATES.format(str(index+1).zfill(2))}'
            if coord_datapath in file_: 
                del file_[coord_datapath] # Remove existing dataset if it exists 

            file_.create_dataset(coord_datapath, data=coords, dtype = 'int32', chunks=utils.def_chunk_size(coords.shape), maxshape=(h5py.UNLIMITED, 2))
            attrs = {
                defs.DoricFile.Attribute.Dataset.USERNAME: bodypart_name,
                defs.DoricFile.Attribute.Dataset.COLOR   : bodypart_colors[index]
            }
            utils.save_attributes(attrs, file_, coord_datapath)

        utils.print_group_path_for_DANSE(operation_path)
    
        file_.close()

def get_info_config_file(config_file_path):

    with open(config_file_path, 'r') as file:
        dataConfig = yaml.safe_load(file)
    task       = dataConfig['Task']
    date       = dataConfig['date']
    trainset   = int(dataConfig['TrainingFraction'][0] * 100)
    iterations = dataConfig['iteration']

    return [task, date, trainset, iterations]