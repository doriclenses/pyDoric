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
import DeepLabCut.deeplabcut_definitions as dlc_defs
import DeepLabCut.deeplabcut_parameters as dlc_params

import deeplabcut

from multiprocessing import freeze_support
freeze_support()

def main(deeplabcut_params: dlc_params.DeepLabCutParameters):

    """
    DeepLabCut algorithm
    """
    # Read danse parameters 
    datapaths: list             = deeplabcut_params.paths[defs.Parameters.Path.H5PATHS]
    data_filepaths: list[str]   = deeplabcut_params.paths[defs.Parameters.Path.FILEPATHS]
    exp_filepath:  str          = deeplabcut_params.params.get(defs.Parameters.Path.EXP_FILE, "")
    project_folder: str         = deeplabcut_params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    bodypart_names: list        = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    extracted_frames: list      = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES]
    extracted_frames_count: int = deeplabcut_params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES_COUNT]
    video_filepaths: list[str]  = deeplabcut_params.params[dlc_defs.Parameters.danse.LABELED_VIDEOS].split(', ')                          

    # Create project and train network
    valid_filepaths, config_filepath = create_project(datapaths, data_filepaths, exp_filepath, project_folder, bodypart_names, extracted_frames_count, extracted_frames, video_filepaths, deeplabcut_params.params)

    training_dataset_info = deeplabcut.create_training_dataset(config_filepath)
    
    shuffle: int = training_dataset_info[0][1]
    update_pytorch_config_file(config_filepath, shuffle) 
    deeplabcut.train_network(config_filepath, batch_size=8, shuffle = shuffle)
    deeplabcut.evaluate_network(config_filepath, Shuffles = [shuffle])

    # Analyze video and save the result
    deeplabcut.analyze_videos(config_filepath, video_filepaths, destfolder = os.path.dirname(config_filepath), shuffle = shuffle)

    save_coords_to_doric(valid_filepaths, datapaths, deeplabcut_params, config_filepath, shuffle)

def preview(deeplabcut_params: dlc_params.DeepLabCutParameters):
    print("hello preview")


def create_project(
    datapaths: list,
    data_filepaths: list,
    exp_filepath: str,
    project_folder: str,
    bodypart_names: list,
    extracted_frames_count: int, 
    extracted_frames: list,
    video_filepaths: list,
    params: dict
):
    """
    Create DeepLabCut project folder with config file and labeled data
    """
    valid_filepaths = []
    for filepath in data_filepaths:
        try:
            file_ = h5py.File(filepath, 'r')
            file_.close()
            valid_filepaths.append(filepath)
        except OSError as e:
            utils.print_error(e, dlc_defs.Messages.FILE_OPENING_ERROR.format(file = filepath))
            continue

    # Exit early if no valid files are found
    if not valid_filepaths:
        utils.print_error(dlc_defs.Messages.NO_VALID_FILE, dlc_defs.Messages.FILE_OPENING_ERROR.format(file = exp_filepath))
        sys.exit()

    path = exp_filepath if len(valid_filepaths) > 1 else valid_filepaths[0]
    task = os.path.splitext(os.path.basename(path))[0]
    user = "danse"

    config_filepath = deeplabcut.create_new_project(task, user, video_filepaths, project_folder, copy_videos = False)
    
    update_config_file(config_filepath, bodypart_names)

    create_labeled_data(config_filepath, extracted_frames_count, extracted_frames, bodypart_names, user, params, video_filepaths)

    return valid_filepaths, config_filepath


def update_config_file(config_filepath, bodypart_names):
    """
    Update label names in the config file
    """

    with open(config_filepath, 'r') as cfg:
        data = yaml.safe_load(cfg)
  
    data['bodyparts'] = bodypart_names

    with open(config_filepath, 'w') as cfg:
        yaml.safe_dump(data, cfg, default_flow_style=False)


def update_pytorch_config_file(
    config_filepath: str, 
    shuffle: int
):
    """
    Update label names in the pytorch config file
    """

    pytorch_config_filepath = get_pytorch_config_file(config_filepath, shuffle)
    with open(pytorch_config_filepath, 'r') as file:
        data = yaml.safe_load(file)

    data['model']['backbone']['freeze_bn_stats']    = False
    data['train_settings']['dataloader_pin_memory'] = True

    with open(pytorch_config_filepath, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)


def create_labeled_data(
    config_filepath: str,
    extracted_frames_count: int,
    extracted_frames: list,
    bodypart_names: list,
    experimenter: str, 
    params: dict, 
    video_filepaths: list
):
    """
    Create labeled data in DeepLabCut format
    """

    # Create folder for labeled data
    project_path = os.path.dirname(config_filepath)
    for i, video_filepath in enumerate(video_filepaths):
        video_name = os.path.splitext(os.path.basename(video_filepath))[0]
        labeled_data_filepath = os.path.join(project_path, "labeled-data", video_name)
        if not os.path.exists(labeled_data_filepath):
            os.makedirs(labeled_data_filepath)

        frames_range = [extracted_frames_count*i, extracted_frames_count*(i+1)]

        # Create pandas dataframe with body part coordinates
        data = [] # [[bp1_x1, bp1_x2, ..., bp1_xn], [bp1_y1, bp1_y2, ..., bp1_yn], [bp2_x1, bp2_x2, ..., bp2_xn], [bp2_y1, bp2_y2, ..., bp2_yn], ...]
        for name in bodypart_names:
            coords = params[name+dlc_defs.Parameters.danse.COORDINATES][frames_range[0]:frames_range[1]]
            data.append([x for x, y in coords])
            data.append([y for x, y in coords]) 
        dataT = list(map(list, zip(*data))) # [[bp1_x1, bp1_y1, bp2_x1, bp2_y1, ...], ..., [bp1_xn, bp1_yn, bp2_xn, bp2_yn, ...]]
         
        frames = extracted_frames[frames_range[0]:frames_range[1]]
        indices = [('labeled-data', video_name, f'img{frame}.png') for frame in frames]
        header1 = []
        for bodypart_name in bodypart_names:
            header1 += [(experimenter, bodypart_name, 'x'),(experimenter, bodypart_name, 'y')]
        header2  = pd.MultiIndex.from_tuples(header1, names = ['scorer', 'bodyparts', 'coords'])

        df = pd.DataFrame(dataT, columns = header2, index = pd.MultiIndex.from_tuples(indices))

        # Save dataframe as hdf file 
        filepath = f'{labeled_data_filepath}/CollectedData_{experimenter}.h5'
        df.to_hdf(filepath, key='keypoints', mode='w')

        # Save extracted frames used for labeling as .png
        cap = cv2.VideoCapture(video_filepath)
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, image = cap.read()
            frame_filename = f'{labeled_data_filepath}/img{frame}.png'
            cv2.imwrite(frame_filename, image)

        cap.release() 
        cv2.destroyAllWindows()


def save_coords_to_doric(
    filepaths: list, 
    datapaths: list,
    deeplabcut_params, 
    config_filepath: str, 
    shuffle: int
):
    """
    Save DeepLabCut analyzed video labels in doric file
    """

    print(dlc_defs.Messages.SAVING_TO_DORIC, flush=True)
    
    bodypart_names  = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodypart_colors = deeplabcut_params.params[dlc_defs.Parameters.danse.BODY_PART_COLORS].split(', ')

    # Update parameters
    deeplabcut_params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER] = os.path.dirname(config_filepath)

    # Get info from config file
    with open(config_filepath, 'r') as file:
        config_data = yaml.safe_load(file)
  
    config_task = config_data['Task']
    config_date = config_data['date']

    # Get info from PyTorch config file
    pytorch_config_filepath = get_pytorch_config_file(config_filepath, shuffle)
    with open(pytorch_config_filepath, 'r') as file:
        pytorch_data = yaml.safe_load(file)

    model  = pytorch_data['net_type'].replace("_", "").capitalize()
    epochs = pytorch_data['train_settings']['epochs']

    # Save results to doric data files
    for filepath in filepaths:
        file_ = h5py.File(filepath, 'a')
        for datapath in datapaths:
            if datapath in file_:
                deeplabcut_params.params[dlc_defs.Parameters.danse.VIDEO_DATAPATH] = datapath
                # Define correct paths for saving operaion results
                _, _, _, series, video_group_name = deeplabcut_params.get_h5path_names(datapath)
                group_path = f"{defs.DoricFile.Group.DATA_BEHAVIOR}/{dlc_defs.Parameters.danse.COORDINATES}/{series}"
                operation_name  = f"{video_group_name}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}"

                operation_count = utils.operation_count(group_path, file_, operation_name, deeplabcut_params.params, {})    
                operation_path  = f'{group_path}/{operation_name + operation_count}'

                # Save time
                video_time = np.array(file_[f"{datapath[:datapath.rfind('/')]}/{defs.DoricFile.Dataset.TIME}"])
                time_datapath = f'{operation_path}/{defs.DoricFile.Dataset.TIME}'
                if time_datapath not in file_:
                    file_.create_dataset(time_datapath, data=video_time, dtype="float64", chunks=utils.def_chunk_size(video_time.shape), maxshape=None)

                # Save operation attributes
                utils.save_attributes(utils.merge_params(params_current = deeplabcut_params.params), file_, operation_path)

                relative_path = file_[datapath].attrs[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]
                video_range   = file_[datapath].attrs[dlc_defs.Parameters.danse.VIDEO_RANGE]

                video_filepath = os.path.join(os.path.dirname(filepath), relative_path.lstrip('/'))
                video_filename = os.path.splitext(os.path.basename(video_filepath))[0]

                # Get coords from hdf file using info (above) from config and pytorch config files
                hdf_data_file = f'{video_filename}DLC_{model}_{config_task}{config_date}shuffle{shuffle}_snapshot_{epochs}.h5'
                hdf_data_file = os.path.join(os.path.dirname(config_filepath), hdf_data_file)
                df_coords = pd.read_hdf(hdf_data_file)
                df_coords = df_coords.iloc[video_range[0]: video_range[1] + 1] 

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


def get_pytorch_config_file(config_filepath, shuffle):

    with open(config_filepath, 'r') as file:
        data = yaml.safe_load(file)
        
    task      = data['Task']
    date      = data['date']
    trainset  = int(data['TrainingFraction'][0] * 100)
    iteration = data['iteration']

    filename     = 'pytorch_config.yaml'
    project_path = os.path.dirname(config_filepath)
    folder_name  = f'{task}{date}-trainset{trainset}shuffle{shuffle}'
    folder_path  = os.path.join(project_path, 'dlc-models-pytorch', f'iteration-{iteration}', folder_name, 'train')

    return os.path.join(folder_path, filename)
