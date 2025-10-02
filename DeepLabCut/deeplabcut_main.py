"""DeepLabCut main functions."""

from multiprocessing import freeze_support

import os
import sys
import cv2
import h5py
import yaml
import glob
import numpy as np
import pandas as pd

import deeplabcut_definitions as dlc_defs
import deeplabcut_parameters as dlc_params
import deeplabcut

sys.path.append("..")
import utilities as utils
import definitions as defs

freeze_support()


def create_project(params: dlc_params.DeepLabCutParameters):

    """
    Create a new DeepLabCut project.
    """
    # Read danse parameters
    experimenter:  str         = params.params.get(dlc_defs.Parameters.danse.EXPERIMENTER)
    project_name:  str         = params.params.get(dlc_defs.Parameters.danse.PROJECT_NAME)
    root_dir:  str             = params.params.get(dlc_defs.Parameters.danse.ROOT_DIR)
    video_filepaths: list[str] = params.params.get(dlc_defs.Parameters.danse.VIDEO_FILEPATHS)

    config_filepath = deeplabcut.create_new_project(project_name, experimenter, video_filepaths, root_dir, copy_videos = False)

    utils.print_to_intercept("[project path]" + os.path.dirname(config_filepath))


def save_labels(params: dlc_params.DeepLabCutParameters):

    """
    Save labels in DeepLabCut format.
    """
    # Read danse parameters
    video_filepaths: list[str]  = params.params[dlc_defs.Parameters.danse.VIDEO_FILEPATHS]
    project_folder: str         = params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    bodypart_names: list        = params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    extracted_frames: list      = params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES]
    extracted_frames_count: int = params.params[dlc_defs.Parameters.danse.EXTRACTED_FRAMES_COUNT]

    config_filepath = os.path.join(project_folder, 'config.yaml')

    update_config_file(config_filepath, bodypart_names)
    deeplabcut.add_new_videos(config_filepath, video_filepaths, extract_frames=False)
    create_labeled_data(config_filepath, extracted_frames_count, extracted_frames, bodypart_names, params.params, video_filepaths)

    video_names = []
    for video_filepath in video_filepaths:
        video_names.append(os.path.splitext(os.path.basename(video_filepath))[0])

    utils.print_to_intercept("[labeled videos]" + ', '.join(video_names))


def train_evaluate(params: dlc_params.DeepLabCutParameters):

    """
    Train and evaluate the DeepLabCut network.
    """

    project_folder: str = params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    config_filepath = os.path.join(project_folder, 'config.yaml')

    training_dataset_info = deeplabcut.create_training_dataset(config_filepath)
    shuffle: int = training_dataset_info[0][1]
    update_pytorch_config_file(config_filepath, shuffle)

    with open(config_filepath, 'r') as cfg:
        data = yaml.safe_load(cfg)
    iteration = data['iteration']

    deeplabcut.train_network(config_filepath, batch_size=8, shuffle=shuffle)
    deeplabcut.evaluate_network(config_filepath, Shuffles=[shuffle])

    utils.print_to_intercept(f"[train info]{iteration}, shuffle-{shuffle}")


def analyze_videos(params: dlc_params.DeepLabCutParameters):

    """
    Analyze videos using the trained DeepLabCut network.
    """
    # Read danse parameters
    video_filepaths: list[str] = params.params[dlc_defs.Parameters.danse.VIDEO_FILEPATHS]
    project_folder: str        = params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    shuffle: int               = params.params[dlc_defs.Parameters.danse.SHUFFLE]

    config_filepath = os.path.join(project_folder, 'config.yaml')

    deeplabcut.analyze_videos(config_filepath, video_filepaths, destfolder=project_folder+'/analyzed-data', shuffle=shuffle)

    for file_ in glob.glob(os.path.join(project_folder, 'analyzed-data', '*.h5')):
        df = pd.read_hdf(file_)
        df.to_csv(file_.replace('.h5', '.csv'))

    video_names = []
    for video_filepath in video_filepaths:
        video_names.append(os.path.splitext(os.path.basename(video_filepath))[0])

    utils.print_to_intercept("[analyzed videos]" + ', '.join(video_names))


def save_coordinates(params: dlc_params.DeepLabCutParameters):

    """
    Save DeepLabCut analyzed video labels in doric file.
    """
    # Read danse parameters
    datapath: str             = params.paths[defs.Parameters.Path.H5PATH]
    data_filepaths: list[str] = params.paths[defs.Parameters.Path.FILEPATHS]

    project_folder: str = params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    shuffle: int        = params.params[dlc_defs.Parameters.danse.SHUFFLE]
    best_snapshot: str  = params.params.get(dlc_defs.Parameters.danse.BEST_SNAPSHOT, None)

    config_filepath = os.path.join(project_folder, 'config.yaml')
    coords_datapaths = save_coords_to_doric(data_filepaths, datapath, params, config_filepath, shuffle, best_snapshot)

    utils.print_to_intercept("[coordinates datapaths]" + ', '.join(coords_datapaths))


def update_config_file(
        config_filepath: str,
        bodypart_names: list[str]
):
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
    extracted_frames: list[int],
    bodypart_names: list[str],
    params: dict,
    video_filepaths: list[str]
):
    """
    Create labeled data in DeepLabCut format
    """

    project_path = os.path.dirname(config_filepath)
    experimenter = "danse"

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
            _, image = cap.read()
            frame_filename = f'{labeled_data_filepath}/img{frame}.png'
            cv2.imwrite(frame_filename, image)

        cap.release()
        cv2.destroyAllWindows()


def save_coords_to_doric(
    filepaths: list[str],
    datapath: str,
    params: dlc_params.DeepLabCutParameters,
    config_filepath: str,
    shuffle: int,
    best_snapshot: str = None
):
    """
    Save DeepLabCut analyzed video labels in doric file
    """
    all_saved_datapaths = []

    print(dlc_defs.Messages.SAVING_TO_DORIC, flush=True)
    
    bodypart_names  = params.params[dlc_defs.Parameters.danse.BODY_PART_NAMES].split(', ')
    bodypart_colors = params.params[dlc_defs.Parameters.danse.BODY_PART_COLORS].split(', ')

    # Define correct paths for saving operaion results
    _, _, _, series, video_group_name = params.get_h5path_names(datapath)
    group_path = f"{defs.DoricFile.Group.DATA_BEHAVIOR}/{dlc_defs.Parameters.danse.COORDINATES}/{series}"
    operation_name  = f"{video_group_name}{dlc_defs.DoricFile.Group.POSE_ESTIMATION}"

    # Update parameters
    params.params[dlc_defs.Parameters.danse.VIDEO_DATAPATH] = datapath
    params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER] = os.path.dirname(config_filepath)

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
    epochs = best_snapshot if best_snapshot is None else pytorch_data['train_settings']['epochs']

    # Save results to doric data files
    for filepath in filepaths:
        try:
            file_ = h5py.File(filepath, 'a')
        except OSError as e:
            utils.print_error(e, dlc_defs.Messages.FILE_OPENING_ERROR.format(file = filepath))
            continue

        operation_count = utils.operation_count(group_path, file_, operation_name, params.params, {})
        operation_path  = f'{group_path}/{operation_name + operation_count}'

        # Save time
        video_time = np.array(file_[f"{datapath[:datapath.rfind('/')]}/{defs.DoricFile.Dataset.TIME}"])
        time_datapath = f'{operation_path}/{defs.DoricFile.Dataset.TIME}'
        if time_datapath not in file_:
            file_.create_dataset(time_datapath, data=video_time, dtype="float64", chunks=utils.def_chunk_size(video_time.shape), maxshape=None)

        # Save operation attributes
        utils.save_attributes(utils.merge_params(params_current = params.params), file_, operation_path)

        relative_path = file_[datapath].attrs[dlc_defs.Parameters.danse.RELATIVE_FILEPATH]
        video_range   = file_[datapath].attrs[dlc_defs.Parameters.danse.VIDEO_RANGE]

        video_filepath = os.path.join(os.path.dirname(filepath), relative_path.lstrip('/'))
        video_filename = os.path.splitext(os.path.basename(video_filepath))[0]

        # Get coords from hdf file using info (above) from config and pytorch config files
        hdf_data_file = f'{video_filename}DLC_{model}_{config_task}{config_date}shuffle{shuffle}_snapshot_{epochs}.h5'
        hdf_data_file = os.path.join(os.path.dirname(config_filepath), 'analyzed-data', hdf_data_file)
        df_coords = pd.read_hdf(hdf_data_file)
        df_coords = df_coords[video_range[0]: video_range[1] + 1]

        # Save coordinates for each body part
        coords_datapaths = []
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

            coords_datapaths.append(coord_datapath)

        file_.close()

        return all_saved_datapaths



def get_pytorch_config_file(config_filepath, shuffle):

    """
    Get the PyTorch config file path based on the main config file and shuffle parameter.
    """

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
