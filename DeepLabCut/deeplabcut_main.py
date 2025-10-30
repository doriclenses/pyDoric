"""DeepLabCut main functions."""

from multiprocessing import freeze_support

import os
import sys
import cv2
import yaml
import glob
import pandas as pd

import deeplabcut_definitions as dlc_defs
import deeplabcut_parameters as dlc_params
import deeplabcut

sys.path.append("..")
import utilities as utils
import definitions as defs

freeze_support()

# --- Main functions called by danse ---
def create_project(params: dlc_params.DeepLabCutParameters):

    """
    Create a new DeepLabCut project.
    """
    # Read danse parameters
    experimenter:  str         = params.params.get(dlc_defs.Parameters.danse.EXPERIMENTER)
    project_name:  str         = params.params.get(dlc_defs.Parameters.danse.PROJECT_NAME)
    root_dir:  str             = params.params.get(dlc_defs.Parameters.danse.ROOT_DIR)
    video_filepaths: list[str] = params.params.get(dlc_defs.Parameters.danse.VIDEO_FILEPATHS)

    config_filepath = deeplabcut.create_new_project(
        project_name,
        experimenter,
        video_filepaths,
        root_dir,
        copy_videos=False
    )

    utils.print_to_intercept("[project path]" + os.path.dirname(config_filepath))


def extract_frames(params: dlc_params.DeepLabCutParameters):
    """
    Extract frames from videos for labeling.
    """
    # Read danse parameters
    project_folder: str         = params.params[dlc_defs.Parameters.danse.PROJECT_FOLDER]
    extraction_method: str      = params.params[dlc_defs.Parameters.danse.EXTRACTION_METHOD]
    num_frames: int             = params.params[dlc_defs.Parameters.danse.NUM_FRAMES]
    video_filepaths: list[str]  = params.params[dlc_defs.Parameters.danse.VIDEO_FILEPATHS]

    config_filepath = os.path.join(project_folder, 'config.yaml')

    deeplabcut.extract_frames(
        config_filepath,
        mode=extraction_method,
        userfeedback=False,
        crop=False
    )

    utils.print_to_intercept("[extracted frames]" + ', '.join([os.path.splitext(os.path.basename(v))[0] for v in video_filepaths]))


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


# --- Helper functions ---
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


def get_pytorch_config_file(
        config_filepath, 
        shuffle
        ) -> str:

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
