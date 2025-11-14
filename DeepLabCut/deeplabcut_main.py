"""DeepLabCut main functions."""

import os
import re
import sys
import yaml
import glob
import pandas as pd
import deeplabcut
from multiprocessing import freeze_support

sys.path.append("..")
import utilities as utils
import deeplabcut_definitions as defs

freeze_support()

# --- Main functions called by danse ---
def create_project(params: dict):

    """
    Create a new DeepLabCut project.
    """
    # Read danse parameters
    experimenter:  str         = params.get(defs.Parameters.danse.EXPERIMENTER)
    project_name:  str         = params.get(defs.Parameters.danse.PROJECT_NAME)
    root_dir:  str             = params.get(defs.Parameters.danse.ROOT_DIR)
    video_filepaths: list[str] = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)
    bodypart_names: list[str]   = params.get(defs.Parameters.danse.BODY_PART_NAMES)

    config_filepath = deeplabcut.create_new_project(
        project=project_name,
        experimenter=experimenter,
        videos=video_filepaths,
        working_directory=root_dir,
        copy_videos=False
    )

    update_config_file(config_filepath, 'bodyparts', bodypart_names)

    utils.print_to_intercept("[project path]" + os.path.dirname(config_filepath))


def extract_frames(params: dict):
    """
    Extract frames from videos for labeling.
    """
    # Read danse parameters
    project_folder: str         = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    extraction_algo: str        = params.get(defs.Parameters.danse.EXTRACTION_ALGO, 'kmeans')
    num_frames: int             = params.get(defs.Parameters.danse.NUM_FRAMES, 20)
    video_filepaths: list[str]  = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)

    config_filepath = os.path.join(project_folder, 'config.yaml')
    update_config_file(config_filepath, 'numframes2pick', num_frames)

    for i, filepath in enumerate(video_filepaths):
        video_filepaths[i] = re.sub(r"/+", r"\\", filepath)

    deeplabcut.extract_frames(
        config=config_filepath,
        mode='automatic',
        algo=extraction_algo,
        userfeedback=False,
        crop=False,
        videos_list=video_filepaths
    )

    frames = []
    videos = []
    for folder in glob.glob(os.path.join(project_folder, 'labeled-data', '*')):
        if not os.listdir(folder):
            continue
        videos.append(os.path.basename(folder))
        for image_file in glob.glob(os.path.join(folder, '*.png')):
            image_index = int(os.path.splitext(os.path.basename(image_file))[0].replace('img',''))
            frames.append(image_index)

    utils.print_to_intercept("[extracted frames]" + 'videos: [' + ', '.join(videos) + '],frames: [' + ', '.join(map(str, frames)) + ']')


def save_labels(params: dict):

    """
    Save labels in DeepLabCut format.
    """
    # Read danse parameters
    project_folder: str         = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    bodypart_names: list[str]   = params.get(defs.Parameters.danse.BODY_PART_NAMES)
    video_filepaths: list[str]  = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)
    video_names: list[str]      = params.get(defs.Parameters.danse.VIDEO_NAMES)

    config_filepath = os.path.join(project_folder, 'config.yaml')
    update_config_file(config_filepath, 'bodyparts', bodypart_names)

    scorer = project_folder.split('-')[1]

    deeplabcut.add_new_videos(
        config_filepath,
        video_filepaths,
        extract_frames=False
    )

    for video_name in video_names:
        labeled_data_path = os.path.join(project_folder, "labeled-data", video_name)
        if not os.path.exists(labeled_data_path):
            os.makedirs(labeled_data_path)

        coords = params.get(video_name + defs.Parameters.danse.COORDINATES)

        frames = os.listdir(labeled_data_path)
        indices = [('labeled-data', video_name, frame) for frame in frames]

        header1 = []
        for bodypart_name in bodypart_names:
            header1 += [(scorer, bodypart_name, 'x'), (scorer, bodypart_name, 'y')]
        header2  = pd.MultiIndex.from_tuples(header1, names = ['scorer', 'bodyparts', 'coords'])

        df = pd.DataFrame(coords, columns = header2, index = pd.MultiIndex.from_tuples(indices))

        filepath = f'{labeled_data_path}/CollectedData_{scorer}.h5'
        df.to_hdf(filepath, key='keypoints', mode='w')

    utils.print_to_intercept("[labeled videos]" + ', '.join(video_names))


def train_evaluate(params: dict):

    """
    Train and evaluate the DeepLabCut network.
    """

    project_folder: str = params.get(defs.Parameters.danse.PROJECT_FOLDER)

    config_filepath = os.path.join(project_folder, 'config.yaml')
    deeplabcut.create_training_dataset(config_filepath)
    deeplabcut.train_network(config_filepath)
    deeplabcut.evaluate_network(config_filepath)

    utils.print_to_intercept("[train info]")

    # training_dataset_info = deeplabcut.create_training_dataset(config_filepath)
    # shuffle: int = training_dataset_info[0][1]
    # update_pytorch_config_file(config_filepath, shuffle)

    # with open(config_filepath, 'r') as cfg:
    #     data = yaml.safe_load(cfg)
    # iteration = data['iteration']

    # deeplabcut.train_network(config_filepath, batch_size=8, shuffle=shuffle)
    # deeplabcut.evaluate_network(config_filepath, Shuffles=[shuffle])

    # utils.print_to_intercept(f"[train info]{iteration}, shuffle-{shuffle}")


def analyze_videos(params: dict):

    """
    Analyze videos using the trained DeepLabCut network.
    """
    # Read danse parameters
    project_folder: str        = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    video_filepaths: list[str] = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)
    shuffle: int               = params.get(defs.Parameters.danse.SHUFFLE)
    iteration: int             = params.get(defs.Parameters.danse.ITERATION)

    config_filepath = os.path.join(project_folder, 'config.yaml')
    destfolder = os.path.join(project_folder, 'analyzed-data')

    update_config_file(config_filepath, 'iteration', iteration)

    deeplabcut.analyze_videos(
        config=config_filepath,
        videos=video_filepaths,
        destfolder=destfolder,
        shuffle=shuffle,
        save_as_csv=True
    )

    video_names = [os.path.splitext(os.path.basename(filepath))[0] for filepath in video_filepaths]

    utils.print_to_intercept("[analyzed videos]" + ', '.join(video_names))


# --- Helper functions ---
def update_config_file(
        config_filepath: str,
        parameter: str,
        value: any
):
    """
    Update label names in the config file
    """

    with open(config_filepath, 'r') as cfg:
        data = yaml.safe_load(cfg)

    data[parameter] = value

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
        config_filepath: str,
        shuffle: int
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
