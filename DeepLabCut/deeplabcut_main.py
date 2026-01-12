"""DeepLabCut main functions."""

import os
import re
import sys
import glob
import pandas as pd
import deeplabcut
from multiprocessing import freeze_support
from collections import defaultdict

sys.path.append('..')
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
        copy_videos=True
    )

    deeplabcut.auxiliaryfunctions.edit_config(config_filepath, {defs.ConfigKeys.BODY_PARTS: bodypart_names})

    print("This is a test", flush=True)

    utils.print_to_intercept(defs.MessageTags.PROJECT_PATH + os.path.dirname(config_filepath))


def extract_frames(params: dict):
    """
    Extract frames from videos for labeling.
    """
    # Read danse parameters
    project_folder: str         = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    extraction_algo: str        = params.get(defs.Parameters.danse.EXTRACTION_ALGO, defs.Defaults.EXTRACTION_ALGO)
    num_frames: int             = params.get(defs.Parameters.danse.NUM_FRAMES, 20)
    video_filepaths: list[str]  = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)

    config_filepath = os.path.join(project_folder, defs.Paths.CONFIG_FILE)

    deeplabcut.auxiliaryfunctions.edit_config(config_filepath, {defs.ConfigKeys.NUM_FRAMES: num_frames})

    # Check if videos are already in the config file
    add_videos = False
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_filepath)
    config_videos = [os.path.basename(filepath) for filepath in cfg[defs.ConfigKeys.VIDEO_SETS]]
    for video_filepath in video_filepaths:
        if os.path.basename(video_filepath) not in config_videos:
            add_videos = True
            break
    
    # Add videos to the project if not already present
    if add_videos:
        deeplabcut.add_new_videos(
            config=config_filepath,
            videos=video_filepaths,
            copy_videos=True,
            coords=None,           # List containing the list of cropping coordinates of the video
            extract_frames=False,
        )

        iteration = cfg[defs.ConfigKeys.ITERATION]
        deeplabcut.auxiliaryfunctions.edit_config(config_filepath, {defs.ConfigKeys.ITERATION: iteration + 1})

    # Update video paths to the copied videos in the project folder
    video_names = []
    new_video_filepaths = [] # The filepaths that deeplabcut saves in the config file
    for filepath in video_filepaths:
        video_filename = os.path.basename(filepath)
        video_names.append(os.path.splitext(video_filename)[0])
        video_path = os.path.join(project_folder, defs.Paths.VIDEOS, video_filename)
        new_video_filepaths.append(re.sub(r"/+", r"\\", video_path))

    # Check if any of the videos were already labeled
    images_with_labels = defaultdict(list)
    for filepath, video_name in zip(new_video_filepaths, video_names):
        h5_file = glob.glob(os.path.join(project_folder, defs.Paths.LABELED_DATA, video_name, defs.Files.COLLECTED_DATA_PATTERN))
        if h5_file:
            df = pd.read_hdf(h5_file[0])
            images_with_labels[video_name] = df.index.get_level_values(2).tolist()
    print(len(images_with_labels), images_with_labels, flush=True)

    deeplabcut.extract_frames(
        config=config_filepath,
        mode=defs.Defaults.EXTRACTION_MODE,
        algo=extraction_algo,
        userfeedback=False,
        crop=False,
        videos_list=new_video_filepaths
    )

    frames_by_video = defaultdict(list)
    for folder in glob.glob(os.path.join(project_folder, defs.Paths.LABELED_DATA, "*")):
        video_name = os.path.basename(folder)
        if not os.listdir(folder) or video_name not in video_names:
            continue
        for image_file in glob.glob(os.path.join(folder, defs.Files.PNG_PATTERN)):
            labeled_images = images_with_labels[video_name]
            basename = os.path.basename(image_file)
            # Skip frames that were already labeled; handle both absolute and basename matches.
            if image_file in labeled_images or basename in labeled_images:
                continue
            image_index = int(os.path.splitext(basename)[0].replace(defs.LabelColumns.IMAGE_PREFIX,''))
            frames_by_video[video_name].append(image_index)
        frames_by_video[video_name].sort()

    extracted_frames_repr = ', '.join(
        f"{video_name}: [" + ', '.join(map(str, frames_by_video[video_name])) + "]"
        for video_name in video_names
    )
    utils.print_to_intercept(defs.MessageTags.EXTRACTED_FRAMES + "{" + extracted_frames_repr + "}")


def save_labels(params: dict):

    """
    Save labels in DeepLabCut format.
    """
    # Read danse parameters
    project_folder: str       = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    bodypart_names: list[str] = params.get(defs.Parameters.danse.BODY_PART_NAMES)
    video_names: list[str]    = params.get(defs.Parameters.danse.VIDEO_NAMES)

    config_filepath = os.path.join(project_folder, defs.Paths.CONFIG_FILE)

    deeplabcut.auxiliaryfunctions.edit_config(config_filepath, {defs.ConfigKeys.BODY_PARTS: bodypart_names})

    scorer = project_folder.split('-')[1]

    for video_name in video_names:
        labeled_data_path = os.path.join(project_folder, defs.Paths.LABELED_DATA, video_name)
        if not os.path.exists(labeled_data_path):
            os.makedirs(labeled_data_path)

        coords = params.get(video_name + defs.Parameters.danse.COORDINATES)

        frames = os.listdir(labeled_data_path)
        indices = [(defs.Paths.LABELED_DATA, video_name, frame) for frame in frames]

        header1 = []
        for bodypart_name in bodypart_names:
            header1 += [(scorer, bodypart_name, defs.LabelColumns.X), (scorer, bodypart_name, defs.LabelColumns.Y)]
        header2  = pd.MultiIndex.from_tuples(
            header1,
            names = [defs.LabelColumns.SCORER, defs.LabelColumns.BODY_PARTS, defs.LabelColumns.COORDS]
        )

        df = pd.DataFrame(coords, columns = header2, index = pd.MultiIndex.from_tuples(indices))

        filepath = os.path.join(
            labeled_data_path,
            f"{defs.Files.COLLECTED_DATA_PREFIX}{scorer}{defs.Files.HDF_EXTENSION}"
        )
        df.to_hdf(filepath, key=defs.Files.HDF_KEYPOINTS, mode='w')

    utils.print_to_intercept(defs.MessageTags.LABELED_VIDEOS + ', '.join(video_names))


def train_evaluate(params: dict):

    """
    Train and evaluate the DeepLabCut network.
    """

    project_folder: str = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    config_filepath = os.path.join(project_folder, defs.Paths.CONFIG_FILE)

    training_dataset_info = deeplabcut.create_training_dataset(config_filepath) # returns list of tupples [(trainFraction, shuffle, ...), ...]
    shuffle: int = training_dataset_info[0][1]

    deeplabcut.train_network(config_filepath, batch_size=8, shuffle=shuffle)
    deeplabcut.evaluate_network(config_filepath, Shuffles=[shuffle])

    utils.print_to_intercept(defs.MessageTags.TRAIN_INFO + defs.MessageTags.SHUFFLE + str(shuffle))


def analyze_videos(params: dict):

    """
    Analyze videos using the trained DeepLabCut network.
    """
    # Read danse parameters
    project_folder: str        = params.get(defs.Parameters.danse.PROJECT_FOLDER)
    video_filepaths: list[str] = params.get(defs.Parameters.danse.VIDEO_FILEPATHS)
    shuffle: int               = params.get(defs.Parameters.danse.SHUFFLE)
    iteration: int             = params.get(defs.Parameters.danse.ITERATION)

    config_filepath = os.path.join(project_folder, defs.Paths.CONFIG_FILE)
    destfolder = os.path.join(project_folder, defs.Paths.ANALYZED_DATA)

    deeplabcut.analyze_videos(
        config=config_filepath,
        videos=video_filepaths,
        destfolder=destfolder,
        shuffle=shuffle,
        save_as_csv=True
    )

    video_names = [os.path.splitext(os.path.basename(filepath))[0] for filepath in video_filepaths]

    utils.print_to_intercept(defs.MessageTags.ANALYZED_VIDEOS + ', '.join(video_names))
