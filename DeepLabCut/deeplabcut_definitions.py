"""Definitions for DeepLabCut"""

class Messages:
    """Messages showed in danse software"""
    ADVANCED_BAD_TYPE  = "One of the advanced parameters is not of a python type"
    LOADING_ARGUMENTS  = "Loading parameters"
    PROCESS_DONE       = "Pose Estimation process is done"
    SAVING_TO_DORIC    = "Saving data to doric file..."
    FILE_OPENING_ERROR = "Error opening file - {file}"
    NO_VALID_FILE      = "No valid files were found. Exiting..."


class Parameters:
    """All the parameters that danse software sends to python"""
    class danse:
        """Parameters of danse Pose Estimation operation"""
        PROJECT_FOLDER = "ProjectFolder"

        EXPERIMENTER  = "Experimenter"
        PROJECT_NAME  = "ProjectName"
        ROOT_DIR      = "RootDir"

        EXTRACTION_ALGO = "ExtractionAlgorithm"
        NUM_FRAMES      = "NumFrames"

        VIDEO_FILEPATHS = "VideoFilepaths"
        VIDEO_NAMES     = "VideoNames"
        BODY_PART_NAMES  = "BodyPartNames"
        COORDINATES      = "Coordinates"

        SHUFFLE   = "Shuffle"
        ITERATION = "Iteration"

class DoricFile:
    """Definitions for Doric file format"""
    class Group:
        """Group names in Doric file"""
        POSE_ESTIMATION = "PoseEstimation"


class Paths:
    """Common DeepLabCut paths"""
    CONFIG_FILE   = "config.yaml"
    VIDEOS        = "videos"
    LABELED_DATA  = "labeled-data"
    ANALYZED_DATA = "analyzed-data"


class ConfigKeys:
    """Keys stored in DeepLabCut config files"""
    BODY_PARTS = "bodyparts"
    NUM_FRAMES = "numframes2pick"
    VIDEO_SETS = "video_sets"
    ITERATION  = "iteration"


class Defaults:
    """Default settings for DeepLabCut operations"""
    EXTRACTION_ALGO = "kmeans"
    EXTRACTION_MODE = "automatic"


class Files:
    """Common file names or patterns"""
    COLLECTED_DATA_PREFIX = "CollectedData_"
    COLLECTED_DATA_PATTERN = "CollectedData_*.h5"
    HDF_KEYPOINTS = "keypoints"
    HDF_EXTENSION = ".h5"
    PNG_PATTERN  = "*.png"


class MessageTags:
    """Prefixes used when printing status messages"""
    PROJECT_PATH     = "[project path]"
    EXTRACTED_FRAMES = "[extracted frames]"
    LABELED_VIDEOS   = "[labeled videos]"
    TRAIN_INFO       = "[train info]"
    ANALYZED_VIDEOS  = "[analyzed videos]"

    SHUFFLE = "shuffle:"

class LabelColumns:
    """Column and index names used for labeled dataframes"""
    SCORER       = "scorer"
    BODY_PARTS   = "bodyparts"
    COORDS       = "coords"
    X            = "x"
    Y            = "y"
    IMAGE_PREFIX = "img"
     
