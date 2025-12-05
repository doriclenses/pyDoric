"""Definitions for DeepLabCut"""

class Messages:
    """
    Messages showed in danse software
    """
    ADVANCED_BAD_TYPE  = "One of the advanced parameters is not of a python type"
    LOADING_ARGUMENTS  = "Loading parameters"
    PROCESS_DONE       = "Pose Estimation process is done"
    SAVING_TO_DORIC    = "Saving data to doric file..."
    FILE_OPENING_ERROR = "Error opening file - {file}"
    NO_VALID_FILE      = "No valid files were found. Exiting..."


class Parameters:

    """
    All the parameters that danse software sends to python
    """
    class danse:

        """
        Parameters of danse Pose Estimation operation
        """
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
     