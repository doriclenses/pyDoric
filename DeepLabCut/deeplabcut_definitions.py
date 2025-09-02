class Messages:
    """
    Messages
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
        PROJECT_FOLDER         = "ProjectFolder"
        BODY_PART_NAMES        = "BodyPartNames"
        BODY_PART_COLORS       = "BodyPartColors"
        COORDINATES            = "Coordinates"
        EXTRACTED_FRAMES       = "ExtractedFrames"
        EXTRACTED_FRAMES_COUNT = "ExtractedFramesCount"
        VIDEO_FILEPATHS        = "VideoFilepaths"
        EXP_FILE               = "ExperimentFile"
        
        SHUFFLE       = "Shuffle"
        BEST_SNAPSHOT = "BestSnapshot"

        RELATIVE_FILEPATH = "RelativeFilePath" 
        VIDEO_DATAPATH    = "VideoDatapath"
        VIDEO_RANGE       = "VideoRange"
        LABELED_VIDEOS         = "LabeledVideos"


class DoricFile:

    class Group:
        POSE_ESTIMATION = "PoseEstimation"


