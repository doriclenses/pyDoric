class Messages:
    """
    Messages
    """
    ADVANCED_BAD_TYPE  = "One of the advanced parameters is not of a python type"
    LOADING_ARGUMENTS  = "Loading parameters"
    PROCESS_DONE       = "Pose Estimation process is done"

class Parameters:

    """
    All the parameters that danse software sends to python
    """
    class danse:

        """
        Parameters of danse Pose Estimation operation
        """
        PROJECT_FOLDER    = "ProjectFolder"
        BODY_PART_NAMES   = "BodyPartNames"
        BODY_PART_COLORS  = "BodyPartColors"
        COORDINATES       = "Coordinates"
        EXTRACTED_FRAMES  = "ExtractedFrames"
        RELATIVE_FILEPATH = "RelativeFilePath"
        VIDEO_DATAPATH    = "VideoDatapath"
        
class DoricFile:

    class Group:
        POSE_ESTIMATION = "PoseEstimation"


