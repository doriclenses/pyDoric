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
        Parameters of danse Find Cells operation
        """
        PROJECT_NAME      = "ProjectName"
        PROJECT_FOLDER    = "ProjectFolder"
        BODY_PART_NAMES   = "BodyPartNames"
        BODY_PART_COLORS  = "BodyPartColors"
        COORDINATES       = "Coordinates"
        EXTRACTED_FRAMES  = "ExtractedFrames"
        VIDEO_FILEPATH    = "VideoFilepath"
        RELATIVE_FILEPATH = "RelativeFilePath"
