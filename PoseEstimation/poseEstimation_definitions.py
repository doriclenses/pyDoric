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
        POSITIONS = "PosePositions"
