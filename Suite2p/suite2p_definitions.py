class Messages:
    """
    Messages
    """
    ADVANCED_BAD_TYPE  = "One of the advanced parameters is not of a python type"
    LOADING_ARGUMENTS  = "Loading parameters"
    PROCESS_DONE       = "Suite2p process is done"
    SAVING_ROIS        = "Saving ROIs"
    SAVING_SPIKES      = "Saving Spikes"
    ROI_NAMES          = "Generating ROI names"

class DoricFile:
    """
    Names of the .doric file structue
    """

    class Group:
        ROISIGNALS = "Suite2pROISignals"
        MEAMIMG    = "Suite2pMeanImg"
        MEAMIMGE   = "Suite2pMeanImgE"
        VCORR      = "Suite2pVcorr"
        SPIKES     = "Suite2pSpikes"