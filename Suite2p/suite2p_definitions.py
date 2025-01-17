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


class Preview:
    """
    Names of preview file structue
    """
    class Dataset:
        MEAN_IMAGE         = "MeanImage"
        MEDIAN_FILTER_MEAN = "MedianFilteredMeanImage"
        CORRELATION_MAP    = "CorrelationMap"
        MAX_PROJECTION     = "MaxProjection"


class DoricFile:
    """
    Names of the .doric file structue
    """
    class Group:
        ROISIGNALS = "Suite2pROISignals"
        SPIKES     = "Suite2pSpikes"