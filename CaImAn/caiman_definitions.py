
class Messages:
    """
    Messages
    """
    WRITE_IMAGE_TIFF    = "Write image in tiff..."
    MOTION_CORREC       = "Motion correction"
    PARAM_WRONG_TYPE    = "One parameter is of the wrong type"
    START_CNMF          = "Starting CNMF..."
    FITTING             = "Fitting..."
    EVA_COMPO           = "Evaluating components..."
    SAVING_DATA         = "Saving data to doric file..."
    NO_CELLS_FOUND      = "No cells where found"
    ADVANCED_BAD_TYPE   = "One of the advanced parameters is not of a python type"
    LOADING_ARGUMENTS   = "Loading parameters"
    PROCESS_DONE        = "CaImAn process is done"
    SAVE_TO_HDF5        = "Saving to hdf5"
    GEN_ROI_NAMES       = "Generating ROI names"
    SAVE_ROI_SIG        = "Saving ROI signals"
    SAVE_IMAGES         = "Saving images"
    SAVE_RES_IMAGES     = "Saving residual images"
    SAVE_SPIKES         = "Saving spikes"
    SAVE_TO             = "Saved to {path}"
    CROSS_REGISTRATING  = "Cross-registering cells between sessions."

class Folder:
    """
    Folder Name
    """

    _ = ""

class DoricFile:

    """
    Names of the .doric file structue
    """
    class Group:
        ROISIGNALS  = 'CaImAnROISignals'
        IMAGES      = 'CaImAnImages'
        RESIDUALS   = 'CaImAnResidualImages'
        SPIKES      = 'CaImAnSpikes'


class Preview:

    """
    Names of the HDF5 preview file structue
    """
    class Group:
        _ = ""

    class Dataset:
        LOCALCORR  = "LocalCorr"
        PNR        = "PNR"

    class Attribute:
        _ = ""

