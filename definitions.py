class DoricFile:

    class Group:
        DATA_PROCESSED  = "DataProcessed"

    class Dataset:
        IMAGE_STACK     = "ImageStack"

    class Attribute:
        FORMAT          = "Format"
        USERNAME        = "Username"
        OPERATION_NAME  = "OperationName"
        OPERATIONS      = "Operations"
        BIT_COUNT       = "BitCount"
        BINNING_FACTOR  = "BinningFactor"


class Parameters:
    """
    All the parameters that danse sends
    """

    class Main:
        """
        Main keys to distinguish different parameters
        """

        PARAMETERS = "Parameters"
        PATHS      = "Paths"
        PREVIEW    = "Preview"
        CROSS_REG  = "CrossReg"

    class Path:
        """
        Paths to file and dataset to process
        """

        TMP_DIR  = "TmpDir"
        FILEPATH = "Filepath"
        H5PATH   = "HDF5Path"

    class danse:
        """
        Parameters of danse Find Cells operation
        """

        NEURO_DIAM_MIN      = "NeuronDiameterMin"
        NEURO_DIAM_MAX      = "NeuronDiameterMax"
        TEMPORAL_DOWNSAMPLE = "TemporalDownsample"
        SPATIAL_DOWNSAMPLE  = "SpatialDownsample"
        NOISE_FREQ          = "NoiseFreq"
        THRES_CORR          = "ThresCorr"
        SPATIAL_PENALTY     = "SpatialPenalty"
        TEMPORAL_PENALTY    = "TemporalPenalty"
        CORRECT_MOTION      = "CorrectMotion"
        ADVANCED_SETTINGS   = "AdvancedSettings"

    class Preview:
        """
        Preview parameters
        """

        FILEPATH                = "PreviewFilepath"
        RANGE                   = "PreviewRange"
        TEMPORAL_DOWNSAMPLE     = "TemporalDownsample"

    class CrossReg:
        """
        Cross Registration parameters
        """

        _ = ""