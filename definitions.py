class HDF5Definitions:

    class Dataset:
        IMAGESTACK = "ImageStack"

    class Attributes:
        OPERATION_NAME  = "OperationName"
        OPERATIONS      = "Operations"
        BIT_COUNT       = "BitCount"
        BINNING_FACTOR  = "BinningFactor"


class Parameters:
    '''
    Parameters_keys
    '''
    NEURO_DIAM_MIN          = "NeuronDiameterMin"
    NEURO_DIAM_MAX          = "NeuronDiameterMax"
    TEMPORAL_DOWN_SAMP      = "TemporalDownsample"
    SPATIAL_DOWN_SAMP       = "SpatialDownsample"
    NOISE_FREQ              = "NoiseFreq"
    THRES_CORR              = "ThresCorr"
    SPATIAL_PENALTY         = "SpatialPenalty"
    TEMPORAL_PENALTY        = "TemporalPenalty"
    CORRECT_MOTION          = "CorrectMotion"
    ADVANCED_SETTINGS       = "AdvancedSettings"


class PythonKeys:
    PARAMETERS              = "Parameters"
    PREVIEW                 = "Preview"

    PATHS                   = "Paths"
    TMP_DIR                 = "TmpDir"
    FNAME                   = "FileName"
    H5PATH                  = "HDF5Path"