class DoricFile:

    """
    Doric file structure names
    """

    class Group:
        DATA_PROCESSED = "DataProcessed"
        BEHAVIOR       = "DataBehavior"

    class Dataset:
        IMAGE_STACK = "ImageStack"
        TIME        = "Time"
        ROI         = "ROI{0}"

    class Attribute:

        class Dataset:
            USERNAME = "Username"
            NAME     = "Name"
            PLANE_ID = "PlaneID"
            COLOR    = "Color"

        class Group:
            OPERATIONS      = "Operations"
            BINNING_FACTOR  = "BinningFactor"

        class Image:
            BIT_COUNT = "BitCount"
            FORMAT    = "Format"
            HEIGHT    = "Height"
            WIDTH     = "Width"

        class ROI:
            ID     = "ID"
            COORDS = "Coordinates"
            SHAPE  = "Shape"

        class Signal:
            RANGE_MIN = "RangeMin"
            RANGE_MAX = "RangeMax"
            UNIT      = "Unit"

    class Deprecated:

        class Dataset:

            IMAGES_STACK = "ImagesStack"



class Parameters:

    """
    All the parameters that danse software sends to python
    """

    class Main:

        """
        Main keys to distinguish different parameters
        """

        PARAMETERS    = "Parameters"
        PATHS         = "Paths"
        PREVIEW       = "Preview"
        IS_MICROSCOPE = "IsMicroscope"


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

        NEURO_DIAM_MIN       = "NeuronDiameterMin"
        NEURO_DIAM_MAX       = "NeuronDiameterMax"
        TEMPORAL_DOWNSAMPLE  = "TemporalDownsample"
        SPATIAL_DOWNSAMPLE   = "SpatialDownsample"
        NOISE_FREQ           = "NoiseFreq"
        THRES_CORR           = "ThresCorr"
        SPATIAL_PENALTY      = "SpatialPenalty"
        TEMPORAL_PENALTY     = "TemporalPenalty"
        CORRECT_MOTION       = "CorrectMotion"
        ADVANCED_SETTINGS    = "AdvancedSettings"
        LOCAL_CORR_THRESHOLD = "CorrelationThreshold"
        PNR_THRESHOLD        = "PNRThreshold"
        CROSS_REG            = "CrossReg"
        REF_FILEPATH         = "ReferenceFilepath"
        REF_IMAGES_PATH      = "ReferenceImagesPath"
        REF_ROIS_PATH        = "ReferenceROIsPath"

    class Preview:

        """
        Preview parameters
        """

        FILEPATH            = "PreviewFilepath"
        RANGE               = "PreviewRange"
        TEMPORAL_DOWNSAMPLE = "TemporalDownsample"
        PREVIEW_TYPE        = "PreviewType"


class Messages:

    """
    Messages
    """

    CANT_SAVE_ATT_VAL       = "Cannot save attribute {attribute} with value {value}"
    PATHGROUP               = "[pathgroup]{path}"
    INTERCEPT_MESSAGE       = "[intercept]{message}[end]"
    ERROR_IN                = "Error in {position}: {type_error_name} - {error}"
    FILE_CLOSE              = "File is closed"
    DATASET_NOT_TIME        = "The dataset is not a time vector"
    F_NOT_H5_FILE_FILEPATH  = "f is not h5py.File or filepath to HDF file"
    DATPATH_DOESNT_EXIST    = "{datasetpath} path does not exist in the file"
    HAS_TO_BE_PATH          = "{path} has to be a path to dataset"
