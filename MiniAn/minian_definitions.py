
class Messages:
    """
    Messages
    """

    class Main:
        """
        # For minian_main
        """

        START_CLUSTER               = "Starting cluster..."
        LOAD_DATA                   = "Loading dataset to MiniAn..."
        PREPROCESS                  = "Pre-processing..."
        PREPROC_REMOVE_GLOW         = "Pre-processing: removing glow..."
        PREPROC_DENOISING           = "Pre-processing: denoising..."
        PREPROC_REMOV_BACKG         = "Pre-processing: removing background..."
        PREPROC_SAVE                = "Pre-processing: saving..."
        CORRECT_MOTION_ESTIM_SHIFT  = "Correcting motion: estimating shifts..."
        CORRECT_MOTION_APPLY_SHIFT  = "Correcting motion: applying shifts..."
        PREP_DATA_INIT              = "Preparing data for initialization..."
        INIT_SEEDS                  = "Initializing seeds..."
        INIT_SEEDS_PNR_REFI         = "Initializing seeds: PNR refinement..."
        INIT_SEEDS_KOLSM_REF        = "Initializing seeds: Kolmogorov-Smirnov refinement..."
        INIT_SEEDS_MERG             = "Initializing seeds: merging..."
        INIT_COMP                   = "Initializing components..."
        INIT_COMP_SPATIAL           = "Initializing components: spatial..."
        INIT_COMP_TEMP              = "Initializing components: temporal..."
        INIT_COMP_MERG              = "Initializing components: merging..."
        INIT_COMP_BACKG             = "Initializing components: background..."
        CNMF_IT                     = "CNMF {0} iteration"
        CNMF_ESTIM_NOISE            = CNMF_IT + ": estimating noise..."
        CNMF_UPDAT_SPATIAL          = CNMF_IT + ": updating spatial components..."
        CNMF_UPDAT_BACKG            = CNMF_IT + ": updating background components..."
        CNMF_UPDAT_TEMP             = CNMF_IT + ": updating temporal components..."
        CNMF_MERG_COMP              = CNMF_IT + ": merging components..."
        CNMF_SAVE_INTERMED          = CNMF_IT + ": saving intermediate results..."
        SAVING_FINAL                = "Saving final results..."
        SAVING_TO_DORIC             = "Saving data to doric file..."

    class Run:
        """
        # For minian_run
        """

        ADVANCED_BAD_TYPE           = "One of the advanced parameters is not of a python type"
        LOADING_ARGUMENTS           = "Loading parameters"

    class Preview:
        """
        # For preview
        """

        SAVE_TO_HDF5                = "Saving to hdf5"

    class Utilities:
        """
        # For minian_utilities
        """

        UNREC_DOWNSAMPLING_STRAT    = "Unrecognized downsampling strategy"
        GEN_ROI_NAMES               = "Generating ROI names"
        SAVE_ROI_SIG                = "Saving ROI signals"
        SAVE_IMAGES                 = "Saving images"
        SAVE_RES_IMAGES             = "Saving residual images"
        SAVE_SPIKES                 = "Saving spikes"
        SAVE_TO                     = "Saved to {0}"
        ONE_PARM_WRONG_TYPE         = "One parameter of {0} function is of the wrong type"
        NO_CELLS_FOUND              = "No cells where found"

    class CrossReg:
        """
        Cross Registration
        """

        _ = ""

class Text:
    """
    Text definitions
    """

    class Main:
        """
        main definitions
        """

        INTERMEDIATE = "intermediate"

    class Utilities:
        """
        Utilities definitions
        """

        ROISIGNALS  = 'MiniAnROISignals'
        IMAGES      = 'MiniAnImages'
        RESIDUALS   = 'MiniAnResidualImages'
        SPIKES      = 'MiniAnSpikes'

    class CrossReg:
        """
        Cross Registration
        """

        _ = ""

class DictionaryKeys:
    """
    Dictionary keys
    """

    class Preview:
        """
        Preview keys
        """
        SEEDS                   = "Seeds"
        MERGED                  = "Merged"
        REFINED                 = "Refined"
        MAX_PROJ_DATASET_NAME   = "MaxProjDatasetName"

    class CrossReg:
            """
            Cross Registration keys
            """

            _ = ""