
class Messages:
    '''
    Messages
    '''

    # For minian_main
    START_CLUSTER               = "Starting cluster..."
    LOAD_DATA                   = "Loading dataset to MiniAn..."
    ONE_PARM_WRONG_TYPE         = "One parameter of {0} function is of the wrong type"
    NO_CELLS_FOUND              = "No cells where found"
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

    # For minian_run
    ADVANCED_BAD_TYPE           = "One of the advanced settings is not of a python type"
    LOADING_ARGUMENTS           = "Loading arguments"

    # For preview
    SAVE_TO_HDF5                = "save to hdf5"

class Preview:
    '''
    Preview keys
    '''

    MAX_PROJ_DATASET_NAME   = "MaxProjDatasetName"
    SEED_GROUP_NAME         = "SeedGroupName"

