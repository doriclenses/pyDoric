import sys
sys.path.append("..")
import definitions as defs
import deeplabcut_main as dlc_main
import deeplabcut_parameters as dlc_params
import deeplabcut_definitions as dlc_defs

dict_params = {
    defs.Parameters.Main.PARAMETERS : {
        'animal1115_NSF'+dlc_defs.Parameters.danse.COORDINATES : [
            [329,298,313,279,308,260],[468,162,450,163,440,177],[319,390,347,390,366,390],[425,418,413,404,405,388],[404,123,410,141,416,157],
            [227,256,219,283,217,302],[488,307,474,302,467,305],[211,152,236,158,251,167],[297,141,320,144,335,156],[384,388,360,389,339,395],
            [335,267,350,285,360,297],[216,336,232,358,248,359],[476,337,468,362,456,375],[357,265,336,270,317,280],[333,263,315,271,299,280],
            [497,176,482,181,470,188],[379,402,379,375,380,357],[248,337,245,314,233,296],[239,211,236,234,229,253],[333,270,349,288,361,297]],
        dlc_defs.Parameters.danse.BODY_PART_NAMES : 'Head, Body, Tail',
        dlc_defs.Parameters.danse.NUM_FRAMES : 20,
        dlc_defs.Parameters.danse.VIDEO_FILEPATHS : ['C:/Users/ING55/dataFiles/Clients/SusanaLima/allResaved/Video/animal1115_NSF.mp4'],
        dlc_defs.Parameters.danse.PROJECT_FOLDER : 'C:/Users/ING55/dataFiles/Clients/SusanaLima/animal1115_NSF-danse-2025-06-11',
        dlc_defs.Parameters.danse.SHUFFLE: 2,
        defs.DoricFile.Attribute.Group.OPERATIONS : "DeepLabCut Pose Estimation"
    },
    defs.Parameters.Main.PATHS : {
        defs.Parameters.Path.FILEPATHS : ['C:/Users/ING55/dataFiles/Clients/SusanaLima/animal1115_NSF.doric'],
        defs.Parameters.Path.H5PATH : '/DataBehavior/Video/Series0001/Anymaze/Video01'
    }
}

dlc_main.create_project(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.extract_frames(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.save_labels(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.train_evaluate(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.analyze_videos(dlc_params.DeepLabCutParameters(dict_params))
