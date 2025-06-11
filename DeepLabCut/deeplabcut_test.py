import sys
sys.path.append("..")
import definitions as defs
import deeplabcut_main as dlc_main
import deeplabcut_parameters as dlc_params
import deeplabcut_definitions as dlc_defs

dict_params = {
    defs.Parameters.Main.PARAMETERS : {
        'Head'+dlc_defs.Parameters.danse.COORDINATES : [(329,298),(468,162),(319,390),(425,418),(404,123),(227,256),(488,307),(211,152),(297,141),(384,388),(335,267),(216,336),(476,337),(357,265),(333,263),(497,176),(379,402),(248,337),(239,211),(333,270)],
        'Body'+dlc_defs.Parameters.danse.COORDINATES : [(313,279),(450,163),(347,390),(413,404),(410,141),(219,283),(474,302),(236,158),(320,144),(360,389),(350,285),(232,358),(468,362),(336,270),(315,271),(482,181),(379,375),(245,314),(236,234),(349,288)],
        'Tail'+dlc_defs.Parameters.danse.COORDINATES : [(308,260),(440,177),(366,390),(405,388),(416,157),(217,302),(467,305),(251,167),(335,156),(339,395),(360,297),(248,359),(456,375),(317,280),(299,280),(470,188),(380,357),(233,296),(229,253),(361,297)],
        dlc_defs.Parameters.danse.BODY_PART_COLORS : '#77aadd, #ee8866, #eedd88',
        dlc_defs.Parameters.danse.BODY_PART_NAMES : 'Head, Body, Tail',
        dlc_defs.Parameters.danse.EXTRACTED_FRAMES : [3165,3750,1061,3980,1887,214,4032,2267,2628,2335,1493,755,4014,2822,599,3698,532,2305,2536,1578],
        dlc_defs.Parameters.danse.EXTRACTED_FRAMES_COUNT : 20,
        dlc_defs.Parameters.danse.VIDEO_FILEPATHS : 'C:/Users/ING55/dataFiles/Clients/SusanaLima/allResaved/Video/animal1115_NSF.mp4',
        dlc_defs.Parameters.danse.PROJECT_FOLDER : 'C:/Users/ING55/dataFiles/Clients/SusanaLima/animal1115_NSF-danse-2025-06-10',
        dlc_defs.Parameters.danse.SHUFFLE: 2,
        dlc_defs.Parameters.danse.BEST_SNAPSHOT: '030',
        defs.DoricFile.Attribute.Group.OPERATIONS : "DeepLabCut Pose Estimation"
    },
    defs.Parameters.Main.PATHS : {
        defs.Parameters.Path.FILEPATHS : ['C:/Users/ING55/dataFiles/Clients/SusanaLima/animal1115_NSF.doric'],
        defs.Parameters.Path.H5PATH : '/DataBehavior/Video/Series0001/Anymaze/Video01'
    }
}

dlc_main.create_project(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.add_labels(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.train_evaluate(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.analyze_videos(dlc_params.DeepLabCutParameters(dict_params))
# dlc_main.save_coordinates(dlc_params.DeepLabCutParameters(dict_params))
