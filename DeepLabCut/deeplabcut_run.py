import sys
sys.path.append("..")
import utilities                         as utils
import DeepLabCut.deeplabcut_main        as dlc_main
import DeepLabCut.deeplabcut_parameters  as dlc_params
import DeepLabCut.deeplabcut_definitions as dlc_defs

from unittest.mock import Mock
submodules_matplotlib = [
    'matplotlib', 
    'matplotlib.colors', 
    'matplotlib.animation', 
    'matplotlib.collections', 
    'matplotlib.pyplot',
    'matplotlib.axes',
    'matplotlib.axes._axes',
    'matplotlib.axes._base',
    'matplotlib.artist',
    'matplotlib.image',
    'matplotlib.lines',
    'matplotlib.patches',
    'matplotlib.container',
    'matplotlib.transforms',
    'matplotlib.tri._triangulation',
    'mpl_toolkits.mplot3d', 
    'mpl_toolkits.mplot3d.axis3d', 
    'mpl_toolkits.mplot3d.art3d'
]
for submodule in submodules_matplotlib:
    sys.modules[submodule] = Mock()

from multiprocessing import freeze_support
freeze_support()

danse_params: dict = {}

try:
    for arg in sys.argv[1:]:
        danse_params = eval(arg)

except SyntaxError:
    utils.print_to_intercept(dlc_defs.Messages.ADVANCED_BAD_TYPE)
    sys.exit()

except Exception as error:
    utils.print_error(error, dlc_defs.Messages.LOADING_ARGUMENTS)
    sys.exit()

if __name__ == "__main__":
    params = dlc_params.DeepLabCutParameters(danse_params)

    if params.stage == "CreateProject":
        dlc_main.create_project(params)

    elif params.stage == "SaveLabels":
        dlc_main.save_labels(params)

    elif params.stage == "TrainEvaluate":
        dlc_main.train_evaluate(params)

    elif params.stage == "AnalyzeVideos":
        dlc_main.analyze_videos(params)

    else:
        utils.print_to_intercept(f"Unknown stage: {params.stage}")
        sys.exit()

    print(dlc_defs.Messages.PROCESS_DONE, flush=True)
