import sys
import os
sys.path.append("..")
path_dev = os.path.abspath(f"{os.getcwd()}\\..\\..\\..\\DNAMainApp\\Libraries\\pythondlls")
path_re  = os.path.abspath(f"{os.getcwd()}\\libraries\\pythondlls")

if os.path.exists(path_dev):
    os.add_dll_directory(path_dev)
elif os.path.exists(path_re):
    os.add_dll_directory(path_re)

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

import utilities                         as utils
import DeepLabCut.deeplabcut_main        as dlc_main
import DeepLabCut.deeplabcut_parameters  as dlc_params
import DeepLabCut.deeplabcut_definitions as dlc_defs

# Import for PyInstaller
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
    deeplabcut_params = dlc_params.DeepLabCutParameters(danse_params)

    # if deeplabcut_params.preview_params:
    # dlc_main.preview(deeplabcut_params)
    # else:
    dlc_main.main(deeplabcut_params)

    print(dlc_defs.Messages.PROCESS_DONE, flush=True)