import os
import sys
import cv2
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import caiman

sys.path.append('..')
from utilities import (
    save_attributes, 
    load_attributes,
    save_roi_signals,
    save_signals,
    save_images,
    print_to_intercept,
    print_group_path_for_DANSE
)

def save_caiman_to_doric(
    Y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    shape,
    fr: int,
    bits_count: int,
    qt_format: int,
    imagesStackUsername: str,
    vname: str = "caiman.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = 'Series1/Sensor1/',
    params_doric: Optional[dict] = None,
    params_source: Optional[dict] = None,
    saveimages: bool = True,
    saveresiduals: bool = True,
    savespikes: bool = True
) -> str:
    """
    Save CaImAn results to .doric file:
    MiniAnImages - `AC` representing cellular activities as computed by :func:`minian.cnmf.compute_AtC`
    MiniAnResidualImages - residule movie computed as the difference between `Y` and `AC`
    MiniAnSignals - `C` with coordinates from `A`
    MiniAnSpikes - `S`
    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing 
    scalar is computed with least square using a subset of frames from `Y` and `AC` 
    such that their numerical values matches.

    """
    ROISIGNALS = 'CaImAnROISignals'
    IMAGES = 'CaImAnImages'
    RESIDUALS = 'CaImAnResidualImages'
    SPIKES = 'CaImAnSpikes'

    print("generating traces")
    AC = A.dot(C)
    res = Y - AC

    AC = AC.reshape(shape, order='F').transpose((-1, 0, 1)) 
    res = res.reshape(shape, order='F').transpose((-1, 0, 1)) 
    A = A.toarray()
    A = A.reshape(shape[0],shape[1],A.shape[1], order='F').transpose((-1, 0, 1)) 

    time_ = np.arange(0, shape[2]/fr, 1/fr, dtype='float64')
    
    print("generating ROI names")
    names = []
    roiUsernames = []
    for i in range(len(C)):
        names.append('ROI'+str(i+1).zfill(4))
        roiUsernames.append('ROI {}'.format(i+1))
    
    with h5py.File(vname, 'a') as f:
          
        # Check if CaImAn results already exist
        operationCount = ''
        if vpath in f:
            operations = [ name for name in f[vpath] if ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = load_attributes(f, vpath+operation)
                    if merge_params(params_doric, params_source) == operationAttrs:
                        if(len(operation) == len(ROISIGNALS)):
                            operationCount = ''
                        else:
                            operationCount = operation[-1]

                        break


        if vpath[-1] != '/':
            vpath += '/'
        
        if vdataset[-1] != '/':
            vdataset += '/'
        
        params_doric["Operations"] += operationCount

        print("saving ROI signals")
        pathROIs = vpath+ROISIGNALS+operationCount+'/'
        save_roi_signals(C, A, time_, f, pathROIs+vdataset, bits_count=bits_count, attrs_add={"Unit": "Intensity"})
        print_group_path_for_DANSE(pathROIs+vdataset)
        if params_doric is not None and params_source is not None:
            save_attributes(merge_params(params_doric, params_source), f, pathROIs)
        
        if saveimages:
            print("saving images")
            pathImages = vpath+IMAGES+operationCount+'/'
            save_images(AC, time_, f, pathImages+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            print_group_path_for_DANSE(pathImages+vdataset)
            if params_doric is not None and params_source is not None:
                save_attributes(merge_params(params_doric, params_source, params_doric["Operations"] + "(Images)"), f, pathImages)
        
        if saveresiduals:
            print("saving residual images")
            pathResiduals = vpath+RESIDUALS+operationCount+'/'
            save_images(res, time_, f, pathResiduals+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            print_group_path_for_DANSE(pathResiduals+vdataset)
            if params_doric is not None and params_source is not None:
                save_attributes(merge_params(params_doric, params_source, params_doric["Operations"] + "(Residuals)"), f, pathResiduals)
            
        if savespikes:
            print("saving spikes")
            pathSpikes = vpath+SPIKES+operationCount+'/'
            save_signals(S > 0, time_, f, pathSpikes+vdataset, names, roiUsernames, range_min=0, range_max=1)
            print_group_path_for_DANSE(pathSpikes+vdataset)
            if params_doric is not None and params_source is not None:
                save_attributes(merge_params(params_doric, params_source), f, pathSpikes)
        
    print("Saved to {}".format(vname))

def set_advanced_parameters(
    param,
    advanced_parameters
):

    '''
    input:
    param: is a class CNMFParams
    advanced_parameters: dict

    ouput:
    new param: is a class CNMFParams
    new advanced_parameters: dict
    '''
    param_dict = param.to_dict()
    advan_param_keys_used = []
    for param_part_key, param_part_value in param_dict.items():
        for part_key, part_value in param_part_value.items():
            if part_key in advanced_parameters:
               param_dict[param_part_key][part_key] = advanced_parameters[part_key]
               advan_param_keys_used.append(part_key)

    #keep only used keys in andvanced parameters
    advanced_parameters = {key: advanced_parameters[key] for key in advan_param_keys_used}

    return [param, advanced_parameters]


def merge_params(
    params_caiman,
    params_source,
    operation_name = None
):
    '''
    Merge params
    '''

    params_final = {}

    if operation_name is None:
        operation_name = params_caiman["Operations"]

    params_final["Operations"] = params_source["Operations"] + " > " + operation_name

    # Set the advanced Settings keys
    for key in params_caiman:
        if key != "Operations":
            if key == "AdvancedSettings":
                for variable_name, variable_value in params_caiman[key].items():
                    params_final["Advanced-"+variable_name] = str(variable_value) if not isinstance(variable_value, str) else '"'+variable_value+'"'
            else:
                params_final[key] = params_caiman[key]

    # Add Operations operation_name- to the keys
    for key in params_final.copy():
        if key != "Operations":
            params_final[operation_name + "-" + key] = params_final.pop(key)

    # Merging with params source
    for key in params_source:
        if key != "Operations":
            params_final[key] = params_source[key]

    return params_final
