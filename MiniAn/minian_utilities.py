import os
import sys
import h5py
import dask as da
import numpy as np
import xarray as xr
import functools as fct
from typing import Optional, Callable
from minian.utilities import custom_arr_optimize
sys.path.append('..')
from utilities import (
    save_roi_signals,
    save_signals,
    save_images,
    load_attributes,
    save_attributes,
    print_group_path_for_DANSE
)
import inspect

def load_doric_to_xarray(
    fname: str,
    h5path: str,
    dtype: type = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Loads images stack from HDF as xarray

    Args:
        fname: full path to the file


    Returns:
    

    Raises:
    

    """

    file_ = h5py.File(fname, 'r')
    varr = da.array.from_array(file_[h5path+'ImagesStack'])
    varr = xr.DataArray(
        varr,
        dims=["height", "width", "frame"],
        coords=dict(
            height=np.arange(varr.shape[0]),
            width=np.arange(varr.shape[1]),
            frame=np.arange(varr.shape[2]) + 1, #Frame number start a 1 not 0
        ),
    )
    varr = varr.transpose('frame', 'height', 'width')
    
    if dtype != varr.dtype:
        if dtype == np.uint8:
            #varr = (varr - varr.values.min()) / (varr.values.max() - varr.values.min()) * 2**8 + 1
            bitsCount = file_[h5path+'ImagesStack'].attrs["BitsCount"]
            varr = varr / 2**bitsCount * 2**8

        varr = varr.astype(dtype)
    
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    
    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)
    
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]

    return varr, file_


def save_minian_to_doric(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    AC: xr.DataArray,
    S: xr.DataArray,
    fr: int,
    bits_count: int = 10,
    qt_format: int = 28,
    imagesStackUsername:str = "MiniAn",
    vname: str = "minian.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = 'Series1/Sensor1/',
    params_doric: Optional[dict] = None,
    params_source: Optional[dict] = None,
    saveimages: bool = True,
    saveresiduals: bool = True,
    savespikes: bool = True
) -> str:
    """
    Save MiniAn results to .doric file:
    MiniAnImages - `AC` representing cellular activities as computed by :func:`minian.cnmf.compute_AtC`
    MiniAnResidualImages - residule movie computed as the difference between `Y` and `AC`
    MiniAnSignals - `C` with coordinates from `A`
    MiniAnSpikes - `S`
    Since the CNMF algorithm contains various arbitrary scaling process, a normalizing 
    scalar is computed with least square using a subset of frames from `Y` and `AC` 
    such that their numerical values matches.
    
    Parameters
    ----------
    varr : xr.DataArray
        Input reference movie data. Should have dimensions ("frame", "height",
        "width"), and should only be chunked along "frame" dimension.
    Y : xr.DataArray
        Movie data representing input to CNMF algorithm. Should have dimensions
        ("frame", "height", "width"), and should only be chunked along "frame"
        dimension.
    A : xr.DataArray, optional
        Spatial footprints of cells. Only used if `AC` is `None`. By default
        `None`.
    C : xr.DataArray, optional
        Temporal activities of cells. Only used if `AC` is `None`. By default
        `None`.
    AC : xr.DataArray, optional
        Spatial-temporal activities of cells. Should have dimensions ("frame",
        "height", "width"), and should only be chunked along "frame" dimension.
        If `None` then both `A` and `C` should be supplied and
        :func:`minian.cnmf.compute_AtC` will be used to compute this variable.
        By default `None`.
    nfm_norm : int, optional
        Number of frames to randomly draw from `Y` and `AC` to compute the
        normalizing factor with least square. By default `None`.
    gain : float, optional
        A gain factor multiplied to `Y`. Useful to make the results visually
        brighter. By default `1.5`.
    vpath : str, optional
        Desired folder containing the resulting video. By default `"."`.
    vname : str, optional
        Desired name of the video. By default `"minian.mp4"`.
    Returns
    -------
    fname : str
        Absolute path of the resulting video.
    """
 
    ROISIGNALS = 'MiniAnROISignals'
    IMAGES = 'MiniAnImages'
    RESIDUALS = 'MiniAnResidualImages'
    SPIKES = 'MiniAnSpikes'

    res = Y - AC # residual images
    
    duration = Y.shape[0]
    time_ = np.arange(0, duration/fr, 1/fr, dtype='float64')
    
    print("generating ROI names")
    names = []
    usernames = []
    for i in range(len(C)):
        names.append('ROI'+str(i+1).zfill(4))
        usernames.append('ROI {}'.format(i+1))
    
    with h5py.File(vname, 'a') as f:
          
        # Check if MiniAn results already exist
        operationCount = ''
        if vpath in f:
            operations = [ name for name in f[vpath] if ROISIGNALS in name ]
            if len(operations) > 0:
                operationCount = str(len(operations))
                for operation in operations:
                    operationAttrs = load_attributes(f, vpath+operation)
                    if create_param_attribut_to_save(params_doric, params_source) == operationAttrs:
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
        save_roi_signals(C.values, A.values, time_, f, pathROIs+vdataset, attrs_add={"RangeMin": 0, "RangeMax": 0, "Unit": "AU"})
        print_group_path_for_DANSE(pathROIs+vdataset)
        if params_doric is not None and params_source is not None:
            save_attributes(create_param_attribut_to_save(params_doric, params_source), f, pathROIs)
        
        if saveimages:
            print("saving images")
            pathImages = vpath+IMAGES+operationCount+'/'
            save_images(AC.values, time_, f, pathImages+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            print_group_path_for_DANSE(pathImages+vdataset)
        if params_doric is not None and params_source is not None:
                save_attributes(create_param_attribut_to_save(params_doric, params_source, params_doric["Operations"] + "(Images)"), f, pathImages)
        
        if saveresiduals:
            print("saving residual images")
            pathResiduals = vpath+RESIDUALS+operationCount+'/'
            save_images(res.values, time_, f, pathResiduals+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            print_group_path_for_DANSE(pathResiduals+vdataset)
        if params_doric is not None and params_source is not None:
                save_attributes(create_param_attribut_to_save(params_doric, params_source,  params_doric["Operations"] + "(Residuals)"), f, pathResiduals)
            
        if savespikes:
            print("saving spikes")
            pathSpikes = vpath+SPIKES+operationCount+'/'
            save_signals(S.values > 0, time_, f, pathSpikes+vdataset, names, usernames, range_min=0, range_max=1)
            print_group_path_for_DANSE(pathSpikes+vdataset)
        if params_doric is not None and params_source is not None:
                save_attributes(create_param_attribut_to_save(params_doric, params_source), f, pathSpikes)
                
    print("Saved to {}".format(vname))

def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def round_down_to_odd(f):
    f = int(np.ceil(f))
    return f - 1 if f % 2 == 0 else f

def remove_keys_not_in_function_argument(
    input_dic:dict, func
) -> dict:
    '''
    This function while keep the keys in input_dic that are not use in the function func

    Returns
    -------
    The new dictionary new_dictionary
    '''
    func_arguments = inspect.getfullargspec(func).args
    new_dictionary = {key: input_dic[key] for key in input_dic if key in func_arguments}
    return new_dictionary
  


def set_advanced_parameters_for_func_params(
    param_func,
    advanced_parameters,
    func
    ):
    '''
    This function while change the value of the key from the dictionary param with the value of the key from the dictionary advanced_parameters.
    It while also remove the keys from the dictionary advanced_parameters that are not used in the function func
    
    Returns
    -------
    it while return the dictionary param_func with the new values and also the new advanced_parameters with only the used keys

    ''' 
    advanced_parameters = remove_keys_not_in_function_argument(advanced_parameters, func)
    for key, value in advanced_parameters.items():
        param_func[key] = value

    return [param_func, advanced_parameters]


def create_param_attribut_to_save(
    params_minian,
    params_source,
    operation_Name = None
    ):

    params_final = {}
    params_operation    = params_minian.copy()
    params_from_source  = params_source.copy()

    if operation_Name:
        params_operation["Operations"] = operation_Name

    params_final["Operations"] = params_from_source["Operations"] + " > " + params_operation["Operations"]
    del params_from_source["Operations"]

    for key in params_operation:
        if key == "Operations": continue

        if key == "AdvancedSettings":
            for funcName, funcValue in params_operation[key].items():
                if type(funcValue) is dict:
                    for variableName, variableValue in funcValue.items():
                        params_final["Advanced-" + funcName + "-" + variableName] = str(variableValue) if type(variableValue) is not str else '"' + variableValue + '"'
        else:
            params_final[key] = params_operation[key]

    for key in params_final.copy():
        if key == "Operations": continue
    
        params_final[params_operation["Operations"] + "-" + key] = params_final.pop(key)

    return {**params_final, **params_from_source}