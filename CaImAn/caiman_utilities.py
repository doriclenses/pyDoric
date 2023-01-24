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
    save_images
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
    imagesStackUsername: str = "caimanImagesStack",
    vname: str = "caiman.doric",
    vpath: str = "DataProcessed/MicroscopeDriver-1stGen1C/",
    vdataset: str = 'Series1/Sensor1/',
    attrs: Optional[dict] = None,
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
                    if attrs == operationAttrs:
                        if(len(operation) == len(ROISIGNALS)):
                            operationCount = ''
                        else:
                            operationCount = operation[-1]

                        break


        if vpath[-1] != '/':
            vpath += '/'
        
        if vdataset[-1] != '/':
            vdataset += '/'
        
        print("saving ROI signals")
        pathROIs = vpath+ROISIGNALS+operationCount+'/'
        save_roi_signals(C, A, time_, f, pathROIs+vdataset, bits_count=bits_count)
        if attrs is not None:
            save_attributes(attrs, f, pathROIs)
        
        if saveimages:
            print("saving images")
            pathImages = vpath+IMAGES+operationCount+'/'
            save_images(AC, time_, f, pathImages+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            if attrs is not None:
                save_attributes(attrs, f, pathImages)
        
        if saveresiduals:
            print("saving residual images")
            pathResiduals = vpath+RESIDUALS+operationCount+'/'
            save_images(res, time_, f, pathResiduals+vdataset, bits_count=bits_count, qt_format=qt_format, username=imagesStackUsername)
            if attrs is not None:
                save_attributes(attrs, f, pathResiduals)
            
        if savespikes:
            print("saving spikes")
            pathSpikes = vpath+SPIKES+operationCount+'/'
            save_signals(S > 0, time_, f, pathSpikes+vdataset, names, roiUsernames, range_min=0, range_max=1)
            if attrs is not None:
                save_attributes(attrs, f, pathSpikes)
        
    print("Saved to {}".format(vname))
