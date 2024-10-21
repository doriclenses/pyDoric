import os, requests
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from tifffile import imwrite, TiffWriter, TiffFile

sys.path.append("..")
import utilities as utils
import definitions as defs
import suite2p_parameters as s2p_params
import suite2p_definitions as s2p_defs

import suite2p

# Import for PyInstaller
from multiprocessing import freeze_support
freeze_support()

def main(suite2p_params: s2p_params.Suite2pParameters):

    """
    Suite2p algorithm
    """
    filePath: str = suite2p_params.paths[defs.Parameters.Path.FILEPATH]
    doricFile = h5py.File(filePath, 'r')

    nTime = -1
    for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
        _, _, timeCount = doricFile[datapath].shape
        
        if nTime == -1:
            nTime = timeCount
        else:
            nTime = min(nTime, timeCount)

    filePathTif: str = suite2p_params.paths[defs.Parameters.Path.TMP_DIR] + "\\images.tif"
    with TiffWriter(filePathTif, bigtiff=True) as tifW:
        for I in range(nTime):
            for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
                tifW.write(doricFile[datapath][:, :, I], contiguous=True)


    output_ops = suite2p.run_s2p(ops = suite2p_params.ops, db = suite2p_params.db)
    
    time_ = np.zeros((suite2p_params.ops['nplanes'], nTime))
    for i in range(suite2p_params.ops['nplanes']):
        time_[i,:] = np.array(doricFile[suite2p_params.paths[defs.Parameters.Path.H5PATH][i].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])[0:nTime]
    
    doricFile.close()

    data, driver, operation, series, sensor = suite2p_params.get_h5path_names()

    print("....Printing split path....", data, driver, operation, series, sensor)

    save_suite2p_to_doric(
        output_ops = output_ops,
        time_ = time_,
        doricFileName = suite2p_params.paths[defs.Parameters.Path.FILEPATH],
        vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
        vdataset = f"{series}/{sensor}"
        )

def preview(suite2p_params: s2p_params.Suite2pParameters):
    print("hello preview")

def save_suite2p_to_doric(
        output_ops: dict,
        time_: np.ndarray,
        doricFileName: str,
        vpath: str,
        vdataset: str
):  
    output_ops['save_path'] = Path("/".join(output_ops['data_path'])).joinpath(output_ops['save_folder'], "combined")
    
    iscell = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(int) #specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    stats_file = Path(output_ops['save_path']).joinpath('stat.npy')
    stats = np.load(stats_file, allow_pickle=True) #list of statistics computed for each cell

    n_cells = len(stats)

    f_cells = np.load(Path(output_ops['save_path']).joinpath('F.npy')) #array of fluorescence traces (ROIs by timepoints)
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy')) # array of neuropil fluorescence traces (ROIs by timepoints)
    spks = np.load(Path(output_ops['save_path']).joinpath('spks.npy')) #array of deconvolved traces (ROIs by timepoints)

    Ly = output_ops["Ly"]
    Lx = output_ops["Lx"]

    #FootPrint to use to save the ROIs contour in doric
    footPrint = np.zeros((n_cells, Ly, Lx))
    for i, stat in enumerate(stats):
        footPrint[i, stat['ypix'] - int(stat['iplane']/2) * Ly, stat['xpix'] - (stat['iplane'] % 2) * Lx] = 1

    with h5py.File(doricFileName, 'a') as f:
        print("Saving")

        print("Saving ROIs", flush=True)
        rois_grouppath = f"{vpath}/ROISignals"
        rois_datapath  = f"{rois_grouppath}/{vdataset}"
        
        save_roi_signals(f_cells, footPrint, time_, f, rois_datapath, PlaneID=[stat['iplane'] + 1 for stat in stats])
        utils.print_group_path_for_DANSE(rois_datapath)


def save_roi_signals(
    signals: np.ndarray,
    footprints: np.ndarray,
    time_: np.ndarray,
    f: h5py.File,
    path: str,
    ids: List[int] = None,
    dataset_names: List[int] = None,
    usernames: List[int] = None,
    attrs: Optional[dict] = None,
    PlaneID: List[int] = None
    ):

    """
    Saves ROI signals, time vector, and ROI coordinates.
    Parameters
    ----------
    signals : np.ndarray
        2D array of signals, with shape (n_ROI, time).
    footprints:
        3D array of spatial cell footprints with shape (n_ROI, height, width)
    time_ : np.array
        1D vector of timestamps
    f : h5py.File
        Opened HDF file where the information should be saved
    path  : str
        Group path in the HDF file
    bit_count : int
        Bits depth of images

    """
    path = utils.clean_path(path)

    for i, footprint in enumerate(footprints):
        id_ = ids[i] if ids is not None else i + 1

        roi_attrs = {
            defs.DoricFile.Attribute.ROI.ID:           id_,
            defs.DoricFile.Attribute.ROI.SHAPE:        0,
            defs.DoricFile.Attribute.ROI.COORDS:       utils.footprint_to_coords(footprint),
            defs.DoricFile.Attribute.Dataset.NAME:     usernames[i] if usernames is not None else defs.DoricFile.Dataset.ROI.format(id_),
            defs.DoricFile.Attribute.Dataset.USERNAME: usernames[i] if usernames is not None else defs.DoricFile.Dataset.ROI.format(id_)
        }

        roi_attrs["PlaneID"] = PlaneID[i]

        if attrs is not None:
            roi_attrs = {**roi_attrs, **attrs}

        dataset_name = dataset_names[i] if dataset_names is not None else defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4))
        
        utils.save_signal(signals[i], f, f"{path}-P{PlaneID[i]}/{dataset_name}", roi_attrs)
        timePath = f"{path}-P{PlaneID[i]}/{defs.DoricFile.Dataset.TIME}"
        
        if timePath not in f:
            utils.save_signal(time_[i], f, timePath)