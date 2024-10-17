import os, requests
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import h5py
import typing
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
 
    time_ = np.array(doricFile[suite2p_params.paths[defs.Parameters.Path.H5PATH][0].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])[0:(nTime-1)]
    
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
        time_: np.array,
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
        footPrint[i, stat['ypix'], stat['xpix'] - stat['iplane']*Lx] = 1

    with h5py.File(doricFileName, 'a') as f:
        print("Saving")

        print("Saving ROIs", flush=True)
        rois_grouppath = f"{vpath}/ROISignals"
        rois_datapath  = f"{rois_grouppath}/{vdataset}"
        
        utils.save_roi_signals(f_cells, footPrint, time_, f, rois_datapath, PlaneID=[stat['iplane'] + 1 for stat in stats])
        utils.print_group_path_for_DANSE(rois_datapath)