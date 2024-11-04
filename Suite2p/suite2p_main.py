import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Optional
from tifffile import TiffWriter

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
    file_ = h5py.File(filePath, 'r')

    filePathTif: str = f"{suite2p_params.paths[defs.Parameters.Path.TMP_DIR]}/images.tif"
    with TiffWriter(filePathTif, bigtiff=True) as tifW:
        for I in range(suite2p_params.timeLength):
            for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
                tifW.write(file_[datapath][:, :, I], contiguous=True)

    output_ops = suite2p.run_s2p(ops = suite2p_params.ops, db = suite2p_params.db)

    data, driver, operation, series, sensor = suite2p_params.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")

    time_ = np.zeros((suite2p_params.ops['nplanes'], suite2p_params.timeLength))
    for i in range(suite2p_params.ops['nplanes']):
        time_[i,:] = np.array(file_[suite2p_params.paths[defs.Parameters.Path.H5PATH][i].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])[0:suite2p_params.timeLength]

    file_.close()

    save_suite2p_to_doric(
        output_ops = output_ops,
        time_ = time_,
        doricFileName = suite2p_params.paths[defs.Parameters.Path.FILEPATH],
        vpath = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
        series = series,
        sensor = sensor,
        params_doric = suite2p_params.params,
        params_source = params_source_data,
        )


def preview(suite2p_params: s2p_params.Suite2pParameters):
    print("Preview")


def save_suite2p_to_doric(
        output_ops: dict,
        time_: np.ndarray,
        doricFileName: str,
        vpath: str,
        series: str,
        sensor: str,
        params_doric: dict = {},
        params_source: dict = {}
):
    output_ops['save_path'] = Path("/".join(output_ops['data_path'])).joinpath(output_ops['save_folder'], "combined")

    iscell      = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool) #specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    stats       = np.load(Path(output_ops['save_path']).joinpath('stat.npy'), allow_pickle=True) #list of statistics computed for each cell
    f_cells     = np.load(Path(output_ops['save_path']).joinpath('F.npy')) #array of fluorescence traces (ROIs by timepoints)
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy')) # array of neuropil fluorescence traces (ROIs by timepoints)
    spks        = np.load(Path(output_ops['save_path']).joinpath('spks.npy')) #array of deconvolved traces (ROIs by timepoints)

    stats   = stats[iscell]
    f_cells = f_cells[iscell, :]
    spks    = spks[iscell, :]
     
    n_cells = len(stats)
    Ly = output_ops["Ly"]
    Lx = output_ops["Lx"]

    #FootPrint to use to save the ROIs contour in doric
    footPrint = np.zeros((n_cells, Ly, Lx))
    for i, stat in enumerate(stats):
        footPrint[i, stat['ypix'] - int(stat['iplane']/2) * Ly, stat['xpix'] - (stat['iplane'] % 2) * Lx] = 1

    print(s2p_defs.Messages.ROI_NAMES, flush = True)
    ids           = [i + 1 for i in range(n_cells)]
    dataset_names = [defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4)) for id_ in ids]
    usernames     = [defs.DoricFile.Dataset.ROI.format(id_) for id_ in ids]

    with h5py.File(doricFileName, 'a') as f:
        # Check if Suite2p results already exist
        operationCount = utils.operation_count(vpath, f, s2p_defs.DoricFile.Group.ROISIGNALS, params_doric, params_source)

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operationCount

        print(s2p_defs.Messages.SAVING_ROIS, flush=True)
        rois_grouppath   = f"{vpath}/{s2p_defs.DoricFile.Group.ROISIGNALS+operationCount}"
        rois_seriespath  = f"{rois_grouppath}/{series}"
        attrs = {"Unit": "AU"}
        save_roi_signals(f_cells, footPrint, time_, f, rois_seriespath, sensor,
                         ids            = ids,
                         dataset_names  = dataset_names,
                         usernames      = usernames,
                         attrs          = attrs,
                         planeID        = [stat['iplane'] + 1 for stat in stats])
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, rois_grouppath)
        for planeSensor in f[rois_seriespath].keys():
            utils.print_group_path_for_DANSE(f"{rois_seriespath}/{planeSensor}")

        print(s2p_defs.Messages.SAVING_SPIKES, flush=True)
        spikes_grouppath   = f"{vpath}/{s2p_defs.DoricFile.Group.SPIKES+operationCount}"
        spikes_seriespath  = f"{spikes_grouppath}/{series}"
        attrs = {"Unit": "AU"}
        spikes = correctSpikesValues(spks, f_cells)
        save_spikes(spikes, time_, f, spikes_seriespath , sensor,
                            dataset_names  = dataset_names,
                            usernames      = usernames,
                            attrs          = attrs,
                            planeID        = [stat['iplane'] + 1 for stat in stats])
        utils.save_attributes(utils.merge_params(params_doric, params_source), f, spikes_grouppath)
        for planeSensor in f[spikes_seriespath].keys():
            utils.print_group_path_for_DANSE(f"{spikes_seriespath}/{planeSensor}")


def correctSpikesValues(spks:np.ndarray, f_cells:np.ndarray)->np.ndarray:
        spikes: np.ndarray = (spks/spks) * f_cells
        spikes[np.isnan(spikes)] = 0

        return spikes


def save_roi_signals(
    signals: np.ndarray,
    footprints: np.ndarray,
    time_: np.ndarray,
    f: h5py.File,
    seriesPath: str,
    sensor: str,
    ids: list[int] = [],
    dataset_names: list[str] = [],
    usernames: list[str] = [],
    attrs: Optional[dict] = {},
    planeID: list[int] = []
    ):

    """
    Saves ROI signals, time vector, and ROI coordinates.
    """
    seriesPath = utils.clean_path(seriesPath)

    for i, footprint in enumerate(footprints):
        id_ = ids[i] if ids else i + 1

        roi_attrs = {
            defs.DoricFile.Attribute.ROI.ID :           id_,
            defs.DoricFile.Attribute.ROI.SHAPE :        0,
            defs.DoricFile.Attribute.ROI.COORDS :       utils.footprint_to_coords(footprint),
            defs.DoricFile.Attribute.Dataset.NAME :     usernames[i] if usernames else defs.DoricFile.Dataset.ROI.format(id_),
            defs.DoricFile.Attribute.Dataset.USERNAME : usernames[i] if usernames else defs.DoricFile.Dataset.ROI.format(id_),
            defs.DoricFile.Attribute.Dataset.PLANE_ID : planeID[i]
        }

        if attrs:
            roi_attrs = {**roi_attrs, **attrs}

        dataset_name = dataset_names[i] if dataset_names else defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4))

        sensorPath = f"{seriesPath}/{sensor}-P{planeID[i]}"
        timePath   = f"{sensorPath}/{defs.DoricFile.Dataset.TIME}"

        utils.save_signal(signals[i], f, f"{sensorPath}/{dataset_name}", roi_attrs)
        if timePath not in f:
            utils.save_signal(time_[planeID[i] - 1], f, timePath)


def save_spikes(
        spikes: np.ndarray,
        time_: np.ndarray,
        f:  h5py.File,
        seriesPath: str,
        sensor: str,
        dataset_names: list[str],
        usernames: list[str],
        attrs: dict,
        planeID: list[int]
):
    seriesPath = utils.clean_path(seriesPath)

    for i, spike in enumerate(spikes):
        sensorPath = f"{seriesPath}/{sensor}-P{planeID[i]}"
        timePath   = f"{sensorPath}/{defs.DoricFile.Dataset.TIME}"

        attrs[defs.DoricFile.Attribute.Dataset.USERNAME] = usernames[i]
        utils.save_signal(spike, f, f"{sensorPath}/{dataset_names[i]}", attrs)
        if timePath not in f:
            utils.save_signal(time_[planeID[i] - 1], f, timePath)
