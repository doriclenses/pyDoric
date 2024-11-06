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
    file_path: str = suite2p_params.paths[defs.Parameters.Path.FILEPATH]
    file_ = h5py.File(file_path, 'r')

    tif_file_path: str = f"{suite2p_params.paths[defs.Parameters.Path.TMP_DIR]}/images.tif"
    with TiffWriter(tif_file_path, bigtiff=True) as tif_file:
        for I in range(suite2p_params.time_length):
            for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
                tif_file.write(file_[datapath][:, :, I], contiguous=True)

    output_ops = suite2p.run_s2p(ops = suite2p_params.ops, db = suite2p_params.db)

    data, driver, operation, series, sensor = suite2p_params.get_h5path_names()
    params_source_data = utils.load_attributes(file_, f"{data}/{driver}/{operation}")

    time_ = np.zeros((suite2p_params.ops['nplanes'], suite2p_params.time_length))
    for i in range(suite2p_params.ops['nplanes']):
        time_[i,:] = np.array(file_[suite2p_params.paths[defs.Parameters.Path.H5PATH][i].replace(defs.DoricFile.Dataset.IMAGE_STACK, defs.DoricFile.Dataset.TIME)])[0:suite2p_params.time_length]

    file_.close()

    save_suite2p_to_doric(
        output_ops = output_ops,
        time_ = time_,
        doric_file_name = suite2p_params.paths[defs.Parameters.Path.FILEPATH],
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
        doric_file_name: str,
        vpath: str,
        series: str,
        sensor: str,
        params_doric: dict = {},
        params_source: dict = {}
):
    output_ops['save_path'] = Path("/".join(output_ops['data_path'])).joinpath(output_ops['save_folder'], "combined")

    iscell      = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool) #specifies whether an ROI is a cell
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
    footprint = np.zeros((n_cells, Ly, Lx))
    for i, stat in enumerate(stats):
        footprint[i, stat['ypix'] - int(stat['iplane']/2) * Ly, stat['xpix'] - (stat['iplane'] % 2) * Lx] = 1

    print(s2p_defs.Messages.ROI_NAMES, flush = True)
    ids           = [i + 1 for i in range(n_cells)]
    dataset_names = [defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4)) for id_ in ids]
    usernames     = [defs.DoricFile.Dataset.ROI.format(id_) for id_ in ids]

    with h5py.File(doric_file_name, 'a') as file_:
        # Check if Suite2p results already exist
        operation_count = utils.operation_count(vpath, file_, s2p_defs.DoricFile.Group.ROISIGNALS, params_doric, params_source)

        params_doric[defs.DoricFile.Attribute.Group.OPERATIONS] += operation_count

        print(s2p_defs.Messages.SAVING_ROIS, flush=True)
        rois_grouppath   = f"{vpath}/{s2p_defs.DoricFile.Group.ROISIGNALS + operation_count}"
        rois_seriespath  = f"{rois_grouppath}/{series}"
        attrs = {"Unit": "AU"}
        save_roi_signals(f_cells, footprint, time_, file_, rois_seriespath, sensor,
                         ids            = ids,
                         dataset_names  = dataset_names,
                         usernames      = usernames,
                         attrs          = attrs,
                         planeID        = [stat['iplane'] + 1 for stat in stats])
        utils.save_attributes(utils.merge_params(params_doric, params_source), file_, rois_grouppath)
        for plane_sensor in file_[rois_seriespath].keys():
            utils.print_group_path_for_DANSE(f"{rois_seriespath}/{plane_sensor}")

        print(s2p_defs.Messages.SAVING_SPIKES, flush=True)
        spikes_grouppath   = f"{vpath}/{s2p_defs.DoricFile.Group.SPIKES+operation_count}"
        spikes_seriespath  = f"{spikes_grouppath}/{series}"
        attrs = {"Unit": "AU"}
        spikes = correctSpikesValues(spks, f_cells)
        save_spikes(spikes, time_, file_, spikes_seriespath , sensor,
                            dataset_names  = dataset_names,
                            usernames      = usernames,
                            attrs          = attrs,
                            plane_ID        = [stat['iplane'] + 1 for stat in stats])
        utils.save_attributes(utils.merge_params(params_doric, params_source), file_, spikes_grouppath)
        for plane_sensor in file_[spikes_seriespath].keys():
            utils.print_group_path_for_DANSE(f"{spikes_seriespath}/{plane_sensor}")


def correctSpikesValues(spks:np.ndarray, f_cells:np.ndarray)->np.ndarray:
        spikes: np.ndarray = (spks/spks) * f_cells
        spikes[np.isnan(spikes)] = 0

        return spikes


def save_roi_signals(
    signals: np.ndarray,
    footprints: np.ndarray,
    time_: np.ndarray,
    file_: h5py.File,
    series_path: str,
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
    series_path = utils.clean_path(series_path)

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

        sensor_path = f"{series_path}/{sensor}-P{planeID[i]}"
        time_path   = f"{sensor_path}/{defs.DoricFile.Dataset.TIME}"

        utils.save_signal(signals[i], file_, f"{sensor_path}/{dataset_name}", roi_attrs)
        if time_path not in file_:
            utils.save_signal(time_[planeID[i] - 1], file_, time_path)


def save_spikes(
        spikes: np.ndarray,
        time_: np.ndarray,
        file_:  h5py.File,
        series_path: str,
        sensor: str,
        dataset_names: list[str],
        usernames: list[str],
        attrs: dict,
        plane_ID: list[int]
):
    series_path = utils.clean_path(series_path)

    for i, spike in enumerate(spikes):
        sensorPath = f"{series_path}/{sensor}-P{plane_ID[i]}"
        timePath   = f"{sensorPath}/{defs.DoricFile.Dataset.TIME}"

        attrs[defs.DoricFile.Attribute.Dataset.USERNAME] = usernames[i]
        utils.save_signal(spike, file_, f"{sensorPath}/{dataset_names[i]}", attrs)
        if timePath not in file_:
            utils.save_signal(time_[plane_ID[i] - 1], file_, timePath)
