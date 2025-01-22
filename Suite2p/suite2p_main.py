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
        output_ops      = output_ops,
        time_           = time_,
        doric_file_name = suite2p_params.paths[defs.Parameters.Preview.FILEPATH],
        vpath           = f"{defs.DoricFile.Group.DATA_PROCESSED}/{driver}",
        series          = series,
        sensor          = sensor,
        params_doric    = suite2p_params.params,
        params_source   = params_source_data,
        plane_IDs       = [int(datapath[-1]) for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]] if not suite2p_params.is_microscope else [-1]
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
    params_source: dict = {},
    plane_IDs: list[int] = []
):
    if len(plane_IDs) > 1:
        output_ops['save_path'] = Path("/".join(output_ops['data_path'])).joinpath(output_ops['save_folder'], "combined")
    else:
        output_ops['save_path'] = Path("/".join(output_ops['data_path'])).joinpath(output_ops['save_folder'], "plane0")

    iscell      = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool) #specifies whether an ROI is a cell
    stats       = np.load(Path(output_ops['save_path']).joinpath('stat.npy'), allow_pickle=True) #list of statistics computed for each cell
    f_cells     = np.load(Path(output_ops['save_path']).joinpath('F.npy')) #array of fluorescence traces (ROIs by timepoints)
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy')) # array of neuropil fluorescence traces (ROIs by timepoints)
    spks        = np.load(Path(output_ops['save_path']).joinpath('spks.npy')) #array of deconvolved traces (ROIs by timepoints)
    ops         = np.load(Path(output_ops['save_path']).joinpath('ops.npy'), allow_pickle=True).item()
  
    n_cells = len(stats)
    Ly = output_ops["Ly"]
    Lx = output_ops["Lx"]
    spikes = correct_spikes_values(spks, f_cells)

    if len(plane_IDs) == 1:
        for stat in stats:
            stat['iplane'] = 0
    
    #FootPrint to use to save the ROIs contour in doric
    footprints = np.zeros((n_cells, Ly, Lx))
    for i, stat in enumerate(stats):
        footprints[i, stat['ypix'] - int(stat['iplane']/2) * Ly, stat['xpix'] - (stat['iplane'] % 2) * Lx] = 1

    print(s2p_defs.Messages.ROI_NAMES, flush = True)
    ids           = [i + 1 for i in range(n_cells)]
    dataset_names = [defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4)) for id_ in ids]
    usernames     = [defs.DoricFile.Dataset.ROI.format(id_) for id_ in ids]

    file_ = h5py.File(doric_file_name, 'w')

    # Check if Suite2p results already exist
    mean_image      = split_by_plane(ops['meanImg'], len(plane_IDs))
    median_filter   = split_by_plane(ops['meanImgE'], len(plane_IDs))
    correlation_map = split_by_plane(ops['Vcorr'], len(plane_IDs))
    max_projection  = split_by_plane(ops['max_proj'], len(plane_IDs))

    print(s2p_defs.Messages.SAVING_IMAGES, flush=True)
    height, width , _ = mean_image.shape
    file_.create_dataset(s2p_defs.Preview.Dataset.MEAN_IMAGE, data = mean_image, dtype = "float64", chunks = (height, width , 1), maxshape = (height, width, None))
    file_.create_dataset(s2p_defs.Preview.Dataset.MEDIAN_FILTER_MEAN, data = median_filter, dtype = "float64", chunks = (height, width , 1), maxshape = (height, width, None))
    file_.create_dataset(s2p_defs.Preview.Dataset.CORRELATION_MAP, data = correlation_map, dtype = "float64", chunks = (height, width , 1), maxshape = (height, width, None))
    file_.create_dataset(s2p_defs.Preview.Dataset.MAX_PROJECTION, data = max_projection, dtype = "float64", chunks = (height, width , 1), maxshape = (height, width, None))

    print(f"{s2p_defs.Messages.SAVING_ROIS} and {s2p_defs.Messages.SAVING_SPIKES}", flush=True)
    for plane_index, plane_ID in enumerate(plane_IDs):
        cell_indexs = [i for i, stat in enumerate(stats) if stat["iplane"] == plane_index]

        attrs = {defs.DoricFile.Attribute.Dataset.PLANE_ID: plane_ID}
        
        utils.save_roi_signals(signals       = f_cells[cell_indexs, :],
                               footprints    = footprints[cell_indexs, :, :],
                               time_         = time_[plane_index],
                               file_         = file_,
                               path          = f"{s2p_defs.Preview.Group.ROISIGNALS}/P{plane_ID}",
                               ids           = [ids[i] for i in cell_indexs],
                               dataset_names = [dataset_names[i] for i in cell_indexs],
                               usernames     = [usernames[i] for i in cell_indexs],
                               other_attrs   = [{"isCell": int(iscell[i])} for i in cell_indexs],
                               common_attrs  = attrs)
            
        utils.save_signals(signals        = spikes[cell_indexs, :],
                           time_         = time_[plane_index],
                           file_         = file_,
                           path          = f"{s2p_defs.Preview.Group.SPIKES}/P{plane_ID}",
                           dataset_names = [dataset_names[i] for i in cell_indexs],
                           usernames     = [usernames[i] for i in cell_indexs],
                           attrs         = attrs)

    file_.close()


def correct_spikes_values(spks:np.ndarray, f_cells:np.ndarray) -> np.ndarray:
    spikes = np.zeros(spks.shape)
    spikes[spks > 0] = f_cells[spks > 0]

    return spikes

def split_by_plane(suite2p_image: np.ndarray, plane_count: int) -> np.ndarray:
    hight, width = suite2p_image.shape
    
    hight_split = np.ceil(plane_count/2)
    width_split = 1 if plane_count == 1 else 2
    
    images = []
    for n in range(plane_count):
        image = suite2p_image[int(n/2)*int(hight/hight_split) : (int(n/2) + 1)*int(hight/hight_split),
                              n%2*int(width/width_split) : (n%2 + 1)*int(width/width_split)]
        images.append(image)

    return np.moveaxis(np.array(images), 0, -1)