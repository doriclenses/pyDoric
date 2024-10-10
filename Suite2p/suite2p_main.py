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

    timeIdx = -1
    for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
        _, _, nTime = doricFile[datapath].shape
        
        if timeIdx == -1:
            timeIdx = nTime
        else:
            timeIdx = min(timeIdx, nTime)

    filePathTif: str = suite2p_params.paths[defs.Parameters.Path.TMP_DIR] + "\\images.tif"
    with TiffWriter(filePathTif, bigtiff=True) as tifW:
        for I in range(timeIdx):
            for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
                tifW.write(doricFile[datapath][:, :, I], contiguous=True)

    doricFile.close()

    output_ops = suite2p.run_s2p(ops = suite2p_params.ops, db = suite2p_params.db)

    save_suite2p_to_doric(output_ops)

def preview(suite2p_params: s2p_params.Suite2pParameters):
    print("hello preview")

def save_suite2p_to_doric(
        output_ops: dict
):

    # Figure Style settings
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.subplot.wspace': .01,
        'figure.subplot.hspace': .01,
        'figure.figsize': (18, 13),
        'ytick.major.left': True,
    })
    jet = mpl.cm.get_cmap('jet')
    jet.set_bad(color='k')

    plt.subplot(1, 4, 1)
    plt.imshow(output_ops['refImg'], cmap='gray', )
    plt.title("Reference Image for Registration");

    # maximum of recording over time
    plt.subplot(1, 4, 2)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection");

    plt.subplot(1, 4, 3)
    plt.imshow(output_ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")

    plt.subplot(1, 4, 4)
    plt.imshow(output_ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image");

    plt.savefig("refImages")

    plt.figure(figsize=(18,8))

    plt.subplot(4,1,1)
    plt.plot(output_ops['yoff'][:1000])
    plt.ylabel('rigid y-offsets')

    plt.subplot(4,1,2)
    plt.plot(output_ops['xoff'][:1000])
    plt.ylabel('rigid x-offsets')

    plt.subplot(4,1,3)
    plt.plot(output_ops['yoff1'][:1000])
    plt.ylabel('nonrigid y-offsets')

    plt.subplot(4,1,4)
    plt.plot(output_ops['xoff1'][:1000])
    plt.ylabel('nonrigid x-offsets')
    plt.xlabel('frames')

    plt.savefig("Graphs")

    iscell = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(int) #specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    
    stats_file = Path(output_ops['save_path']).joinpath('stat.npy')
    stats = np.load(stats_file, allow_pickle=True) #list of statistics computed for each cell

    n_cells = len(stats)

    h = np.random.rand(n_cells)
    
    Ly, Lx = output_ops["Ly"], output_ops["Lx"]

    hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)

    for i, stat in enumerate(stats):
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
        hsvs[iscell[i], ypix, xpix, 0] = h[i]
        hsvs[iscell[i], ypix, xpix, 1] = 1
        hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()

    from colorsys import hsv_to_rgb
    rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

    plt.figure(figsize=(18,18))
    plt.subplot(3, 1, 1)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(3, 1, 2)
    plt.imshow(rgbs[1])
    plt.title("All Cell ROIs")

    plt.subplot(3, 1, 3)
    plt.imshow(rgbs[0])
    plt.title("All non-Cell ROIs");

    plt.savefig("ROIS")

    f_cells = np.load(Path(output_ops['save_path']).joinpath('F.npy')) #array of fluorescence traces (ROIs by timepoints)
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy')) # array of neuropil fluorescence traces (ROIs by timepoints)
    spks = np.load(Path(output_ops['save_path']).joinpath('spks.npy')) #array of deconvolved traces (ROIs by timepoints)

    #FootPrint to use to save the ROIs contour in doric
    footPrint = np.zeros((n_cells, Ly, Lx))
    for i, stat in enumerate(stats):
        footPrint[i, stat['ypix'], stat['xpix']] = 1
