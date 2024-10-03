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

    with TiffWriter(filePath + ".tif", bigtiff=True) as tifW:
        for I in range(timeIdx):
            for datapath in suite2p_params.paths[defs.Parameters.Path.H5PATH]:
                tifW.write(doricFile[datapath][:, :, I], contiguous=True)

    doricFile.close()

    with TiffFile(filePath + ".tif") as tifF:
        n_time= len(tifF.pages)
        Ly, Lx = tifF.pages[0].shape

    print(Lx, Ly, n_time)

    output_ops = suite2p.run_s2p(ops = suite2p_params.ops, db = suite2p_params.db)


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



    # #@title Run cell to look at registered frames
    # from ipywidgets import interact, interactive, fixed, interact_manual
    # import ipywidgets as widgets
    # from suite2p.io import BinaryFile

    # widget = widgets.IntSlider(
    #     value=7,
    #     min=0,
    #     max=10,
    #     step=1,
    #     description='Test:',
    #     disabled=False,
    #     continuous_update=False,
    #     orientation='horizontal',
    #     readout=True,
    #     readout_format='d'
    # )


    # def plot_frame(t):
    #     with BinaryFile(Ly=output_ops['Ly'],
    #                 Lx=output_ops['Lx'],
    #                 filename=output_ops['reg_file']) as f:
    #         plt.imshow(f[t])

    # interact(plot_frame, t=(0, output_ops['nframes']- 1, 1)); # zero-indexed so have to subtract 1



    stats_file = Path(output_ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(int)
    stats = np.load(stats_file, allow_pickle=True)
    print(stats[0].keys())


    n_cells = len(stats)

    h = np.random.rand(n_cells)
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

    # plt.tight_layout()
    plt.savefig("ROIS")



    f_cells = np.load(Path(output_ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(output_ops['save_path']).joinpath('spks.npy'))
    f_cells.shape, f_neuropils.shape, spks.shape


    plt.figure(figsize=(20,20))
    plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92);
    rois = np.arange(len(f_cells))[::200]
    for i, roi in enumerate(rois):
        plt.subplot(len(rois), 1, i+1, )
        f = f_cells[roi]
        f_neu = f_neuropils[roi]
        sp = spks[roi]
        # Adjust spks range to match range of fluroescence traces
        fmax = np.maximum(f.max(), f_neu.max())
        fmin = np.minimum(f.min(), f_neu.min())
        frange = fmax - fmin 
        sp /= sp.max()
        sp *= frange
        plt.plot(f, label="Cell Fluorescence")
        plt.plot(f_neu, label="Neuropil Fluorescence")
        plt.plot(sp + fmin, label="Deconvolved")
        plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
        plt.ylabel(f"ROI {roi}", rotation=0)
        plt.xlabel("frame")
        if i == 0:
            plt.legend(bbox_to_anchor=(0.93, 2))

    plt.savefig("Fluorescence Traces")

def preview(suite2p_params: s2p_params.Suite2pParameters):
    print("hello preview")