#!/usr/bin/env python
# coding: utf-8

# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import numpy as np
from tifffile import imread
from tifffile import imwrite

# Import for CaimAn lib
from summary_images import correlation_pnr



if __name__ == "__main__":
    with h5py.File(kwargs["fname"], 'r') as f:
        images = np.array(f[kwargs['h5path']+'ImagesStack'])


    images = images.transpose(2, 0, 1)
    h5path_list = kwargs['h5path'].split('/')
    fname_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif'


    print("Write image in tiff...", flush=True)
    imwrite(fname_tif, images)

    cr, pnr = correlation_pnr(images,swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0


    cr_th = cr
    #cr_th[cr_th < params_doric['CorrelationThreshold']] = 0

    pnr_th = pnr
    #pnr_th[pnr_th < params_doric['PNRThreshold']] = 0


    fname_Correlation_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + 'correlation.tif'
    fname_PNR_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + 'PNR.tif'

    print("Write Correlation and PNR tiff images...", flush=True)
    imwrite(fname_Correlation_tif, cr_th)
    imwrite(fname_PNR_tif, pnr_th)

