#!/usr/bin/env python
# coding: utf-8

# Import miscellaneous and utilities librarys
import os
import sys
import h5py
import numpy as np
from tifffile import imread
from tifffile import imwrite
from PIL import Image

# Import for CaimAn lib
from summary_images import correlation_pnr

for arg in sys.argv[1:]:
    exec(arg)


if __name__ == "__main__":
    with h5py.File(kwargs["fname"], 'r') as f:
        images = np.array(f[kwargs['h5path']+'ImagesStack'])


    images = images.transpose(2, 0, 1)
    h5path_list = kwargs['h5path'].split('/')

    cr, pnr = correlation_pnr(images,swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0


    fname_Correlation_tif = os.path.splitext(kwargs["fnamecorr"])[0]+ '.tif'
    fname_PNR_tif = os.path.splitext(kwargs["fnamePNR"])[0]+ '.tif'
    
    print("Write Correlation and PNR tiff images...", flush=True)
    imwrite(fname_Correlation_tif, cr)
    imwrite(fname_PNR_tif, pnr)


