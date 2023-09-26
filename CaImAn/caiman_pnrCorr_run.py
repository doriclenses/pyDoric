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

video_start_frame   = params_doric["VideoStartFrame"]
video_stop_frame    = params_doric["VideoStopFrame"]

if __name__ == "__main__":

    with h5py.File(kwargs["fname"], 'r') as f:
        try:
            images = np.array(f[kwargs['h5path']+'ImageStack'])
        except:
            images = np.array(f[kwargs['h5path']+'ImagesStack'])

    images = images[:, :, (video_start_frame-1):video_stop_frame]
    images = images[:, :, ::params_doric['TemporalDownsample']]

    images = images.transpose(2, 0, 1)

    cr, pnr = correlation_pnr(images, swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0

    print("Write Correlation and PNR tiff images...", flush=True)
    imwrite(kwargs["fnamecorr"], cr)
    imwrite(kwargs["fnamePNR"], pnr)


