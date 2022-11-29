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
    #fname_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + '.tif'
    fname_tif = os.path.splitext(kwargs["fname"])[0] + '_' + 'tmp' + '.tif'


    print("Write image in tiff...", flush=True)
    imwrite(fname_tif, images)

    cr, pnr = correlation_pnr(images,swap_dim = False)

    cr[np.isnan(cr)] = 0
    pnr[np.isnan(pnr)] = 0


    cr_th = cr
    #cr_th[cr_th < params_doric['CorrelationThreshold']] = 0

    pnr_th = pnr
    #pnr_th[pnr_th < params_doric['PNRThreshold']] = 0


    #fname_Correlation_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + 'correlation.tif'
    #fname_PNR_tif = os.path.splitext(kwargs["fname"])[0] + '_' + h5path_list[3] + h5path_list[4] + h5path_list[5] + 'PNR.tif'
    
    fname_Correlation_tif = os.path.splitext(kwargs["fnamecorr"])[0]+ '.tif'
    fname_PNR_tif = os.path.splitext(kwargs["fnamePNR"])[0]+ '.tif'
    
    print("Write Correlation and PNR tiff images...", flush=True)
    #imwrite(fname_Correlation_tif, cr_th)
    #imwrite(fname_PNR_tif, pnr_th)
    
    # cr_th_image = Image.fromarray(cr_th)
    # pnr_th_image = Image.fromarray(pnr_th)
    
    # print(cr_th_image)
    # cr_th_image = cr_th_image.convert('RGB')
    # pnr_th_image = pnr_th_image.convert('RGB')
    
    
    # cr_th_image.save(fname_Correlation_tif)
    # pnr_th_image.save(fname_PNR_tif)
    
    # imwrite(fname_Correlation_tif, np.dstack((cr_th,cr_th,cr_th)))
    # imwrite(fname_PNR_tif, np.dstack((pnr_th, pnr_th ,pnr_th)))

    imwrite(fname_Correlation_tif, cr_th)
    imwrite(fname_PNR_tif, pnr_th)


