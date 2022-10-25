import os
import h5py
import dask as da
import numpy as np
import xarray as xr
import functools as fct


from typing import Tuple, Optional, Callable
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation, get_optimal_chk, custom_arr_optimize, save_minian, open_minian
from minian.preprocessing import denoise, remove_background
from minian.initialization import seeds_init, pnr_refine, ks_refine, seeds_merge, initA, initC
from minian.cnmf import compute_trace, get_noise_fft, update_spatial, update_temporal, unit_merge, update_background, compute_AtC
from minian_utilities import load_doric_to_xarray, save_minian_to_doric, round_up_to_odd, round_down_to_odd
from utilities import get_frequency, load_attributes, save_attributes

from multiprocessing import freeze_support
freeze_support()


import sys

kwargs = {}
for arg in sys.argv[1:]:
    exec(arg)

dpath = os.path.join(os.path.dirname(kwargs["fname"]), "minian")
fr = get_frequency(kwargs["fname"], kwargs['h5path']+'Time')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = os.path.join(dpath, "intermediate")

#params = {
#    "CorrectMotion": bool(kwargs["CorrectMotion"]),
#    "NeuronDiameter": eval(kwargs["NeuronDiameter"]),
#    "NoiseFreq": kwargs["NoiseFreq"],
#    "ThresCorr": kwargs["ThresCorr"],
#    "SpatialPenalty": kwargs["SpatialPenalty"],
#    "TemporalPenalty": kwargs["TemporalPenalty"],
#    "SpatialDownsample": kwargs["SpatialDownsample"],
#    "TemporalDownsample": kwargs["TemporalDownsample"],
#}

params = params_doric

neuron_diameter             = tuple([params_doric["NeuronDiameterMin"], params_doric["NeuronDiameterMax"]])
noise_freq: float           = params["NoiseFreq"]
thres_corr: float           = params["ThresCorr"]
spatial_penalty: float      = params["SpatialPenalty"]
temporal_penalty: float     = params["TemporalPenalty"]
spatial_downsample: int     = params["SpatialDownsample"]
temporal_downsample: int    = params["TemporalDownsample"]

for params_, dict_ in kwargs.items():
    if type(dict_) is dict:
        for key, value in dict_.items():
            params[params_.replace('params_','')+'-'+key] = value

params_LocalCluster = dict(
    n_workers=int(os.getenv("MINIAN_NWORKERS", 4)),
    memory_limit="3GB",
    resources={"MEM": 1},
    threads_per_worker=2,
    dashboard_address=":8787",
    local_directory=dpath
)
if "params_LocalCluster" in kwargs:
    for key, value in kwargs["params_LocalCluster"].items():
        params_LocalCluster[key] = value
    del kwargs["params_LocalCluster"]

params_load_doric = {
    "fname": kwargs["fname"],
    "h5path": kwargs['h5path'],
    "dtype": np.uint8,
    "downsample": dict(frame=temporal_downsample, 
                       height=spatial_downsample, 
                       width=spatial_downsample),
    "downsample_strategy": "subset",
}

params_save_minian = {
    "dpath": os.path.join(dpath, "final"),
    "meta_dict": dict(session=-1, animal=-2),
    "overwrite": True,
}

params_get_optimal_chk = {
    "dtype": float
}
if "params_get_optimal_chk" in kwargs:
    for key, value in kwargs["params_get_optimal_chk"].items():
        params_get_optimal_chk[key] = value

params_denoise = {
    'method': 'median',
    'ksize': round_down_to_odd((neuron_diameter[0]+neuron_diameter[-1])/4.0) # half of average size
}
if "params_denoise" in kwargs:
    for key, value in kwargs["params_denoise"].items():
        params_denoise[key] = value

params_remove_background = {
    'method': 'tophat',
    'wnd': np.ceil(neuron_diameter[-1]) # largest neuron diameter
}
if "params_remove_background" in kwargs:
    for key, value in kwargs["params_remove_background"].items():
        params_remove_background[key] = value

params_estimate_motion = {
    'dim': 'frame'
}
if "params_estimate_motion" in kwargs:
    for key, value in kwargs["params_estimate_motion"].items():
        params_estimate_motion[key] = value
params_apply_transform = {
    'fill': 0
}
if "params_apply_transform" in kwargs:
    for key, value in kwargs["params_apply_transform"].items():
        params_apply_transform[key] = value

wnd = 60 # time window of 60 seconds
params_seeds_init = {
        'wnd_size': fr*wnd,
        'method': 'rolling',
        'stp_size': fr*wnd / 2,
        'max_wnd': neuron_diameter[-1],
        'diff_thres': 3
}
if "params_seeds_init" in kwargs:
    for key, value in kwargs["params_seeds_init"].items():
        params_seeds_init[key] = value
params_pnr_refine = {
    "noise_freq": noise_freq,
    "thres": 1
}
if "params_pnr_refine" in kwargs:
    for key, value in kwargs["params_pnr_refine"].items():
        params_pnr_refine[key] = value
params_ks_refine = {
    "sig": 0.05
}
if "params_ks_refine" in kwargs:
    for key, value in kwargs["params_ks_refine"].items():
        params_ks_refine[key] = value
params_seeds_merge = {
    'thres_dist': neuron_diameter[0],
    'thres_corr': thres_corr,
    'noise_freq': noise_freq
}
if "params_seeds_merge" in kwargs:
    for key, value in kwargs["params_seeds_merge"].items():
        params_seeds_merge[key] = value

params_initA = {
    'thres_corr': thres_corr,
    'wnd': neuron_diameter[-1],
    'noise_freq': noise_freq
}
if "params_initA" in kwargs:
    for key, value in kwargs["params_initA"].items():
        params_initA[key] = value
params_unit_merge = {
    'thres_corr': thres_corr
}
if "params_unit_merge" in kwargs:
    for key, value in kwargs["params_unit_merge"].items():
        params_unit_merge[key] = value

params_get_noise_fft = {
    'noise_range': (noise_freq, 0.5)
}
if "params_get_noise_fft" in kwargs:
    for key, value in kwargs["params_get_noise_fft"].items():
        params_get_noise_fft[key] = value
params_update_spatial = {
    'dl_wnd': neuron_diameter[-1],
    'sparse_penal': spatial_penalty,
    'size_thres': (np.ceil(0.9*np.pi*neuron_diameter[0]), np.ceil(1.1*np.pi*neuron_diameter[-1]**2))
}
if "params_update_spatial" in kwargs:
    for key, value in kwargs["params_update_spatial"].items():
        params_update_spatial[key] = value
params_update_temporal = {
    'noise_freq': noise_freq,
    'sparse_penal': temporal_penalty,
    'p': 1,
    'add_lag': 20,
    'jac_thres': 0.2
}
if "params_update_temporal" in kwargs:
    for key, value in kwargs["params_update_temporal"].items():
        params_update_temporal[key] = value

if __name__ == "__main__":

    # Start cluster
    print("Starting cluster...")
    cluster = LocalCluster(**params_LocalCluster)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # MiniAn CNMF
    intpath = os.path.join(dpath, "intermediate")
    subset = dict(frame=slice(0, None))

    ### Load and chunk the data ###
    print("Loading dataset to MiniAn...")
    varr, file_ = load_doric_to_xarray(**params_load_doric)
    chk, _ = get_optimal_chk(varr, **params_get_optimal_chk)
    varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"), 
                       intpath, overwrite=True)
    varr_ref = varr.sel(subset)

    ### Pre-process data ###
    print("Pre-processing...")
    # 1. Glow removal
    print("Pre-processing: removing glow...")
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    # 2. Denoise
    print("Pre-processing: denoising...")
    varr_ref = denoise(varr_ref, **params_denoise)
    # 3. Background removal
    print("Pre-processing: removing background...")
    varr_ref = remove_background(varr_ref, **params_remove_background)
    # Save
    print("Pre-processing: saving...")
    varr_ref = save_minian(varr_ref.rename("varr_ref"), intpath, overwrite=True)

    ### Motion correction ###
    if params["CorrectMotion"]:
        print("Correcting motion: estimating shifts...")
        motion = estimate_motion(varr_ref, **params_estimate_motion)
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **params_save_minian)
        print("Correcting motion: applying shifts...")
        Y = apply_transform(varr_ref, motion, **params_apply_transform)

    else:
        Y = varr_ref

    print("Preparing data for initialization...")
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    ### Seed initialization ###
    print("Initializing seeds...")
    # 1. Compute max projection
    max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **params_save_minian).compute()
    # 2. Generating over-complete set of seeds
    seeds = seeds_init(Y_fm_chk, **params_seeds_init)
    # 3. Peak-Noise-Ratio refine
    print("Initializing seeds: PNR refinement...")
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **params_pnr_refine)
    # 4. Kolmogorov-Smirnov refine
    print("Initializing seeds: Kolmogorov-Smirnov refinement...")
    seeds = ks_refine(Y_hw_chk, seeds, **params_ks_refine)
    # 5. Merge seeds
    print("Initializing seeds: merging...")
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **params_seeds_merge)
    
    ### Component initialization ###
    print("Initializing components...")
    # 1. Initialize spatial
    print("Initializing components: spatial...")
    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **params_initA)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    # 2. Initialize temporal
    print("Initializing components: temporal...")
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True, 
                         chunks={"unit_id": 1, "frame": -1})
    # 3. Merge components
    print("Initializing components: merging...")
    A, C = unit_merge(A_init, C_init, **params_unit_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True,
                        chunks={"unit_id": -1, "frame": chk["frame"]})
    # 4. Initialize background
    print("Initializing components: background...")
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    ### CNMF 1st itteration ###
    # 1. Estimate spatial noise
    print("Running CNMF 1st itteration: estimating noise...")
    sn_spatial = get_noise_fft(Y_hw_chk, **params_get_noise_fft)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    # 2. First spatial update
    print("Running CNMF 1st itteration: updating spatial components...")
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **params_update_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    # 3. Update background
    print("Running CNMF 1st itteration: updating background components...")
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    # 4. First temporal update
    print("Running CNMF 1st itteration: updating temporal components...")
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                      chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **params_update_temporal)
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)
    # 5. Merge components
    print("Running CNMF 1st itteration: merging components...")
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **params_unit_merge)
    # Save
    print("Running CNMF 1st itteration: saving intermediate results...")
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True,
                        chunks={"unit_id": -1, "frame": chk["frame"]})
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    ### CNMF 2nd itteration ###
    # 5. Second spatial update
    print("Running CNMF 2nd itteration: updating spatial components...")
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **params_update_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    # 6. Second background update
    print("Running CNMF 2nd itteration: updating background components...")
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    # 7. Second temporal update
    print("Running CNMF 2nd itteration: updating temporal components...")
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True,
                      chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **params_update_temporal)
    # Save
    print("Running CNMF 2nd itteration: saving intermediate results...")
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True,
                        chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    AC = compute_AtC(A, C_chk)

    ### Save final results ###
    print("Saving final results...")
    A = save_minian(A.rename("A"), **params_save_minian)
    C = save_minian(C.rename("C"), **params_save_minian)
    AC = save_minian(AC.rename("AC"), **params_save_minian)
    S = save_minian(S.rename("S"), **params_save_minian)
    c0 = save_minian(c0.rename("c0"), **params_save_minian)
    b0 = save_minian(b0.rename("b0"), **params_save_minian)
    b = save_minian(b.rename("b"), **params_save_minian)
    f = save_minian(f.rename("f"), **params_save_minian)

    ### Save results to doric file ###
    print("Saving data to doric file...")
    # Get the path from the source data
    h5path = params_load_doric['h5path']
    if h5path[0] == '/': 
        h5path = h5path[1:]
    if h5path[-1] == '/':
        h5path = h5path[:-1]
    h5path_names = h5path.split('/')
    data = h5path_names[0]
    driver = h5path_names[1]
    operation = h5path_names[2]
    series = h5path_names[-2]
    sensor = h5path_names[-1]
    # Get paramaters of the operation on source data
    params_source_data = load_attributes(file_, data+'/'+driver+'/'+operation)
    # Get the attributes of the images stack
    attrs = load_attributes(file_, h5path+'/ImagesStack')
    file_.close()
    
    # Parameters
    params["OperationName"] = "MiniAn"
    if "OperationName" in params_source_data:
        if "Operations" in params_source_data:
            params["Operations"] = params_source_data["Operations"] + " > MiniAn"
            del params_source_data["Operations"]
        else:
            params["Operations"] = params_source_data["OperationName"] + " > MiniAn"
        del params_source_data["OperationName"]
    params = {**params, **params_source_data}

    
    save_minian_to_doric(
        Y, A, C, AC, S,
        fr=fr,
        bits_count=attrs['BitsCount'],
        qt_format=attrs['Format'],
        vname=params_load_doric['fname'], 
        vpath='DataProcessed/'+driver+'/',
        vdataset=series+'/'+sensor+'/',
        attrs=params, 
        saveimages=True, 
        saveresiduals=True, 
        savespikes=True
    )
    
    # Close cluster
    client.close()
    cluster.close()
