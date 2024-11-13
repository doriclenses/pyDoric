import os
import cv2
import h5py
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import definitions as defs

def load_attributes(
    file_: Union[h5py.File, str],
    path: str = ""
    ) -> dict:

    """
    Loads group/dataset attributes as a dictionary

    Args:
        f: opened h5py file or full path to a file
        path: path to the group/dataset

    Returns:
        dictionary of dataset/group attributes

    Raises:
        TypeError: If file is not h5py.File or hdf5 filepath
        ValueError: If file is closed
        KeyError: If path does not exist in the file
    """

    path = clean_path(path)

    if not isinstance(file_, h5py.File):
        if not h5py.is_hdf5(file_):
            raise TypeError(defs.Messages.F_NOT_H5_FILE_FILEPATH)

    if isinstance(file_, h5py.File):
        f = file_
        if not f.__bool__():
            raise ValueError(defs.Messages.FILE_CLOSE)
    else:
        f = h5py.File(file_, 'r')

    if path not in f:
        raise KeyError(defs.Messages.DATPATH_DOESNT_EXIST.format(datasetpath = path))

    params = {}
    for key, value in f[path].attrs.items():
        if isinstance(value, np.ndarray) and value.shape == (2,) and value.size == 2:
            params[key] = tuple(value)
        elif isinstance(value, np.ndarray):
            params[key] = value.tolist()
        elif isinstance(value, bytes):
            params[key] = value.decode("utf-8")
        else:
            params[key] = value

    if not isinstance(file_, h5py.File) and h5py.is_hdf5(file_):
        f.close()

    return params


def get_frequency(
    file_: Union[h5py.File, str],
    vdataset: str
    ) -> float:

    if not isinstance(file_, h5py.File):
        if not h5py.is_hdf5(file_):
            raise TypeError(defs.Messages.F_NOT_H5_FILE_FILEPATH)

    if isinstance(file_, h5py.File):
        f = file_
        if not f.__bool__():
            raise ValueError(defs.Messages.FILE_CLOSE)
    else:
        f = h5py.File(file_, 'r')

    if vdataset not in f:
        raise KeyError(defs.Messages.DATPATH_DOESNT_EXIST.format(datasetpath = vdataset))

    if not isinstance(f[vdataset], h5py.Dataset):
        raise ValueError(defs.Messages.HAS_TO_BE_PATH.format(path = vdataset))

    t = np.array(f[vdataset])

    if len(t.shape) > 1:
        warnings.warn(defs.Messages.DATASET_NOT_TIME)

    dt = np.diff(t)
    T = np.round(np.median(dt),10)

    if not isinstance(file_, h5py.File) and h5py.is_hdf5(file_):
        f.close()

    return 1/T


def get_dims(
    file_: Union[h5py.File, str],
    path: str
    ) -> Tuple[Tuple[int, int], int]:

    path = clean_path(path)

    if not isinstance(file_, h5py.File):
        if not h5py.is_hdf5(file_):
            raise TypeError(defs.Messages.F_NOT_H5_FILE_FILEPATH)

    if isinstance(file_, h5py.File):
        f = file_
        if not f.__bool__():
            raise ValueError(defs.Messages.FILE_CLOSE)
    else:
        f = h5py.File(file_, 'r')

    if path not in f:
        raise KeyError(defs.Messages.DATPATH_DOESNT_EXIST.format(datasetpath = path))

    shape = np.array(f[path]).shape

    if not isinstance(file_, h5py.File) and h5py.is_hdf5(file_):
        f.close()

    return shape[:-1], shape[-1]

#*************************************************************** SAVE FUNCTIONS ********************************************
def save_images(
    images: np.ndarray,
    time_: np.array,
    f: h5py.File,
    path: str,
    bit_count: int = 16,
    qt_format: int = 28,
    username: Optional[str] = defs.DoricFile.Dataset.IMAGE_STACK
    ):
    """
    Saves images and time vector in HDF file as 'ImageStack' and 'Time'
    datasets in `path` group. Saves images width, height, `bit_count`, and
    `qt_format` as 'ImageStack' dataset attribute

    Args:
        images : np.ndarray
            3D images stack, with shape (frame, height, width).
        time_ : np.array
            1D vector of timestamps
        f : h5py.File
            Opened HDF file where the information should be saved
        path  : str
            Group path in the HDF file
        bit_count : int
            Bits depth of images
        qt_format : int
            QImage_Format, necessary to display images in DNS. For reference, please
            see https://doc.qt.io/qt-6/qimage.html
        username: Optional[str]
            Give an username for Danse
    """
    path = clean_path(path)

    duration, height, width = images.shape

    path_image_stack = f"{path}/{defs.DoricFile.Dataset.IMAGE_STACK}"
    if path_image_stack in f:
        del f[path_image_stack]

    dataset = f.create_dataset(path_image_stack, (height,width,duration), dtype="uint16",
                                  chunks=(height,width,1), maxshape=(height,width,None))

    for i, image in enumerate(images):
        dataset[:,:,i] = image

    dataset.attrs[defs.DoricFile.Attribute.Dataset.USERNAME] = username
    dataset.attrs[defs.DoricFile.Attribute.Image.BIT_COUNT]  = bit_count
    dataset.attrs[defs.DoricFile.Attribute.Image.FORMAT]     = qt_format
    dataset.attrs[defs.DoricFile.Attribute.Image.HEIGHT]     = height
    dataset.attrs[defs.DoricFile.Attribute.Image.WIDTH]      = width

    path_time = f"{path}/{defs.DoricFile.Dataset.TIME}"
    if path_time in f:
        del f[path_time]

    f.create_dataset(path_time, data=time_, dtype="float64", chunks=def_chunk_size(time_.shape), maxshape=None)


def save_roi_signals(
    signals: np.ndarray,
    footprints: np.ndarray,
    time_: np.ndarray,
    file_: h5py.File,
    path: str,
    ids: list[int] = [],
    dataset_names: list[str] = [],
    usernames: list[str] = [],
    attrs: Optional[dict] = {}
    ):

    """
    Saves ROI signals, time vector, and ROI coordinates.
    Parameters
    ----------
    signals : np.ndarray
        2D array of signals, with shape (n_ROI, time).
    footprints:
        3D array of spatial cell footprints with shape (n_ROI, height, width)
    time_ : np.array
        1D vector of timestamps
    f : h5py.File
        Opened HDF file where the information should be saved
    path  : str
        Group path in the HDF file
    bit_count : int
        Bits depth of images

    """
    path = clean_path(path)

    for i, footprint in enumerate(footprints):
        id_ = ids[i] if ids else i + 1

        roi_attrs = {
            defs.DoricFile.Attribute.ROI.ID:           id_,
            defs.DoricFile.Attribute.ROI.SHAPE:        0,
            defs.DoricFile.Attribute.ROI.COORDS:       footprint_to_coords(footprint),
            defs.DoricFile.Attribute.Dataset.NAME:     usernames[i] if usernames else defs.DoricFile.Dataset.ROI.format(id_),
            defs.DoricFile.Attribute.Dataset.USERNAME: usernames[i] if usernames else defs.DoricFile.Dataset.ROI.format(id_)
        }

        if attrs:
            roi_attrs = {**roi_attrs, **attrs}

        dataset_name = dataset_names[i] if dataset_names else defs.DoricFile.Dataset.ROI.format(str(id_).zfill(4))
        save_signal(signals[i], file_, f"{path}/{dataset_name}", roi_attrs)

    save_signal(time_, file_, f"{path}/{defs.DoricFile.Dataset.TIME}")


def save_signals(
    signals: np.ndarray,
    time_: np.ndarray,
    file_: h5py.File,
    path: str,
    dataset_names: list[str],
    usernames: Optional[list[str]] = [],
    attrs: Optional[dict] = {}
    ):

    path = clean_path(path)

    for i, name in enumerate(dataset_names):
        attrs[defs.DoricFile.Attribute.Dataset.USERNAME] = usernames[i] if usernames else name

        save_signal(signals[i], file_, f"{path}/{name}", attrs)

    save_signal(time_, file_, f"{path}/{defs.DoricFile.Dataset.TIME}")


def save_signal(
    signal: np.array,
    f: h5py.File,
    path: str,
    attrs: Optional[dict] = None
    ):

    if path in f:
        del f[path]

    f.create_dataset(path, data=signal, dtype="float64", chunks=def_chunk_size(signal.shape), maxshape=None)

    if attrs is not None:
        save_attributes(attrs, f, path)


def save_attributes(
    attributes: dict,
    f: h5py.File,
    path: str
    ):
    path = clean_path(path)

    for key in attributes.keys():
        try:
            f[path].attrs[key] = attributes[key]
        except:
            print(defs.Messages.CANT_SAVE_ATT_VAL.format(attribute = key, value = attributes[key]))


def footprint_to_coords(
    footprint: np.array
    ) -> np.array:

    _, mask = cv2.threshold(footprint, 0, 255, 0)
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxI = 0
    maxArea = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > maxArea:
            maxArea = cv2.contourArea(contour)
            maxI = i

    coords = np.squeeze(contours[maxI])

    return coords

#*************************************************************** PARAMETER FUNCTIONS ********************************************
def merge_params(
    params_current,
    params_source = {},
    operation_name = None
    ):

    """
    Merge parameters of the current operation and the previous ones
    """

    params_final = {}

    if operation_name is None:
        operation_name = params_current[defs.DoricFile.Attribute.Group.OPERATIONS]

    if defs.DoricFile.Attribute.Group.OPERATIONS not in params_source :
        params_final[defs.DoricFile.Attribute.Group.OPERATIONS] = operation_name
    else:
        params_final[defs.DoricFile.Attribute.Group.OPERATIONS] = f"{params_source[defs.DoricFile.Attribute.Group.OPERATIONS]} > {operation_name}"

    # Set the advanced Settings keys
    for key in params_current:
        if key == defs.DoricFile.Attribute.Group.OPERATIONS:
            continue

        if key == defs.Parameters.danse.ADVANCED_SETTINGS:
            for variable_name, variable_value in params_current[key].items():
                # MiniAn
                if isinstance(variable_value, dict):
                    for sub_variable_name, sub_variable_value in variable_value.items():
                        params_final.update(create_params_item(["Advanced", variable_name, sub_variable_name], sub_variable_value))
                # CaImAn
                else:
                    params_final.update(create_params_item(["Advanced", variable_name], variable_value))
        else:
            params_final[key] = params_current[key]

    # Add Operations operation_name- to the keys
    for key in params_final.copy():
        if key != defs.DoricFile.Attribute.Group.OPERATIONS:
            params_final[f"{operation_name}-{key}"] = params_final.pop(key)

    # Merging with params source
    for key in params_source:
        if key != defs.DoricFile.Attribute.Group.OPERATIONS:
            params_final[key] = params_source[key]

    return params_final


def create_params_item(
    key: List[str],
    value
    ):

    """
    Create dictionary instance
    """

    key_final = "-".join(key)

    if not isinstance(value, str):
        value_final = str(value)
    else:
        value_final = f'"{value}"'

    return {key_final: value_final}

#*************************************************************** PRINT FUNCTIONS ********************************************
def print_to_intercept(msg):
    if not isinstance(msg, str):
        msg = str(msg)

    print(defs.Messages.INTERCEPT_MESSAGE.format(message = msg), flush = True, end = '\n')


def print_group_path_for_DANSE(path):
    path = clean_path(path)

    print_to_intercept(defs.Messages.PATHGROUP.format(path = f"/{path}"))


def print_error(error, position):
    print(defs.Messages.ERROR_IN.format(position = position, type_error_name = type(error).__name__, error = error), flush=True)

#*************************************************************** OTHER FUNCTIONS ********************************************
def def_chunk_size(data_shape):

    if len(data_shape) == 3:
        height   = data_shape[1]
        width    = data_shape[2]

        return (height,width,1)

    if len(data_shape) == 1:
        chunk_size = 65536
        durantion = data_shape[0]

        if durantion < chunk_size:
            chunk_size = durantion

        return  (chunk_size,)


def clean_path(path):

    if path[0] == '/':
        path = path[1:]

    if path[-1] == '/':
        path = path[:-1]

    return path


def operation_count(path, file_, operation_name, params_doric, params_source):

    count = ""
    if path not in file_:
        return count

    operations = [ name for name in file_[path] if operation_name in name ]
    if len(operations) == 0:
        return count

    count = str(len(operations))
    for operation in operations:
        operation_attrs = load_attributes(file_, f"{path}/{operation}")
        if merge_params(params_doric, params_source) != operation_attrs:
            continue

        if(len(operation) == len(operation_name)):
            count = ""
        else:
            count = operation[-1]

        break

    return count